from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from dino_slam3.geometry.projection import unproject, transform, project
from dino_slam3.geometry.se3 import se3_error
from dino_slam3.slam.keypoints_torch import extract_keypoints_torch, KeypointsTorch

import kornia

def _depth_at(depth: torch.Tensor, xy: torch.Tensor) -> torch.Tensor:
    """
    depth: (B,1,H,W)
    xy: (B,N,2) pixels
    returns z: (B,N)
    """
    B, _, H, W = depth.shape
    x = xy[..., 0].round().clamp(0, W - 1).long()
    y = xy[..., 1].round().clamp(0, H - 1).long()
    z = depth[:, 0].reshape(B, -1).gather(1, y * W + x)
    return z

def _soft_refine(
    desc2_map: torch.Tensor,
    centers_f: torch.Tensor,   # (B,M,2) feature coords
    query_desc: torch.Tensor,  # (B,M,D)
    window: int,
) -> torch.Tensor:
    """
    Parameter-free local correlation + soft-argmax refinement on the descriptor map.

    returns refined feature coords (B,M,2)
    """
    B, D, Hf, Wf = desc2_map.shape
    w = int(window)
    r = w // 2

    dx = torch.arange(-r, r + 1, device=desc2_map.device).float()
    dy = torch.arange(-r, r + 1, device=desc2_map.device).float()
    grid_y, grid_x = torch.meshgrid(dy, dx, indexing="ij")
    disp = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=-1)  # (w*w,2)

    M = centers_f.shape[1]
    coords = centers_f.unsqueeze(2) + disp.view(1, 1, -1, 2)  # (B,M,w*w,2)

    gx = (coords[..., 0] / max(Wf - 1, 1)) * 2 - 1
    gy = (coords[..., 1] / max(Hf - 1, 1)) * 2 - 1
    grid = torch.stack([gx, gy], dim=-1)  # (B,M,w*w,2)

    # grid_sample expects (B, outH, outW, 2); use outH=M, outW=w*w
    sampled = F.grid_sample(desc2_map.float(), grid.float(), mode="bilinear", align_corners=True)
    # sampled: (B,D,M,w*w)
    sampled = sampled.permute(0, 2, 3, 1)  # (B,M,w*w,D)

    q = query_desc.unsqueeze(2)  # (B,M,1,D)
    corr = (sampled * q).sum(dim=-1)  # (B,M,w*w)
    p = F.softmax(corr, dim=-1)

    disp_xy = disp.view(1, 1, -1, 2)  # (1,1,w*w,2)
    delta = (p.unsqueeze(-1) * disp_xy).sum(dim=2)  # (B,M,2)
    return centers_f + delta


@dataclass
class LossStats:
    num_samples: int
    num_valid: int
    valid_ratio: float


def compute_losses(
    batch: Dict[str, torch.Tensor],
    out1,
    out2,
    cfg: Dict,
    epoch: int,
    stride: int,
) -> Tuple[Dict[str, torch.Tensor], LossStats]:
    """
    Geometry-supervised 2-view training loss that avoids any strided-batched GEMM
    (no einsum/bmm) to be stable on H100.

    Uses:
      - sample pixels in image1 within valid depth
      - unproject using depth + intrinsics
      - transform with relative pose
      - project to image2
      - sample descriptors/heatmap/offset/reliability using grid_sample / gather
      - descriptor InfoNCE using torch.mm (single GEMM) on gathered positives

    Returns loss dict with keys:
      loss_total, loss_desc, loss_repeat, loss_sparsity, loss_refine, loss_rel, loss_pose
    """

    device = batch["rgb1"].device
    B, _, H, W = batch["rgb1"].shape

    depth1 = batch["depth1"]
    depth2 = batch["depth2"]
    valid1 = batch.get("valid_depth1", (depth1 > 0).float())
    valid2 = batch.get("valid_depth2", (depth2 > 0).float())
    K = batch["K"]
    T21 = batch["relative_pose"]

    if K.dim() == 2:
        K = K.unsqueeze(0).expand(B, -1, -1)
    if T21.dim() == 2:
        T21 = T21.unsqueeze(0).expand(B, -1, -1)

    # ---------------- helpers (self-contained) ----------------
    def stratified_sample(valid: torch.Tensor, num: int, border: int = 8) -> torch.Tensor:
        """valid: (B,1,H,W) -> xy: (B,num,2) in image coords"""
        Bv, _, Hv, Wv = valid.shape
        g = int(max(1, round(num ** 0.5)))
        ys = torch.linspace(border, Hv - 1 - border, g, device=valid.device)
        xs = torch.linspace(border, Wv - 1 - border, g, device=valid.device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        base = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=-1)

        if base.shape[0] < num:
            rep = int((num + base.shape[0] - 1) // base.shape[0])
            base = base.repeat(rep, 1)
        base = base[:num]

        out = []
        for b in range(Bv):
            jitter = (torch.rand((num, 2), device=valid.device) - 0.5) * 4.0
            xy = base + jitter
            xy[:, 0] = xy[:, 0].clamp(0, Wv - 1)
            xy[:, 1] = xy[:, 1].clamp(0, Hv - 1)

            xi = xy[:, 0].round().long()
            yi = xy[:, 1].round().long()
            ok = (valid[b, 0, yi, xi] > 0.5)

            # fallback: sample from valid pixels if too few valid
            if ok.float().mean() < 0.3:
                idx = torch.nonzero(valid[b, 0] > 0.5, as_tuple=False)
                if idx.numel() > 0:
                    sel = idx[torch.randint(0, idx.shape[0], (num,), device=valid.device)]
                    xy = torch.stack([sel[:, 1].float(), sel[:, 0].float()], dim=-1)  # x,y
            out.append(xy)
        return torch.stack(out, dim=0)

    def sample_depth(depth: torch.Tensor, xy: torch.Tensor) -> torch.Tensor:
        """depth: (B,1,H,W), xy:(B,N,2) -> (B,N) nearest-neighbor"""
        Bd, _, Hd, Wd = depth.shape
        x = xy[..., 0].round().clamp(0, Wd - 1).long()
        y = xy[..., 1].round().clamp(0, Hd - 1).long()
        lin = y * Wd + x
        flat = depth[:, 0].reshape(Bd, -1)
        return torch.gather(flat, 1, lin)

    def xy_to_grid(xy_f: torch.Tensor, Hf: int, Wf: int) -> torch.Tensor:
        """xy_f (B,N,2) in feature coords -> grid (B,N,1,2) in [-1,1]"""
        denom_x = max(Wf - 1, 1)
        denom_y = max(Hf - 1, 1)
        gx = (xy_f[..., 0] / float(denom_x)) * 2.0 - 1.0
        gy = (xy_f[..., 1] / float(denom_y)) * 2.0 - 1.0
        return torch.stack([gx, gy], dim=-1).unsqueeze(2)

    def grid_sample_1c(logits: torch.Tensor, xy_f: torch.Tensor) -> torch.Tensor:
        """logits:(B,1,Hf,Wf), xy_f:(B,N,2) -> (B,N)"""
        Bx, _, Hf, Wf = logits.shape
        logits = logits.float()
        grid = xy_to_grid(xy_f.float(), Hf, Wf).to(dtype=logits.dtype)
        s = F.grid_sample(logits, grid, mode="bilinear", align_corners=True)  # (B,1,N,1)
        return s[:, 0, :, 0]

    def grid_sample_desc(desc: torch.Tensor, xy_f: torch.Tensor) -> torch.Tensor:
        """desc:(B,D,Hf,Wf), xy_f:(B,N,2) -> (B,N,D)"""
        Bx, D, Hf, Wf = desc.shape
        desc = desc.float()
        grid = xy_to_grid(xy_f.float(), Hf, Wf).to(dtype=desc.dtype)
        samp = F.grid_sample(desc, grid, mode="bilinear", align_corners=True)  # (B,D,N,1)
        return samp[:, :, :, 0].transpose(1, 2).contiguous()  # (B,N,D)

    def gather_map_at_xy_int(m: torch.Tensor, xy_int: torch.Tensor) -> torch.Tensor:
        """m:(B,C,Hf,Wf), xy_int:(B,N,2) integer feature coords -> (B,N,C)"""
        Bm, C, Hf, Wf = m.shape
        x = xy_int[..., 0].clamp(0, Wf - 1).long()
        y = xy_int[..., 1].clamp(0, Hf - 1).long()
        lin = y * Wf + x
        flat = m.reshape(Bm, C, -1).transpose(1, 2)  # (B,Hf*Wf,C)
        return torch.gather(flat, 1, lin.unsqueeze(-1).expand(-1, -1, C))

    def focal_bce_with_logits(logits: torch.Tensor, targets: torch.Tensor, alpha=0.25, gamma=2.0) -> torch.Tensor:
        ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * ((1 - p_t) ** gamma) * ce
        return loss.mean()

    # ---------------- cfg defaults ----------------
    geom = cfg.get("geom", {})
    contrastive = cfg.get("contrastive", {})
    detector = cfg.get("detector", {})
    offset_cfg = cfg.get("offset", {})
    rel_cfg = cfg.get("reliability", {})

    N = int(geom.get("sample_points", 1024))
    border = int(geom.get("border", 8))
    depth_cons_m = float(geom.get("depth_consistency_m", 0.05))

    temperature = float(contrastive.get("temperature", 0.07))
    max_pos = int(contrastive.get("max_positives", 512))
    w_desc = float(contrastive.get("weight", 1.0))

    w_repeat = float(detector.get("weight", 1.0))
    w_sparse = float(detector.get("sparsity_weight", 0.2))
    det_alpha = float(detector.get("alpha", 0.25))
    det_gamma = float(detector.get("gamma", 2.0))
    target_mean = float(detector.get("target_mean", 0.01))

    w_refine = float(offset_cfg.get("weight", 0.2))
    w_rel = float(rel_cfg.get("weight", 0.05))
    rel_target = float(rel_cfg.get("target_mean", 0.1))

    # ---------------- geometry correspondences ----------------
    xy1 = stratified_sample(valid1, N, border=border)  # (B,N,2)
    pts1 = unproject(depth1, K, xy1)
    if pts1.numel() == 0 or pts1.shape[1] == 0:
        z = torch.zeros([], device=device)
        losses = {
            "loss_total": z,
            "loss_desc": z,
            "loss_repeat": z,
            "loss_sparsity": z,
            "loss_refine": z,
            "loss_rel": z,
            "loss_pose": z,
        }
        stats = LossStats(num_samples=B * N, num_valid=0, valid_ratio=0.0)
        return losses, stats

    pts2 = transform(T21, pts1)
    xy2 = project(pts2, K)

    x2, y2 = xy2[..., 0], xy2[..., 1]
    inb = (x2 >= 0) & (x2 <= (W - 1)) & (y2 >= 0) & (y2 <= (H - 1)) & (pts2[..., 2] > 1e-3)

    z2 = pts2[..., 2]
    d2_obs = sample_depth(depth2, xy2)
    depth_ok = (d2_obs <= 0.0) | (torch.abs(d2_obs - z2) < depth_cons_m)

    mask = inb & depth_ok
    m = mask.float()
    num_valid = int(mask.sum().item())
    stats = LossStats(num_samples=B * N, num_valid=num_valid, valid_ratio=float(m.mean().item()))

    # Convert to feature coords
    xy1_f = xy1 / float(stride)
    xy2_f = xy2 / float(stride)

    # ---------------- descriptor InfoNCE (stable: uses torch.mm only) ----------------
    valid_flat = mask.view(-1)
    if valid_flat.sum() < 16:
        loss_desc = torch.tensor(0.0, device=device)
    else:
        d1 = grid_sample_desc(out1.desc, xy1_f).reshape(-1, out1.desc.shape[1])
        d2 = grid_sample_desc(out2.desc, xy2_f).reshape(-1, out2.desc.shape[1])

        d1v = d1[valid_flat]
        d2v = d2[valid_flat]

        M = int(d1v.shape[0])
        if M > max_pos:
            idx = torch.randperm(M, device=device)[:max_pos]
            d1v = d1v[idx]
            d2v = d2v[idx]
            M = max_pos

        if M == 0:
            loss_desc = torch.tensor(0.0, device=device)
        else:
            d1v = F.normalize(d1v.float(), dim=-1, eps=1e-6).contiguous()
            d2v = F.normalize(d2v.float(), dim=-1, eps=1e-6).contiguous()

            # IMPORTANT: torch.mm only (no batched GEMM)
            logits = torch.mm(d1v, d2v.t()) / float(temperature)  # (M,M)
            labels = torch.arange(M, device=device)

            loss_a = F.cross_entropy(logits, labels)
            loss_b = F.cross_entropy(logits.t(), labels)
            loss_desc = 0.5 * (loss_a + loss_b)

    # ---------------- detector/repeatability loss ----------------
    heat1 = grid_sample_1c(out1.heatmap, xy1_f)  # (B,N)
    loss_rep1 = focal_bce_with_logits(heat1, m, alpha=det_alpha, gamma=det_gamma)

    heat2 = grid_sample_1c(out2.heatmap, xy2_f)
    if mask.sum() > 0:
        loss_rep2 = focal_bce_with_logits(heat2[mask], torch.ones_like(heat2[mask]), alpha=det_alpha, gamma=det_gamma)
    else:
        loss_rep2 = torch.tensor(0.0, device=device)

    loss_repeat = 0.5 * (loss_rep1 + loss_rep2)

    # Sparsity regularizer (global)
    p1 = torch.sigmoid(out1.heatmap.float()).mean()
    p2 = torch.sigmoid(out2.heatmap.float()).mean()
    loss_sparsity = (p1 - target_mean).abs() + (p2 - target_mean).abs()

    # ---------------- offset refinement loss ----------------
    loss_refine = torch.tensor(0.0, device=device)
    if getattr(out1, "offset", None) is not None and out1.offset is not None:
        xy1_int = xy1_f.round()
        xy2_int = xy2_f.round()

        tgt1 = (xy1_f - xy1_int).detach()
        tgt2 = (xy2_f - xy2_int).detach()

        pred1 = gather_map_at_xy_int(out1.offset.float(), xy1_int)[..., 0:2]
        pred2 = gather_map_at_xy_int(out2.offset.float(), xy2_int)[..., 0:2]

        w = m.unsqueeze(-1)
        denom = w.sum().clamp(min=1.0)
        loss_refine_1 = (F.smooth_l1_loss(pred1, tgt1, reduction="none") * w).sum() / denom

        if mask.sum() > 0:
            w2 = mask.float().unsqueeze(-1)
            denom2 = w2.sum().clamp(min=1.0)
            loss_refine_2 = (F.smooth_l1_loss(pred2, tgt2, reduction="none") * w2).sum() / denom2
            loss_refine = 0.5 * (loss_refine_1 + loss_refine_2)
        else:
            loss_refine = loss_refine_1

    # ---------------- reliability regularizer ----------------
    loss_rel = torch.tensor(0.0, device=device)
    if getattr(out1, "reliability", None) is not None and out1.reliability is not None:
        r_mean = torch.sigmoid(out1.reliability.float()).mean()
        loss_rel = (r_mean - rel_target).abs()

    # ---------------- pose loss placeholder (0) ----------------
    # (You can add pose supervision later if desired.)
    loss_pose = torch.tensor(0.0, device=device)

    # ---------------- total ----------------
    losses = {
        "loss_desc": loss_desc * w_desc,
        "loss_repeat": loss_repeat * w_repeat,
        "loss_sparsity": loss_sparsity * w_sparse,
        "loss_refine": loss_refine * w_refine,
        "loss_rel": loss_rel * w_rel,
        "loss_pose": loss_pose,
    }
    losses["loss_total"] = (
        losses["loss_desc"]
        + losses["loss_repeat"]
        + losses["loss_sparsity"]
        + losses["loss_refine"]
        + losses["loss_rel"]
        + losses["loss_pose"]
    )
    return losses, stats
