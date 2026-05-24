from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from dino_slam3.geometry.projection import unproject, transform, project


def _stratified_sample(valid: torch.Tensor, num: int, border: int = 8) -> torch.Tensor:
    B, _, H, W = valid.shape
    device = valid.device

    g = int(max(1, round(num ** 0.5)))
    ys = torch.linspace(border, H - 1 - border, g, device=device)
    xs = torch.linspace(border, W - 1 - border, g, device=device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    base = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=-1)
    if base.shape[0] < num:
        rep = int((num + base.shape[0] - 1) // base.shape[0])
        base = base.repeat(rep, 1)
    base = base[:num]

    xy_out = []
    for b in range(B):
        jitter = (torch.rand((num, 2), device=device) - 0.5) * 4.0
        xy = base + jitter
        xy[:, 0] = xy[:, 0].clamp(0, W - 1)
        xy[:, 1] = xy[:, 1].clamp(0, H - 1)

        xi = xy[:, 0].round().long()
        yi = xy[:, 1].round().long()
        is_valid = (valid[b, 0, yi, xi] > 0.5)
        if is_valid.float().mean() < 0.3:
            idx = torch.nonzero(valid[b, 0] > 0.5, as_tuple=False)
            if idx.numel() > 0:
                sel = idx[torch.randint(0, idx.shape[0], (num,), device=device)]
                xy = torch.stack([sel[:, 1].float(), sel[:, 0].float()], dim=-1)  # x,y
        xy_out.append(xy)
    return torch.stack(xy_out, dim=0)


def _sample_depth(depth: torch.Tensor, xy: torch.Tensor) -> torch.Tensor:
    B, _, H, W = depth.shape
    x = xy[..., 0].round().clamp(0, W - 1).long()
    y = xy[..., 1].round().clamp(0, H - 1).long()
    lin = y * W + x
    flat = depth[:, 0].reshape(B, -1)
    return torch.gather(flat, 1, lin)


def _xy_to_grid(xy_f: torch.Tensor, Hf: int, Wf: int) -> torch.Tensor:
    """
    xy_f: (B,N,2) in FEATURE coordinates (0..Wf-1, 0..Hf-1)
    returns grid: (B,N,1,2) in [-1,1] for grid_sample
    """
    denom_x = max(Wf - 1, 1)
    denom_y = max(Hf - 1, 1)
    gx = (xy_f[..., 0] / float(denom_x)) * 2.0 - 1.0
    gy = (xy_f[..., 1] / float(denom_y)) * 2.0 - 1.0
    return torch.stack([gx, gy], dim=-1).unsqueeze(2)


def _grid_sample_1c_from_xyf(logits: torch.Tensor, xy_f: torch.Tensor) -> torch.Tensor:
    """
    logits: (B,1,Hf,Wf)
    xy_f: (B,N,2) in feature coords (NOT normalized)
    returns: (B,N)
    """
    B, _, Hf, Wf = logits.shape
    logits = logits.float()
    grid = _xy_to_grid(xy_f.float(), Hf, Wf).to(dtype=logits.dtype)
    s = F.grid_sample(logits, grid, mode="bilinear", align_corners=True)  # (B,1,N,1)
    return s[:, 0, :, 0]


def _grid_sample_desc_from_xyf(desc: torch.Tensor, xy_f: torch.Tensor) -> torch.Tensor:
    """
    desc: (B,D,Hf,Wf)
    xy_f: (B,N,2) in feature coords (NOT normalized)
    returns: (B,N,D)
    """
    B, D, Hf, Wf = desc.shape
    desc = desc.float()
    grid = _xy_to_grid(xy_f.float(), Hf, Wf).to(dtype=desc.dtype)
    samp = F.grid_sample(desc, grid, mode="bilinear", align_corners=True)  # (B,D,N,1)
    samp = samp[:, :, :, 0].transpose(1, 2).contiguous()  # (B,N,D)
    return samp


def _gather_map_at_xy_int(m: torch.Tensor, xy_int: torch.Tensor) -> torch.Tensor:
    B, C, Hf, Wf = m.shape
    x = xy_int[..., 0].clamp(0, Wf - 1).long()
    y = xy_int[..., 1].clamp(0, Hf - 1).long()
    lin = y * Wf + x
    flat = m.reshape(B, C, -1).transpose(1, 2)  # (B,Hf*Wf,C)
    out = torch.gather(flat, 1, lin.unsqueeze(-1).expand(-1, -1, C))
    return out


def focal_bce_with_logits(logits: torch.Tensor, targets: torch.Tensor, alpha: float = 0.25, gamma: float = 2.0) -> torch.Tensor:
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p = torch.sigmoid(logits)
    p_t = p * targets + (1 - p) * (1 - targets)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    loss = alpha_t * ((1 - p_t) ** gamma) * ce
    return loss.mean()


@dataclass
class LossStats:
    num_samples: int
    num_valid: int
    valid_ratio: float


def compute_losses(
    batch: Dict[str, torch.Tensor],
    out1,
    out2,
    stride: int,
    cfg: Dict,
) -> Tuple[Dict[str, torch.Tensor], LossStats]:
    rgb1, rgb2 = batch["rgb1"], batch["rgb2"]
    depth1, depth2 = batch["depth1"], batch["depth2"]
    valid1, valid2 = batch["valid_depth1"], batch["valid_depth2"]
    K = batch["K"]
    T21 = batch["relative_pose"]

    if K.dim() == 2:
        K = K.unsqueeze(0).expand(rgb1.shape[0], -1, -1)

    B, _, H, W = rgb1.shape
    device = rgb1.device

    # Sample candidate pixels in frame1 (image coords)
    N = int(cfg["geom"]["sample_points"])
    xy1 = _stratified_sample(valid1, N, border=int(cfg["geom"].get("border", 8)))  # (B,N,2)

    # Project to frame2 using depth+pose
    pts1 = unproject(depth1, K, xy1)
    if pts1.numel() == 0 or pts1.shape[1] == 0:
        losses = {k: torch.zeros([], device=device) for k in [
            "loss_total","loss_desc","loss_det","loss_sparse","loss_offset","loss_rel"
        ]}
        stats = LossStats(num_samples=B * N, num_valid=0, valid_ratio=0.0)
        return losses, stats

    pts2 = transform(T21, pts1)
    xy2 = project(pts2, K)

    x2, y2 = xy2[..., 0], xy2[..., 1]
    inb = (x2 >= 0) & (x2 <= (W - 1)) & (y2 >= 0) & (y2 <= (H - 1)) & (pts2[..., 2] > 1e-3)

    z2 = pts2[..., 2]
    d2_obs = _sample_depth(depth2, xy2)
    depth_ok = (d2_obs <= 0.0) | (torch.abs(d2_obs - z2) < float(cfg["geom"].get("depth_consistency_m", 0.05)))

    mask = inb & depth_ok
    m = mask.float()

    num_valid = int(mask.sum().item())
    stats = LossStats(num_samples=B * N, num_valid=num_valid, valid_ratio=float(m.mean().item()))

    # Convert to FEATURE coords
    xy1_f = xy1 / float(stride)
    xy2_f = xy2 / float(stride)

    # ---------- Descriptor InfoNCE ----------
    dcfg = cfg["contrastive"]
    temperature = float(dcfg.get("temperature", 0.07))
    max_pos = int(dcfg.get("max_positives", 512))

    valid_flat = mask.view(-1)
    if valid_flat.sum() < 16:
        loss_desc = torch.tensor(0.0, device=device)
        sim_margin = torch.tensor(0.0, device=device)
    else:
        d1 = _grid_sample_desc_from_xyf(out1.desc, xy1_f).reshape(-1, out1.desc.shape[1])
        d2 = _grid_sample_desc_from_xyf(out2.desc, xy2_f).reshape(-1, out2.desc.shape[1])

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
            sim_margin = torch.tensor(0.0, device=device)
        else:
            d1v = F.normalize(d1v.float(), dim=-1, eps=1e-6).contiguous()
            d2v = F.normalize(d2v.float(), dim=-1, eps=1e-6).contiguous()

            logits = torch.mm(d1v, d2v.t()) / float(temperature)  # (M,M)
            labels = torch.arange(M, device=device)

            loss_a = F.cross_entropy(logits, labels)
            loss_b = F.cross_entropy(logits.t(), labels)
            loss_desc = 0.5 * (loss_a + loss_b)

            with torch.no_grad():
                top2 = torch.topk(logits, k=2, dim=1).values
                sim_margin = (top2[:, 0] - top2[:, 1]).mean()

    # ---------- Detector loss ----------
    heat1 = _grid_sample_1c_from_xyf(out1.heatmap, xy1_f)  # (B,N)
    det_targets1 = m
    loss_det1 = focal_bce_with_logits(
        heat1, det_targets1,
        alpha=float(cfg["detector"].get("alpha", 0.25)),
        gamma=float(cfg["detector"].get("gamma", 2.0))
    )

    heat2 = _grid_sample_1c_from_xyf(out2.heatmap, xy2_f)
    if mask.sum() > 0:
        loss_det2 = focal_bce_with_logits(heat2[mask], torch.ones_like(heat2[mask]), alpha=0.25, gamma=2.0)
    else:
        loss_det2 = torch.tensor(0.0, device=device)

    loss_det = 0.5 * (loss_det1 + loss_det2)

    # Sparsity control
    target_mean = float(cfg["detector"].get("target_mean", 0.01))
    p1 = torch.sigmoid(out1.heatmap.float()).mean()
    p2 = torch.sigmoid(out2.heatmap.float()).mean()
    loss_sparse = (p1 - target_mean).abs() + (p2 - target_mean).abs()

    # ---------- Offset loss ----------
    loss_off = torch.tensor(0.0, device=device)
    if out1.offset is not None:
        xy1_int = xy1_f.round()
        xy2_int = xy2_f.round()

        tgt1 = (xy1_f - xy1_int).detach()
        tgt2 = (xy2_f - xy2_int).detach()

        pred1 = _gather_map_at_xy_int(out1.offset.float(), xy1_int)[..., 0:2]
        pred2 = _gather_map_at_xy_int(out2.offset.float(), xy2_int)[..., 0:2]

        w = det_targets1.unsqueeze(-1)
        denom = w.sum().clamp(min=1.0)
        loss_off = (F.smooth_l1_loss(pred1, tgt1, reduction="none") * w).sum() / denom
        if mask.sum() > 0:
            w2 = mask.float().unsqueeze(-1)
            denom2 = w2.sum().clamp(min=1.0)
            loss_off = 0.5 * (loss_off + (F.smooth_l1_loss(pred2, tgt2, reduction="none") * w2).sum() / denom2)

    # ---------- Reliability loss (simple scalar regularizer) ----------
    loss_rel = torch.tensor(0.0, device=device)
    if out1.reliability is not None:
        r_mean = torch.sigmoid(out1.reliability.float()).mean()
        loss_rel = (r_mean - float(cfg["reliability"].get("target_mean", 0.1))).abs()

    losses = {
        "loss_desc": loss_desc * float(cfg["contrastive"].get("weight", 1.0)),
        "loss_det": loss_det * float(cfg["detector"].get("weight", 1.0)),
        "loss_sparse": loss_sparse * float(cfg["detector"].get("sparsity_weight", 0.2)),
        "loss_offset": loss_off * float(cfg["offset"].get("weight", 0.2)),
        "loss_rel": loss_rel * float(cfg["reliability"].get("weight", 0.05)),
        "loss_margin_dbg": sim_margin.detach() if torch.is_tensor(sim_margin) else torch.tensor(0.0, device=device),
    }
    losses["loss_total"] = losses["loss_desc"] + losses["loss_det"] + losses["loss_sparse"] + losses["loss_offset"] + losses["loss_rel"]
    return losses, stats
