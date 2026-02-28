from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
import math

import torch
import torch.nn.functional as F

from dino_slam3.geometry.projection import unproject, transform, project

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
    occlusion_ratio: float = 0.0


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
    def stratified_sample(
        valid: torch.Tensor,
        num: int,
        border: int = 8,
        heatmap: Optional[torch.Tensor] = None,
        guided_ratio: float = 0.0,
    ) -> torch.Tensor:
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

            # Mix in detector-guided samples so training aligns with evaluation keypoints.
            if heatmap is not None and guided_ratio > 0.0:
                Hf, Wf = heatmap.shape[-2:]
                vf = F.interpolate(valid[b : b + 1], size=(Hf, Wf), mode="nearest")[0, 0] > 0.5
                hf = torch.sigmoid(heatmap[b, 0].float())
                bf = int(max(1, round(border / max(float(stride), 1.0))))
                if bf > 0:
                    hf[:bf, :] = -1e9
                    hf[-bf:, :] = -1e9
                    hf[:, :bf] = -1e9
                    hf[:, -bf:] = -1e9
                flat = hf.reshape(-1)
                mask_flat = vf.reshape(-1)
                if bool(mask_flat.any()):
                    scores = flat.masked_fill(~mask_flat, -1e9)
                    n_guided = int(min(num, max(1, round(num * float(guided_ratio)))))
                    k = int(min(n_guided, int(mask_flat.sum().item())))
                    if k > 0:
                        idx_top = torch.topk(scores, k=k, dim=0, largest=True).indices
                        y = torch.div(idx_top, Wf, rounding_mode="floor")
                        x = idx_top - y * Wf
                        guided_xy = torch.stack([x.float() * float(stride), y.float() * float(stride)], dim=-1)
                        guided_xy[:, 0] = guided_xy[:, 0].clamp(0.0, float(Wv - 1))
                        guided_xy[:, 1] = guided_xy[:, 1].clamp(0.0, float(Hv - 1))
                        xy[:k] = guided_xy
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

    def sample_depth_bilinear(depth: torch.Tensor, xy: torch.Tensor) -> torch.Tensor:
        """depth: (B,1,H,W), xy:(B,N,2) -> (B,N) bilinear sampling in pixel coords"""
        Bd, _, Hd, Wd = depth.shape
        gx = (xy[..., 0] / float(max(Wd - 1, 1))) * 2.0 - 1.0
        gy = (xy[..., 1] / float(max(Hd - 1, 1))) * 2.0 - 1.0
        grid = torch.stack([gx, gy], dim=-1).unsqueeze(2)  # (B,N,1,2)
        z = F.grid_sample(depth.float(), grid.float(), mode="bilinear", align_corners=True)
        return z[:, 0, :, 0]

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

    def epipolar_distance_px(
        xy1_img: torch.Tensor,  # (B,N,2)
        xy2_img: torch.Tensor,  # (B,N,2)
        Kb: torch.Tensor,       # (B,3,3)
        T21b: torch.Tensor,     # (B,4,4)
    ) -> torch.Tensor:
        """Symmetric-free epipolar distance of x2 to line F x1 in pixels."""
        out = []
        Bn = int(xy1_img.shape[0])
        for b in range(Bn):
            x1 = xy1_img[b].float()
            x2 = xy2_img[b].float()
            x1h = torch.cat([x1, torch.ones_like(x1[:, :1])], dim=-1)  # (N,3)
            x2h = torch.cat([x2, torch.ones_like(x2[:, :1])], dim=-1)  # (N,3)

            R = T21b[b, :3, :3].float()
            t = T21b[b, :3, 3].float()
            tx = torch.zeros((3, 3), device=xy1_img.device, dtype=torch.float32)
            tx[0, 1] = -t[2]
            tx[0, 2] = t[1]
            tx[1, 0] = t[2]
            tx[1, 2] = -t[0]
            tx[2, 0] = -t[1]
            tx[2, 1] = t[0]
            E = tx @ R
            Kinv = torch.linalg.inv(Kb[b].float())
            Fm = Kinv.t() @ E @ Kinv  # (3,3)

            l2 = x1h @ Fm.t()  # (N,3)
            numer = torch.abs((x2h * l2).sum(dim=-1))
            denom = torch.sqrt(l2[:, 0] * l2[:, 0] + l2[:, 1] * l2[:, 1] + 1e-6)
            out.append(numer / denom)
        return torch.stack(out, dim=0)

    # ---------------- cfg defaults ----------------
    geom = cfg.get("geom", {})
    contrastive = cfg.get("contrastive", {})
    detector = cfg.get("detector", {})
    offset_cfg = cfg.get("offset", {})
    rel_cfg = cfg.get("reliability", {})

    def _pick(
        nested: Dict[str, Any],
        nested_key: str,
        legacy_key: Optional[str],
        default: Any,
    ) -> Any:
        if nested_key in nested:
            return nested[nested_key]
        if legacy_key is not None and legacy_key in cfg:
            return cfg[legacy_key]
        return default

    N = int(_pick(geom, "sample_points", "sample_points", 1024))
    border = int(_pick(geom, "border", "border", 8))
    depth_cons_m = float(_pick(geom, "depth_consistency_m", "depth_consistency_m", 0.05))
    depth_cons_rel = float(_pick(geom, "depth_consistency_rel", None, 0.03))
    guided_ratio = float(_pick(geom, "heatmap_guided_ratio", None, 0.0))
    soft_refine_window = int(_pick(geom, "soft_refine_window", None, 5))
    if soft_refine_window % 2 == 0:
        soft_refine_window += 1
    require_valid_depth2 = bool(_pick(geom, "require_valid_depth2", None, True))
    fb_consistency_px = float(_pick(geom, "fb_consistency_px", None, 2.0))
    z_min_m = float(_pick(cfg, "z_min_m", None, 0.10))
    w_pose = float(_pick(geom, "pose_weight", "w_pose", 0.0))
    w_epipolar = float(_pick(geom, "epipolar_weight", None, 0.0))
    pose_det_weight = float(_pick(geom, "pose_det_weight", None, 0.0))
    pose_det_topk = int(_pick(geom, "pose_det_topk", None, 256))

    temperature = float(_pick(contrastive, "temperature", "temperature", 0.07))
    max_pos = int(_pick(contrastive, "max_positives", None, contrastive.get("num_negatives", 512)))
    min_pairs = int(_pick(contrastive, "min_pairs", None, 8))
    triplet_margin = float(_pick(contrastive, "triplet_margin", None, 0.10))
    triplet_weight = float(_pick(contrastive, "triplet_weight", None, 0.0))
    mnn_consistency_weight = float(_pick(contrastive, "mnn_consistency_weight", None, 0.0))
    w_desc = float(_pick(contrastive, "weight", "w_desc", 1.0))

    w_repeat = float(_pick(detector, "weight", "w_repeat", 1.0))
    w_sparse = float(_pick(detector, "sparsity_weight", "w_sparsity", 0.2))
    det_alpha = float(_pick(detector, "alpha", None, 0.25))
    det_gamma = float(_pick(detector, "gamma", None, 2.0))
    target_mean = float(_pick(detector, "target_mean", None, 0.01))
    peak_w = float(_pick(detector, "peakiness_weight", None, 0.0))
    peak_margin = float(_pick(detector, "peak_margin", None, 0.1))
    coverage_weight = float(_pick(detector, "coverage_weight", None, 0.0))
    coverage_target = float(_pick(detector, "coverage_target", None, 0.35))
    coverage_tile = int(_pick(detector, "coverage_tile", None, 4))
    coverage_thresh = float(_pick(detector, "coverage_thresh", None, 0.30))
    entropy_weight = float(_pick(detector, "entropy_weight", None, 0.0))
    entropy_target = float(_pick(detector, "entropy_target", None, 0.65))

    w_refine = float(_pick(offset_cfg, "weight", "w_refine", 0.2))
    offset_soft_target_mix = float(_pick(offset_cfg, "soft_target_mix", None, 0.5))
    w_rel = float(_pick(rel_cfg, "weight", "w_reliability", 0.05))
    rel_target = float(_pick(rel_cfg, "target_mean", None, 0.1))
    rel_mode = str(_pick(rel_cfg, "mode", None, "cosine")).lower()
    rel_pos_weight = float(_pick(rel_cfg, "pos_weight", None, 2.0))
    rel_mean_reg_weight = float(_pick(rel_cfg, "mean_reg_weight", None, 0.1))
    rel_reproj_sigma_px = float(_pick(rel_cfg, "reproj_sigma_px", None, 2.0))
    rel_hybrid_mix = float(_pick(rel_cfg, "hybrid_mix", None, 0.5))

    # ---------------- geometry correspondences ----------------
    xy1 = stratified_sample(
        valid1,
        N,
        border=border,
        heatmap=out1.heatmap if guided_ratio > 0.0 else None,
        guided_ratio=guided_ratio,
    )  # (B,N,2)
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
        stats = LossStats(num_samples=B * N, num_valid=0, valid_ratio=0.0, occlusion_ratio=0.0)
        return losses, stats

    pts2 = transform(T21, pts1)
    xy2 = project(pts2, K)

    x2, y2 = xy2[..., 0], xy2[..., 1]
    inb = (x2 >= 0) & (x2 <= (W - 1)) & (y2 >= 0) & (y2 <= (H - 1)) & (pts2[..., 2] > 1e-3)

    z2 = pts2[..., 2]
    d2_obs = sample_depth_bilinear(depth2, xy2)
    valid2_obs = sample_depth_bilinear(valid2, xy2) > 0.5
    d2_valid = torch.isfinite(d2_obs) & (d2_obs > z_min_m)
    if require_valid_depth2:
        d2_valid = d2_valid & valid2_obs

    depth_bound = depth_cons_m + depth_cons_rel * torch.abs(z2)
    depth_ok = torch.abs(d2_obs - z2) < depth_bound

    fb_ok = torch.ones_like(inb, dtype=torch.bool)
    if fb_consistency_px > 0.0:
        fx = K[:, 0, 0].unsqueeze(1)
        fy = K[:, 1, 1].unsqueeze(1)
        cx = K[:, 0, 2].unsqueeze(1)
        cy = K[:, 1, 2].unsqueeze(1)
        X2 = (x2 - cx) * d2_obs / fx
        Y2 = (y2 - cy) * d2_obs / fy
        pts2_obs = torch.stack([X2, Y2, d2_obs], dim=-1)
        T12 = torch.linalg.inv(T21)
        pts1_back = transform(T12, pts2_obs)
        xy1_back = project(pts1_back, K)
        fb_err = torch.linalg.norm(xy1_back - xy1, dim=-1)
        fb_ok = fb_err < float(fb_consistency_px)

    mask = inb & d2_valid & depth_ok & fb_ok
    m = mask.float()
    num_valid = int(mask.sum().item())
    denom_occ = inb.float().sum().clamp(min=1.0)
    occlusion_ratio = float(((inb & ~mask).float().sum() / denom_occ).item())
    stats = LossStats(
        num_samples=B * N,
        num_valid=num_valid,
        valid_ratio=float(m.mean().item()),
        occlusion_ratio=occlusion_ratio,
    )

    # Convert to feature coords
    xy1_f = xy1 / float(stride)
    xy2_f = xy2 / float(stride)

    # ---------------- descriptor InfoNCE (stable: uses torch.mm only) ----------------
    valid_flat = mask.view(-1)
    if int(valid_flat.sum().item()) < max(2, min_pairs):
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

            # Extra hard-negative pressure improves descriptor ranking for inlier metrics.
            if triplet_weight > 0.0 and M > 1:
                sim = torch.mm(d1v, d2v.t())  # cosine similarity (unnormalized temp)
                eye = torch.eye(M, device=device, dtype=torch.bool)
                pos = sim.diag()
                row_hard = sim.masked_fill(eye, -1e9).max(dim=1).values
                col_hard = sim.masked_fill(eye, -1e9).max(dim=0).values
                tri_row = F.relu(float(triplet_margin) - pos + row_hard).mean()
                tri_col = F.relu(float(triplet_margin) - pos + col_hard).mean()
                loss_desc = loss_desc + float(triplet_weight) * 0.5 * (tri_row + tri_col)
                if mnn_consistency_weight > 0.0:
                    p12 = F.softmax(logits, dim=1)
                    p21 = F.softmax(logits.t(), dim=1).t()
                    loss_mnn = F.smooth_l1_loss(p12, p21, reduction="mean")
                    loss_desc = loss_desc + float(mnn_consistency_weight) * loss_mnn

    # ---------------- detector/repeatability loss ----------------
    heat1 = grid_sample_1c(out1.heatmap, xy1_f)  # (B,N)
    loss_rep1 = focal_bce_with_logits(heat1, m, alpha=det_alpha, gamma=det_gamma)

    heat2 = grid_sample_1c(out2.heatmap, xy2_f)
    if mask.sum() > 0:
        loss_rep2 = focal_bce_with_logits(heat2[mask], torch.ones_like(heat2[mask]), alpha=det_alpha, gamma=det_gamma)
    else:
        loss_rep2 = torch.tensor(0.0, device=device)

    loss_repeat = 0.5 * (loss_rep1 + loss_rep2)

    if peak_w > 0.0:
        neigh = torch.tensor(
            [[-1.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.0, 1.0], [-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]],
            device=device,
            dtype=xy1_f.dtype,
        )

        def _peak_penalty(logits: torch.Tensor, xy_f: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
            c = grid_sample_1c(logits, xy_f)
            neigh_vals = []
            for dxy in neigh:
                neigh_vals.append(grid_sample_1c(logits, xy_f + dxy.view(1, 1, 2)))
            nmax = torch.stack(neigh_vals, dim=0).amax(dim=0)
            penalty = F.relu(nmax + float(peak_margin) - c)
            denom = weights.sum().clamp(min=1.0)
            return (penalty * weights).sum() / denom

        loss_peak = 0.5 * (
            _peak_penalty(out1.heatmap, xy1_f, m)
            + _peak_penalty(out2.heatmap, xy2_f, m)
        )
        loss_repeat = loss_repeat + float(peak_w) * loss_peak

    if coverage_weight > 0.0:
        def _coverage_loss(logits: torch.Tensor) -> torch.Tensor:
            p = torch.sigmoid(logits.float())
            t = max(1, int(coverage_tile))
            pooled = F.max_pool2d(p, kernel_size=t, stride=t)
            occ = torch.sigmoid((pooled - float(coverage_thresh)) / 0.05).mean()
            return (occ - float(coverage_target)).abs()
        loss_repeat = loss_repeat + float(coverage_weight) * 0.5 * (
            _coverage_loss(out1.heatmap) + _coverage_loss(out2.heatmap)
        )

    if entropy_weight > 0.0:
        def _entropy_loss(logits: torch.Tensor) -> torch.Tensor:
            p = torch.sigmoid(logits.float()).reshape(B, -1)
            p = p / p.sum(dim=1, keepdim=True).clamp(min=1e-8)
            ent = -(p * p.clamp(min=1e-8).log()).sum(dim=1) / max(math.log(max(p.shape[1], 2)), 1e-8)
            return (ent - float(entropy_target)).abs().mean()
        loss_repeat = loss_repeat + float(entropy_weight) * 0.5 * (
            _entropy_loss(out1.heatmap) + _entropy_loss(out2.heatmap)
        )

    # Sparsity regularizer (global)
    p1 = torch.sigmoid(out1.heatmap.float()).mean()
    p2 = torch.sigmoid(out2.heatmap.float()).mean()
    loss_sparsity = (p1 - target_mean).abs() + (p2 - target_mean).abs()

    # ---------------- offset refinement loss ----------------
    loss_refine = torch.tensor(0.0, device=device)
    if getattr(out1, "offset", None) is not None and out1.offset is not None:
        xy2_int = xy2_f.round()
        pred2 = gather_map_at_xy_int(out2.offset.float(), xy2_int)[..., 0:2]

        if mask.sum() > 0:
            tgt2 = xy2_f.detach()
            if offset_soft_target_mix > 0.0:
                with torch.no_grad():
                    q_desc = grid_sample_desc(out1.desc.detach(), xy1_f)
                    soft_xy2 = _soft_refine(
                        out2.desc.detach(),
                        centers_f=xy2_int.float(),
                        query_desc=q_desc,
                        window=soft_refine_window,
                    )
                    mix = float(min(max(offset_soft_target_mix, 0.0), 1.0))
                    tgt2 = (1.0 - mix) * tgt2 + mix * soft_xy2
            tgt2 = (tgt2 - xy2_int).clamp(min=-0.5, max=0.5)

            w2 = mask.float().unsqueeze(-1)
            denom2 = w2.sum().clamp(min=1.0)
            loss_refine = (F.smooth_l1_loss(pred2, tgt2, reduction="none") * w2).sum() / denom2

    # ---------------- reliability / uncertainty ----------------
    loss_rel = torch.tensor(0.0, device=device)
    if getattr(out1, "reliability", None) is not None and out1.reliability is not None:
        r_mean = torch.sigmoid(out1.reliability.float()).mean()
        mean_reg = (r_mean - rel_target).abs()

        if rel_mode in {"none", "off"}:
            loss_rel = mean_reg
        else:
            rel_logits = grid_sample_1c(out1.reliability, xy1_f)
            with torch.no_grad():
                rel_target_cos = None
                rel_target_reproj = None
                if rel_mode in {"cosine", "hybrid"}:
                    d1_rel = grid_sample_desc(out1.desc.detach(), xy1_f)
                    d2_rel = grid_sample_desc(out2.desc.detach(), xy2_f)
                    d1_rel = F.normalize(d1_rel, dim=-1, eps=1e-6)
                    d2_rel = F.normalize(d2_rel, dim=-1, eps=1e-6)
                    cos_sim = (d1_rel * d2_rel).sum(dim=-1)  # (B,N), in [-1,1]
                    rel_target_cos = ((cos_sim + 1.0) * 0.5).clamp(0.0, 1.0)

                if rel_mode in {"reproj", "hybrid"}:
                    q_desc_rel = grid_sample_desc(out1.desc.detach(), xy1_f)
                    soft_xy2_rel = _soft_refine(
                        out2.desc.detach(),
                        centers_f=xy2_f.detach(),
                        query_desc=q_desc_rel,
                        window=soft_refine_window,
                    )
                    err_px = torch.linalg.norm((soft_xy2_rel - xy2_f.detach()) * float(stride), dim=-1)
                    sigma = max(float(rel_reproj_sigma_px), 1e-3)
                    rel_target_reproj = torch.exp(-0.5 * (err_px / sigma) ** 2).clamp(0.0, 1.0)

                if rel_mode == "reproj":
                    rel_target_map = rel_target_reproj
                elif rel_mode == "hybrid":
                    mix = float(min(max(rel_hybrid_mix, 0.0), 1.0))
                    if rel_target_cos is None:
                        rel_target_map = rel_target_reproj
                    elif rel_target_reproj is None:
                        rel_target_map = rel_target_cos
                    else:
                        rel_target_map = (1.0 - mix) * rel_target_cos + mix * rel_target_reproj
                else:
                    rel_target_map = rel_target_cos

                if rel_target_map is None:
                    rel_target_map = torch.zeros_like(m)
                rel_target_map = rel_target_map * m
                rel_weights = torch.where(
                    rel_target_map > 0.5,
                    torch.full_like(rel_target_map, rel_pos_weight),
                    torch.ones_like(rel_target_map),
                )

            rel_bce = F.binary_cross_entropy_with_logits(
                rel_logits,
                rel_target_map,
                weight=rel_weights,
                reduction="mean",
            )
            loss_rel = rel_bce + rel_mean_reg_weight * mean_reg

    loss_pose = torch.tensor(0.0, device=device)
    if w_pose > 0.0 and mask.sum() > 0:
        q_desc = grid_sample_desc(out1.desc, xy1_f)  # (B,N,D)
        soft_xy2 = _soft_refine(
            out2.desc,
            centers_f=xy2_f.detach(),
            query_desc=q_desc,
            window=soft_refine_window,
        )
        dxy_px = (soft_xy2 - xy2_f.detach()) * float(stride)
        e = torch.sqrt((dxy_px * dxy_px).sum(dim=-1) + 1e-6)
        denom = m.sum().clamp(min=1.0)
        loss_pose = (e * m).sum() / denom
        if w_epipolar > 0.0:
            epi = epipolar_distance_px(
                xy1_img=xy1.detach(),
                xy2_img=soft_xy2 * float(stride),
                Kb=K,
                T21b=T21,
            )
            epi_loss = (epi * m).sum() / denom
            loss_pose = loss_pose + float(w_epipolar) * epi_loss

    if w_pose > 0.0 and pose_det_weight > 0.0:
        with torch.no_grad():
            Hf, Wf = out1.heatmap.shape[-2:]
            flat = torch.sigmoid(out1.heatmap.float()).reshape(B, -1)
            v1f = F.interpolate(valid1.float(), size=(Hf, Wf), mode="nearest").reshape(B, -1) > 0.5
            scores = flat.masked_fill(~v1f, -1e9)
            Ksel = int(min(max(8, pose_det_topk), Hf * Wf))
            idx_top = torch.topk(scores, k=Ksel, dim=1, largest=True).indices
            yy = torch.div(idx_top, Wf, rounding_mode="floor")
            xx = idx_top - yy * Wf
            xy1_top = torch.stack([xx.float(), yy.float()], dim=-1) * float(stride)

            pts1_top = unproject(depth1, K, xy1_top)
            pts2_top = transform(T21, pts1_top)
            xy2_top = project(pts2_top, K)
            inb_top = (
                (xy2_top[..., 0] >= 0.0) & (xy2_top[..., 0] <= float(W - 1)) &
                (xy2_top[..., 1] >= 0.0) & (xy2_top[..., 1] <= float(H - 1)) &
                (pts2_top[..., 2] > z_min_m)
            )
            d2_top = sample_depth_bilinear(depth2, xy2_top)
            v2_top = sample_depth_bilinear(valid2, xy2_top) > 0.5
            top_mask = inb_top & v2_top & (d2_top > z_min_m)

        xy1_top_f = xy1_top / float(stride)
        xy2_top_f = xy2_top / float(stride)
        q_desc_top = grid_sample_desc(out1.desc, xy1_top_f)
        soft_xy2_top = _soft_refine(
            out2.desc,
            centers_f=xy2_top_f.detach(),
            query_desc=q_desc_top,
            window=soft_refine_window,
        )
        dxy_top_px = (soft_xy2_top - xy2_top_f.detach()) * float(stride)
        e_top = torch.sqrt((dxy_top_px * dxy_top_px).sum(dim=-1) + 1e-6)
        m_top = top_mask.float()
        denom_top = m_top.sum().clamp(min=1.0)
        det_pose_loss = (e_top * m_top).sum() / denom_top
        if w_epipolar > 0.0:
            epi_top = epipolar_distance_px(
                xy1_img=xy1_top.detach(),
                xy2_img=soft_xy2_top * float(stride),
                Kb=K,
                T21b=T21,
            )
            det_pose_loss = det_pose_loss + float(w_epipolar) * (epi_top * m_top).sum() / denom_top
        loss_pose = loss_pose + float(pose_det_weight) * det_pose_loss

    # ---------------- total ----------------
    losses = {
        "loss_desc": loss_desc * w_desc,
        "loss_repeat": loss_repeat * w_repeat,
        "loss_sparsity": loss_sparsity * w_sparse,
        "loss_refine": loss_refine * w_refine,
        "loss_rel": loss_rel * w_rel,
        "loss_pose": loss_pose * w_pose,
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
