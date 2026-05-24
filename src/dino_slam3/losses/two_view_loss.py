from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
import math

import torch
import torch.nn.functional as F

from dino_slam3.geometry.projection import unproject, transform, project


_DESC_MEMORY_BANK: Optional[torch.Tensor] = None

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
    total_epochs = int(cfg.get("__total_epochs", max(1, int(epoch))))

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
    pose_error_mode = str(_pick(geom, "pose_error_mode", None, "l1")).strip().lower()
    if pose_error_mode not in {"l1", "charbonnier"}:
        pose_error_mode = "l1"
    pose_charb_eps = float(_pick(geom, "pose_charb_eps", None, 1.0))

    temperature = float(_pick(contrastive, "temperature", "temperature", 0.07))
    temperature_schedule_raw = contrastive.get("temperature_schedule", None)
    adaptive_temperature_by_gap = bool(_pick(contrastive, "adaptive_temperature_by_gap", None, False))
    temperature_gap_low = float(_pick(contrastive, "temperature_gap_low", None, max(1.0e-4, temperature * 0.6)))
    temperature_gap_high = float(_pick(contrastive, "temperature_gap_high", None, temperature))
    contrastive_mode = str(_pick(contrastive, "mode", None, "infonce")).strip().lower()
    if contrastive_mode not in {"infonce", "circle", "multisim"}:
        contrastive_mode = "infonce"
    max_pos = int(_pick(contrastive, "max_positives", None, contrastive.get("num_negatives", 512)))
    min_pairs = int(_pick(contrastive, "min_pairs", None, 8))
    triplet_margin = float(_pick(contrastive, "triplet_margin", None, 0.10))
    triplet_weight = float(_pick(contrastive, "triplet_weight", None, 0.0))
    mnn_consistency_weight = float(_pick(contrastive, "mnn_consistency_weight", None, 0.0))
    hard_sample_fraction = float(_pick(contrastive, "hard_sample_fraction", None, 0.0))
    hard_mining_start_epoch = int(_pick(contrastive, "hard_mining_start_epoch", None, 1))
    hard_min_displacement_px = float(_pick(contrastive, "hard_min_displacement_px", None, 4.0))
    long_disp_px = float(_pick(contrastive, "long_disp_px", None, 0.0))
    long_weight = float(_pick(contrastive, "long_weight", None, 1.0))
    circle_margin = float(_pick(contrastive, "circle_margin", None, 0.25))
    circle_gamma = float(_pick(contrastive, "circle_gamma", None, 32.0))
    circle_pos_weight = float(_pick(contrastive, "circle_pos_weight", None, 1.0))
    circle_neg_weight = float(_pick(contrastive, "circle_neg_weight", None, 1.0))
    multisim_alpha = float(_pick(contrastive, "multisim_alpha", None, 2.0))
    multisim_beta = float(_pick(contrastive, "multisim_beta", None, 50.0))
    multisim_lambda = float(_pick(contrastive, "multisim_lambda", None, 0.5))
    memory_bank_size = int(_pick(contrastive, "memory_bank_size", None, 0))
    memory_momentum = float(_pick(contrastive, "memory_momentum", None, 0.0))
    uniformity_weight = float(_pick(contrastive, "uniformity_weight", None, 0.0))
    depth_edge_separation_weight = float(_pick(contrastive, "depth_edge_separation_weight", None, 0.0))
    gap_weight_schedule_raw = contrastive.get("gap_weight_schedule", None)
    w_desc = float(_pick(contrastive, "weight", "w_desc", 1.0))

    w_repeat = float(_pick(detector, "weight", "w_repeat", 1.0))
    w_sparse = float(_pick(detector, "sparsity_weight", "w_sparsity", 0.2))
    det_alpha = float(_pick(detector, "alpha", None, 0.25))
    det_gamma = float(_pick(detector, "gamma", None, 2.0))
    teacher_weight = float(_pick(detector, "teacher_weight", None, 0.0))
    teacher_alpha = float(_pick(detector, "teacher_alpha", None, 0.25))
    teacher_gamma = float(_pick(detector, "teacher_gamma", None, 2.0))
    saliency_consistency_weight = float(_pick(detector, "saliency_consistency_weight", None, 0.0))
    depth_edge_consistency_weight = float(_pick(detector, "depth_edge_consistency_weight", None, 0.0))
    semantic_stability_weight = float(_pick(detector, "semantic_stability_weight", None, 0.0))
    saliency_teacher_mix = float(_pick(detector, "saliency_teacher_mix", None, 0.0))
    saliency_entropy_weight = float(_pick(detector, "saliency_entropy_weight", None, 0.0))
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
    offset_enable = bool(_pick(offset_cfg, "enable", None, True))
    offset_soft_target_mix = float(_pick(offset_cfg, "soft_target_mix", None, 0.5))
    offset_smoothness_weight = float(_pick(offset_cfg, "smoothness_weight", None, 0.0))
    offset_bias_weight = float(_pick(offset_cfg, "bias_weight", None, 0.0))
    w_rel = float(_pick(rel_cfg, "weight", "w_reliability", 0.05))
    rel_target = float(_pick(rel_cfg, "target_mean", None, 0.1))
    rel_mode = str(_pick(rel_cfg, "mode", None, "cosine")).lower()
    rel_pos_weight = float(_pick(rel_cfg, "pos_weight", None, 2.0))
    rel_mean_reg_weight = float(_pick(rel_cfg, "mean_reg_weight", None, 0.1))
    rel_reproj_sigma_px = float(_pick(rel_cfg, "reproj_sigma_px", None, 2.0))
    rel_hybrid_mix = float(_pick(rel_cfg, "hybrid_mix", None, 0.5))
    cycle_weight = float(_pick(geom, "cycle_weight", None, 0.0))
    cycle_max_px = float(_pick(geom, "cycle_max_px", None, 8.0))
    loop_consistency_weight = float(_pick(geom, "loop_consistency_weight", None, 0.0))
    dynamic_suppression_weight = float(_pick(geom, "dynamic_suppression_weight", None, 0.0))
    loop_min_gap = int(_pick(geom, "loop_min_gap", None, 80))
    loop_pose_dist_m = float(_pick(geom, "loop_pose_dist_m", None, 0.40))
    loop_yaw_deg = float(_pick(geom, "loop_yaw_deg", None, 20.0))
    pose_error_transition_epoch = int(_pick(geom, "pose_error_transition_epoch", None, 0))

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

    gap_bins: list[tuple[float, float, float]] = []
    if isinstance(gap_weight_schedule_raw, (list, tuple)):
        for item in gap_weight_schedule_raw:
            if isinstance(item, dict):
                lo = float(item.get("min_disp_px", item.get("min", 0.0)))
                hi = float(item.get("max_disp_px", item.get("max", 1.0e9)))
                w = float(item.get("weight", 1.0))
            elif isinstance(item, (list, tuple)) and len(item) >= 3:
                lo = float(item[0])
                hi = float(item[1])
                w = float(item[2])
            else:
                continue
            if not math.isfinite(lo):
                lo = 0.0
            if not math.isfinite(hi):
                hi = 1.0e9
            if hi < lo:
                lo, hi = hi, lo
            gap_bins.append((lo, hi, w))
    if gap_bins:
        gap_bins.sort(key=lambda x: (x[0], x[1]))

    temp_start = None
    temp_end = None
    if isinstance(temperature_schedule_raw, dict):
        try:
            temp_start = float(temperature_schedule_raw.get("start"))
            temp_end = float(temperature_schedule_raw.get("end"))
        except Exception:
            temp_start = None
            temp_end = None
    scheduled_temperature = float(temperature)
    if temp_start is not None and temp_end is not None:
        t = 0.0 if total_epochs <= 1 else float(epoch - 1) / float(total_epochs - 1)
        t = float(min(max(t, 0.0), 1.0))
        scheduled_temperature = (1.0 - t) * float(temp_start) + t * float(temp_end)

    effective_pose_mode = str(pose_error_mode)
    if pose_error_transition_epoch > 0 and int(epoch) >= int(pose_error_transition_epoch):
        effective_pose_mode = "charbonnier"

    def _pose_point_error(dxy_px: torch.Tensor) -> torch.Tensor:
        sq = (dxy_px * dxy_px).sum(dim=-1)
        if effective_pose_mode == "charbonnier":
            eps = max(1.0e-6, float(pose_charb_eps))
            return torch.sqrt(sq + (eps * eps)) - eps
        return torch.sqrt(sq + 1e-6)

    def _uniformity_term(z: torch.Tensor, max_samples: int = 512) -> torch.Tensor:
        if z.dim() != 2 or int(z.shape[0]) < 2:
            return torch.tensor(0.0, device=device)
        if int(z.shape[0]) > int(max_samples):
            idx = torch.randperm(int(z.shape[0]), device=z.device)[: int(max_samples)]
            z = z[idx]
        sq_d = torch.pdist(z, p=2.0).pow(2)
        if int(sq_d.numel()) == 0:
            return torch.tensor(0.0, device=device)
        return torch.log(torch.exp(-2.0 * sq_d).mean().clamp(min=1e-8))

    def _depth_edge_map(depth: torch.Tensor, valid: torch.Tensor, out_h: int, out_w: int) -> torch.Tensor:
        # Depth edge saliency target in [0,1].
        d = depth.float()
        v = (valid.float() > 0.5).float()
        dx = torch.abs(d[..., 1:] - d[..., :-1])
        dy = torch.abs(d[..., 1:, :] - d[..., :-1, :])
        vx = v[..., 1:] * v[..., :-1]
        vy = v[..., 1:, :] * v[..., :-1, :]
        dx = dx * vx
        dy = dy * vy
        dx = F.pad(dx, (0, 1, 0, 0), value=0.0)
        dy = F.pad(dy, (0, 0, 0, 1), value=0.0)
        e = dx + dy
        e = torch.log1p(20.0 * e)
        e = F.interpolate(e, size=(int(out_h), int(out_w)), mode="bilinear", align_corners=False)
        vmax = e.amax(dim=(2, 3), keepdim=True).clamp(min=1e-6)
        return (e / vmax).clamp(0.0, 1.0)

    def _depth_edge_descriptor_term(
        desc: torch.Tensor,
        edge_map: torch.Tensor,
    ) -> torch.Tensor:
        # Encourage descriptor separation across strong depth edges.
        d = F.normalize(desc.float(), dim=1, eps=1e-6)
        ex = 0.5 * (edge_map[..., 1:] + edge_map[..., :-1])
        ey = 0.5 * (edge_map[..., 1:, :] + edge_map[..., :-1, :])
        sim_x = (d[..., 1:] * d[..., :-1]).sum(dim=1, keepdim=True)
        sim_y = (d[..., 1:, :] * d[..., :-1, :]).sum(dim=1, keepdim=True)
        sep = (F.relu(sim_x - 0.2) * ex).mean() + (F.relu(sim_y - 0.2) * ey).mean()
        smooth = ((1.0 - sim_x) * (1.0 - ex)).mean() + ((1.0 - sim_y) * (1.0 - ey)).mean()
        return 0.5 * sep + 0.1 * smooth

    def _render_sparse_targets(
        xy_f: torch.Tensor,
        weights: torch.Tensor,
        out_h: int,
        out_w: int,
    ) -> torch.Tensor:
        # xy_f: (B,N,2), weights: (B,N)
        bsz = int(xy_f.shape[0])
        npts = int(xy_f.shape[1])
        out = torch.zeros((bsz, 1, int(out_h), int(out_w)), device=xy_f.device, dtype=torch.float32)
        if npts <= 0:
            return out
        xi = torch.clamp(xy_f[..., 0].round().long(), 0, int(out_w) - 1)
        yi = torch.clamp(xy_f[..., 1].round().long(), 0, int(out_h) - 1)
        for b in range(bsz):
            idx = yi[b] * int(out_w) + xi[b]
            flat = out[b, 0].view(-1)
            flat.index_put_((idx,), weights[b].float(), accumulate=True)
        return out.clamp_(0.0, 1.0)

    def _gap_pair_weights(disp: torch.Tensor) -> torch.Tensor:
        w = torch.ones_like(disp, dtype=torch.float32)
        if gap_bins:
            for i, (lo, hi, wi) in enumerate(gap_bins):
                if i == len(gap_bins) - 1:
                    sel = disp >= float(lo)
                else:
                    sel = (disp >= float(lo)) & (disp < float(hi))
                if bool(sel.any()):
                    w = torch.where(sel, torch.full_like(w, float(wi)), w)
        elif long_weight > 1.0 and long_disp_px > 0.0:
            w = torch.where(
                disp >= float(long_disp_px),
                torch.full_like(w, float(long_weight)),
                w,
            )
        return w / w.mean().clamp(min=1e-6)

    # ---------------- descriptor objective ----------------
    valid_flat = mask.view(-1)
    if int(valid_flat.sum().item()) < max(2, min_pairs):
        loss_desc = torch.tensor(0.0, device=device)
    else:
        d1 = grid_sample_desc(out1.desc, xy1_f).reshape(-1, out1.desc.shape[1])
        d2 = grid_sample_desc(out2.desc, xy2_f).reshape(-1, out2.desc.shape[1])
        disp_flat = torch.linalg.norm((xy2 - xy1).float(), dim=-1).reshape(-1)

        d1v = d1[valid_flat]
        d2v = d2[valid_flat]
        disp_v = disp_flat[valid_flat]

        M = int(d1v.shape[0])
        if M > max_pos:
            hard_frac = float(min(max(hard_sample_fraction, 0.0), 1.0))
            if epoch < int(max(1, hard_mining_start_epoch)):
                hard_frac = 0.0
            if hard_frac > 0.0:
                n_hard_target = int(round(float(max_pos) * hard_frac))
                hard_pool = torch.nonzero(disp_v >= float(hard_min_displacement_px), as_tuple=False).squeeze(1)
                if hard_pool.numel() > 0 and n_hard_target > 0:
                    n_hard = int(min(max_pos, n_hard_target, int(hard_pool.numel())))
                    hard_sel = hard_pool[torch.randperm(int(hard_pool.numel()), device=device)[:n_hard]]
                    sel_mask = torch.zeros((M,), device=device, dtype=torch.bool)
                    sel_mask[hard_sel] = True
                    rem_pool = torch.nonzero(~sel_mask, as_tuple=False).squeeze(1)
                    n_rest = int(max_pos - n_hard)
                    if n_rest > 0 and rem_pool.numel() > 0:
                        rem_sel = rem_pool[torch.randperm(int(rem_pool.numel()), device=device)[:n_rest]]
                        idx = torch.cat([hard_sel, rem_sel], dim=0)
                    else:
                        idx = hard_sel
                    if idx.numel() < max_pos:
                        remaining = torch.nonzero(~sel_mask, as_tuple=False).squeeze(1)
                        need = int(max_pos - idx.numel())
                        if need > 0 and remaining.numel() > 0:
                            extra = remaining[torch.randperm(int(remaining.numel()), device=device)[:need]]
                            idx = torch.cat([idx, extra], dim=0)
                else:
                    idx = torch.randperm(M, device=device)[:max_pos]
            else:
                idx = torch.randperm(M, device=device)[:max_pos]
            d1v = d1v[idx]
            d2v = d2v[idx]
            disp_v = disp_v[idx]
            M = int(d1v.shape[0])

        if M == 0:
            loss_desc = torch.tensor(0.0, device=device)
        else:
            global _DESC_MEMORY_BANK
            d1v = F.normalize(d1v.float(), dim=-1, eps=1e-6).contiguous()
            d2v = F.normalize(d2v.float(), dim=-1, eps=1e-6).contiguous()
            pair_weights = _gap_pair_weights(disp_v.float())
            temp_vec = torch.full(
                (int(d1v.shape[0]),),
                float(max(1e-6, scheduled_temperature)),
                device=device,
                dtype=torch.float32,
            )
            if adaptive_temperature_by_gap and int(disp_v.numel()) > 0:
                low = float(max(1.0e-6, min(temperature_gap_low, temperature_gap_high)))
                high = float(max(low, temperature_gap_high))
                ref = float(max(1.0, long_disp_px))
                frac = torch.clamp(disp_v.float() / ref, min=0.0, max=1.0)
                # Larger displacement -> lower temperature (sharper positives).
                temp_vec = high + (low - high) * frac

            sim = torch.mm(d1v, d2v.t())  # (M,M), cosine similarity
            labels = torch.arange(M, device=device)
            eye = torch.eye(M, device=device, dtype=torch.bool)
            neg_bank = None
            if memory_bank_size > 0 and _DESC_MEMORY_BANK is not None and _DESC_MEMORY_BANK.numel() > 0:
                bank = _DESC_MEMORY_BANK
                if bank.device != d1v.device:
                    bank = bank.to(device=d1v.device, dtype=d1v.dtype)
                if bank.shape[1] == d1v.shape[1]:
                    neg_bank = F.normalize(bank.float(), dim=-1, eps=1e-6)

            if contrastive_mode == "circle":
                pos = sim.diag()  # (M,)
                neg = sim.masked_fill(eye, -1e9)
                if neg_bank is not None and int(neg_bank.shape[0]) > 0:
                    neg_extra = torch.mm(d1v, neg_bank.t())
                    neg = torch.cat([neg, neg_extra], dim=1)
                ap = torch.clamp_min(1.0 + float(circle_margin) - pos.detach(), 0.0)
                an = torch.clamp_min(neg.detach() + float(circle_margin), 0.0)
                delta_p = 1.0 - float(circle_margin)
                delta_n = float(circle_margin)
                gamma = float(max(1e-6, circle_gamma))

                logit_p = -gamma * ap * (pos - delta_p) * float(max(1e-6, circle_pos_weight))
                logit_n = gamma * an * (neg - delta_n) * float(max(1e-6, circle_neg_weight))

                lse_n_row = torch.logsumexp(logit_n, dim=1)
                lse_n_col = torch.logsumexp(logit_n, dim=0)
                loss_row = F.softplus(logit_p + lse_n_row)
                loss_col = F.softplus(logit_p + lse_n_col)
                loss_desc = 0.5 * ((loss_row * pair_weights).mean() + (loss_col * pair_weights).mean())
            elif contrastive_mode == "multisim":
                alpha = float(max(1e-6, multisim_alpha))
                beta = float(max(1e-6, multisim_beta))
                lam = float(multisim_lambda)
                pos = sim.diag().unsqueeze(1)
                neg = sim.masked_fill(eye, -1e9)
                if neg_bank is not None and int(neg_bank.shape[0]) > 0:
                    neg_extra = torch.mm(d1v, neg_bank.t())
                    neg = torch.cat([neg, neg_extra], dim=1)
                pos_term = torch.log1p(torch.exp(-alpha * (pos - lam)).sum(dim=1)) / alpha
                neg_term = torch.log1p(torch.exp(beta * (neg - lam)).sum(dim=1)) / beta
                loss_desc = ((pos_term + neg_term) * pair_weights).mean()
            else:
                logits_row = sim / temp_vec.unsqueeze(1).clamp(min=1e-6)
                if neg_bank is not None and int(neg_bank.shape[0]) > 0:
                    logits_bank = torch.mm(d1v, neg_bank.t()) / temp_vec.unsqueeze(1).clamp(min=1e-6)
                    logits_row = torch.cat([logits_row, logits_bank], dim=1)
                logits_col = sim.t() / temp_vec.unsqueeze(1).clamp(min=1e-6)
                loss_a = F.cross_entropy(logits_row, labels, reduction="none")
                loss_b = F.cross_entropy(logits_col, labels, reduction="none")
                loss_desc = 0.5 * ((loss_a * pair_weights).mean() + (loss_b * pair_weights).mean())

            # Extra hard-negative pressure improves descriptor ranking for inlier metrics.
            if triplet_weight > 0.0 and M > 1:
                pos = sim.diag()
                row_hard = sim.masked_fill(eye, -1e9).max(dim=1).values
                col_hard = sim.masked_fill(eye, -1e9).max(dim=0).values
                tri_row = F.relu(float(triplet_margin) - pos + row_hard).mean()
                tri_col = F.relu(float(triplet_margin) - pos + col_hard).mean()
                loss_desc = loss_desc + float(triplet_weight) * 0.5 * (tri_row + tri_col)
                if mnn_consistency_weight > 0.0:
                    logits_for_mnn = sim / float(max(1e-6, scheduled_temperature))
                    p12 = F.softmax(logits_for_mnn, dim=1)
                    p21 = F.softmax(logits_for_mnn.t(), dim=1).t()
                    loss_mnn = F.smooth_l1_loss(p12, p21, reduction="mean")
                    loss_desc = loss_desc + float(mnn_consistency_weight) * loss_mnn

            if uniformity_weight > 0.0:
                uni = 0.5 * (_uniformity_term(d1v) + _uniformity_term(d2v))
                loss_desc = loss_desc + float(uniformity_weight) * uni

            if memory_bank_size > 0:
                cur = d2v.detach()
                if (
                    _DESC_MEMORY_BANK is not None
                    and _DESC_MEMORY_BANK.numel() > 0
                    and _DESC_MEMORY_BANK.shape[1] == cur.shape[1]
                    and memory_momentum > 0.0
                ):
                    bank_prev = _DESC_MEMORY_BANK.to(device=cur.device, dtype=cur.dtype)
                    n_blend = int(min(bank_prev.shape[0], cur.shape[0]))
                    if n_blend > 0:
                        cur = cur.clone()
                        cur[:n_blend] = F.normalize(
                            float(memory_momentum) * bank_prev[:n_blend]
                            + float(1.0 - memory_momentum) * cur[:n_blend],
                            dim=-1,
                            eps=1e-6,
                        )
                    merged = torch.cat([cur, bank_prev], dim=0)
                else:
                    merged = cur if _DESC_MEMORY_BANK is None else torch.cat([cur, _DESC_MEMORY_BANK.to(device=cur.device, dtype=cur.dtype)], dim=0)
                _DESC_MEMORY_BANK = merged[: int(memory_bank_size)].detach().cpu()
            else:
                _DESC_MEMORY_BANK = None

    if depth_edge_separation_weight > 0.0:
        edge1_desc = _depth_edge_map(depth1, valid1, out1.desc.shape[-2], out1.desc.shape[-1])
        edge2_desc = _depth_edge_map(depth2, valid2, out2.desc.shape[-2], out2.desc.shape[-1])
        depth_sep = 0.5 * (
            _depth_edge_descriptor_term(out1.desc, edge1_desc)
            + _depth_edge_descriptor_term(out2.desc, edge2_desc)
        )
        loss_desc = loss_desc + float(depth_edge_separation_weight) * depth_sep

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

    if teacher_weight > 0.0:
        teacher_heat1 = batch.get("teacher_heatmap1")
        teacher_heat2 = batch.get("teacher_heatmap2")
        if torch.is_tensor(teacher_heat1) and torch.is_tensor(teacher_heat2):
            t1 = teacher_heat1.float()
            t2 = teacher_heat2.float()
            if t1.dim() == 3:
                t1 = t1.unsqueeze(1)
            if t2.dim() == 3:
                t2 = t2.unsqueeze(1)
            if t1.shape[-2:] != out1.heatmap.shape[-2:]:
                t1 = F.interpolate(t1, size=out1.heatmap.shape[-2:], mode="nearest")
            if t2.shape[-2:] != out2.heatmap.shape[-2:]:
                t2 = F.interpolate(t2, size=out2.heatmap.shape[-2:], mode="nearest")
            t1 = t1.clamp(0.0, 1.0)
            t2 = t2.clamp(0.0, 1.0)
            teacher_loss = 0.5 * (
                focal_bce_with_logits(out1.heatmap, t1, alpha=teacher_alpha, gamma=teacher_gamma)
                + focal_bce_with_logits(out2.heatmap, t2, alpha=teacher_alpha, gamma=teacher_gamma)
            )
            loss_repeat = loss_repeat + float(teacher_weight) * teacher_loss

    if saliency_consistency_weight > 0.0:
        h1, w1 = int(out1.heatmap.shape[-2]), int(out1.heatmap.shape[-1])
        h2, w2 = int(out2.heatmap.shape[-2]), int(out2.heatmap.shape[-1])
        geom_t1 = _render_sparse_targets(xy1_f.detach(), m.detach(), h1, w1)
        geom_t2 = _render_sparse_targets(xy2_f.detach(), m.detach(), h2, w2)

        mix = float(min(max(saliency_teacher_mix, 0.0), 1.0))
        teacher_heat1 = batch.get("teacher_heatmap1")
        teacher_heat2 = batch.get("teacher_heatmap2")
        if mix > 0.0 and torch.is_tensor(teacher_heat1) and torch.is_tensor(teacher_heat2):
            tt1 = teacher_heat1.float()
            tt2 = teacher_heat2.float()
            if tt1.dim() == 3:
                tt1 = tt1.unsqueeze(1)
            if tt2.dim() == 3:
                tt2 = tt2.unsqueeze(1)
            if tt1.shape[-2:] != out1.heatmap.shape[-2:]:
                tt1 = F.interpolate(tt1, size=out1.heatmap.shape[-2:], mode="nearest")
            if tt2.shape[-2:] != out2.heatmap.shape[-2:]:
                tt2 = F.interpolate(tt2, size=out2.heatmap.shape[-2:], mode="nearest")
            geom_t1 = (1.0 - mix) * geom_t1 + mix * tt1.clamp(0.0, 1.0)
            geom_t2 = (1.0 - mix) * geom_t2 + mix * tt2.clamp(0.0, 1.0)

        sal_loss = 0.5 * (
            F.binary_cross_entropy_with_logits(out1.heatmap.float(), geom_t1, reduction="mean")
            + F.binary_cross_entropy_with_logits(out2.heatmap.float(), geom_t2, reduction="mean")
        )
        loss_repeat = loss_repeat + float(saliency_consistency_weight) * sal_loss

    if semantic_stability_weight > 0.0:
        p1m = torch.sigmoid(heat1.float())
        p2m = torch.sigmoid(heat2.float())
        with torch.no_grad():
            d1_sem = grid_sample_desc(out1.desc.detach(), xy1_f)
            d2_sem = grid_sample_desc(out2.desc.detach(), xy2_f)
            d1_sem = F.normalize(d1_sem, dim=-1, eps=1e-6)
            d2_sem = F.normalize(d2_sem, dim=-1, eps=1e-6)
            sem_pen = (1.0 - (d1_sem * d2_sem).sum(dim=-1)).clamp(min=0.0, max=2.0)
        denom_sem = m.sum().clamp(min=1.0)
        stab = ((0.5 * (p1m - p2m).abs() + 0.5 * sem_pen) * m).sum() / denom_sem
        loss_repeat = loss_repeat + float(semantic_stability_weight) * stab

    if saliency_entropy_weight > 0.0:
        p_h1 = torch.sigmoid(out1.heatmap.float()).clamp(min=1e-6, max=1.0 - 1e-6)
        p_h2 = torch.sigmoid(out2.heatmap.float()).clamp(min=1e-6, max=1.0 - 1e-6)
        ent1 = -(p_h1 * p_h1.log() + (1.0 - p_h1) * (1.0 - p_h1).log()).mean()
        ent2 = -(p_h2 * p_h2.log() + (1.0 - p_h2) * (1.0 - p_h2).log()).mean()
        loss_repeat = loss_repeat + float(saliency_entropy_weight) * 0.5 * (ent1 + ent2)

    if depth_edge_consistency_weight > 0.0:
        edge1 = _depth_edge_map(depth1, valid1, out1.heatmap.shape[-2], out1.heatmap.shape[-1])
        edge2 = _depth_edge_map(depth2, valid2, out2.heatmap.shape[-2], out2.heatmap.shape[-1])
        depth_sal = 0.5 * (
            F.binary_cross_entropy_with_logits(out1.heatmap.float(), edge1, reduction="mean")
            + F.binary_cross_entropy_with_logits(out2.heatmap.float(), edge2, reduction="mean")
        )
        loss_repeat = loss_repeat + float(depth_edge_consistency_weight) * depth_sal

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

    if dynamic_suppression_weight > 0.0:
        unstable = (inb & ~(d2_valid & depth_ok & fb_ok)).float()
        denom_dyn = unstable.sum().clamp(min=1.0)
        dyn_score = (
            (torch.sigmoid(heat1.float()) * unstable).sum()
            + (torch.sigmoid(heat2.float()) * unstable).sum()
        ) / (2.0 * denom_dyn)
        loss_repeat = loss_repeat + float(dynamic_suppression_weight) * dyn_score

    # Sparsity regularizer (global)
    p1 = torch.sigmoid(out1.heatmap.float()).mean()
    p2 = torch.sigmoid(out2.heatmap.float()).mean()
    loss_sparsity = (p1 - target_mean).abs() + (p2 - target_mean).abs()

    # ---------------- offset refinement loss ----------------
    loss_refine = torch.tensor(0.0, device=device)
    if (
        offset_enable
        and getattr(out1, "offset", None) is not None
        and out1.offset is not None
        and getattr(out2, "offset", None) is not None
        and out2.offset is not None
    ):
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

        if offset_smoothness_weight > 0.0:
            def _tv(o: torch.Tensor) -> torch.Tensor:
                dx = (o[..., 1:] - o[..., :-1]).abs().mean()
                dy = (o[..., 1:, :] - o[..., :-1, :]).abs().mean()
                return dx + dy

            tv = 0.5 * (_tv(out1.offset.float()) + _tv(out2.offset.float()))
            loss_refine = loss_refine + float(offset_smoothness_weight) * tv
        if offset_bias_weight > 0.0:
            bias = 0.5 * (out1.offset.float().mean(dim=(1, 2, 3)).abs().mean() + out2.offset.float().mean(dim=(1, 2, 3)).abs().mean())
            loss_refine = loss_refine + float(offset_bias_weight) * bias

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

                if rel_mode in {"reproj", "hybrid", "geom_correctness"}:
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
                elif rel_mode == "geom_correctness":
                    geom_target = (inb & d2_valid & depth_ok & fb_ok).float()
                    if rel_target_reproj is None:
                        rel_target_map = geom_target
                    else:
                        rel_target_map = (geom_target * rel_target_reproj).clamp(0.0, 1.0)
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
    loss_cycle = torch.tensor(0.0, device=device)
    loss_loop = torch.tensor(0.0, device=device)
    if (w_pose > 0.0 or cycle_weight > 0.0 or loop_consistency_weight > 0.0) and mask.sum() > 0:
        q_desc = grid_sample_desc(out1.desc, xy1_f)  # (B,N,D)
        soft_xy2 = _soft_refine(
            out2.desc,
            centers_f=xy2_f.detach(),
            query_desc=q_desc,
            window=soft_refine_window,
        )
        denom = m.sum().clamp(min=1.0)
        if w_pose > 0.0:
            dxy_px = (soft_xy2 - xy2_f.detach()) * float(stride)
            e = _pose_point_error(dxy_px)
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
        if cycle_weight > 0.0:
            q_desc_back = grid_sample_desc(out2.desc, soft_xy2)
            soft_xy1 = _soft_refine(
                out1.desc,
                centers_f=xy1_f.detach(),
                query_desc=q_desc_back,
                window=soft_refine_window,
            )
            cyc_err = torch.linalg.norm((soft_xy1 - xy1_f.detach()) * float(stride), dim=-1)
            if cycle_max_px > 0.0:
                cyc_err = torch.clamp(cyc_err, max=float(cycle_max_px))
            loss_cycle = float(cycle_weight) * ((cyc_err * m).sum() / denom)
        if loop_consistency_weight > 0.0:
            with torch.no_grad():
                # Identify loop-like pairs in the batch.
                t_rel = torch.linalg.norm(T21[:, :3, 3].float(), dim=-1)
                r_rel = T21[:, :3, :3].float()
                yaw_rel = torch.rad2deg(torch.atan2(r_rel[:, 1, 0], r_rel[:, 0, 0]).abs())
                frame_delta = batch.get("frame_delta")
                if torch.is_tensor(frame_delta):
                    delta_abs = frame_delta.to(device=device).float().abs().reshape(-1)
                    if int(delta_abs.numel()) != int(B):
                        delta_abs = torch.full((B,), float(loop_min_gap), device=device)
                else:
                    delta_abs = torch.full((B,), float(loop_min_gap), device=device)
                is_loop = (
                    (delta_abs >= float(max(1, loop_min_gap)))
                    & (t_rel <= float(max(0.0, loop_pose_dist_m)))
                    & (yaw_rel <= float(max(0.0, loop_yaw_deg)))
                )
            if bool(is_loop.any()):
                lm = m * is_loop.float().unsqueeze(1)
                denom_loop = lm.sum().clamp(min=1.0)
                reproj_loop = torch.linalg.norm((soft_xy2 - xy2_f.detach()) * float(stride), dim=-1)
                d2_soft = grid_sample_desc(out2.desc, soft_xy2)
                qn = F.normalize(q_desc, dim=-1, eps=1e-6)
                d2n = F.normalize(d2_soft, dim=-1, eps=1e-6)
                desc_loop = (1.0 - (qn * d2n).sum(dim=-1)).clamp(min=0.0)
                loop_obj = ((reproj_loop + 2.0 * desc_loop) * lm).sum() / denom_loop
                loss_loop = float(loop_consistency_weight) * loop_obj

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
        e_top = _pose_point_error(dxy_top_px)
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
        "loss_pose": (loss_pose * w_pose) + loss_cycle + loss_loop,
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
