from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from dino_slam3.geometry.projection import unproject, transform, project

def _sample_depth_at_xy(depth: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    depth: (B,1,H,W)
    x,y: (B,N)
    returns: (B,N) depth
    """
    B, _, H, W = depth.shape
    xi = x.round().clamp(0, W - 1).long()
    yi = y.round().clamp(0, H - 1).long()

    depth_hw = depth[:, 0]                # (B,H,W)
    depth_flat = depth_hw.reshape(B, -1)  # (B,H*W)
    lin = yi * W + xi                     # (B,N)
    return torch.gather(depth_flat, 1, lin)

def _sample_valid_pixels(valid: torch.Tensor, num: int, border: int = 16) -> torch.Tensor:
    """
    valid: (B,1,H,W) float/bool
    Returns xy: (B,num,2) in pixel coords.

    Fix:
      - avoids sampling from image borders where padding/artifacts often cause collapse
    """
    B, _, H, W = valid.shape
    xy_out = []
    for b in range(B):
        v = valid[b, 0] > 0.5

        # apply border margin
        if border > 0 and H > 2 * border and W > 2 * border:
            v[:border, :] = False
            v[-border:, :] = False
            v[:, :border] = False
            v[:, -border:] = False

        idx = torch.nonzero(v, as_tuple=False)  # (M,2) [y,x]
        if idx.numel() == 0:
            ys = torch.randint(border, max(border + 1, H - border), (num,), device=valid.device)
            xs = torch.randint(border, max(border + 1, W - border), (num,), device=valid.device)
        else:
            sel = idx[torch.randint(0, idx.shape[0], (num,), device=valid.device)]
            ys, xs = sel[:, 0], sel[:, 1]

        xy_out.append(torch.stack([xs.float(), ys.float()], dim=-1))
    return torch.stack(xy_out, dim=0)


def _grid_sample_desc(desc: torch.Tensor, xy: torch.Tensor) -> torch.Tensor:
    """desc: (B,D,Hf,Wf). xy in image pixels at full res (H,W).
    Assumes desc resolution matches full res / stride s. We infer scale from shapes.
    Returns (B,N,D).
    """
    B, D, Hf, Wf = desc.shape
    # Convert xy to feature coords
    # full resolution H,W is unknown; infer scale using ratio with K? Instead pass xy already at feature res.
    # Here we assume xy is at full resolution (same as input images).
    # Map to normalized coords for grid_sample.
    # Compute scale s so that H = Hf*s. But H isn't available, so accept that xy is already in feature coords
    # if user pre-scales. In our pipeline we DO pre-scale.
    x = xy[...,0]
    y = xy[...,1]
    # normalized [-1,1]
    gx = (x / (Wf - 1)) * 2 - 1
    gy = (y / (Hf - 1)) * 2 - 1
    grid = torch.stack([gx, gy], dim=-1).unsqueeze(2)  # B,N,1,2
    samp = F.grid_sample(desc, grid, mode="bilinear", align_corners=True)  # B,D,N,1
    samp = samp.squeeze(-1).transpose(1,2)  # B,N,D
    return F.normalize(samp, dim=-1, eps=1e-6)


def _grid_sample_map(m: torch.Tensor, xy: torch.Tensor) -> torch.Tensor:
    """m: (B,1,Hf,Wf) map, xy in feature coords."""
    B, _, Hf, Wf = m.shape
    x, y = xy[...,0], xy[...,1]
    gx = (x / (Wf - 1)) * 2 - 1
    gy = (y / (Hf - 1)) * 2 - 1
    grid = torch.stack([gx, gy], dim=-1).unsqueeze(2)
    samp = F.grid_sample(m, grid, mode="bilinear", align_corners=True)  # B,1,N,1
    return samp[:,0,:,0]  # B,N


def contrastive_info_nce(
    desc1: torch.Tensor,
    desc2: torch.Tensor,
    temperature: float,
    num_negatives: int,
) -> torch.Tensor:
    """
    Memory-safe InfoNCE.
    desc1, desc2: (B,N,D) positives aligned by index.
    Negatives are sampled by indexing desc2 directly (no (B,N,N,D) expansion).
    """
    B, N, D = desc1.shape
    pos = (desc1 * desc2).sum(dim=-1) / temperature  # (B,N)

    with torch.no_grad():
        neg_idx = torch.randint(0, N, (B, N, num_negatives), device=desc1.device)

    b_idx = torch.arange(B, device=desc1.device).view(B, 1, 1).expand(B, N, num_negatives)
    neg = desc2[b_idx, neg_idx]  # (B,N,K,D)

    neg_logits = (desc1.unsqueeze(2) * neg).sum(dim=-1) / temperature  # (B,N,K)

    logits = torch.cat([pos.unsqueeze(-1), neg_logits], dim=-1)  # (B,N,1+K)
    labels = torch.zeros((B, N), dtype=torch.long, device=desc1.device)
    return F.cross_entropy(logits.view(-1, logits.shape[-1]), labels.view(-1))



def repeatability_loss(h1: torch.Tensor, h2: torch.Tensor, xy1_f: torch.Tensor, xy2_f: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """
    Masked repeatability loss.
    """
    s1 = torch.sigmoid(_grid_sample_map(h1, xy1_f))
    s2 = torch.sigmoid(_grid_sample_map(h2, xy2_f))
    per = (s1 - s2).abs()  # (B,N)
    denom = m.sum().clamp(min=1.0)
    return (per * m).sum() / denom



    return (per * m).sum() / denom

def offset_regression_loss(offset: torch.Tensor, xy_f: torch.Tensor, xy_target_f: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """
    Masked offset regression loss.
    offset: (B,2,Hf,Wf), xy_f/xy_target_f: (B,N,2), m: (B,N)
    """
    B, _, Hf, Wf = offset.shape
    x, y = xy_f[..., 0], xy_f[..., 1]
    gx = (x / (Wf - 1)) * 2 - 1
    gy = (y / (Hf - 1)) * 2 - 1
    grid = torch.stack([gx, gy], dim=-1).unsqueeze(2)  # (B,N,1,2)

    samp = F.grid_sample(offset, grid, mode="bilinear", align_corners=True)  # (B,2,N,1)
    samp = samp.squeeze(-1).transpose(1, 2)  # (B,N,2)

    target_delta = (xy_target_f - xy_f).detach()  # (B,N,2)

    per = F.smooth_l1_loss(samp, target_delta, reduction="none").sum(dim=-1)  # (B,N)
    denom = m.sum().clamp(min=1.0)
    return (per * m).sum() / denom

def entropy_collapse_loss(heat: torch.Tensor) -> torch.Tensor:
    """
    Positive loss that penalizes low-entropy heatmaps.
    Uses softmax distribution over pixels.
    """
    B, _, H, W = heat.shape
    logits = heat.view(B, -1)
    p = torch.softmax(logits, dim=-1)  # (B,HW)
    ent = -(p * (p + 1e-12).log()).sum(dim=-1)  # (B,)
    max_ent = float(torch.log(torch.tensor(H * W, device=heat.device, dtype=heat.dtype)))
    return (max_ent - ent).mean()  # 0 is best, higher is worse


def entropy_regularizer(heat: torch.Tensor) -> torch.Tensor:
    """Prevent collapse: encourage spread out activations."""
    B, _, H, W = heat.shape
    p = torch.sigmoid(heat).view(B, -1)
    p = p / (p.sum(dim=-1, keepdim=True) + 1e-6)
    ent = -(p * (p + 1e-9).log()).sum(dim=-1).mean()
    return -ent  # maximize entropy -> minimize negative entropy


def uniformity_regularizer(heat: torch.Tensor) -> torch.Tensor:
    """Encourage spatial uniformity (weak) by penalizing strong low-frequency bias."""
    p = torch.sigmoid(heat)
    mean_row = p.mean(dim=-1)
    mean_col = p.mean(dim=-2)
    return mean_row.var() + mean_col.var()

def grid_coverage_loss(heat: torch.Tensor, bins: int = 8) -> torch.Tensor:
    """
    Penalize concentration in one region (e.g., top band) by encouraging uniform mass over a coarse grid.
    """
    B, _, H, W = heat.shape
    p = torch.softmax(heat.view(B, -1), dim=-1).view(B, 1, H, W)  # probability mass map

    # pool mass into bins x bins
    ph = F.adaptive_avg_pool2d(p, (bins, bins)).view(B, -1)  # (B, bins*bins)
    ph = ph / (ph.sum(dim=-1, keepdim=True) + 1e-12)

    target = torch.full_like(ph, 1.0 / ph.shape[-1])
    return F.mse_loss(ph, target)


def compute_losses(
    batch: Dict[str, torch.Tensor],
    out1,
    out2,
    stride: int,
    cfg: Dict,
) -> Dict[str, torch.Tensor]:
    """
    Stage-1 depth+pose supervision with masked geometry.

    Fixes:
      - border-aware sampling to stop top/border collapse
      - safe depth2 sampling (no broken advanced indexing)
      - masked contrastive / repeatability / offset
      - regularizers that are positive + non-constant
    """
    rgb1, rgb2 = batch["rgb1"], batch["rgb2"]
    depth1, depth2 = batch["depth1"], batch["depth2"]
    valid1 = batch["valid_depth1"]
    K = batch["K"]
    T21 = batch["relative_pose"]

    B, _, H, W = rgb1.shape

    # depth filtering
    min_d = float(cfg["geom"]["min_depth"])
    max_d = float(cfg["geom"]["max_depth"])
    v1 = valid1 * (depth1 >= min_d).float() * (depth1 <= max_d).float()

    # sample pixels at full res (with border margin)
    N = int(cfg["geom"]["sample_points"])
    border = int(cfg["geom"].get("border_margin", 16))
    xy1 = _sample_valid_pixels(v1, N, border=border)  # (B,N,2)

    # unproject, transform, project
    pts1 = unproject(depth1, K, xy1)         # (B,N,3)
    pts2 = transform(T21, pts1)              # (B,N,3)
    xy2 = project(pts2, K)                   # (B,N,2)

    # in-bounds + positive depth
    x2, y2 = xy2[..., 0], xy2[..., 1]
    inb = (x2 >= 0) & (x2 <= (W - 1)) & (y2 >= 0) & (y2 <= (H - 1)) & (pts2[..., 2] > 1e-3)

    # depth agreement in frame2 (optional)
    d2_obs = _sample_depth_at_xy(depth2, x2, y2)  # (B,N)
    z2 = pts2[..., 2]
    depth_tol = float(cfg["geom"].get("depth_tolerance", 0.05))
    reproj_ok = (torch.abs(d2_obs - z2) < depth_tol) | (d2_obs <= 0.0)
    mask = inb & reproj_ok

    # feature coords
    xy1_f = xy1 / float(stride)
    xy2_f = xy2 / float(stride)

    m = mask.float()  # (B,N)

    # descriptors
    d1 = _grid_sample_desc(out1.desc, xy1_f)  # (B,N,D)
    d2 = _grid_sample_desc(out2.desc, xy2_f)  # (B,N,D)

    # masked contrastive: flatten valid across batch
    c_cfg = cfg["contrastive"]
    temperature = float(c_cfg["temperature"])
    num_neg = int(c_cfg["num_negatives"])

    valid_flat = (m > 0.5)
    if valid_flat.sum() < 8:
        con = torch.tensor(0.0, device=rgb1.device)
    else:
        d1v = d1[valid_flat]  # (Nv,D)
        d2v = d2[valid_flat]  # (Nv,D)
        con = contrastive_info_nce(d1v.unsqueeze(0), d2v.unsqueeze(0), temperature, num_neg)

    # masked repeatability
    rep = repeatability_loss(out1.heatmap, out2.heatmap, xy1_f, xy2_f, m)

    # masked offset regression
    off_loss = torch.tensor(0.0, device=rgb1.device)
    if getattr(out1, "offset", None) is not None:
        off_loss = offset_regression_loss(out1.offset, xy1_f, xy2_f, m)

    # regularizers (positive, non-constant)
    reg_cfg = cfg["regularizers"]
    ent_w = float(reg_cfg.get("entropy_weight", 1.0))
    cov_w = float(reg_cfg.get("coverage_weight", 1.0))
    bins = int(reg_cfg.get("coverage_bins", 8))

    reg_loss = (
        ent_w * (entropy_collapse_loss(out1.heatmap) + entropy_collapse_loss(out2.heatmap)) +
        cov_w * (grid_coverage_loss(out1.heatmap, bins=bins) + grid_coverage_loss(out2.heatmap, bins=bins))
    )

    losses = {
        "loss_contrastive": con * float(c_cfg["weight"]),
        "loss_repeat": rep * float(cfg["repeatability"]["weight"]),
        "loss_offset": off_loss * float(cfg["offset"]["weight"]) if "offset" in cfg else off_loss,
        "loss_reg": reg_loss * float(reg_cfg.get("weight", 1.0)),
    }
    losses["loss_total"] = sum(losses.values())
    return losses