from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from dino_slam3.geometry.projection import unproject, transform, project

def _sample_depth_at_xy(depth: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    depth: (B,1,H,W)
    x,y: (B,N) in pixel coords (float OK)
    returns: (B,N)
    """
    B, _, H, W = depth.shape
    xi = x.round().clamp(0, W - 1).long()
    yi = y.round().clamp(0, H - 1).long()

    depth_hw = depth[:, 0]              # (B,H,W)
    depth_flat = depth_hw.reshape(B, -1)  # (B,H*W)
    lin = yi * W + xi                   # (B,N)
    return torch.gather(depth_flat, 1, lin)  # (B,N)


def _sample_valid_pixels(valid: torch.Tensor, num: int) -> torch.Tensor:
    """valid: (B,1,H,W) float/bool. Returns xy: (B,num,2)"""
    B, _, H, W = valid.shape
    xy_out = []
    for b in range(B):
        idx = torch.nonzero(valid[b,0] > 0.5, as_tuple=False)
        if idx.numel() == 0:
            # fallback: random
            ys = torch.randint(0, H, (num,), device=valid.device)
            xs = torch.randint(0, W, (num,), device=valid.device)
        else:
            sel = idx[torch.randint(0, idx.shape[0], (num,), device=valid.device)]
            ys, xs = sel[:,0], sel[:,1]
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


def contrastive_info_nce(desc1: torch.Tensor, desc2: torch.Tensor, temperature: float, num_negatives: int) -> torch.Tensor:
    """desc1, desc2: (B,N,D) positive pairs aligned by index."""
    B, N, D = desc1.shape
    # positives
    pos = (desc1 * desc2).sum(dim=-1) / temperature  # B,N

    # negatives: within-batch, within-sample (shuffle indices)
    # sample negatives from desc2
    with torch.no_grad():
        neg_idx = torch.randint(0, N, (B, N, num_negatives), device=desc1.device)
    neg = torch.gather(
        desc2.unsqueeze(2).expand(B, N, N, D),
        2,
        neg_idx.unsqueeze(-1).expand(B, N, num_negatives, D)
    )
    neg_logits = (desc1.unsqueeze(2) * neg).sum(dim=-1) / temperature  # B,N,K

    logits = torch.cat([pos.unsqueeze(-1), neg_logits], dim=-1)  # B,N,1+K
    labels = torch.zeros((B, N), dtype=torch.long, device=desc1.device)
    loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), labels.view(-1))
    return loss


def repeatability_loss(h1: torch.Tensor, h2: torch.Tensor, xy1_f: torch.Tensor, xy2_f: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    s1 = torch.sigmoid(_grid_sample_map(h1, xy1_f))
    s2 = torch.sigmoid(_grid_sample_map(h2, xy2_f))
    per = (s1 - s2).abs()  # (B,N)
    denom = m.sum().clamp(min=1.0)
    return (per * m).sum() / denom



def offset_regression_loss(offset: torch.Tensor, xy_f: torch.Tensor, xy_target_f: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """offset: (B,2,Hf,Wf), xy_f/xy_target_f: (B,N,2), m: (B,N) {0,1}"""
    B, _, Hf, Wf = offset.shape
    x, y = xy_f[..., 0], xy_f[..., 1]
    gx = (x / (Wf - 1)) * 2 - 1
    gy = (y / (Hf - 1)) * 2 - 1
    grid = torch.stack([gx, gy], dim=-1).unsqueeze(2)  # (B,N,1,2)

    samp = F.grid_sample(offset, grid, mode="bilinear", align_corners=True)  # (B,2,N,1)
    samp = samp.squeeze(-1).transpose(1, 2)  # (B,N,2)

    target_delta = (xy_target_f - xy_f).detach()  # (B,N,2)

    # per-point smooth L1
    per = F.smooth_l1_loss(samp, target_delta, reduction="none").sum(dim=-1)  # (B,N)
    denom = m.sum().clamp(min=1.0)
    return (per * m).sum() / denom



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


def compute_losses(
    batch: Dict[str, torch.Tensor],
    out1,
    out2,
    stride: int,
    cfg: Dict,
) -> Dict[str, torch.Tensor]:
    """Main loss computation for Stage-1 depth+pose supervision."""
    rgb1, rgb2 = batch["rgb1"], batch["rgb2"]
    depth1, depth2 = batch["depth1"], batch["depth2"]
    valid1, valid2 = batch["valid_depth1"], batch["valid_depth2"]
    K = batch["K"]
    T21 = batch["relative_pose"]

    B, _, H, W = rgb1.shape
    # depth filtering
    min_d = float(cfg["geom"]["min_depth"])
    max_d = float(cfg["geom"]["max_depth"])
    v1 = valid1 * (depth1 >= min_d).float() * (depth1 <= max_d).float()

    # sample pixels at full res
    N = int(cfg["geom"]["sample_points"])
    xy1 = _sample_valid_pixels(v1, N)  # full-res pixels

    # unproject, transform, project
    pts1 = unproject(depth1, K, xy1)
    pts2 = transform(T21, pts1)
    xy2 = project(pts2, K)

    # valid reprojection inside image
    x2, y2 = xy2[...,0], xy2[...,1]
    inb = (x2 >= 0) & (x2 <= (W-1)) & (y2 >= 0) & (y2 <= (H-1)) & (pts2[...,2] > 1e-3)

    # depth agreement check in frame2 (optional): compare predicted depth with observed depth2
    d2_obs = _sample_depth_at_xy(depth2, x2, y2)  # (B,N) SAFE
    z2 = pts2[..., 2]
    reproj_ok = (torch.abs(d2_obs - z2) < 0.05) | (d2_obs <= 0.0)
    mask = inb & reproj_ok
    m = mask.float()  # (B,N)

    if torch.rand(()) < 0.01:
    print("valid %:", (mask.float().mean().item() * 100),
          "x2 min/max:", x2.min().item(), x2.max().item(),
          "y2 min/max:", y2.min().item(), y2.max().item())


    # convert to feature coords at stride
    xy1_f = xy1 / float(stride)
    xy2_f = xy2 / float(stride)

    # descriptors at corresponding points
    d1 = _grid_sample_desc(out1.desc, xy1_f)
    d2 = _grid_sample_desc(out2.desc, xy2_f)

    # contrastive
    c_cfg = cfg["contrastive"]
    con = contrastive_info_nce(d1, d2, float(c_cfg["temperature"]), int(c_cfg["num_negatives"]))

    # repeatability (detector map consistency)
    rep = repeatability_loss(out1.heatmap, out2.heatmap, xy1_f, xy2_f, m)

    # offset regression (optional): predict delta towards projected point
    off_loss = torch.tensor(0.0, device=rgb1.device)
    if out1.offset is not None:
        off_loss = offset_regression_loss(out1.offset, xy1_f, xy2_f, m)

    # regularizers
    reg_cfg = cfg["regularizers"]
    ent = entropy_regularizer(out1.heatmap) + entropy_regularizer(out2.heatmap)
    uni = uniformity_regularizer(out1.heatmap) + uniformity_regularizer(out2.heatmap)
    reg = float(reg_cfg["entropy_weight"]) * ent + float(reg_cfg["uniformity_weight"]) * uni

    losses = {
        "loss_contrastive": con * float(c_cfg["weight"]),
        "loss_repeat": rep * float(cfg["repeatability"]["weight"]),
        "loss_offset": off_loss * float(cfg["offset"]["weight"]) if "offset" in cfg else off_loss,
        "loss_reg": reg * float(reg_cfg["weight"]),
    }
    losses["loss_total"] = sum(losses.values())
    return losses
