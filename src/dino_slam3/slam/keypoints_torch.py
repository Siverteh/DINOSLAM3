from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import math
import torch
import torch.nn.functional as F

@dataclass
class KeypointsTorch:
    xy_img: torch.Tensor    # (B,N,2) pixels
    xy_f: torch.Tensor      # (B,N,2) feature coords (float)
    desc: torch.Tensor      # (B,N,D)
    score: torch.Tensor     # (B,N)

def _simple_nms(heat: torch.Tensor, radius: int) -> torch.Tensor:
    pad = int(radius)
    maxpool = F.max_pool2d(heat, kernel_size=2 * pad + 1, stride=1, padding=pad)
    keep = (heat == maxpool).float()
    return heat * keep

def _grid_sample_vec(map_: torch.Tensor, xy_f: torch.Tensor) -> torch.Tensor:
    """
    map_: (B,C,Hf,Wf)
    xy_f: (B,N,2) in feature coords
    returns: (B,N,C)
    """
    B, C, Hf, Wf = map_.shape
    gx = (xy_f[..., 0] / max(Wf - 1, 1)) * 2 - 1
    gy = (xy_f[..., 1] / max(Hf - 1, 1)) * 2 - 1
    grid = torch.stack([gx, gy], dim=-1).unsqueeze(2)  # (B,N,1,2)
    out = F.grid_sample(map_.float(), grid.float(), mode="bilinear", align_corners=True)
    out = out.squeeze(-1).transpose(1, 2)  # (B,N,C)
    return out

def extract_keypoints_torch(
    heat_logits: torch.Tensor,
    desc_map: torch.Tensor,
    offset_map: Optional[torch.Tensor],
    reliability_logits: Optional[torch.Tensor],
    *,
    stride: int = 4,
    nms_radius: int = 4,
    tile_size: int = 16,
    k_per_tile: int = 8,
    max_keypoints: int = 1024,
    valid_mask_img: Optional[torch.Tensor] = None,
    use_reliability_in_score: bool = False,
    adaptive_tiling: bool = False,
    adaptive_k_min: int = 1,
    adaptive_k_max: Optional[int] = None,
) -> KeypointsTorch:
    """
    Vectorized tile-topk keypoints (fast, stable, avoids one-side collapse).
    """
    assert heat_logits.dim() == 4 and heat_logits.shape[1] == 1
    B, _, Hf, Wf = heat_logits.shape

    heat = _simple_nms(heat_logits, radius=nms_radius)
    score = torch.sigmoid(heat)  # (B,1,Hf,Wf)

    if valid_mask_img is not None:
        vm = valid_mask_img
        if vm.dim() == 3:
            vm = vm.unsqueeze(1)
        vm_f = F.interpolate(vm.float(), size=(Hf, Wf), mode="nearest")
        score = score * (vm_f > 0.5).float()

    if use_reliability_in_score and reliability_logits is not None:
        rel = torch.sigmoid(reliability_logits)
        score = score * rel

    # pad to tile multiple
    t = int(tile_size)
    pad_h = (t - (Hf % t)) % t
    pad_w = (t - (Wf % t)) % t
    score_p = F.pad(score, (0, pad_w, 0, pad_h), value=0.0)
    Hfp, Wfp = score_p.shape[-2:]
    Ht, Wt = Hfp // t, Wfp // t
    tiles = score_p.view(B, 1, Ht, t, Wt, t).permute(0, 2, 4, 3, 5, 1).reshape(B, Ht * Wt, t * t)

    k_base = min(int(k_per_tile), t * t)
    if adaptive_tiling:
        flat = score.reshape(B, -1).float()
        norm = flat.sum(dim=1, keepdim=True).clamp(min=1e-9)
        probs = flat / norm
        entropy = -(probs * probs.clamp(min=1e-9).log()).sum(dim=1) / max(math.log(max(flat.shape[1], 2)), 1e-9)
        if valid_mask_img is not None:
            vm = valid_mask_img
            if vm.dim() == 3:
                vm = vm.unsqueeze(1)
            vm_f = F.interpolate(vm.float(), size=(Hf, Wf), mode="nearest")
            valid_ratio = vm_f.reshape(B, -1).mean(dim=1).clamp(0.0, 1.0)
        else:
            valid_ratio = torch.ones((B,), dtype=torch.float32, device=score.device)
        adapt = (0.5 * entropy + 0.5 * valid_ratio).clamp(0.0, 1.0)
        k_target = int(torch.round(torch.tensor(float(k_base), device=score.device) * adapt.mean()).item())
        k_hi = int(k_base if adaptive_k_max is None else max(1, int(adaptive_k_max)))
        k_lo = max(1, int(adaptive_k_min))
        k = max(k_lo, min(k_hi, k_target, t * t))
    else:
        k = k_base
    vals, inds = torch.topk(tiles, k=k, dim=-1, largest=True, sorted=False)  # (B,T,k)
    vals = vals.squeeze(-1) if vals.dim() == 4 else vals  # keep
    inds = inds

    tile_ids = torch.arange(Ht * Wt, device=heat_logits.device)
    tile_y = (tile_ids // Wt).view(1, -1, 1).expand(B, -1, k)
    tile_x = (tile_ids % Wt).view(1, -1, 1).expand(B, -1, k)

    loc_y = (inds // t)
    loc_x = (inds % t)

    ys = tile_y * t + loc_y
    xs = tile_x * t + loc_x

    # flatten
    xs = xs.reshape(B, -1).float()
    ys = ys.reshape(B, -1).float()
    sc = vals.reshape(B, -1)

    # drop padded positions
    keep_in = (xs < Wf) & (ys < Hf) & (sc > 0.0)
    sc = sc.masked_fill(~keep_in, -1.0)

    # global topk
    N = sc.shape[1]
    K = min(int(max_keypoints), N)
    topv, topi = torch.topk(sc, k=K, dim=1, largest=True, sorted=False)
    xs = torch.gather(xs, 1, topi)
    ys = torch.gather(ys, 1, topi)
    sc = topv

    xy_f = torch.stack([xs, ys], dim=-1)  # (B,K,2)

    # add offsets
    if offset_map is not None:
        off = _grid_sample_vec(offset_map, xy_f)  # (B,K,2)
        xy_f = xy_f + off

    # descriptors
    d = _grid_sample_vec(desc_map, xy_f)  # (B,K,D)
    d = F.normalize(d, dim=-1, eps=1e-6)

    xy_img = xy_f * float(stride)
    return KeypointsTorch(xy_img=xy_img, xy_f=xy_f, desc=d, score=sc)
