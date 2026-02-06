from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import cv2


def simple_nms(heat: torch.Tensor, nms_radius: int) -> torch.Tensor:
    assert heat.dim() == 4 and heat.shape[1] == 1
    pad = int(nms_radius)
    maxpool = F.max_pool2d(heat, kernel_size=2 * pad + 1, stride=1, padding=pad)
    keep = (heat == maxpool).float()
    return heat * keep


@dataclass
class KeypointBatch:
    kpts: np.ndarray   # (N,2) float32 (x,y) in image pixels
    desc: np.ndarray   # (N,D) float32
    scores: np.ndarray # (N,) float32


def _tile_topk(score: torch.Tensor, tile: int, k_per_tile: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    score: (Hf,Wf) probabilities
    returns (ys, xs) of selected points
    """
    Hf, Wf = score.shape
    ys_all = []
    xs_all = []

    for y0 in range(0, Hf, tile):
        for x0 in range(0, Wf, tile):
            y1 = min(Hf, y0 + tile)
            x1 = min(Wf, x0 + tile)
            patch = score[y0:y1, x0:x1].reshape(-1)
            if patch.numel() == 0:
                continue
            k = min(int(k_per_tile), patch.numel())
            vals, inds = torch.topk(patch, k=k, largest=True, sorted=False)
            keep = vals > 0.0
            inds = inds[keep]
            if inds.numel() == 0:
                continue
            yy = inds // (x1 - x0) + y0
            xx = inds % (x1 - x0) + x0
            ys_all.append(yy)
            xs_all.append(xx)

    if not ys_all:
        return (
            torch.empty((0,), dtype=torch.long, device=score.device),
            torch.empty((0,), dtype=torch.long, device=score.device),
        )
    ys = torch.cat(ys_all, dim=0)
    xs = torch.cat(xs_all, dim=0)
    return ys, xs


def extract_keypoints_and_descriptors(
    heatmap_logits: torch.Tensor,
    desc_map: torch.Tensor,
    offset_map: Optional[torch.Tensor],
    reliability_logits: Optional[torch.Tensor],
    *,
    stride: int = 4,
    nms_radius: int = 4,
    max_keypoints: int = 1024,
    tile_size: int = 16,        # in feature cells
    k_per_tile: int = 8,
    # NEW: optional valid depth mask in IMAGE space (B,1,H,W) or (B,H,W)
    valid_mask_img: Optional[torch.Tensor] = None,
) -> List[KeypointBatch]:
    """
    Produces spatially-covered keypoints using tile-topk (prevents one-side collapse).

    NEW:
      - If valid_mask_img is provided, we downsample to feature size and
        multiply the keypoint score by that mask. This prevents selecting
        keypoints where depth is invalid (critical for geometry + SLAM).
    """
    B, _, Hf, Wf = heatmap_logits.shape

    heat = simple_nms(heatmap_logits, nms_radius=nms_radius)
    score = torch.sigmoid(heat)  # (B,1,Hf,Wf)

    # Depth-valid gating (downsample to feature map)
    if valid_mask_img is not None:
        vm = valid_mask_img
        if vm.dim() == 3:
            vm = vm.unsqueeze(1)  # (B,1,H,W)
        vm = vm.float()
        vm_f = F.interpolate(vm, size=(Hf, Wf), mode="nearest")  # (B,1,Hf,Wf)
        score = score * (vm_f > 0.5).float()

    batches: List[KeypointBatch] = []
    for b in range(B):
        s = score[b, 0]

        # tie-break noise (tiny) to avoid identical values collapsing selection
        s = s + (torch.rand_like(s) * 1e-6)

        ys, xs = _tile_topk(s, tile=int(tile_size), k_per_tile=int(k_per_tile))
        if ys.numel() == 0:
            batches.append(
                KeypointBatch(
                    kpts=np.zeros((0, 2), np.float32),
                    desc=np.zeros((0, desc_map.shape[1]), np.float32),
                    scores=np.zeros((0,), np.float32),
                )
            )
            continue

        vals = s[ys, xs]

        # global prune if too many
        if vals.numel() > int(max_keypoints):
            topv, topi = torch.topk(vals, k=int(max_keypoints), largest=True, sorted=False)
            ys = ys[topi]
            xs = xs[topi]
            vals = topv

        xy_f = torch.stack([xs.float(), ys.float()], dim=-1).unsqueeze(0)  # (1,N,2)

        # grid in [-1,1] for feature coords
        gx = (xy_f[..., 0] / max(Wf - 1, 1)) * 2 - 1
        gy = (xy_f[..., 1] / max(Hf - 1, 1)) * 2 - 1
        grid = torch.stack([gx, gy], dim=-1).unsqueeze(2)  # (1,N,1,2)

        # descriptor at points (force float for grid_sample stability)
        dmap = desc_map[b:b+1].float()
        d = F.grid_sample(dmap, grid.float(), mode="bilinear", align_corners=True)
        d = d.squeeze(-1).transpose(1, 2).squeeze(0)  # (N,D)
        d = F.normalize(d, dim=-1, eps=1e-6)

        # offsets (sub-cell) refine positions
        xy_f2 = xy_f.squeeze(0)
        if offset_map is not None:
            omap = offset_map[b:b+1].float()
            o = F.grid_sample(omap, grid.float(), mode="bilinear", align_corners=True)
            o = o.squeeze(-1).transpose(1, 2).squeeze(0)  # (N,2)
            xy_f2 = xy_f2 + o

        # reliability scales scores
        if reliability_logits is not None:
            rmap = reliability_logits[b:b+1].float()
            r = F.grid_sample(rmap, grid.float(), mode="bilinear", align_corners=True)
            r = torch.sigmoid(r.squeeze(-1).transpose(1, 2).squeeze(0))[:, 0]
            vals = vals * r

        xy_img = xy_f2 * float(stride)

        batches.append(
            KeypointBatch(
                kpts=xy_img.detach().cpu().numpy().astype(np.float32),
                desc=d.detach().cpu().numpy().astype(np.float32),
                scores=vals.detach().cpu().numpy().astype(np.float32),
            )
        )

    return batches


def to_cv2(kp_batch: KeypointBatch, size: float = 1.0) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
    kps = [
        cv2.KeyPoint(float(x), float(y), float(size), _response=float(s))
        for (x, y), s in zip(kp_batch.kpts, kp_batch.scores)
    ]
    return kps, kp_batch.desc
