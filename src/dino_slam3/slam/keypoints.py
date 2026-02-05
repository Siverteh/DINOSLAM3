from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional, List
import numpy as np
import torch
import torch.nn.functional as F
import cv2

def simple_nms(heat: torch.Tensor, nms_radius: int) -> torch.Tensor:
    """Non-maximum suppression on heatmap logits.
    heat: (B,1,H,W)
    returns suppressed heat (same shape)
    """
    assert heat.dim() == 4 and heat.shape[1] == 1
    pad = nms_radius
    maxpool = F.max_pool2d(heat, kernel_size=2*pad+1, stride=1, padding=pad)
    keep = (heat == maxpool).float()
    return heat * keep

@dataclass
class KeypointBatch:
    kpts: np.ndarray  # (N,2) float32 (x,y) at image resolution
    desc: np.ndarray  # (N,D) float32
    scores: np.ndarray  # (N,) float32

def extract_keypoints_and_descriptors(
    heatmap_logits: torch.Tensor,
    desc_map: torch.Tensor,
    offset_map: Optional[torch.Tensor],
    reliability_logits: Optional[torch.Tensor],
    stride: int,
    nms_radius: int,
    max_keypoints: int,
) -> List[KeypointBatch]:
    """Convert dense maps to sparse keypoints for each batch element."""
    B, _, Hf, Wf = heatmap_logits.shape
    heat = simple_nms(heatmap_logits, nms_radius=nms_radius)
    score = torch.sigmoid(heat)  # (B,1,Hf,Wf)

    # flatten
    score_flat = score.view(B, -1)
    topk = min(max_keypoints, score_flat.shape[1])
    vals, inds = torch.topk(score_flat, k=topk, dim=-1)

    batches = []
    for b in range(B):
        ind = inds[b]
        s = vals[b]
        y = (ind // Wf).float()
        x = (ind % Wf).float()
        xy_f = torch.stack([x, y], dim=-1).unsqueeze(0)  # 1,N,2 (feature coords)

        # sample descriptors (bilinear)
        gx = (xy_f[...,0] / (Wf - 1)) * 2 - 1
        gy = (xy_f[...,1] / (Hf - 1)) * 2 - 1
        grid = torch.stack([gx, gy], dim=-1).unsqueeze(2)  # 1,N,1,2

        d = F.grid_sample(desc_map[b:b+1], grid, mode="bilinear", align_corners=True)
        d = d.squeeze(-1).transpose(1,2).squeeze(0)  # N,D
        d = F.normalize(d, dim=-1, eps=1e-6)

        # offsets in feature coords (delta at feature scale)
        if offset_map is not None:
            o = F.grid_sample(offset_map[b:b+1], grid, mode="bilinear", align_corners=True)
            o = o.squeeze(-1).transpose(1,2).squeeze(0)  # N,2
            xy_f = xy_f.squeeze(0) + o  # refined in feature coords
        else:
            xy_f = xy_f.squeeze(0)

        # reliability as additional score factor
        if reliability_logits is not None:
            r = F.grid_sample(reliability_logits[b:b+1], grid, mode="bilinear", align_corners=True)
            r = torch.sigmoid(r.squeeze(-1).transpose(1,2).squeeze(0))[:,0]  # N
            s = s * r

        # map to image resolution
        xy_img = xy_f * float(stride)

        batches.append(
            KeypointBatch(
                kpts=xy_img.detach().cpu().numpy().astype(np.float32),
                desc=d.detach().cpu().numpy().astype(np.float32),
                scores=s.detach().cpu().numpy().astype(np.float32),
            )
        )
    return batches

def to_cv2(
    kp_batch: KeypointBatch,
    size: float = 1.0
) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
    kps = [cv2.KeyPoint(float(x), float(y), size, _response=float(s))
           for (x,y), s in zip(kp_batch.kpts, kp_batch.scores)]
    return kps, kp_batch.desc
