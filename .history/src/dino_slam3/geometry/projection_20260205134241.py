from __future__ import annotations
from typing import Tuple
import torch

def unproject(depth: torch.Tensor, K: torch.Tensor, xy: torch.Tensor) -> torch.Tensor:
    """Unproject pixels to camera points.

    depth: (B,1,H,W) meters
    K: (B,3,3)
    xy: (B,N,2) pixel coords in resized image space
    returns: (B,N,3)
    """
    B, N, _ = xy.shape
    fx = K[:, 0, 0].unsqueeze(1)
    fy = K[:, 1, 1].unsqueeze(1)
    cx = K[:, 0, 2].unsqueeze(1)
    cy = K[:, 1, 2].unsqueeze(1)

    x = xy[..., 0]
    y = xy[..., 1]

    # sample depth at xy (nearest)
    H, W = depth.shape[-2:]
    xi = x.round().clamp(0, W - 1).long()
    yi = y.round().clamp(0, H - 1).long()
    z = depth[:, 0, yi, xi]  # (B,N)

    X = (x - cx) * z / fx
    Y = (y - cy) * z / fy
    pts = torch.stack([X, Y, z], dim=-1)
    return pts

def project(pts: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """Project camera points to pixels.

    pts: (B,N,3)
    K: (B,3,3)
    returns: (B,N,2)
    """
    fx = K[:, 0, 0].unsqueeze(1)
    fy = K[:, 1, 1].unsqueeze(1)
    cx = K[:, 0, 2].unsqueeze(1)
    cy = K[:, 1, 2].unsqueeze(1)
    X, Y, Z = pts[..., 0], pts[..., 1], pts[..., 2].clamp(min=1e-6)
    u = fx * (X / Z) + cx
    v = fy * (Y / Z) + cy
    return torch.stack([u, v], dim=-1)

def transform(T: torch.Tensor, pts: torch.Tensor) -> torch.Tensor:
    """Apply SE(3) transform T (B,4,4) to pts (B,N,3)."""
    B, N, _ = pts.shape
    R = T[:, :3, :3]
    t = T[:, :3, 3].unsqueeze(1)
    return (R @ pts.transpose(1,2)).transpose(1,2) + t
