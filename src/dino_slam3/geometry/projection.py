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
    if depth.dim() != 4 or depth.shape[1] != 1:
        raise ValueError(f"depth must be (B,1,H,W), got {tuple(depth.shape)}")
    if K.shape[-2:] != (3, 3):
        raise ValueError(f"K must be (B,3,3), got {tuple(K.shape)}")
    if xy.dim() != 3 or xy.shape[-1] != 2:
        raise ValueError(f"xy must be (B,N,2), got {tuple(xy.shape)}")

    B, N, _ = xy.shape
    H, W = depth.shape[-2:]

    fx = K[:, 0, 0].unsqueeze(1)  # (B,1)
    fy = K[:, 1, 1].unsqueeze(1)
    cx = K[:, 0, 2].unsqueeze(1)
    cy = K[:, 1, 2].unsqueeze(1)

    x = xy[..., 0]
    y = xy[..., 1]

    # nearest pixel indices
    xi = x.round().clamp(0, W - 1).long()  # (B,N)
    yi = y.round().clamp(0, H - 1).long()  # (B,N)

    # SAFE batched depth sampling via flatten + gather
    depth_hw = depth[:, 0]                      # (B,H,W)
    depth_flat = depth_hw.reshape(B, -1)        # (B,H*W)
    lin = yi * W + xi                           # (B,N)
    z = torch.gather(depth_flat, 1, lin)        # (B,N)

    X = (x - cx) * z / fx
    Y = (y - cy) * z / fy
    pts = torch.stack([X, Y, z], dim=-1)        # (B,N,3)
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
    """Apply SE(3) transform T (B,4,4) to pts (B,N,3). Returns (B,N,3).
    Avoids cuBLAS batched GEMM (workaround for CUBLAS_STATUS_INVALID_VALUE on some stacks).
    """
    if T.dim() != 3 or T.shape[-2:] != (4, 4):
        raise ValueError(f"T must be (B,4,4), got {tuple(T.shape)}")
    if pts.dim() != 3 or pts.shape[-1] != 3:
        raise ValueError(f"pts must be (B,N,3), got {tuple(pts.shape)}")

    # Ensure device/dtype match
    T = T.to(device=pts.device, dtype=pts.dtype)

    R = T[:, :3, :3]        # (B,3,3)
    t = T[:, :3, 3]         # (B,3)

    x = pts[..., 0]         # (B,N)
    y = pts[..., 1]
    z = pts[..., 2]

    # Explicit rotation (no matmul / no bmm)
    X = R[:, 0, 0].unsqueeze(1) * x + R[:, 0, 1].unsqueeze(1) * y + R[:, 0, 2].unsqueeze(1) * z
    Y = R[:, 1, 0].unsqueeze(1) * x + R[:, 1, 1].unsqueeze(1) * y + R[:, 1, 2].unsqueeze(1) * z
    Z = R[:, 2, 0].unsqueeze(1) * x + R[:, 2, 1].unsqueeze(1) * y + R[:, 2, 2].unsqueeze(1) * z

    out = torch.stack([X, Y, Z], dim=-1)  # (B,N,3)
    out = out + t.unsqueeze(1)            # (B,N,3)
    return out

