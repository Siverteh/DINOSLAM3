from __future__ import annotations
import torch

def unproject(depth: torch.Tensor, K: torch.Tensor, xy: torch.Tensor) -> torch.Tensor:
    """
    depth: (B,1,H,W) meters
    K: (B,3,3) or (3,3)
    xy: (B,N,2) pixels
    returns: (B,N,3) in camera coords
    """
    if K.dim() == 2:
        K = K.unsqueeze(0).expand(depth.shape[0], -1, -1)

    B, _, H, W = depth.shape
    fx = K[:, 0, 0].unsqueeze(1)
    fy = K[:, 1, 1].unsqueeze(1)
    cx = K[:, 0, 2].unsqueeze(1)
    cy = K[:, 1, 2].unsqueeze(1)

    x = xy[..., 0]
    y = xy[..., 1]
    xi = x.round().clamp(0, W - 1).long()
    yi = y.round().clamp(0, H - 1).long()

    z = depth[:, 0].reshape(B, -1).gather(1, yi * W + xi)
    X = (x - cx) * z / fx
    Y = (y - cy) * z / fy
    return torch.stack([X, Y, z], dim=-1)

def project(pts: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    if K.dim() == 2:
        K = K.unsqueeze(0).expand(pts.shape[0], -1, -1)

    fx = K[:, 0, 0].unsqueeze(1)
    fy = K[:, 1, 1].unsqueeze(1)
    cx = K[:, 0, 2].unsqueeze(1)
    cy = K[:, 1, 2].unsqueeze(1)

    X, Y, Z = pts[..., 0], pts[..., 1], pts[..., 2].clamp(min=1e-6)
    u = fx * (X / Z) + cx
    v = fy * (Y / Z) + cy
    return torch.stack([u, v], dim=-1)

def transform(T: torch.Tensor, pts: torch.Tensor) -> torch.Tensor:
    """
    Apply SE(3) transform to 3D points.

    Args:
        T:  (B,4,4) or (4,4) transform mapping pts frame -> new frame
        pts: (B,N,3) or (N,3)

    Returns:
        pts_out: (B,N,3) or (N,3)

    Notes:
        This implementation avoids cuBLAS GEMM entirely (no bmm/matmul),
        because H100 + certain shapes/layouts can trigger CUBLAS_STATUS_INVALID_VALUE.
    """
    if pts.numel() == 0:
        return pts

    # Normalize shapes to batched
    batched_pts = True
    if pts.dim() == 2:
        pts = pts.unsqueeze(0)
        batched_pts = False

    if T.dim() == 2:
        T = T.unsqueeze(0)

    # Now T: (B,4,4), pts: (B,N,3) with possible broadcasting
    Bp = pts.shape[0]
    Bt = T.shape[0]
    if Bt != Bp:
        if Bt == 1:
            T = T.expand(Bp, -1, -1).contiguous()
        elif Bp == 1:
            pts = pts.expand(Bt, -1, -1).contiguous()
        else:
            raise ValueError(f"Batch mismatch: T batch={Bt}, pts batch={Bp}")

    # Ensure stable dtype/layout
    pts = pts.contiguous()
    T = T.contiguous()
    if pts.dtype != torch.float32:
        pts = pts.float()
    if T.dtype != torch.float32:
        T = T.float()

    R = T[:, :3, :3]          # (B,3,3)
    t = T[:, :3, 3]           # (B,3)
    Rt = R.transpose(1, 2).contiguous()  # (B,3,3)

    # Elementwise matmul: pts @ Rt (no GEMM)
    # pts: (B,N,3) -> (B,N,3,1)
    # Rt:  (B,3,3) -> (B,1,3,3)
    # product -> (B,N,3,3), sum over dim=2 -> (B,N,3)
    pts_out = (pts.unsqueeze(-1) * Rt.unsqueeze(1)).sum(dim=2) + t.unsqueeze(1)

    if not batched_pts:
        pts_out = pts_out.squeeze(0)
    return pts_out