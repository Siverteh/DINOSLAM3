from __future__ import annotations
import torch


def unproject(depth: torch.Tensor, K: torch.Tensor, xy: torch.Tensor) -> torch.Tensor:
    """
    depth: (B,1,H,W)
    K: (B,3,3) or (3,3)
    xy: (B,N,2) in pixels
    returns: (B,N,3)
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

    depth_hw = depth[:, 0]
    depth_flat = depth_hw.reshape(B, -1)
    lin = yi * W + xi
    z = torch.gather(depth_flat, 1, lin)

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

    T:   (B,4,4) or (4,4)
    pts: (B,M,3) or (M,3)
    returns: (B,M,3) or (M,3) matching input batching
    """
    # normalize batch dims
    squeeze_T = False
    squeeze_pts = False
    if T.dim() == 2:
        T = T.unsqueeze(0)
        squeeze_T = True
    if pts.dim() == 2:
        pts = pts.unsqueeze(0)
        squeeze_pts = True

    if pts.numel() == 0 or pts.shape[1] == 0:
        return pts.squeeze(0) if squeeze_pts else pts

    # Force fp32 + contiguous for stability
    T = T.float().contiguous()
    pts = pts.float().contiguous()

    R = T[:, :3, :3].contiguous()          # (B,3,3)
    t = T[:, :3, 3].contiguous().unsqueeze(1)  # (B,1,3)

    # pts is (B,M,3)
    X = pts[..., 0]  # (B,M)
    Y = pts[..., 1]
    Z = pts[..., 2]

    # Manual multiply-add (NO GEMM, NO cuBLAS)
    x2 = R[:, 0, 0].unsqueeze(1) * X + R[:, 0, 1].unsqueeze(1) * Y + R[:, 0, 2].unsqueeze(1) * Z
    y2 = R[:, 1, 0].unsqueeze(1) * X + R[:, 1, 1].unsqueeze(1) * Y + R[:, 1, 2].unsqueeze(1) * Z
    z2 = R[:, 2, 0].unsqueeze(1) * X + R[:, 2, 1].unsqueeze(1) * Y + R[:, 2, 2].unsqueeze(1) * Z

    out = torch.stack([x2, y2, z2], dim=-1) + t  # (B,M,3)

    if squeeze_pts:
        out = out.squeeze(0)
    return out