from __future__ import annotations
import torch

def rotation_geodesic_rad(R1: torch.Tensor, R2: torch.Tensor) -> torch.Tensor:
    """
    R1,R2: (B,3,3)
    returns: (B,) geodesic angle in radians
    """
    R = R1 @ R2.transpose(1, 2)
    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    cos = (trace - 1.0) * 0.5
    cos = torch.clamp(cos, -1.0 + 1e-6, 1.0 - 1e-6)
    return torch.acos(cos)

def se3_error(T_hat: torch.Tensor, T_gt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    returns: (rot_rad, trans_l2) both (B,)
    """
    R_hat = T_hat[:, :3, :3]
    t_hat = T_hat[:, :3, 3]
    R_gt = T_gt[:, :3, :3]
    t_gt = T_gt[:, :3, 3]
    rot = rotation_geodesic_rad(R_hat, R_gt)
    trans = torch.linalg.norm(t_hat - t_gt, dim=-1)
    return rot, trans
