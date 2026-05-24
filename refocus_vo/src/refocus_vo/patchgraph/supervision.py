from __future__ import annotations

from typing import Any

import torch
from scipy.spatial.transform import Rotation as R

from refocus_vo.data.tartanair import pose_vector_to_matrix

from .teacher import DinoPatchTeacher


def relative_pose_matrix(src_pose: torch.Tensor, tgt_pose: torch.Tensor) -> torch.Tensor:
    src = torch.as_tensor(pose_vector_to_matrix(src_pose.detach().cpu().numpy()), device=src_pose.device, dtype=torch.float32)
    tgt = torch.as_tensor(pose_vector_to_matrix(tgt_pose.detach().cpu().numpy()), device=tgt_pose.device, dtype=torch.float32)
    return torch.linalg.inv(src) @ tgt


def rotvec_from_matrix(rot: torch.Tensor) -> torch.Tensor:
    rv = R.from_matrix(rot.detach().cpu().numpy()).as_rotvec()
    return torch.as_tensor(rv, device=rot.device, dtype=torch.float32)


def project_patch_centers(
    pixel_xy: torch.Tensor,
    depth_values: torch.Tensor,
    rel_pose: torch.Tensor,
    intrinsics: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    fx, fy, cx, cy = intrinsics
    z = depth_values
    x = ((pixel_xy[:, 0] - cx) / fx) * z
    y = ((pixel_xy[:, 1] - cy) / fy) * z
    xyz = torch.stack([x, y, z, torch.ones_like(z)], dim=1)
    xyz_t = (rel_pose @ xyz.t()).t()[:, :3]
    z_t = xyz_t[:, 2].clamp_min(1e-6)
    u = (fx * xyz_t[:, 0] / z_t) + cx
    v = (fy * xyz_t[:, 1] / z_t) + cy
    return torch.stack([u, v], dim=1), xyz_t[:, 2]


def build_teacher_scores_from_fused(
    fused: torch.Tensor,
    batch: dict[str, Any],
    teacher: DinoPatchTeacher,
) -> torch.Tensor:
    images = batch["images"].to(fused.device)
    depths = batch["depths"].to(fused.device)
    poses = batch["poses"].to(fused.device)
    intrinsics = batch["intrinsics"].to(fused.device)
    b, t = images.shape[:2]
    scores = torch.zeros(
        (b, t, fused.shape[-2], fused.shape[-1]),
        device=fused.device,
        dtype=torch.float32,
    )
    counts = torch.zeros_like(scores)
    for bi in range(b):
        K = intrinsics[bi]
        for ti in range(t - 1):
            rel = relative_pose_matrix(poses[bi, ti], poses[bi, ti + 1]).to(fused.device)
            pair_score = teacher.score_adjacent_pair(
                fused[bi, ti].unsqueeze(0),
                fused[bi, ti + 1].unsqueeze(0),
                images[bi, ti].unsqueeze(0),
                images[bi, ti + 1].unsqueeze(0),
                depths[bi, ti].unsqueeze(0),
                depths[bi, ti + 1].unsqueeze(0),
                rel.unsqueeze(0),
                K.unsqueeze(0),
            )[0]
            scores[bi, ti] += pair_score
            counts[bi, ti] += 1.0
            scores[bi, ti + 1] += pair_score
            counts[bi, ti + 1] += 1.0
    return scores / counts.clamp_min(1.0)
