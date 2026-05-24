from __future__ import annotations

import math

import numpy as np

from refocus_vo.eval.kitti_odometry_metrics import compute_kitti_official_metrics


def _straight_line_poses(count: int = 200, step_m: float = 5.0) -> np.ndarray:
    poses = np.tile(np.eye(4, dtype=np.float64), (count, 1, 1))
    poses[:, 0, 3] = np.arange(count, dtype=np.float64) * float(step_m)
    return poses


def test_compute_kitti_official_metrics_zero_for_identical_poses() -> None:
    poses = _straight_line_poses()
    metrics = compute_kitti_official_metrics(poses, poses)
    assert math.isclose(metrics["kitti_trans_percent"], 0.0, abs_tol=1e-12)
    assert math.isclose(metrics["kitti_rot_deg_per_m"], 0.0, abs_tol=1e-12)
    assert metrics["kitti_segment_count"] > 0

