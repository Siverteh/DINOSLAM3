"""
Headless-safe evaluation helpers.

Key point:
- DO NOT import evo.tools.plot in headless environments, because it may force a Qt backend (qtagg).
- Computing ATE/RPE does not require plotting.
"""

import os
from typing import Any

# Headless-friendly matplotlib backend (must be set before importing matplotlib)
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib
matplotlib.use("Agg", force=True)
import numpy as np

from evo.core import sync
from evo.core.metrics import PoseRelation, Unit
from evo.core.metrics import APE, RPE
from evo.tools import file_interface


def _umeyama_align(src_points: np.ndarray, dst_points: np.ndarray, with_scale: bool = False):
    """
    Estimate transform T (4x4) that maps src_points -> dst_points.
    """
    src = np.asarray(src_points, dtype=np.float64)
    dst = np.asarray(dst_points, dtype=np.float64)
    if src.ndim != 2 or dst.ndim != 2 or src.shape[1] != 3 or dst.shape[1] != 3:
        raise ValueError("Expected Nx3 source and destination points")
    if src.shape[0] != dst.shape[0] or src.shape[0] < 2:
        raise ValueError("Need at least 2 matched 3D points")

    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_centered = src - mu_src
    dst_centered = dst - mu_dst

    cov = (dst_centered.T @ src_centered) / src.shape[0]
    u, d, vt = np.linalg.svd(cov)
    s = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        s[2, 2] = -1.0

    r = u @ s @ vt
    c = 1.0
    if with_scale:
        var_src = np.mean(np.sum(src_centered * src_centered, axis=1))
        if var_src > 1e-12:
            c = float(np.trace(np.diag(d) @ s) / var_src)

    t = mu_dst - c * (r @ mu_src)
    tform = np.eye(4, dtype=np.float64)
    tform[:3, :3] = c * r
    tform[:3, 3] = t
    return tform


def _poses_to_points(poses: Any) -> np.ndarray:
    arr = np.asarray(poses)
    if arr.ndim != 3 or arr.shape[1:] != (4, 4):
        raise ValueError("Expected poses shaped [N,4,4]")
    return arr[:, :3, 3].astype(np.float64, copy=False)


def _stats_from_errors(errors: np.ndarray):
    err = np.asarray(errors, dtype=np.float64).reshape(-1)
    if err.size == 0:
        raise ValueError("No errors available")
    sq = err * err
    return {
        "rmse": float(np.sqrt(np.mean(sq))),
        "mean": float(np.mean(err)),
        "median": float(np.median(err)),
        "std": float(np.std(err)),
        "min": float(np.min(err)),
        "max": float(np.max(err)),
        "sse": float(np.sum(sq)),
        "num_samples": int(err.size),
    }


def _eval_ate_from_poses(poses_est: Any, poses_gt: Any, is_monocular: bool = False):
    est_points = _poses_to_points(poses_est)
    gt_points = _poses_to_points(poses_gt)
    n = min(len(est_points), len(gt_points))
    if n < 2:
        raise ValueError("Need at least 2 associated poses for ATE")
    est_points = est_points[:n]
    gt_points = gt_points[:n]

    t_gt_est = _umeyama_align(est_points, gt_points, with_scale=bool(is_monocular))
    est_aligned = (t_gt_est[:3, :3] @ est_points.T).T + t_gt_est[:3, 3]
    errors = np.linalg.norm(gt_points - est_aligned, axis=1)
    return _stats_from_errors(errors), t_gt_est


def _eval_ate_from_files(gt_file: str, est_file: str):
    """
    Evaluate ATE (Absolute Pose Error) using evo, without plotting.
    Returns evo statistics dict.
    """
    traj_ref = file_interface.read_tum_trajectory_file(gt_file)
    traj_est = file_interface.read_tum_trajectory_file(est_file)

    traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est)

    ape_metric = APE(PoseRelation.translation_part)
    ape_metric.process_data((traj_ref, traj_est))
    return ape_metric.get_all_statistics()


def eval_ate(*args, **kwargs):
    """
    Backward-compatible ATE API.

    Supported signatures:
    - eval_ate(gt_file, est_file) -> stats_dict
    - eval_ate(poses_est=..., poses_gt=..., is_monocular=...) -> (stats_dict, T_gt_est)
    """
    if "poses_est" in kwargs or "poses_gt" in kwargs:
        if "poses_est" not in kwargs or "poses_gt" not in kwargs:
            raise TypeError("eval_ate legacy mode requires poses_est and poses_gt")
        is_monocular = bool(kwargs.get("is_monocular", False))
        return _eval_ate_from_poses(
            poses_est=kwargs["poses_est"],
            poses_gt=kwargs["poses_gt"],
            is_monocular=is_monocular,
        )

    if len(args) == 2 and not kwargs:
        return _eval_ate_from_files(str(args[0]), str(args[1]))

    raise TypeError(
        "eval_ate expected either (gt_file, est_file) or keyword args with poses_est and poses_gt"
    )


def _eval_rpe_from_files(gt_file: str, est_file: str, delta: int = 1):
    """
    Evaluate RPE (Relative Pose Error) using evo, without plotting.
    Returns evo statistics dict.
    """
    traj_ref = file_interface.read_tum_trajectory_file(gt_file)
    traj_est = file_interface.read_tum_trajectory_file(est_file)

    traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est)

    rpe_metric = RPE(PoseRelation.translation_part, delta=delta, unit=Unit.frames)
    rpe_metric.process_data((traj_ref, traj_est))
    return rpe_metric.get_all_statistics()


def eval_rpe(*args, **kwargs):
    """
    Backward-compatible RPE API.
    Supported signature: eval_rpe(gt_file, est_file, delta=1) -> stats_dict
    """
    if len(args) >= 2:
        delta = kwargs.get("delta", 1)
        return _eval_rpe_from_files(str(args[0]), str(args[1]), delta=int(delta))
    if "gt_file" in kwargs and "est_file" in kwargs:
        delta = kwargs.get("delta", 1)
        return _eval_rpe_from_files(
            str(kwargs["gt_file"]),
            str(kwargs["est_file"]),
            delta=int(delta),
        )
    raise TypeError("eval_rpe expected (gt_file, est_file, delta=1)")
