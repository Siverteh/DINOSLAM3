from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R


KITTI_SEQUENCE_ORDER = [f"{idx:02d}" for idx in range(11)]
KITTI_OFFICIAL_LENGTHS = [100, 200, 300, 400, 500, 600, 700, 800]
KITTI_OFFICIAL_STEP_SIZE = 10


def _as_pose_array(poses: np.ndarray | list[np.ndarray]) -> np.ndarray:
    arr = np.asarray(poses, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[1:] != (4, 4):
        raise ValueError(f"expected poses with shape [N,4,4], got {arr.shape}")
    return arr


def load_kitti_poses(path: str | Path) -> np.ndarray:
    raw = np.loadtxt(str(path), dtype=np.float64)
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)
    if raw.size == 0:
        return np.zeros((0, 4, 4), dtype=np.float64)
    if raw.shape[1] != 12:
        raise ValueError(f"KITTI pose file must have 12 columns, got shape {raw.shape}")
    poses = np.tile(np.eye(4, dtype=np.float64), (raw.shape[0], 1, 1))
    poses[:, :3, :] = raw.reshape(-1, 3, 4)
    return poses


def write_kitti_poses(path: str | Path, poses: np.ndarray | list[np.ndarray]) -> None:
    pose_arr = _as_pose_array(poses)
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for pose in pose_arr:
        rows.append(" ".join(f"{float(value):.9f}" for value in pose[:3, :].reshape(-1)))
    out_path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")


def write_tum_trajectory_from_kitti(
    path: str | Path,
    poses: np.ndarray | list[np.ndarray],
    *,
    timestamps: np.ndarray | list[float] | None = None,
) -> None:
    pose_arr = _as_pose_array(poses)
    if timestamps is None:
        ts = np.arange(pose_arr.shape[0], dtype=np.float64)
    else:
        ts = np.asarray(timestamps, dtype=np.float64)
    if ts.shape[0] != pose_arr.shape[0]:
        raise ValueError("timestamps length must match pose count")
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for timestamp, pose in zip(ts, pose_arr):
        quat = R.from_matrix(np.asarray(pose[:3, :3], dtype=np.float64)).as_quat()
        tx, ty, tz = [float(v) for v in pose[:3, 3]]
        qx, qy, qz, qw = [float(v) for v in quat]
        rows.append(
            f"{float(timestamp):.9f} {tx:.9f} {ty:.9f} {tz:.9f} "
            f"{qx:.9f} {qy:.9f} {qz:.9f} {qw:.9f}"
        )
    out_path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")


def poses_from_dpvo_output(poses_xyz_xyzw: np.ndarray) -> np.ndarray:
    arr = np.asarray(poses_xyz_xyzw, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 7:
        raise ValueError(f"expected DPVO poses with shape [N,7], got {arr.shape}")
    quat = np.asarray(arr[:, 3:7], dtype=np.float64)
    quat_norm = np.linalg.norm(quat, axis=1)
    if np.any(~np.isfinite(quat_norm)) or np.any(quat_norm <= 0.0):
        raise RuntimeError("Found zero-norm or non-finite quaternion in DPVO output")
    out = np.tile(np.eye(4, dtype=np.float64), (arr.shape[0], 1, 1))
    out[:, :3, 3] = arr[:, :3]
    rotations = R.from_quat(quat / quat_norm[:, None]).as_matrix()
    out[:, :3, :3] = rotations
    return out


def _trajectory_distances(poses: np.ndarray) -> np.ndarray:
    distances = np.zeros((poses.shape[0],), dtype=np.float64)
    for idx in range(1, poses.shape[0]):
        delta = poses[idx - 1, :3, 3] - poses[idx, :3, 3]
        distances[idx] = distances[idx - 1] + float(np.linalg.norm(delta))
    return distances


def _last_frame_from_segment_length(distances: np.ndarray, first_frame: int, length_m: float) -> int:
    target = float(distances[first_frame]) + float(length_m)
    idx = np.searchsorted(distances, target, side="left")
    return int(idx) if idx < distances.shape[0] else -1


def _rotation_error_rad(pose_error: np.ndarray) -> float:
    trace_term = 0.5 * (float(pose_error[0, 0]) + float(pose_error[1, 1]) + float(pose_error[2, 2]) - 1.0)
    return float(np.arccos(np.clip(trace_term, -1.0, 1.0)))


def _translation_error_m(pose_error: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(pose_error[:3, 3], dtype=np.float64)))


def calculate_kitti_sequence_errors(
    poses_gt: np.ndarray | list[np.ndarray],
    poses_est: np.ndarray | list[np.ndarray],
    *,
    lengths_m: list[int] | None = None,
    step_size: int = KITTI_OFFICIAL_STEP_SIZE,
) -> list[dict[str, float]]:
    gt = _as_pose_array(poses_gt)
    est = _as_pose_array(poses_est)
    if gt.shape[0] != est.shape[0]:
        raise ValueError("ground truth and estimate must have the same number of poses")
    if gt.shape[0] < 2:
        return []
    lengths = list(lengths_m or KITTI_OFFICIAL_LENGTHS)
    distances = _trajectory_distances(gt)
    errors: list[dict[str, float]] = []
    for first_frame in range(0, gt.shape[0], int(step_size)):
        for length_m in lengths:
            last_frame = _last_frame_from_segment_length(distances, first_frame, float(length_m))
            if last_frame < 0:
                continue
            pose_delta_gt = np.linalg.inv(gt[first_frame]) @ gt[last_frame]
            pose_delta_est = np.linalg.inv(est[first_frame]) @ est[last_frame]
            pose_error = np.linalg.inv(pose_delta_est) @ pose_delta_gt
            num_frames = float(last_frame - first_frame + 1)
            speed = float(length_m) / (0.1 * num_frames)
            errors.append(
                {
                    "first_frame": float(first_frame),
                    "rotation_error_per_m_rad": _rotation_error_rad(pose_error) / float(length_m),
                    "translation_error_per_m": _translation_error_m(pose_error) / float(length_m),
                    "segment_length_m": float(length_m),
                    "speed_m_per_s": speed,
                }
            )
    return errors


def compute_kitti_official_metrics(
    poses_gt: np.ndarray | list[np.ndarray],
    poses_est: np.ndarray | list[np.ndarray],
    *,
    lengths_m: list[int] | None = None,
    step_size: int = KITTI_OFFICIAL_STEP_SIZE,
) -> dict[str, float]:
    errors = calculate_kitti_sequence_errors(
        poses_gt,
        poses_est,
        lengths_m=lengths_m,
        step_size=step_size,
    )
    if not errors:
        raise RuntimeError("Not enough valid KITTI pose pairs for official metrics")
    rot_values = [float(item["rotation_error_per_m_rad"]) for item in errors]
    trans_values = [float(item["translation_error_per_m"]) for item in errors]
    return {
        "kitti_trans_percent": float(np.mean(trans_values) * 100.0),
        "kitti_rot_deg_per_m": float(np.mean(rot_values) * 180.0 / math.pi),
        "kitti_segment_count": float(len(errors)),
    }
