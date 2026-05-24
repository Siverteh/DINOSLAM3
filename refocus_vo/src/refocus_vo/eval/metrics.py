from __future__ import annotations

import math
from pathlib import Path

import numpy as np


def _safe_stats_from_errors(errors: np.ndarray) -> dict[str, float]:
    if errors.size == 0:
        return {"rmse": math.nan, "mean": math.nan, "median": math.nan}
    sq = np.square(errors, dtype=np.float64)
    return {
        "rmse": float(np.sqrt(np.mean(sq))),
        "mean": float(np.mean(errors)),
        "median": float(np.median(errors)),
    }


def _read_expected_timestamps(path: str | None) -> np.ndarray:
    if path is None:
        return np.zeros((0,), dtype=np.float64)
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"expected timestamps file not found: {path}")
    out = []
    for idx, raw in enumerate(p.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        token = line.split()[0]
        try:
            ts = float(token)
        except Exception as exc:
            raise ValueError(f"invalid timestamp in expected-timestamps file at line {idx}: '{token}'") from exc
        if math.isfinite(ts):
            out.append(ts)
    if not out:
        return np.zeros((0,), dtype=np.float64)
    arr = np.asarray(out, dtype=np.float64)
    if np.any(np.diff(arr) < 0):
        arr = np.sort(arr)
    return arr


def _nearest_indices(sorted_values: np.ndarray, queries: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    idx = np.searchsorted(sorted_values, queries)
    idx_r = np.clip(idx, 0, sorted_values.shape[0] - 1)
    idx_l = np.clip(idx - 1, 0, sorted_values.shape[0] - 1)
    dl = np.abs(sorted_values[idx_l] - queries)
    dr = np.abs(sorted_values[idx_r] - queries)
    choose_r = dr < dl
    best_idx = np.where(choose_r, idx_r, idx_l)
    best_diff = np.where(choose_r, dr, dl)
    return best_idx, best_diff


def compute_metrics(
    gt_path: str,
    traj_path: str,
    max_dt: float = 0.02,
    missing_penalty: float = 1.0,
    expected_timestamps_file: str | None = None,
    correct_scale: bool = False,
):
    from evo.core import metrics, sync
    from evo.core.metrics import PoseRelation, Unit
    from evo.tools import file_interface

    if not math.isfinite(missing_penalty) or missing_penalty <= 0.0:
        raise ValueError(f"missing_penalty must be > 0 and finite, got: {missing_penalty}")

    traj_ref = file_interface.read_tum_trajectory_file(gt_path)
    traj_est = file_interface.read_tum_trajectory_file(traj_path)
    num_gt = int(traj_ref.num_poses)
    num_est = int(traj_est.num_poses)

    traj_ref_assoc, traj_est_assoc = sync.associate_trajectories(traj_ref, traj_est, max_diff=max_dt)
    num_assoc = int(traj_est_assoc.num_poses)
    if traj_est_assoc.num_poses < 2:
        raise RuntimeError("Not enough associated poses for metrics")

    _, _, scale = traj_est_assoc.align(traj_ref_assoc, correct_scale=bool(correct_scale))
    scale_correction = float(scale) if scale is not None else 1.0
    if not math.isfinite(scale_correction) or scale_correction <= 0.0:
        scale_correction = math.nan
    scale_error_abs = (
        abs(scale_correction - 1.0)
        if math.isfinite(scale_correction)
        else math.nan
    )
    scale_error_abs_log = (
        abs(math.log(scale_correction))
        if math.isfinite(scale_correction) and scale_correction > 0.0
        else math.nan
    )

    ape = metrics.APE(PoseRelation.translation_part)
    ape.process_data((traj_ref_assoc, traj_est_assoc))
    ape_stats = ape.get_all_statistics()
    ape_errors_assoc = np.asarray(ape.error, dtype=np.float64)

    rpe_t = metrics.RPE(PoseRelation.translation_part, delta=1, delta_unit=Unit.frames, all_pairs=False)
    rpe_t.process_data((traj_ref_assoc, traj_est_assoc))
    rpe_t_stats = rpe_t.get_all_statistics()

    rpe_r = metrics.RPE(PoseRelation.rotation_angle_deg, delta=1, delta_unit=Unit.frames, all_pairs=False)
    rpe_r.process_data((traj_ref_assoc, traj_est_assoc))
    rpe_r_stats = rpe_r.get_all_statistics()

    expected_ts = _read_expected_timestamps(expected_timestamps_file)
    if expected_ts.size == 0:
        expected_ts = np.asarray(traj_ref.timestamps, dtype=np.float64)

    assoc_est_ts = np.asarray(traj_est_assoc.timestamps, dtype=np.float64)
    full_errors = np.full(expected_ts.shape[0], float(missing_penalty), dtype=np.float64)
    matched_mask = np.zeros(expected_ts.shape[0], dtype=bool)
    if assoc_est_ts.size > 0 and expected_ts.size > 0:
        nearest_idx, nearest_diff = _nearest_indices(assoc_est_ts, expected_ts)
        matched_mask = nearest_diff <= float(max_dt)
        full_errors[matched_mask] = ape_errors_assoc[nearest_idx[matched_mask]]

    full_stats = _safe_stats_from_errors(full_errors)
    coverage = (
        float(np.count_nonzero(matched_mask)) / float(expected_ts.shape[0])
        if expected_ts.size > 0
        else 0.0
    )

    return {
        "ate_rmse": float(full_stats["rmse"]),
        "ate_mean": float(full_stats["mean"]),
        "ate_median": float(full_stats["median"]),
        "ate_rmse_associated": float(ape_stats.get("rmse", math.nan)),
        "ate_mean_associated": float(ape_stats.get("mean", math.nan)),
        "ate_median_associated": float(ape_stats.get("median", math.nan)),
        "rpe_trans_rmse": float(rpe_t_stats.get("rmse", math.nan)),
        "rpe_rot_rmse": float(rpe_r_stats.get("rmse", math.nan)),
        "scale_correction": scale_correction,
        "scale_error_abs": float(scale_error_abs),
        "scale_error_abs_log": float(scale_error_abs_log),
        "num_gt_poses": num_gt,
        "num_est_poses": num_est,
        "num_associated_poses": num_assoc,
        "num_eval_timestamps": int(expected_ts.shape[0]),
        "num_eval_matched": int(np.count_nonzero(matched_mask)),
        "coverage": coverage,
        "missing_penalty_m": float(missing_penalty),
        "coverage_basis": (
            "expected_timestamps" if expected_timestamps_file is not None else "groundtruth_timestamps"
        ),
        "correct_scale": bool(correct_scale),
    }
