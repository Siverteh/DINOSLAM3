#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def compute_metrics(gt_path: str, traj_path: str, max_dt: float = 0.02):
    from evo.core import metrics, sync
    from evo.core.metrics import PoseRelation, Unit
    from evo.tools import file_interface

    traj_ref = file_interface.read_tum_trajectory_file(gt_path)
    traj_est = file_interface.read_tum_trajectory_file(traj_path)

    traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est, max_diff=max_dt)
    if traj_est.num_poses < 2:
        raise RuntimeError("Not enough associated poses for metrics")

    traj_est.align(traj_ref, correct_scale=False)

    ape = metrics.APE(PoseRelation.translation_part)
    ape.process_data((traj_ref, traj_est))
    ape_stats = ape.get_all_statistics()

    rpe_t = metrics.RPE(PoseRelation.translation_part, delta=1, delta_unit=Unit.frames, all_pairs=False)
    rpe_t.process_data((traj_ref, traj_est))
    rpe_t_stats = rpe_t.get_all_statistics()

    rpe_r = metrics.RPE(PoseRelation.rotation_angle_deg, delta=1, delta_unit=Unit.frames, all_pairs=False)
    rpe_r.process_data((traj_ref, traj_est))
    rpe_r_stats = rpe_r.get_all_statistics()

    return {
        "ate_rmse": float(ape_stats.get("rmse", math.nan)),
        "ate_mean": float(ape_stats.get("mean", math.nan)),
        "ate_median": float(ape_stats.get("median", math.nan)),
        "rpe_trans_rmse": float(rpe_t_stats.get("rmse", math.nan)),
        "rpe_rot_rmse": float(rpe_r_stats.get("rmse", math.nan)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groundtruth", required=True)
    ap.add_argument("--trajectory", required=True)
    ap.add_argument("--max-dt", type=float, default=0.02)
    ap.add_argument("--output-json", default=None)
    args = ap.parse_args()

    code = 1
    out = {}
    try:
        out = compute_metrics(args.groundtruth, args.trajectory, args.max_dt)
        code = 0
    except Exception as exc:
        out = {
            "ate_rmse": math.nan,
            "ate_mean": math.nan,
            "ate_median": math.nan,
            "rpe_trans_rmse": math.nan,
            "rpe_rot_rmse": math.nan,
            "error": str(exc),
        }

    if args.output_json:
        Path(args.output_json).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out))
    raise SystemExit(code)


if __name__ == "__main__":
    main()
