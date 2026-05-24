from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


def compute_dpvo_style_metrics(
    gt_path: str | Path,
    traj_path: str | Path,
    *,
    max_dt: float = 0.02,
) -> dict[str, float]:
    import evo.main_ape as main_ape
    from evo.core import sync
    from evo.core.metrics import PoseRelation
    from evo.tools import file_interface

    traj_ref = file_interface.read_tum_trajectory_file(str(gt_path))
    traj_est = file_interface.read_tum_trajectory_file(str(traj_path))
    traj_ref_assoc, traj_est_assoc = sync.associate_trajectories(traj_ref, traj_est, max_diff=float(max_dt))
    if traj_est_assoc.num_poses < 2:
        raise RuntimeError("Not enough associated poses for DPVO-style metrics")
    _, _, scale = traj_est_assoc.align(traj_ref_assoc, correct_scale=True)
    scale_correction = float(scale) if scale is not None else 1.0
    if not math.isfinite(scale_correction) or scale_correction <= 0.0:
        scale_correction = math.nan

    result = main_ape.ape(
        traj_ref_assoc,
        traj_est_assoc,
        est_name="traj",
        pose_relation=PoseRelation.translation_part,
        align=True,
        correct_scale=True,
    )
    stats = dict(result.stats)
    return {
        "ate_rmse": float(stats.get("rmse", math.nan)),
        "ate_mean": float(stats.get("mean", math.nan)),
        "ate_median": float(stats.get("median", math.nan)),
        "num_associated_poses": int(traj_est_assoc.num_poses),
        "scale_correction": scale_correction,
        "scale_error_abs": (
            abs(scale_correction - 1.0)
            if math.isfinite(scale_correction)
            else math.nan
        ),
        "scale_error_abs_log": (
            abs(math.log(scale_correction))
            if math.isfinite(scale_correction) and scale_correction > 0.0
            else math.nan
        ),
        "correct_scale": True,
    }


def write_dpvo_style_csv(
    input_csv: str | Path,
    output_csv: str | Path,
) -> None:
    input_path = Path(input_csv).expanduser().resolve()
    output_path = Path(output_csv).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        with output_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "sequence",
                    "feature_type",
                    "status",
                    "ate_rmse",
                    "ate_mean",
                    "ate_median",
                    "ate_rmse_associated",
                    "ate_mean_associated",
                    "ate_median_associated",
                    "rpe_trans_rmse",
                    "rpe_rot_rmse",
                    "scale_correction",
                    "scale_error_abs",
                    "scale_error_abs_log",
                    "coverage",
                ]
            )
        return

    fieldnames = list(rows[0].keys())
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            out["ate_rmse"] = row.get("ate_rmse_associated", "NaN")
            out["ate_mean"] = row.get("ate_mean_associated", "NaN")
            out["ate_median"] = row.get("ate_median_associated", "NaN")
            writer.writerow(out)


def main() -> None:
    ap = argparse.ArgumentParser(description="Create a DPVO-style comparison CSV from an existing metrics CSV.")
    ap.add_argument("--input-csv", required=True)
    ap.add_argument("--output-csv", required=True)
    args = ap.parse_args()
    write_dpvo_style_csv(args.input_csv, args.output_csv)


if __name__ == "__main__":
    main()
