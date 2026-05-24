from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np
import torch

from refocus_vo.data import (
    DPVO_VALIDATION_SPLIT,
    discover_tartanair_sequences,
    matrix_to_pose_vector,
    read_tartanair_rgb,
    scale_intrinsics,
)
from refocus_vo.eval.metrics import compute_metrics
from refocus_vo.eval.plot_trajectory_3d import main as _plot_trajectory_main
from refocus_vo.eval.sparse_vo import REPO_ROOT, _append_csv_row
from refocus_vo.eval.validate_trajectory import validate
from refocus_vo.patchgraph import DinoPatchGraphTracker, load_patchgraph_config
from refocus_vo.train_patchgraph import _build_model, load_patchgraph_state_dict


def _write_tum_pose_file(path: Path, timestamps: list[float], poses: list[np.ndarray]) -> None:
    lines = []
    for ts, pose in zip(timestamps, poses):
        pose_vec = matrix_to_pose_vector(pose)
        tx, ty, tz, qx, qy, qz, qw = pose_vec.tolist()
        lines.append(f"{ts:.6f} {tx:.9f} {ty:.9f} {tz:.9f} {qx:.9f} {qy:.9f} {qz:.9f} {qw:.9f}")
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _run_plot(gt_file: Path, traj_file: Path, output_png: Path, title: str, max_dt: float) -> None:
    argv_backup = sys.argv[:]
    try:
        sys.argv = [
            "plot_trajectory_3d.py",
            "--groundtruth",
            str(gt_file),
            "--trajectory",
            str(traj_file),
            "--title",
            title,
            "--output",
            str(output_png),
            "--max-dt",
            str(max_dt),
        ]
        _plot_trajectory_main()
    finally:
        sys.argv = argv_backup


def _load_checkpoint(checkpoint: Path, config_path: Path | None, device: str):
    ckpt = torch.load(checkpoint, map_location="cpu")
    cfg = load_patchgraph_config(config_path or Path(ckpt["config_path"]))
    cfg.raw.setdefault("training", {})["device"] = str(device)
    model = _build_model(cfg)
    load_patchgraph_state_dict(model, ckpt["state_dict"])
    model.eval()
    return cfg, model


def _select_eval_sequences(dataset_root: Path):
    available = discover_tartanair_sequences(dataset_root, include_validation=True, environments=())
    validation = [seq for seq in available if seq.key in DPVO_VALIDATION_SPLIT]
    if validation:
        return validation, "dpvo_validation"
    if available:
        return available, "available_subset"
    return [], "missing"


def evaluate_sequence(
    *,
    sequence,
    cfg,
    model,
    output_dir: Path,
    max_dt: float,
    missing_penalty_m: float,
    min_coverage_ok: float,
) -> tuple[str, dict | None]:
    seq_short = sequence.key.replace("/", "__")
    run_dir = output_dir / seq_short
    traj_dir = output_dir / "trajectories"
    plot_dir = output_dir / "plots" / seq_short
    run_dir.mkdir(parents=True, exist_ok=True)
    traj_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    expected_ts = run_dir / "expected_timestamps.txt"
    gt_file = run_dir / "groundtruth_tum.txt"
    traj_file = traj_dir / f"{seq_short}_dino_patchgraph.txt"
    plot_file = plot_dir / "trajectory_3d.png"

    timestamps = [float(i) for i in range(sequence.num_frames)]
    expected_ts.write_text("\n".join(f"{ts:.6f}" for ts in timestamps) + "\n", encoding="utf-8")
    gt_poses = [sequence.poses[i] for i in range(sequence.num_frames)]
    gt_mats = []
    from refocus_vo.data import pose_vector_to_matrix

    for pose in gt_poses:
        gt_mats.append(pose_vector_to_matrix(pose))
    _write_tum_pose_file(gt_file, timestamps, gt_mats)

    intrinsics = scale_intrinsics(
        sequence.intrinsics,
        src_height=480,
        src_width=640,
        dst_height=int(cfg.model.get("image_size", [240, 320])[0]),
        dst_width=int(cfg.model.get("image_size", [240, 320])[1]),
    )
    tracker = DinoPatchGraphTracker(
        model,
        intrinsics=intrinsics,
        track_confidence_threshold=float(cfg.eval.get("track_confidence_threshold", 0.35)),
        max_history=int(cfg.model.get("max_history", 3)),
        use_geometric_pose=bool(cfg.eval.get("use_geometric_pose", True)),
    )

    est_poses = []
    try:
        for frame_idx, image_path in enumerate(sequence.image_files):
            rgb = read_tartanair_rgb(image_path, tuple(int(v) for v in cfg.model.get("image_size", [240, 320])))
            pose = tracker.step(rgb, float(frame_idx))
            est_poses.append(pose.copy())
    except Exception as exc:
        (run_dir / "error.txt").write_text(str(exc), encoding="utf-8")
        if len(est_poses) < 2:
            return "tracking_failed", None
        _write_tum_pose_file(traj_file, timestamps[: len(est_poses)], est_poses)
        try:
            metrics = compute_metrics(
                str(gt_file),
                str(traj_file),
                max_dt=max_dt,
                missing_penalty=missing_penalty_m,
                expected_timestamps_file=str(expected_ts),
                correct_scale=True,
            )
        except Exception:
            return "partial_failed", None
        return "partial_failed", metrics

    _write_tum_pose_file(traj_file, timestamps, est_poses)
    try:
        validate(traj_file, gt_file, max_dt)
        metrics = compute_metrics(
            str(gt_file),
            str(traj_file),
            max_dt=max_dt,
            missing_penalty=missing_penalty_m,
            expected_timestamps_file=str(expected_ts),
            correct_scale=True,
        )
    except Exception as exc:
        (run_dir / "error.txt").write_text(str(exc), encoding="utf-8")
        return "invalid_trajectory", None

    try:
        _run_plot(gt_file, traj_file, plot_file, f"{sequence.key} - dino_patchgraph", max_dt)
    except Exception:
        pass

    coverage = float(metrics.get("coverage", float("nan")))
    if math.isfinite(coverage) and coverage < float(min_coverage_ok):
        return "partial_low_coverage", metrics
    return "ok", metrics


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate DINO patch-graph VO on TartanAir validation split.")
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--csv-path", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-dt", type=float, default=0.02)
    ap.add_argument("--missing-penalty-m", type=float, default=3.0)
    ap.add_argument("--min-coverage-ok", type=float, default=0.95)
    args = ap.parse_args()

    cfg, model = _load_checkpoint(
        Path(args.checkpoint).expanduser().resolve(),
        Path(args.config).expanduser().resolve() if args.config else None,
        args.device,
    )
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = Path(args.csv_path).expanduser().resolve() if args.csv_path else (output_dir / "metrics_summary.csv")

    sequences, sequence_mode = _select_eval_sequences(dataset_root)
    if not sequences:
        raise FileNotFoundError(f"No TartanAir sequences found under {dataset_root}")
    if sequence_mode != "dpvo_validation":
        print(
            f"[dino_patchgraph_tartanair] WARNING: DPVO validation split not present under {dataset_root}; "
            f"falling back to {len(sequences)} available converted sequence(s)."
        )

    with csv_path.open("w", encoding="utf-8", newline="") as f:
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
                "coverage",
            ]
        )

    for seq in sequences:
        status, metrics = evaluate_sequence(
            sequence=seq,
            cfg=cfg,
            model=model,
            output_dir=output_dir,
            max_dt=float(args.max_dt),
            missing_penalty_m=float(args.missing_penalty_m),
            min_coverage_ok=float(args.min_coverage_ok),
        )
        _append_csv_row(
            csv_path,
            sequence=seq.key,
            feature_type=str(cfg.feature_type),
            status=status,
            metrics=metrics,
        )
        print(f"[{seq.key}] method=dino_patchgraph status={status}")

    print(f"DINO patch-graph TartanAir results written to {csv_path}")


if __name__ == "__main__":
    main()
