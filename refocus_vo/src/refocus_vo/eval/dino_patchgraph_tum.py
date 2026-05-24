from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from refocus_vo.data import TUMRGBDSequence
from refocus_vo.eval.dense_rgbd import DEFAULT_SEQUENCES, _full_sequence_name, _normalize_sequence_name, _short_alias
from refocus_vo.eval.metrics import compute_metrics
from refocus_vo.eval.plot_trajectory_3d import main as _plot_trajectory_main
from refocus_vo.eval.sparse_vo import _append_csv_row
from refocus_vo.eval.validate_trajectory import validate
from refocus_vo.patchgraph import DinoPatchGraphTracker, load_patchgraph_config
from refocus_vo.train_patchgraph import _build_model, load_patchgraph_state_dict


TUM_CALIBRATIONS = {
    "freiburg1": {
        "fx": 517.306408,
        "fy": 516.469215,
        "cx": 318.643040,
        "cy": 255.313989,
        "dist": [0.262383, -0.953104, -0.005358, 0.002628, 1.163314],
    },
    "freiburg2": {
        "fx": 520.908620,
        "fy": 521.007327,
        "cx": 325.141442,
        "cy": 249.701764,
        "dist": [0.231222, -0.784899, -0.003257, -0.000105, 0.917205],
    },
    "freiburg3": {
        "fx": 535.4,
        "fy": 539.2,
        "cx": 320.1,
        "cy": 247.6,
        "dist": [0.0, 0.0, 0.0, 0.0, 0.0],
    },
}


def _tum_family(sequence: str) -> str:
    seq = _normalize_sequence_name(sequence)
    if seq.startswith("freiburg1_"):
        return "freiburg1"
    if seq.startswith("freiburg2_"):
        return "freiburg2"
    if seq.startswith("freiburg3_"):
        return "freiburg3"
    raise ValueError(f"Unsupported TUM sequence: {sequence}")


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


def _write_expected_timestamps(path: Path, frames) -> None:
    path.write_text(
        "\n".join(f"{float(frame['t_rgb']):.6f}" for frame in frames) + "\n",
        encoding="utf-8",
    )


def _write_tum_trajectory(path: Path, timestamps: list[float], poses: list[np.ndarray]) -> None:
    from refocus_vo.data import matrix_to_pose_vector

    lines = []
    for ts, pose in zip(timestamps, poses):
        tx, ty, tz, qx, qy, qz, qw = matrix_to_pose_vector(pose).tolist()
        lines.append(f"{ts:.6f} {tx:.9f} {ty:.9f} {tz:.9f} {qx:.9f} {qy:.9f} {qz:.9f} {qw:.9f}")
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _prepare_tum_image(image_bgr: np.ndarray, sequence: str, *, target_height: int, target_width: int):
    calib = TUM_CALIBRATIONS[_tum_family(sequence)]
    K = np.array(
        [
            [float(calib["fx"]), 0.0, float(calib["cx"])],
            [0.0, float(calib["fy"]), float(calib["cy"])],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    image = cv2.undistort(image_bgr, K, np.asarray(calib["dist"], dtype=np.float64))
    image = cv2.resize(image, (int(target_width), int(target_height)), interpolation=cv2.INTER_LINEAR)
    sx = float(target_width) / 640.0
    sy = float(target_height) / 480.0
    intrinsics = np.asarray(
        [
            float(calib["fx"]) * sx,
            float(calib["fy"]) * sy,
            float(calib["cx"]) * sx,
            float(calib["cy"]) * sy,
        ],
        dtype=np.float64,
    )
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), intrinsics


def _load_checkpoint(checkpoint: Path, config_path: Path | None, device: str):
    ckpt = torch.load(checkpoint, map_location="cpu")
    cfg = load_patchgraph_config(config_path or Path(ckpt["config_path"]))
    cfg.raw.setdefault("training", {})["device"] = str(device)
    model = _build_model(cfg)
    load_patchgraph_state_dict(model, ckpt["state_dict"])
    model.eval()
    return cfg, model


def evaluate_sequence(
    *,
    dataset_root: Path,
    sequence: str,
    cfg,
    model,
    output_dir: Path,
    max_dt: float,
    missing_penalty_m: float,
    min_coverage_ok: float,
) -> tuple[str, dict | None]:
    seq_short = _normalize_sequence_name(sequence)
    seq_full = _full_sequence_name(sequence)
    seq_dir = dataset_root / seq_full
    if not seq_dir.exists():
        return "skipped_missing_sequence", None

    gt_file = seq_dir / "groundtruth.txt"
    if not gt_file.exists():
        return "skipped_missing_groundtruth", None

    ds = TUMRGBDSequence(dataset_root, seq_full)
    if len(ds.frames) < 2:
        return "tracking_failed", None

    run_dir = output_dir / seq_short
    traj_dir = output_dir / "trajectories"
    plot_dir = output_dir / "plots" / seq_short
    run_dir.mkdir(parents=True, exist_ok=True)
    traj_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    expected_ts = run_dir / "expected_timestamps.txt"
    traj_file = traj_dir / f"{_short_alias(seq_short)}_dino_patchgraph.txt"
    plot_file = plot_dir / "trajectory_3d.png"
    _write_expected_timestamps(expected_ts, ds.frames)

    image_size = tuple(int(v) for v in cfg.model.get("image_size", [240, 320]))
    tracker = None
    timestamps = []
    est_poses = []
    try:
        for frame_idx, frame in enumerate(ds.frames):
            bgr = cv2.imread(frame["rgb"], cv2.IMREAD_COLOR)
            if bgr is None:
                raise FileNotFoundError(frame["rgb"])
            rgb, intrinsics = _prepare_tum_image(
                bgr,
                seq_short,
                target_height=image_size[0],
                target_width=image_size[1],
            )
            if tracker is None:
                tracker = DinoPatchGraphTracker(
                    model,
                    intrinsics=intrinsics,
                    track_confidence_threshold=float(cfg.eval.get("track_confidence_threshold", 0.35)),
                    max_history=int(cfg.model.get("max_history", 3)),
                    use_geometric_pose=bool(cfg.eval.get("use_geometric_pose", True)),
                )
            pose = tracker.step(rgb, float(frame["t_rgb"]))
            timestamps.append(float(frame["t_rgb"]))
            est_poses.append(pose.copy())
    except Exception as exc:
        (run_dir / "error.txt").write_text(str(exc), encoding="utf-8")
        if len(est_poses) < 2:
            return "tracking_failed", None
        _write_tum_trajectory(traj_file, timestamps, est_poses)
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

    _write_tum_trajectory(traj_file, timestamps, est_poses)
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
        _run_plot(gt_file, traj_file, plot_file, f"{seq_short} - dino_patchgraph", max_dt)
    except Exception:
        pass

    coverage = float(metrics.get("coverage", float("nan")))
    if math.isfinite(coverage) and coverage < float(min_coverage_ok):
        return "partial_low_coverage", metrics
    return "ok", metrics


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate DINO patch-graph VO on TUM RGB-D using RGB-only transfer.")
    ap.add_argument("--dataset-root", default=str(Path(__file__).resolve().parents[4] / "src" / "dino_slam3" / "data" / "tum_rgbd"))
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--csv-path", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--sequences", default=",".join(DEFAULT_SEQUENCES))
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
    sequences = [_normalize_sequence_name(s) for s in str(args.sequences).split(",") if s.strip()]

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
            dataset_root=dataset_root,
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
            sequence=seq,
            feature_type=str(cfg.feature_type),
            status=status,
            metrics=metrics,
        )
        print(f"[{seq}] method=dino_patchgraph status={status}")

    print(f"DINO patch-graph TUM transfer results written to {csv_path}")


if __name__ == "__main__":
    main()
