from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Iterable, List

import cv2
import numpy as np
import yaml
from scipy.spatial.transform import Rotation as R

from refocus_vo.eval.metrics import compute_metrics
from refocus_vo.eval.plot_trajectory_3d import main as _plot_trajectory_main
from refocus_vo.eval.validate_trajectory import validate


REPO_ROOT = Path(__file__).resolve().parents[4]
PYSLAM_ROOT = REPO_ROOT / "pyslam_integration" / "pyslam"


DEFAULT_SEQUENCES = [
    "freiburg1_desk",
    "freiburg1_plant",
    "freiburg1_room",
    "freiburg2_desk_with_person",
    "freiburg3_large_cabinet",
    "freiburg3_walking_static",
]


def _normalize_sequence_name(name: str) -> str:
    short = str(name).strip().replace("rgbd_dataset_", "", 1)
    return short


def _full_sequence_name(name: str) -> str:
    return f"rgbd_dataset_{_normalize_sequence_name(name)}"


def _short_alias(sequence: str) -> str:
    mapping = {
        "freiburg1_desk": "f1_desk",
        "freiburg1_plant": "f1_plant",
        "freiburg1_room": "f1_room",
        "freiburg2_desk_with_person": "f2_desk_person",
        "freiburg3_large_cabinet": "f3_lcabinet",
        "freiburg3_walking_static": "f3_wstatic",
    }
    return mapping.get(sequence, sequence)


def _tum_settings_name(sequence: str) -> str:
    seq = str(sequence).replace("rgbd_dataset_", "", 1)
    if seq.startswith("freiburg1_"):
        return "TUM1.yaml"
    if seq.startswith("freiburg2_"):
        return "TUM2.yaml"
    if seq.startswith("freiburg3_"):
        return "TUM3.yaml"
    raise ValueError(f"Unsupported TUM sequence name: {sequence}")


def _load_assoc_frames(sequence_dir: Path) -> list[dict]:
    assoc_path = sequence_dir / "associations.txt"
    if not assoc_path.exists():
        raise FileNotFoundError(f"Missing associations file: {assoc_path}")

    frames = []
    for raw in assoc_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        ts_rgb, rgb_rel, ts_depth, depth_rel = line.split()[:4]
        frames.append(
            {
                "t_rgb": float(ts_rgb),
                "rgb": str((sequence_dir / rgb_rel).resolve()),
                "t_depth": float(ts_depth),
                "depth": str((sequence_dir / depth_rel).resolve()),
            }
        )
    return frames


def _write_expected_timestamps(path: Path, frames: Iterable[dict]) -> None:
    lines = [f"{float(f['t_rgb']):.6f}" for f in frames]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _write_tum_trajectory(
    path: Path,
    timestamps: List[float],
    poses_w_c: List[np.ndarray],
) -> None:
    rows: List[str] = []
    for ts, pose in zip(timestamps, poses_w_c):
        rot = R.from_matrix(np.asarray(pose[:3, :3], dtype=np.float64))
        qx, qy, qz, qw = rot.as_quat()
        tx, ty, tz = [float(v) for v in pose[:3, 3]]
        rows.append(
            f"{float(ts):.6f} {tx:.9f} {ty:.9f} {tz:.9f} "
            f"{float(qx):.9f} {float(qy):.9f} {float(qz):.9f} {float(qw):.9f}"
        )
    path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")


def _fmt_metric(value) -> str:
    try:
        num = float(value)
    except Exception:
        return "NaN"
    return "NaN" if not math.isfinite(num) else f"{num:.6f}"


def _append_csv_row(
    csv_path: Path,
    *,
    sequence: str,
    feature_type: str,
    status: str,
    metrics: dict | None,
) -> None:
    metrics = dict(metrics or {})
    with csv_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                sequence,
                feature_type,
                status,
                _fmt_metric(metrics.get("ate_rmse")),
                _fmt_metric(metrics.get("ate_mean")),
                _fmt_metric(metrics.get("ate_median")),
                _fmt_metric(metrics.get("ate_rmse_associated")),
                _fmt_metric(metrics.get("ate_mean_associated")),
                _fmt_metric(metrics.get("ate_median_associated")),
                _fmt_metric(metrics.get("rpe_trans_rmse")),
                _fmt_metric(metrics.get("rpe_rot_rmse")),
                _fmt_metric(metrics.get("scale_correction")),
                _fmt_metric(metrics.get("scale_error_abs")),
                _fmt_metric(metrics.get("scale_error_abs_log")),
                _fmt_metric(metrics.get("coverage")),
            ]
        )


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


def _load_pyslam_camera(sequence: str, settings_dir: str | Path | None = None):
    from pyslam.slam.camera import PinholeCamera

    settings_root = (
        Path(settings_dir)
        if settings_dir is not None
        else PYSLAM_ROOT / "settings"
    )
    settings_path = settings_root / _tum_settings_name(sequence)
    if not settings_path.exists():
        raise FileNotFoundError(f"TUM settings file not found: {settings_path}")
    cfg = yaml.safe_load(settings_path.read_text(encoding="utf-8")) or {}
    return PinholeCamera(
        {
            "cam_settings": cfg,
            "dataset_settings": {
                "sensor_type": "mono",
            },
        }
    )


def evaluate_sequence(
    *,
    dataset_root: Path,
    sequence: str,
    output_dir: Path,
    max_dt: float,
    missing_penalty_m: float,
    min_coverage_ok: float,
    feature_config_name: str,
    feature_type: str,
) -> tuple[str, dict | None]:
    from pyslam.io.ground_truth import TumGroundTruth
    from pyslam.local_features.feature_tracker import feature_tracker_factory
    from pyslam.local_features.feature_tracker_configs import FeatureTrackerConfigs
    from pyslam.slam.visual_odometry import VisualOdometryEducational

    seq_short = _normalize_sequence_name(sequence)
    seq_full = _full_sequence_name(sequence)
    seq_dir = dataset_root / seq_full
    if not seq_dir.exists():
        return "skipped_missing_sequence", None

    gt_file = seq_dir / "groundtruth.txt"
    if not gt_file.exists():
        return "skipped_missing_groundtruth", None

    frames = _load_assoc_frames(seq_dir)
    if len(frames) < 2:
        return "tracking_failed", None

    run_dir = output_dir / seq_short
    traj_dir = output_dir / "trajectories"
    plot_dir = output_dir / "plots" / seq_short
    run_dir.mkdir(parents=True, exist_ok=True)
    traj_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    expected_ts = run_dir / "expected_timestamps.txt"
    traj_suffix = feature_type.lower().replace(" ", "_")
    traj_path = traj_dir / f"{_short_alias(seq_short)}_{traj_suffix}.txt"
    plot_path = plot_dir / "trajectory_3d.png"
    _write_expected_timestamps(expected_ts, frames)

    gt = TumGroundTruth(
        str(dataset_root),
        seq_full,
        associations="associations.txt",
    )
    camera = _load_pyslam_camera(seq_full)
    tracker_cfg_raw = FeatureTrackerConfigs.get_config_from_name(feature_config_name)
    if tracker_cfg_raw is None:
        raise ValueError(f"Unknown pySLAM feature tracker config: {feature_config_name}")
    tracker_cfg = dict(tracker_cfg_raw)
    feature_tracker = feature_tracker_factory(**tracker_cfg)
    vo = VisualOdometryEducational(camera, gt, feature_tracker)

    timestamps: List[float] = [float(frames[0]["t_rgb"])]
    poses_w_c: List[np.ndarray] = [np.eye(4, dtype=np.float64)]

    try:
        for frame_idx, frame in enumerate(frames):
            bgr = cv2.imread(frame["rgb"], cv2.IMREAD_COLOR)
            if bgr is None:
                raise FileNotFoundError(frame["rgb"])
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            vo.track(rgb, None, None, frame_idx, float(frame["t_rgb"]))
            if frame_idx == 0:
                continue
            pose = np.eye(4, dtype=np.float64)
            pose[:3, :3] = np.asarray(vo.cur_R, dtype=np.float64)
            pose[:3, 3] = np.asarray(vo.cur_t, dtype=np.float64).reshape(3)
            timestamps.append(float(frame["t_rgb"]))
            poses_w_c.append(pose)
    except Exception as exc:
        (run_dir / "error.txt").write_text(str(exc), encoding="utf-8")
        if len(poses_w_c) < 2:
            return "tracking_failed", None
        _write_tum_trajectory(traj_path, timestamps, poses_w_c)
        try:
            validate(traj_path, gt_file, max_dt)
            metrics = compute_metrics(
                str(gt_file),
                str(traj_path),
                max_dt=max_dt,
                missing_penalty=missing_penalty_m,
                expected_timestamps_file=str(expected_ts),
            )
        except Exception:
            return "partial_failed", None
        return "partial_failed", metrics

    _write_tum_trajectory(traj_path, timestamps, poses_w_c)
    try:
        validate(traj_path, gt_file, max_dt)
        metrics = compute_metrics(
            str(gt_file),
            str(traj_path),
            max_dt=max_dt,
            missing_penalty=missing_penalty_m,
            expected_timestamps_file=str(expected_ts),
        )
    except Exception:
        return "invalid_trajectory", None

    try:
        _run_plot(gt_file, traj_path, plot_path, f"{seq_short} - {feature_type}", max_dt)
    except Exception:
        pass

    coverage = float(metrics.get("coverage", float("nan")))
    if math.isfinite(coverage) and coverage < float(min_coverage_ok):
        return "partial_low_coverage", metrics
    return "ok", metrics


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate sparse pySLAM visual odometry on TUM RGB-D.")
    ap.add_argument("--dataset-root", default=str(REPO_ROOT / "src" / "dino_slam3" / "data" / "tum_rgbd"))
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--csv-path", default=None)
    ap.add_argument("--sequences", default=",".join(DEFAULT_SEQUENCES))
    ap.add_argument("--max-dt", type=float, default=0.02)
    ap.add_argument("--missing-penalty-m", type=float, default=3.0)
    ap.add_argument("--min-coverage-ok", type=float, default=0.95)
    ap.add_argument("--feature-config", default="ORB2", help="FeatureTrackerConfigs name, e.g. ORB2, ORB, SIFT.")
    ap.add_argument("--feature-type", default=None, help="Label written to the metrics CSV.")
    args = ap.parse_args()

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = (
        Path(args.csv_path).expanduser().resolve()
        if args.csv_path is not None
        else output_dir / "metrics_summary.csv"
    )
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    sequences = [_normalize_sequence_name(s) for s in str(args.sequences).split(",") if s.strip()]
    if not sequences:
        raise ValueError("No sequences selected")
    feature_config_name = str(args.feature_config).strip()
    feature_type = str(args.feature_type or f"{feature_config_name}_VO").strip()

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
            output_dir=output_dir,
            max_dt=float(args.max_dt),
            missing_penalty_m=float(args.missing_penalty_m),
            min_coverage_ok=float(args.min_coverage_ok),
            feature_config_name=feature_config_name,
            feature_type=feature_type,
        )
        _append_csv_row(
            csv_path,
            sequence=seq,
            feature_type=feature_type,
            status=status,
            metrics=metrics,
        )
        print(f"[{seq}] method={feature_type} status={status}")


if __name__ == "__main__":
    main()
