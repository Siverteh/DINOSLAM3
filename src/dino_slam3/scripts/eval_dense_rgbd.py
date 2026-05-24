from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Iterable, List

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R


REPO_ROOT = Path(__file__).resolve().parents[3]
PYSLAM_ROOT = REPO_ROOT / "pyslam"
PYSLAM_INTEGRATION_SCRIPTS = REPO_ROOT / "pyslam_integration" / "scripts"

for extra_path in (REPO_ROOT / "src", PYSLAM_ROOT, PYSLAM_INTEGRATION_SCRIPTS):
    extra_str = str(extra_path)
    if extra_str not in sys.path:
        sys.path.insert(0, extra_str)

from compute_metrics import compute_metrics  # type: ignore  # noqa: E402
from plot_trajectory_3d import main as _plot_trajectory_main  # type: ignore  # noqa: E402
from validate_trajectory import validate  # type: ignore  # noqa: E402
from dino_slam3.data.tum_rgbd import TUMRGBDDataset  # noqa: E402
from dino_slam3.vo import (  # noqa: E402
    DinoGuidedVisualOdometryRgbdTensor,
    DinoStabilityScorer,
    VisualOdometryRgbdTensor,
    load_tum_camera,
)


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
    short = _normalize_sequence_name(name)
    return f"rgbd_dataset_{short}"


def _write_expected_timestamps(
    path: Path,
    frames: Iterable[dict],
    *,
    associations_file: Path | None = None,
) -> None:
    if associations_file is not None and associations_file.exists():
        lines = []
        for raw in associations_file.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            lines.append(line.split()[0])
    else:
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
                _fmt_metric(metrics.get("coverage")),
            ]
        )


def _fmt_metric(value) -> str:
    try:
        num = float(value)
    except Exception:
        return "NaN"
    return "NaN" if not math.isfinite(num) else f"{num:.6f}"


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


def evaluate_sequence(
    *,
    dataset_root: Path,
    sequence: str,
    method: str,
    scorer: DinoStabilityScorer | None,
    output_dir: Path,
    max_dt: float,
    missing_penalty_m: float,
    min_coverage_ok: float,
    keep_ratio: float,
    device: str,
) -> tuple[str, dict | None]:
    seq_short = _normalize_sequence_name(sequence)
    seq_full = _full_sequence_name(sequence)

    seq_dir = dataset_root / seq_full
    if not seq_dir.exists():
        return "skipped_missing_sequence", None

    gt_file = seq_dir / "groundtruth.txt"
    if not gt_file.exists():
        return "skipped_missing_groundtruth", None

    ds = TUMRGBDDataset(
        dataset_root=dataset_root,
        sequence=seq_full,
        frame_spacing_min=1,
        frame_spacing_max=1,
        is_train=False,
        pad_to=16,
        max_rgb_depth_dt=max_dt,
        max_rgb_gt_dt=max_dt,
    )
    if len(ds.frames) < 2:
        return "tracking_failed", None

    run_dir = output_dir / seq_short
    traj_dir = output_dir / "trajectories"
    plot_dir = output_dir / "plots" / seq_short
    run_dir.mkdir(parents=True, exist_ok=True)
    traj_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    expected_ts = run_dir / "expected_timestamps.txt"
    traj_path = traj_dir / f"{_short_alias(seq_short)}_{method}.txt"
    plot_path = plot_dir / "trajectory_3d.png"

    _write_expected_timestamps(
        expected_ts,
        ds.frames,
        associations_file=(seq_dir / "associations.txt"),
    )

    camera = load_tum_camera(seq_full)
    if method == "dino_guided":
        if scorer is None:
            raise ValueError("dino_guided mode requires an initialized DinoStabilityScorer")
        vo = DinoGuidedVisualOdometryRgbdTensor(
            camera,
            scorer,
            keep_ratio=keep_ratio,
            coarse_method="hybrid",
            refine_method="point_to_plane",
            device=device,
        )
    else:
        vo = VisualOdometryRgbdTensor(camera, groundtruth=None, method_name=method, device=device)

    timestamps: List[float] = []
    poses_w_c: List[np.ndarray] = []
    poses_w_c.append(np.eye(4, dtype=np.float64))
    timestamps.append(float(ds.frames[0]["t_rgb"]))

    try:
        for frame_idx, frame in enumerate(ds.frames):
            rgb = ds._read_rgb_np(frame["rgb"])
            depth = ds._read_depth_np(frame["depth"])
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            vo.track(bgr, None, depth, frame_idx, float(frame["t_rgb"]))
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
        _run_plot(gt_file, traj_path, plot_path, f"{seq_short} - {method}", max_dt)
    except Exception:
        pass

    coverage = float(metrics.get("coverage", float("nan")))
    if math.isfinite(coverage) and coverage < float(min_coverage_ok):
        return "partial_low_coverage", metrics
    return "ok", metrics


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate dense RGB-D odometry baselines and DINO-guided masking on TUM RGB-D.")
    ap.add_argument("--dataset-root", default=str(REPO_ROOT / "src" / "dino_slam3" / "data" / "tum_rgbd"))
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--csv-path", default=None)
    ap.add_argument("--method", choices=["hybrid", "point_to_plane", "dino_guided"], default="dino_guided")
    ap.add_argument("--sequences", default=",".join(DEFAULT_SEQUENCES))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-dt", type=float, default=0.02)
    ap.add_argument("--missing-penalty-m", type=float, default=3.0)
    ap.add_argument("--min-coverage-ok", type=float, default=0.95)
    ap.add_argument("--keep-ratio", type=float, default=0.35)
    ap.add_argument("--dino-name-or-path", default="facebook/dinov3-vits16-pretrain-lvd1689m")
    ap.add_argument("--dino-layers", default="6,11")
    ap.add_argument("--dino-dtype", default="bf16")
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

    scorer = None
    if args.method == "dino_guided":
        layer_indices = tuple(int(v.strip()) for v in str(args.dino_layers).split(",") if v.strip())
        scorer = DinoStabilityScorer(
            backbone_name_or_path=str(args.dino_name_or_path),
            layer_indices=layer_indices,
            dtype=str(args.dino_dtype),
            device=str(args.device),
            keep_ratio=float(args.keep_ratio),
        )

    feature_label = {
        "hybrid": "RGBD_TENSOR_HYBRID",
        "point_to_plane": "RGBD_TENSOR_POINT_TO_PLANE",
        "dino_guided": "DINO_RGBD_GUIDED",
    }[args.method]

    for seq in sequences:
        status, metrics = evaluate_sequence(
            dataset_root=dataset_root,
            sequence=seq,
            method=str(args.method),
            scorer=scorer,
            output_dir=output_dir,
            max_dt=float(args.max_dt),
            missing_penalty_m=float(args.missing_penalty_m),
            min_coverage_ok=float(args.min_coverage_ok),
            keep_ratio=float(args.keep_ratio),
            device=str(args.device),
        )
        _append_csv_row(
            csv_path,
            sequence=seq,
            feature_type=feature_label,
            status=status,
            metrics=metrics,
        )
        print(f"[{seq}] method={args.method} status={status}")


if __name__ == "__main__":
    main()
