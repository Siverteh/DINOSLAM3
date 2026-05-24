from __future__ import annotations

import argparse
import csv
import gc
import math
import sys
from argparse import Namespace
from pathlib import Path

import cv2
import numpy as np
import torch

from refocus_vo.eval.metrics import compute_metrics
from refocus_vo.eval.plot_trajectory_3d import main as _plot_trajectory_main
from refocus_vo.eval.sparse_vo import (
    DEFAULT_SEQUENCES,
    REPO_ROOT,
    _append_csv_row,
    _full_sequence_name,
    _normalize_sequence_name,
    _short_alias,
    _write_expected_timestamps,
)
from refocus_vo.eval.validate_trajectory import validate


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


def _cleanup_cuda(obj: object | None = None) -> None:
    if obj is not None:
        del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


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


def _tum_family(sequence: str) -> str:
    seq = _normalize_sequence_name(sequence)
    if seq.startswith("freiburg1_"):
        return "freiburg1"
    if seq.startswith("freiburg2_"):
        return "freiburg2"
    if seq.startswith("freiburg3_"):
        return "freiburg3"
    raise ValueError(f"Unsupported TUM sequence: {sequence}")


def _sequence_image_files(seq_dir: Path, stride: int) -> list[Path]:
    rgb_dir = seq_dir / "rgb"
    files = sorted(rgb_dir.glob("*.png"))
    if stride > 1:
        files = files[::stride]
    return files


def _build_args(
    *,
    weights: Path,
    buffer: int,
    asynchronous: bool,
    frontend_device: str,
    backend_device: str,
    disable_vis: bool,
) -> Namespace:
    return Namespace(
        datapath="",
        weights=str(weights),
        buffer=int(buffer),
        image_size=[240, 320],
        disable_vis=bool(disable_vis),
        beta=0.3,
        filter_thresh=1.5,
        warmup=12,
        keyframe_thresh=2.0,
        frontend_thresh=12.0,
        frontend_window=25,
        frontend_radius=2,
        frontend_nms=1,
        backend_thresh=20.0,
        backend_radius=2,
        backend_nms=3,
        upsample=False,
        asynchronous=bool(asynchronous),
        frontend_device=frontend_device,
        backend_device=backend_device,
        motion_damping=0.5,
        stereo=False,
    )


def _prepare_image(image_bgr: np.ndarray, sequence: str) -> tuple[torch.Tensor, torch.Tensor]:
    calib = TUM_CALIBRATIONS[_tum_family(sequence)]
    fx = float(calib["fx"])
    fy = float(calib["fy"])
    cx = float(calib["cx"])
    cy = float(calib["cy"])
    dist = np.asarray(calib["dist"], dtype=np.float64)

    K = np.array([fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0], dtype=np.float64).reshape(3, 3)
    image = cv2.undistort(image_bgr, K, dist)
    image = cv2.resize(image, (320 + 32, 240 + 16))
    image_t = torch.from_numpy(image).permute(2, 0, 1)

    intrinsics = torch.as_tensor([fx, fy, cx, cy], dtype=torch.float32)
    intrinsics[0] *= image_t.shape[2] / 640.0
    intrinsics[1] *= image_t.shape[1] / 480.0
    intrinsics[2] *= image_t.shape[2] / 640.0
    intrinsics[3] *= image_t.shape[1] / 480.0
    intrinsics[2] -= 16
    intrinsics[3] -= 8
    image_t = image_t[:, 8:-8, 16:-16]
    return image_t[None], intrinsics


def _load_droid_classes(droid_root: Path):
    sys.path.insert(0, str(droid_root))
    sys.path.insert(0, str(droid_root / "droid_slam"))
    from droid import Droid  # type: ignore
    from droid_async import DroidAsync  # type: ignore

    return Droid, DroidAsync


def evaluate_sequence(
    *,
    dataset_root: Path,
    sequence: str,
    droid_root: Path,
    weights: Path,
    output_dir: Path,
    max_dt: float,
    missing_penalty_m: float,
    min_coverage_ok: float,
    image_stride: int,
    buffer: int,
    asynchronous: bool,
    disable_vis: bool,
    frontend_device: str,
    backend_device: str,
    vo_mode: bool,
) -> tuple[str, dict | None]:
    from evo.core.trajectory import PoseTrajectory3D
    from evo.tools import file_interface

    seq_short = _normalize_sequence_name(sequence)
    seq_full = _full_sequence_name(sequence)
    seq_dir = dataset_root / seq_full
    if not seq_dir.exists():
        return "skipped_missing_sequence", None

    gt_file = seq_dir / "groundtruth.txt"
    if not gt_file.exists():
        return "skipped_missing_groundtruth", None

    image_files = _sequence_image_files(seq_dir, image_stride)
    if len(image_files) < 2:
        return "tracking_failed", None

    run_dir = output_dir / seq_short
    traj_dir = output_dir / "trajectories"
    plot_dir = output_dir / "plots" / seq_short
    run_dir.mkdir(parents=True, exist_ok=True)
    traj_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    expected_ts = run_dir / "expected_timestamps.txt"
    method_slug = "droid_vo" if vo_mode else "droid_slam"
    method_label = "DROID_VO" if vo_mode else "DROID_SLAM"
    traj_path = traj_dir / f"{_short_alias(seq_short)}_{method_slug}.txt"
    plot_path = plot_dir / "trajectory_3d.png"
    error_path = run_dir / "error.txt"
    _write_expected_timestamps(
        expected_ts,
        [{"t_rgb": float(path.stem)} for path in image_files],
    )

    Droid, DroidAsync = _load_droid_classes(droid_root)
    args = _build_args(
        weights=weights,
        buffer=buffer,
        asynchronous=asynchronous,
        frontend_device=frontend_device,
        backend_device=backend_device,
        disable_vis=disable_vis,
    )
    torch.multiprocessing.set_start_method("spawn", force=True)
    if vo_mode and asynchronous:
        error_path.write_text("DROID-VO mode is not supported with asynchronous backend execution\n", encoding="utf-8")
        return "skipped_invalid_config", None

    droid = DroidAsync(args) if asynchronous else Droid(args)

    stream = []
    processed_paths = []
    try:
        for frame_idx, image_path in enumerate(image_files):
            bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if bgr is None:
                raise FileNotFoundError(str(image_path))
            image, intrinsics = _prepare_image(bgr, seq_short)
            # DROID's TUM evaluation uses contiguous frame indices internally;
            # real RGB timestamps are attached only when writing/evaluating the trajectory.
            droid.track(frame_idx, image, intrinsics=intrinsics)
            stream.append((frame_idx, image, intrinsics))
            processed_paths.append(image_path)
    except Exception as exc:
        error_path.write_text(str(exc), encoding="utf-8")
        _cleanup_cuda(droid)
        return "tracking_failed", None

    try:
        if vo_mode:
            # DROID-VO-style evaluation: keep the online frontend result and
            # skip Droid.terminate(), which runs full backend/global refinement.
            if hasattr(droid, "frontend"):
                del droid.frontend
            torch.cuda.empty_cache()
            traj_est = droid.traj_filler(stream).inv().data.cpu().numpy()
        else:
            traj_est = droid.terminate(stream)
    except Exception as exc:
        error_path.write_text(str(exc), encoding="utf-8")
        _cleanup_cuda(droid)
        return "tracking_failed", None

    timestamps = np.asarray([float(path.stem) for path in processed_paths], dtype=np.float64)
    if traj_est is None or len(traj_est) < 2 or timestamps.size < 2:
        return "tracking_failed", None
    n = min(len(traj_est), int(timestamps.size))
    traj_est = np.asarray(traj_est[:n], dtype=np.float64)
    timestamps = timestamps[:n]

    traj = PoseTrajectory3D(
        positions_xyz=traj_est[:, :3],
        orientations_quat_wxyz=traj_est[:, 3:],
        timestamps=timestamps,
    )
    file_interface.write_tum_trajectory_file(str(traj_path), traj)

    try:
        validate(traj_path, gt_file, max_dt)
        metrics = compute_metrics(
            str(gt_file),
            str(traj_path),
            max_dt=max_dt,
            missing_penalty=missing_penalty_m,
            expected_timestamps_file=str(expected_ts),
            correct_scale=True,
        )
    except Exception as exc:
        error_path.write_text(str(exc), encoding="utf-8")
        return "invalid_trajectory", None

    try:
        _run_plot(gt_file, traj_path, plot_path, f"{seq_short} - {method_label}", max_dt)
    except Exception:
        pass

    coverage = float(metrics.get("coverage", float("nan")))
    if math.isfinite(coverage) and coverage < float(min_coverage_ok):
        status = "partial_low_coverage"
    else:
        status = "ok"
    if error_path.exists():
        error_path.unlink()
    _cleanup_cuda(droid)
    return status, metrics


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate DROID-SLAM on selected TUM RGB-D sequences.")
    ap.add_argument("--dataset-root", default=str(REPO_ROOT / "src" / "dino_slam3" / "data" / "tum_rgbd"))
    ap.add_argument("--droid-root", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--csv-path", default=None)
    ap.add_argument("--sequences", default=",".join(DEFAULT_SEQUENCES))
    ap.add_argument("--max-dt", type=float, default=0.02)
    ap.add_argument("--missing-penalty-m", type=float, default=3.0)
    ap.add_argument("--min-coverage-ok", type=float, default=0.95)
    ap.add_argument("--image-stride", type=int, default=2)
    ap.add_argument("--buffer", type=int, default=512)
    ap.add_argument("--asynchronous", action="store_true")
    ap.add_argument("--disable-vis", action="store_true")
    ap.add_argument("--frontend-device", default="cuda:0")
    ap.add_argument("--backend-device", default="cuda:0")
    ap.add_argument(
        "--vo-mode",
        action="store_true",
        help="Skip DROID backend refinement at termination and evaluate the online VO-style frontend trajectory.",
    )
    args = ap.parse_args()

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    droid_root = Path(args.droid_root).expanduser().resolve()
    weights = Path(args.weights).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = (
        Path(args.csv_path).expanduser().resolve()
        if args.csv_path is not None
        else output_dir / "metrics_summary.csv"
    )
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    if not droid_root.exists():
        raise FileNotFoundError(f"DROID-SLAM repo not found: {droid_root}")
    if not weights.exists():
        raise FileNotFoundError(f"DROID-SLAM weights not found: {weights}")
    if not torch.cuda.is_available():
        raise RuntimeError("DROID-SLAM requires CUDA for this wrapper.")

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
                "scale_correction",
                "scale_error_abs",
                "scale_error_abs_log",
                "coverage",
            ]
        )

    for seq in sequences:
        status, metrics = evaluate_sequence(
            dataset_root=dataset_root,
            sequence=seq,
            droid_root=droid_root,
            weights=weights,
            output_dir=output_dir,
            max_dt=float(args.max_dt),
            missing_penalty_m=float(args.missing_penalty_m),
            min_coverage_ok=float(args.min_coverage_ok),
            image_stride=max(int(args.image_stride), 1),
            buffer=int(args.buffer),
            asynchronous=bool(args.asynchronous),
            disable_vis=bool(args.disable_vis),
            frontend_device=str(args.frontend_device),
            backend_device=str(args.backend_device),
            vo_mode=bool(args.vo_mode),
        )
        _append_csv_row(
            csv_path,
            sequence=seq,
            feature_type="DROID_VO" if args.vo_mode else "DROID_SLAM",
            status=status,
            metrics=metrics,
        )
        method = "droid_vo" if args.vo_mode else "droid_slam"
        print(f"[{seq}] method={method} status={status}")


if __name__ == "__main__":
    main()
