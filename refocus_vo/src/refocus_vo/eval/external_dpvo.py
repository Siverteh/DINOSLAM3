from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import math
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from refocus_vo.dino_dpvo import (
    DinoDPVOTracker,
    build_dino_dpvo_frontend,
    load_dino_dpvo_config,
    load_dino_dpvo_frontend_checkpoint,
)
from refocus_vo.dino_dpvo.diagnostics import (
    GroundTruthFrameContext,
    PatchDiagnosticsRecorder,
    append_diagnostics_summary,
    append_patch_diagnostics,
    init_diagnostics_outputs,
)
from refocus_vo.data.tum_rgbd import TUMRGBDSequence
from refocus_vo.eval.dpvo_style_metrics import compute_dpvo_style_metrics
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


def _seed_runtime(seed: int, *, deterministic: bool) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)


def _frontend_eval_seed(frontend_cfg, *, sequence: str, frontend_mode: str, stride: int, backend_thresh: float) -> tuple[int | None, bool]:
    if frontend_cfg is None:
        return None, False
    eval_cfg = frontend_cfg.eval if hasattr(frontend_cfg, "eval") else {}
    training_cfg = frontend_cfg.training if hasattr(frontend_cfg, "training") else {}
    deterministic = bool(eval_cfg.get("deterministic", training_cfg.get("deterministic", False)))
    seed_runtime = bool(eval_cfg.get("seed_runtime", training_cfg.get("seed_runtime", deterministic)))
    if not seed_runtime:
        return None, deterministic
    raw_seed = eval_cfg.get("seed", training_cfg.get("seed"))
    if raw_seed in (None, ""):
        return None, deterministic
    digest = hashlib.sha256(
        f"{int(raw_seed)}|{sequence}|{frontend_mode}|{int(stride)}|{float(backend_thresh):.6f}".encode("utf-8")
    ).digest()
    offset = int.from_bytes(digest[:8], byteorder="little", signed=False) % (2**31 - 1)
    return (int(raw_seed) + offset) % (2**31 - 1), deterministic


def _cleanup_cuda(obj: object | None = None) -> None:
    if obj is not None:
        del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _dpvo_style_row_metrics(metrics: dict | None) -> dict | None:
    if metrics is None:
        return None
    row = dict(metrics)
    row["ate_rmse"] = row.get("dpvo_style_ate_rmse", row.get("ate_rmse_associated", math.nan))
    row["ate_mean"] = row.get("dpvo_style_ate_mean", row.get("ate_mean_associated", math.nan))
    row["ate_median"] = row.get("dpvo_style_ate_median", row.get("ate_median_associated", math.nan))
    row["ate_rmse_associated"] = row["ate_rmse"]
    row["ate_mean_associated"] = row["ate_mean"]
    row["ate_median_associated"] = row["ate_median"]
    return row


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


def _prepare_image(
    image_bgr: np.ndarray,
    sequence: str,
    *,
    target_height: int | None = None,
    target_width: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    calib = TUM_CALIBRATIONS[_tum_family(sequence)]
    fx = float(calib["fx"])
    fy = float(calib["fy"])
    cx = float(calib["cx"])
    cy = float(calib["cy"])
    dist = np.asarray(calib["dist"], dtype=np.float64)

    K = np.array([fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0], dtype=np.float64).reshape(3, 3)
    image = cv2.undistort(image_bgr, K, dist)

    if target_height is not None and target_width is not None:
        pad_h = 16
        pad_w = 32
        image = cv2.resize(image, (int(target_width) + pad_w, int(target_height) + pad_h))
        sx = image.shape[1] / 640.0
        sy = image.shape[0] / 480.0
        intrinsics = np.asarray([fx * sx, fy * sy, cx * sx, cy * sy], dtype=np.float32)
    else:
        intrinsics = np.asarray([fx, fy, cx, cy], dtype=np.float32)

    image = image.transpose(2, 0, 1)
    intrinsics[2] -= 16.0
    intrinsics[3] -= 8.0
    image = image[:, 8:-8, 16:-16]
    return image, intrinsics


def _prepare_depth(
    depth_path: str | Path,
    *,
    target_height: int | None = None,
    target_width: int | None = None,
) -> np.ndarray:
    depth = TUMRGBDSequence.read_depth_np(str(depth_path)).astype(np.float32) / 5000.0
    if target_height is not None and target_width is not None:
        pad_h = 16
        pad_w = 32
        depth = cv2.resize(depth, (int(target_width) + pad_w, int(target_height) + pad_h), interpolation=cv2.INTER_NEAREST)
    depth = depth[8:-8, 16:-16]
    return depth


def _load_gt_pose_lookup(gt_file: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = []
    for raw in gt_file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        vals = [float(v) for v in line.split()[:8]]
        if len(vals) < 8:
            continue
        rows.append(vals)
    arr = np.asarray(rows, dtype=np.float64)
    if arr.size == 0:
        return np.zeros((0,), dtype=np.float64), np.zeros((0, 3), dtype=np.float64), np.zeros((0, 4), dtype=np.float64)
    return arr[:, 0], arr[:, 1:4], arr[:, 4:8]


def _associate_gt_poses(gt_file: Path, timestamps: list[float], max_dt: float) -> list[np.ndarray | None]:
    gt_ts, gt_xyz, gt_quat = _load_gt_pose_lookup(gt_file)
    poses: list[np.ndarray | None] = []
    if gt_ts.size == 0:
        return [None for _ in timestamps]
    for ts in timestamps:
        idx = int(np.argmin(np.abs(gt_ts - float(ts))))
        if abs(float(gt_ts[idx]) - float(ts)) > float(max_dt):
            poses.append(None)
            continue
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R.from_quat(gt_quat[idx]).as_matrix()
        T[:3, 3] = gt_xyz[idx]
        poses.append(T)
    return poses


def _build_tum_gt_context(
    *,
    seq_dir: Path,
    sequence: str,
    image_files: list[Path],
    gt_file: Path,
    max_dt: float,
    target_height: int | None,
    target_width: int | None,
) -> GroundTruthFrameContext:
    ds = TUMRGBDSequence(seq_dir.parent, seq_dir.name)
    depth_times = np.asarray([float(frame["t_rgb"]) for frame in ds.frames], dtype=np.float64)
    timestamps = [float(path.stem) for path in image_files]
    poses = _associate_gt_poses(gt_file, timestamps, max_dt=max_dt)
    depths = []
    for ts in timestamps:
        if depth_times.size == 0:
            depths.append(None)
            continue
        idx = int(np.argmin(np.abs(depth_times - float(ts))))
        if abs(float(depth_times[idx]) - float(ts)) > float(max_dt):
            depths.append(None)
            continue
        depth_path = ds.frames[idx]["depth"]
        depths.append(_prepare_depth(depth_path, target_height=target_height, target_width=target_width))
    intrinsics = _prepare_image(
        np.zeros((480, 640, 3), dtype=np.uint8),
        sequence,
        target_height=target_height,
        target_width=target_width,
    )[1]
    image_h = int(target_height) if target_height is not None else int(depths[0].shape[0])
    image_w = int(target_width) if target_width is not None else int(depths[0].shape[1])
    return GroundTruthFrameContext(
        poses=poses,
        depths=depths,
        intrinsics=np.asarray(intrinsics, dtype=np.float32),
        image_size=(image_h, image_w),
    )


def _load_dpvo_modules(dpvo_root: Path):
    sys.path.insert(0, str(dpvo_root))
    from dpvo.config import cfg as base_cfg  # type: ignore
    from dpvo.dpvo import DPVO  # type: ignore

    return base_cfg, DPVO


def _load_frontend(
    *,
    frontend_mode: str,
    frontend_config: Path | None,
    frontend_checkpoint: Path | None,
    device: str,
):
    mode = str(frontend_mode).lower()
    if mode == "dpvo_native":
        if frontend_config is None:
            return None, None
        return load_dino_dpvo_config(frontend_config), None
    if frontend_checkpoint is not None:
        return load_dino_dpvo_frontend_checkpoint(frontend_checkpoint, config=frontend_config, device=device)
    if frontend_config is None:
        raise ValueError(f"frontend_config is required when frontend_mode={frontend_mode}")
    cfg = load_dino_dpvo_config(frontend_config)
    cfg.raw.setdefault("training", {})["device"] = str(device)
    model = build_dino_dpvo_frontend(cfg)
    model.eval()
    return cfg, model


def evaluate_sequence(
    *,
    dataset_root: Path,
    sequence: str,
    dpvo_root: Path,
    weights: Path,
    config_path: Path,
    output_dir: Path,
    max_dt: float,
    missing_penalty_m: float,
    min_coverage_ok: float,
    stride: int,
    backend_thresh: float,
    viz: bool,
    opts: list[str],
    target_height: int | None,
    target_width: int | None,
    frontend_mode: str = "dpvo_native",
    frontend_cfg=None,
    frontend=None,
    collect_diagnostics: bool = False,
    diagnostics_summary_path: Path | None = None,
    patch_diagnostics_path: Path | None = None,
    feature_type: str = "DPVO",
    write_plots: bool = True,
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

    image_files = _sequence_image_files(seq_dir, stride)
    if len(image_files) < 2:
        return "tracking_failed", None
    runtime_seed, deterministic = _frontend_eval_seed(
        frontend_cfg,
        sequence=seq_short,
        frontend_mode=frontend_mode,
        stride=stride,
        backend_thresh=backend_thresh,
    )
    if runtime_seed is not None:
        _seed_runtime(runtime_seed, deterministic=deterministic)
    diagnostics = PatchDiagnosticsRecorder() if collect_diagnostics else None
    gt_context = _build_tum_gt_context(
        seq_dir=seq_dir,
        sequence=seq_short,
        image_files=image_files,
        gt_file=gt_file,
        max_dt=max_dt,
        target_height=target_height,
        target_width=target_width,
    ) if collect_diagnostics else None

    run_dir = output_dir / seq_short
    traj_dir = output_dir / "trajectories"
    run_dir.mkdir(parents=True, exist_ok=True)
    traj_dir.mkdir(parents=True, exist_ok=True)
    plot_path: Path | None = None
    if write_plots:
        plot_dir = output_dir / "plots" / seq_short
        plot_dir.mkdir(parents=True, exist_ok=True)
        plot_path = plot_dir / "trajectory_3d.png"

    expected_ts = run_dir / "expected_timestamps.txt"
    traj_path = traj_dir / f"{_short_alias(seq_short)}_dpvo.txt"
    error_path = run_dir / "error.txt"
    _write_expected_timestamps(
        expected_ts,
        [{"t_rgb": float(path.stem)} for path in image_files],
    )

    base_cfg, DPVO = _load_dpvo_modules(dpvo_root)
    cfg = base_cfg.clone()
    cfg.merge_from_file(str(config_path))
    cfg.BACKEND_THRESH = float(backend_thresh)
    if opts:
        cfg.merge_from_list(list(opts))

    slam = None
    tracker = None
    timestamps = []
    try:
        with torch.no_grad():
            for image_path in image_files:
                bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if bgr is None:
                    raise FileNotFoundError(str(image_path))
                image_np, intrinsics_np = _prepare_image(
                    bgr,
                    seq_short,
                    target_height=target_height,
                    target_width=target_width,
                )
                image = torch.as_tensor(image_np, device="cuda")
                intrinsics = torch.as_tensor(intrinsics_np, dtype=torch.float32, device="cuda")
                if slam is None:
                    slam = DPVO(cfg, str(weights), ht=image.shape[-2], wd=image.shape[-1], viz=viz)
                    tracker = DinoDPVOTracker(
                        slam,
                        frontend=frontend,
                        frontend_cfg=frontend_cfg,
                        frontend_mode=frontend_mode,
                        patch_budget=int(cfg.PATCHES_PER_FRAME),
                        collect_diagnostics=collect_diagnostics,
                        hybrid_grid_rows=int((frontend_cfg.model if frontend_cfg is not None else {}).get("hybrid_grid_rows", 6)),
                        hybrid_grid_cols=int((frontend_cfg.model if frontend_cfg is not None else {}).get("hybrid_grid_cols", 8)),
                    )
                tracker.step(float(image_path.stem), image, intrinsics)
                timestamps.append(float(image_path.stem))
                if diagnostics is not None:
                    diagnostics.observe_step(slam)
    except Exception as exc:
        error_path.write_text(str(exc), encoding="utf-8")
        _cleanup_cuda(slam)
        return "tracking_failed", None

    if slam is None:
        return "tracking_failed", None

    try:
        poses, pose_timestamps = slam.terminate()
    except Exception as exc:
        error_path.write_text(str(exc), encoding="utf-8")
        _cleanup_cuda(slam)
        return "tracking_failed", None

    pose_timestamps = np.asarray(pose_timestamps, dtype=np.float64)
    poses = np.asarray(poses, dtype=np.float64)
    if poses.shape[0] < 2 or pose_timestamps.size < 2:
        return "tracking_failed", None
    n = min(poses.shape[0], pose_timestamps.size)
    poses = poses[:n]
    pose_timestamps = pose_timestamps[:n]

    traj = PoseTrajectory3D(
        positions_xyz=poses[:, :3],
        orientations_quat_wxyz=poses[:, [6, 3, 4, 5]],
        timestamps=pose_timestamps,
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
        dpvo_style = compute_dpvo_style_metrics(
            gt_file,
            traj_path,
            max_dt=max_dt,
        )
        metrics.update(
            {
                "dpvo_style_ate_rmse": float(dpvo_style.get("ate_rmse", math.nan)),
                "dpvo_style_ate_mean": float(dpvo_style.get("ate_mean", math.nan)),
                "dpvo_style_ate_median": float(dpvo_style.get("ate_median", math.nan)),
            }
        )
    except Exception as exc:
        error_path.write_text(str(exc), encoding="utf-8")
        return "invalid_trajectory", None

    if write_plots and plot_path is not None:
        try:
            _run_plot(gt_file, traj_path, plot_path, f"{seq_short} - DPVO", max_dt)
        except Exception:
            pass

    coverage = float(metrics.get("coverage", float("nan")))
    if diagnostics is not None and gt_context is not None and diagnostics_summary_path is not None:
        summary, patch_rows = diagnostics.summarize(
            sequence=seq_short,
            feature_type=feature_type,
            status="partial_low_coverage" if math.isfinite(coverage) and coverage < float(min_coverage_ok) else "ok",
            metrics=metrics,
            gt_context=gt_context,
        )
        append_diagnostics_summary(diagnostics_summary_path, summary)
        append_patch_diagnostics(patch_diagnostics_path, patch_rows)
    if math.isfinite(coverage) and coverage < float(min_coverage_ok):
        status = "partial_low_coverage"
    else:
        status = "ok"

    if error_path.exists():
        error_path.unlink()
    _cleanup_cuda(slam)
    return status, metrics


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate DPVO on selected TUM RGB-D sequences.")
    ap.add_argument("--dataset-root", default=str(REPO_ROOT / "src" / "dino_slam3" / "data" / "tum_rgbd"))
    ap.add_argument("--dpvo-root", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--csv-path", default=None)
    ap.add_argument("--dpvo-style-csv-path", default=None)
    ap.add_argument("--sequences", default=",".join(DEFAULT_SEQUENCES))
    ap.add_argument("--max-dt", type=float, default=0.02)
    ap.add_argument("--missing-penalty-m", type=float, default=3.0)
    ap.add_argument("--min-coverage-ok", type=float, default=0.95)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--backend-thresh", type=float, default=64.0)
    ap.add_argument("--viz", action="store_true")
    ap.add_argument("--opts", nargs="*", default=[])
    ap.add_argument("--image-height", type=int, default=None)
    ap.add_argument("--image-width", type=int, default=None)
    ap.add_argument("--frontend-mode", default="dpvo_native", choices=["dpvo_native", "dino_proposals", "dino_full", "dino_hybrid"])
    ap.add_argument("--frontend-config", default=None)
    ap.add_argument("--frontend-checkpoint", default=None)
    ap.add_argument("--collect-diagnostics", action="store_true")
    args = ap.parse_args()

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    dpvo_root = Path(args.dpvo_root).expanduser().resolve()
    weights = Path(args.weights).expanduser().resolve()
    config_path = (
        Path(args.config).expanduser().resolve()
        if args.config is not None
        else (dpvo_root / "config" / "default.yaml")
    )
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = (
        Path(args.csv_path).expanduser().resolve()
        if args.csv_path is not None
        else output_dir / "metrics_summary.csv"
    )
    dpvo_style_csv_path = (
        Path(args.dpvo_style_csv_path).expanduser().resolve()
        if args.dpvo_style_csv_path is not None
        else output_dir / "dpvo_style_metrics_summary.csv"
    )
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    dpvo_style_csv_path.parent.mkdir(parents=True, exist_ok=True)

    if not dpvo_root.exists():
        raise FileNotFoundError(f"DPVO repo not found: {dpvo_root}")
    if not weights.exists():
        raise FileNotFoundError(f"DPVO weights not found: {weights}")
    if not config_path.exists():
        raise FileNotFoundError(f"DPVO config not found: {config_path}")
    if not torch.cuda.is_available():
        raise RuntimeError("DPVO requires CUDA for this wrapper.")

    frontend_cfg, frontend = _load_frontend(
        frontend_mode=str(args.frontend_mode),
        frontend_config=Path(args.frontend_config).expanduser().resolve() if args.frontend_config else None,
        frontend_checkpoint=Path(args.frontend_checkpoint).expanduser().resolve() if args.frontend_checkpoint else None,
        device="cuda",
    )
    feature_type = "DPVO" if frontend_cfg is None else frontend_cfg.feature_type
    diagnostics_summary_path = output_dir / "diagnostics_summary.csv"
    patch_diagnostics_path = output_dir / "patch_diagnostics.jsonl"
    if args.collect_diagnostics:
        init_diagnostics_outputs(diagnostics_summary_path, patch_diagnostics_path)

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
    with dpvo_style_csv_path.open("w", encoding="utf-8", newline="") as f:
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
            dpvo_root=dpvo_root,
            weights=weights,
            config_path=config_path,
            output_dir=output_dir,
            max_dt=float(args.max_dt),
            missing_penalty_m=float(args.missing_penalty_m),
            min_coverage_ok=float(args.min_coverage_ok),
            stride=max(int(args.stride), 1),
            backend_thresh=float(args.backend_thresh),
            viz=bool(args.viz),
            opts=list(args.opts),
            target_height=args.image_height,
            target_width=args.image_width,
            frontend_mode=str(args.frontend_mode),
            frontend_cfg=frontend_cfg,
            frontend=frontend,
            collect_diagnostics=bool(args.collect_diagnostics),
            diagnostics_summary_path=diagnostics_summary_path,
            patch_diagnostics_path=patch_diagnostics_path,
            feature_type=feature_type,
        )
        _append_csv_row(
            csv_path,
            sequence=seq,
            feature_type=feature_type,
            status=status,
            metrics=metrics,
        )
        _append_csv_row(
            dpvo_style_csv_path,
            sequence=seq,
            feature_type=feature_type,
            status=status,
            metrics=_dpvo_style_row_metrics(metrics),
        )
        print(f"[{seq}] method={str(args.frontend_mode)} status={status}")


if __name__ == "__main__":
    main()
