from __future__ import annotations

import argparse
import csv
import gc
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from evo.core.trajectory import PoseTrajectory3D
from evo.tools import file_interface

from refocus_vo.dino_dpvo import (
    DinoDPVOTracker,
    build_dino_dpvo_frontend,
    load_dino_dpvo_config,
    load_dino_dpvo_frontend_checkpoint,
)
from refocus_vo.eval.dpvo_style_metrics import compute_dpvo_style_metrics
from refocus_vo.eval.metrics import compute_metrics
from refocus_vo.eval.plot_trajectory_3d import main as _plot_trajectory_main
from refocus_vo.eval.sparse_vo import _append_csv_row
from refocus_vo.eval.validate_trajectory import validate


EUROC_SEQUENCES = {
    "MH01": "MH_01_easy",
    "MH02": "MH_02_easy",
    "MH03": "MH_03_medium",
    "MH04": "MH_04_difficult",
    "MH05": "MH_05_difficult",
    "V101": "V1_01_easy",
    "V102": "V1_02_medium",
    "V103": "V1_03_difficult",
    "V201": "V2_01_easy",
    "V202": "V2_02_medium",
    "V203": "V2_03_difficult",
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


def _resolve_scene_dir(dataset_root: Path, sequence: str) -> tuple[str, Path] | tuple[None, None]:
    alias = str(sequence).strip()
    scene_name = EUROC_SEQUENCES.get(alias, alias)
    scene_dir = dataset_root / scene_name
    if scene_dir.exists():
        return alias, scene_dir
    return None, None


def _groundtruth_path(scene_dir: Path, groundtruth_root: Path | None, alias: str) -> Path | None:
    candidates: list[Path] = []
    if groundtruth_root is not None:
        candidates.extend(
            [
                groundtruth_root / f"{alias}.txt",
                groundtruth_root / f"{EUROC_SEQUENCES.get(alias, alias)}.txt",
            ]
        )
    candidates.append(scene_dir / "mav0" / "state_groundtruth_estimate0" / "data.tum")
    for path in candidates:
        if path.exists():
            return path
    return None


def _normalize_tum_groundtruth(input_path: Path, output_path: Path) -> None:
    rows = []
    raw_ts = []
    for raw in input_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 8:
            continue
        raw_ts.append(float(parts[0]))
        rows.append(parts[:8])
    if not rows:
        output_path.write_text("", encoding="utf-8")
        return
    # EuRoC raw CSV groundtruth uses nanosecond timestamps (~1e18), while the
    # normalized TUM files use epoch seconds (~1e9). Using a low threshold here
    # wrongly rescales already-normalized TUM files and breaks timestamp overlap.
    scale = 1e9 if np.median(np.asarray(raw_ts, dtype=np.float64)) > 1e12 else 1.0
    normalized = []
    for parts in rows:
        ts = float(parts[0]) / scale
        normalized.append(" ".join([f"{ts:.9f}", *parts[1:8]]))
    output_path.write_text("\n".join(normalized) + "\n", encoding="utf-8")


def _load_calibration(calib_path: Path) -> tuple[np.ndarray, np.ndarray]:
    calib = np.loadtxt(str(calib_path), dtype=np.float64)
    fx, fy, cx, cy = [float(v) for v in calib[:4]]
    K = np.eye(3, dtype=np.float64)
    K[0, 0] = fx
    K[0, 2] = cx
    K[1, 1] = fy
    K[1, 2] = cy
    dist = np.asarray(calib[4:], dtype=np.float64) if calib.shape[0] > 4 else np.zeros((0,), dtype=np.float64)
    return np.asarray([fx, fy, cx, cy], dtype=np.float32), np.concatenate([K.reshape(-1), dist]).astype(np.float64)


def _prepare_image(
    image_bgr: np.ndarray,
    *,
    intrinsics: np.ndarray,
    calib_vec: np.ndarray,
    target_height: int | None,
    target_width: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    fx, fy, cx, cy = [float(v) for v in intrinsics]
    K = np.array([fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0], dtype=np.float64).reshape(3, 3)
    if calib_vec.shape[0] > 9:
        image_bgr = cv2.undistort(image_bgr, K, calib_vec[9:])

    intr = np.asarray(intrinsics, dtype=np.float32)
    if target_height is not None and target_width is not None:
        src_h, src_w = image_bgr.shape[:2]
        image_bgr = cv2.resize(image_bgr, (int(target_width), int(target_height)), interpolation=cv2.INTER_LINEAR)
        sx = float(target_width) / float(src_w) if src_w > 0 else 1.0
        sy = float(target_height) / float(src_h) if src_h > 0 else 1.0
        intr = np.asarray([intr[0] * sx, intr[1] * sy, intr[2] * sx, intr[3] * sy], dtype=np.float32)

    h, w = image_bgr.shape[:2]
    crop_h = h - (h % 16)
    crop_w = w - (w % 16)
    image_bgr = image_bgr[:crop_h, :crop_w]
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
    return image, intr


def evaluate_sequence(
    *,
    dataset_root: Path,
    groundtruth_root: Path | None,
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
    image_height: int | None,
    image_width: int | None,
    frontend_mode: str = "dpvo_native",
    frontend_cfg=None,
    frontend=None,
    feature_type: str = "DPVO",
    calib_path: Path | None = None,
) -> tuple[str, dict | None]:
    alias, scene_dir = _resolve_scene_dir(dataset_root, sequence)
    if alias is None or scene_dir is None:
        return "skipped_missing_sequence", None
    gt_source = _groundtruth_path(scene_dir, groundtruth_root, alias)
    if gt_source is None:
        return "skipped_missing_groundtruth", None

    image_dir = scene_dir / "mav0" / "cam0" / "data"
    image_files = sorted(image_dir.glob("*.png"))[:: max(int(stride), 1)]
    if len(image_files) < 2:
        return "tracking_failed", None

    run_dir = output_dir / alias
    traj_dir = output_dir / "trajectories"
    plot_dir = output_dir / "plots" / alias
    run_dir.mkdir(parents=True, exist_ok=True)
    traj_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    gt_file = run_dir / "groundtruth_tum.txt"
    expected_ts = run_dir / "expected_timestamps.txt"
    traj_file = traj_dir / f"{alias}_dpvo.txt"
    plot_file = plot_dir / "trajectory_3d.png"
    error_file = run_dir / "error.txt"
    _normalize_tum_groundtruth(gt_source, gt_file)

    timestamps = []
    for image_path in image_files:
        raw_ts = float(image_path.stem)
        timestamps.append(raw_ts / 1e9 if raw_ts > 1e6 else raw_ts)
    expected_ts.write_text("\n".join(f"{ts:.9f}" for ts in timestamps) + "\n", encoding="utf-8")

    base_cfg, DPVO = _load_dpvo_modules(dpvo_root)
    cfg = base_cfg.clone()
    cfg.merge_from_file(str(config_path))
    cfg.BACKEND_THRESH = float(backend_thresh)
    if opts:
        cfg.merge_from_list(list(opts))

    intrinsics, calib_vec = _load_calibration(calib_path or (dpvo_root / "calib" / "euroc.txt"))

    slam = None
    tracker = None
    try:
        with torch.no_grad():
            for image_path, ts in zip(image_files, timestamps):
                bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if bgr is None:
                    raise FileNotFoundError(str(image_path))
                image_np, intrinsics_np = _prepare_image(
                    bgr,
                    intrinsics=intrinsics,
                    calib_vec=calib_vec,
                    target_height=image_height,
                    target_width=image_width,
                )
                image = torch.from_numpy(np.ascontiguousarray(image_np)).to("cuda")
                intrinsics_t = torch.as_tensor(intrinsics_np, dtype=torch.float32, device="cuda")
                if slam is None:
                    slam = DPVO(cfg, str(weights), ht=image.shape[-2], wd=image.shape[-1], viz=viz)
                    tracker = DinoDPVOTracker(
                        slam,
                        frontend=frontend,
                        frontend_cfg=frontend_cfg,
                        frontend_mode=frontend_mode,
                        patch_budget=int(cfg.PATCHES_PER_FRAME),
                        collect_diagnostics=False,
                        hybrid_grid_rows=int((frontend_cfg.model if frontend_cfg is not None else {}).get("hybrid_grid_rows", 6)),
                        hybrid_grid_cols=int((frontend_cfg.model if frontend_cfg is not None else {}).get("hybrid_grid_cols", 8)),
                    )
                tracker.step(float(ts), image, intrinsics_t)
    except Exception as exc:
        error_file.write_text(str(exc), encoding="utf-8")
        _cleanup_cuda(slam)
        return "tracking_failed", None

    if slam is None:
        return "tracking_failed", None

    try:
        poses, pose_timestamps = slam.terminate()
    except Exception as exc:
        error_file.write_text(str(exc), encoding="utf-8")
        _cleanup_cuda(slam)
        return "tracking_failed", None
    finally:
        _cleanup_cuda(slam)

    poses = np.asarray(poses, dtype=np.float64)
    pose_timestamps = np.asarray(pose_timestamps, dtype=np.float64)
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
    file_interface.write_tum_trajectory_file(str(traj_file), traj)

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
        dpvo_style = compute_dpvo_style_metrics(gt_file, traj_file, max_dt=max_dt)
        metrics.update(
            {
                "dpvo_style_ate_rmse": float(dpvo_style.get("ate_rmse", math.nan)),
                "dpvo_style_ate_mean": float(dpvo_style.get("ate_mean", math.nan)),
                "dpvo_style_ate_median": float(dpvo_style.get("ate_median", math.nan)),
            }
        )
    except Exception as exc:
        error_file.write_text(str(exc), encoding="utf-8")
        return "invalid_trajectory", None

    try:
        _run_plot(gt_file, traj_file, plot_file, f"{alias} - dpvo", max_dt)
    except Exception:
        pass

    coverage = float(metrics.get("coverage", float("nan")))
    if math.isfinite(coverage) and coverage < float(min_coverage_ok):
        return "partial_low_coverage", metrics
    return "ok", metrics


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate DINO-DPVO / DPVO on EuRoC sequences.")
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--groundtruth-root", default=None)
    ap.add_argument("--dpvo-root", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--csv-path", default=None)
    ap.add_argument("--dpvo-style-csv-path", default=None)
    ap.add_argument("--calib-path", default=None)
    ap.add_argument("--sequences", default=",".join(EUROC_SEQUENCES.keys()))
    ap.add_argument("--max-dt", type=float, default=0.02)
    ap.add_argument("--missing-penalty-m", type=float, default=3.0)
    ap.add_argument("--min-coverage-ok", type=float, default=0.95)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--backend-thresh", type=float, default=64.0)
    ap.add_argument("--image-height", type=int, default=None)
    ap.add_argument("--image-width", type=int, default=None)
    ap.add_argument("--viz", action="store_true")
    ap.add_argument("--opts", nargs="*", default=[])
    ap.add_argument("--frontend-mode", default="dpvo_native", choices=["dpvo_native", "dino_proposals", "dino_full", "dino_hybrid"])
    ap.add_argument("--frontend-config", default=None)
    ap.add_argument("--frontend-checkpoint", default=None)
    args = ap.parse_args()

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    groundtruth_root = Path(args.groundtruth_root).expanduser().resolve() if args.groundtruth_root else None
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = Path(args.csv_path).expanduser().resolve() if args.csv_path else (output_dir / "metrics_summary.csv")
    dpvo_style_csv_path = (
        Path(args.dpvo_style_csv_path).expanduser().resolve()
        if args.dpvo_style_csv_path
        else (output_dir / "dpvo_style_metrics_summary.csv")
    )

    frontend_cfg, frontend = _load_frontend(
        frontend_mode=str(args.frontend_mode),
        frontend_config=Path(args.frontend_config).expanduser().resolve() if args.frontend_config else None,
        frontend_checkpoint=Path(args.frontend_checkpoint).expanduser().resolve() if args.frontend_checkpoint else None,
        device="cuda",
    )
    feature_type = "DPVO" if frontend_cfg is None else frontend_cfg.feature_type

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

    sequences = [str(s).strip() for s in str(args.sequences).split(",") if str(s).strip()]
    for seq in sequences:
        status, metrics = evaluate_sequence(
            dataset_root=dataset_root,
            groundtruth_root=groundtruth_root,
            sequence=seq,
            dpvo_root=Path(args.dpvo_root).expanduser().resolve(),
            weights=Path(args.weights).expanduser().resolve(),
            config_path=Path(args.config).expanduser().resolve(),
            output_dir=output_dir,
            max_dt=float(args.max_dt),
            missing_penalty_m=float(args.missing_penalty_m),
            min_coverage_ok=float(args.min_coverage_ok),
            stride=int(args.stride),
            backend_thresh=float(args.backend_thresh),
            viz=bool(args.viz),
            opts=list(args.opts),
            image_height=args.image_height,
            image_width=args.image_width,
            frontend_mode=str(args.frontend_mode),
            frontend_cfg=frontend_cfg,
            frontend=frontend,
            feature_type=feature_type,
            calib_path=Path(args.calib_path).expanduser().resolve() if args.calib_path else None,
        )
        _append_csv_row(csv_path, sequence=seq, feature_type=feature_type, status=status, metrics=metrics)
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
