from __future__ import annotations

import argparse
import csv
import gc
import math
import random
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


TEST_SPLIT = [f"ME{i:03d}" for i in range(8)] + [f"MH{i:03d}" for i in range(8)]
POSE_PERM = [1, 2, 0, 4, 5, 3, 6]
DEFAULT_INTRINSICS = np.asarray([320.0, 320.0, 320.0, 240.0], dtype=np.float32)


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


def _find_image_dir(dataset_root: Path, sequence: str) -> Path | None:
    candidates = [
        dataset_root / sequence / "image_left",
        dataset_root / sequence,
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def _groundtruth_path(dataset_root: Path, groundtruth_root: Path | None, sequence: str) -> Path | None:
    candidates = []
    if groundtruth_root is not None:
        candidates.append(groundtruth_root / f"{sequence}.txt")
    candidates.append(dataset_root / "mono_gt" / f"{sequence}.txt")
    for path in candidates:
        if path.exists():
            return path
    return None


def _load_pose_vectors(path: Path) -> np.ndarray:
    poses = np.loadtxt(path, dtype=np.float64)
    if poses.ndim == 1:
        poses = poses.reshape(1, -1)
    poses = poses[:, :7]
    return poses[:, POSE_PERM]


def _write_tum_pose_vectors(path: Path, timestamps: list[float], pose_vectors: np.ndarray) -> None:
    lines = []
    n = min(len(timestamps), int(pose_vectors.shape[0]))
    for idx in range(n):
        vals = [float(v) for v in pose_vectors[idx].reshape(-1)[:7]]
        lines.append(
            f"{float(timestamps[idx]):.6f} {vals[0]:.9f} {vals[1]:.9f} {vals[2]:.9f} "
            f"{vals[3]:.9f} {vals[4]:.9f} {vals[5]:.9f} {vals[6]:.9f}"
        )
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _prepare_image(
    image_bgr: np.ndarray,
    *,
    target_height: int | None,
    target_width: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    intrinsics = np.asarray(DEFAULT_INTRINSICS, dtype=np.float32)
    if target_height is not None and target_width is not None:
        image_bgr = cv2.resize(image_bgr, (int(target_width), int(target_height)), interpolation=cv2.INTER_LINEAR)
        sx = float(target_width) / 640.0
        sy = float(target_height) / 480.0
        intrinsics = np.asarray(
            [intrinsics[0] * sx, intrinsics[1] * sy, intrinsics[2] * sx, intrinsics[3] * sy],
            dtype=np.float32,
        )
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return image.transpose(2, 0, 1), intrinsics


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
) -> tuple[str, dict | None]:
    sequence = str(sequence)
    image_dir = _find_image_dir(dataset_root, sequence)
    gt_pose_path = _groundtruth_path(dataset_root, groundtruth_root, sequence)
    if image_dir is None:
        return "skipped_missing_sequence", None
    if gt_pose_path is None:
        return "skipped_missing_groundtruth", None

    image_files = sorted(list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpg")))[:: max(int(stride), 1)]
    if len(image_files) < 2:
        return "tracking_failed", None
    pose_vectors = _load_pose_vectors(gt_pose_path)[:: max(int(stride), 1)]
    n = min(len(image_files), int(pose_vectors.shape[0]))
    if n < 2:
        return "tracking_failed", None
    image_files = image_files[:n]
    pose_vectors = pose_vectors[:n]

    run_dir = output_dir / sequence
    traj_dir = output_dir / "trajectories"
    plot_dir = output_dir / "plots" / sequence
    run_dir.mkdir(parents=True, exist_ok=True)
    traj_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    gt_file = run_dir / "groundtruth_tum.txt"
    expected_ts = run_dir / "expected_timestamps.txt"
    traj_file = traj_dir / f"{sequence}_dpvo.txt"
    plot_file = plot_dir / "trajectory_3d.png"
    error_file = run_dir / "error.txt"

    gt_timestamps = [float(i) for i in range(n)]
    _write_tum_pose_vectors(gt_file, gt_timestamps, pose_vectors)
    expected_ts.write_text("\n".join(f"{ts:.6f}" for ts in gt_timestamps) + "\n", encoding="utf-8")

    base_cfg, DPVO = _load_dpvo_modules(dpvo_root)
    cfg = base_cfg.clone()
    cfg.merge_from_file(str(config_path))
    cfg.BACKEND_THRESH = float(backend_thresh)
    if opts:
        cfg.merge_from_list(list(opts))

    slam = None
    tracker = None
    try:
        with torch.no_grad():
            for local_idx, image_path in enumerate(image_files):
                bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if bgr is None:
                    raise FileNotFoundError(str(image_path))
                image_np, intrinsics_np = _prepare_image(
                    bgr,
                    target_height=image_height,
                    target_width=image_width,
                )
                image = torch.from_numpy(np.ascontiguousarray(image_np)).to("cuda")
                intrinsics = torch.as_tensor(intrinsics_np, dtype=torch.float32, device="cuda")
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
                tracker.step(float(local_idx), image, intrinsics)
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
    n = min(poses.shape[0], pose_timestamps.size, len(gt_timestamps))
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
        _run_plot(gt_file, traj_file, plot_file, f"{sequence} - dpvo", max_dt)
    except Exception:
        pass

    coverage = float(metrics.get("coverage", float("nan")))
    if math.isfinite(coverage) and coverage < float(min_coverage_ok):
        return "partial_low_coverage", metrics
    return "ok", metrics


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate DINO-DPVO / DPVO on TartanAir mono test scenes.")
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--groundtruth-root", default=None)
    ap.add_argument("--dpvo-root", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--csv-path", default=None)
    ap.add_argument("--dpvo-style-csv-path", default=None)
    ap.add_argument("--sequences", default=",".join(TEST_SPLIT))
    ap.add_argument("--max-dt", type=float, default=0.02)
    ap.add_argument("--missing-penalty-m", type=float, default=3.0)
    ap.add_argument("--min-coverage-ok", type=float, default=0.95)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--backend-thresh", type=float, default=18.0)
    ap.add_argument("--image-height", type=int, default=240)
    ap.add_argument("--image-width", type=int, default=320)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--viz", action="store_true")
    ap.add_argument("--opts", nargs="*", default=[])
    ap.add_argument("--frontend-mode", default="dpvo_native", choices=["dpvo_native", "dino_proposals", "dino_full", "dino_hybrid"])
    ap.add_argument("--frontend-config", default=None)
    ap.add_argument("--frontend-checkpoint", default=None)
    args = ap.parse_args()

    if args.seed is not None:
        seed = int(args.seed)
        random.seed(seed)
        np.random.seed(seed % (2**32))
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

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
