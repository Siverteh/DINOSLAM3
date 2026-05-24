from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import cv2
import numpy as np
import torch
from evo.core import sync
from evo.core.trajectory import PoseTrajectory3D
from scipy.spatial.transform import Rotation as R

from refocus_vo.dino_dpvo import DinoDPVOTracker
from refocus_vo.eval.dpvo_style_metrics import compute_dpvo_style_metrics
from refocus_vo.eval.external_dpvo_euroc import (
    _cleanup_cuda,
    _load_dpvo_modules,
    _load_frontend,
)
from refocus_vo.eval.kitti_odometry_metrics import (
    KITTI_SEQUENCE_ORDER,
    compute_kitti_official_metrics,
    poses_from_dpvo_output,
    write_kitti_poses,
    write_tum_trajectory_from_kitti,
)
from refocus_vo.eval.metrics import compute_metrics
from refocus_vo.eval.sparse_vo import _fmt_metric
from refocus_vo.eval.validate_trajectory import validate


def _read_kitti_calib(calib_path: Path) -> np.ndarray:
    data: dict[str, np.ndarray] = {}
    for raw in calib_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        try:
            data[key.strip()] = np.asarray([float(item) for item in value.split()], dtype=np.float64)
        except ValueError:
            continue
    calib = data.get("P2")
    if calib is None:
        calib = data.get("P0")
    if calib is None or calib.shape[0] < 12:
        raise RuntimeError(f"Could not read P2/P0 intrinsics from {calib_path}")
    return np.asarray([calib[0], calib[5], calib[2], calib[6]], dtype=np.float32)


def _prepare_image(
    image_bgr: np.ndarray,
    *,
    intrinsics: np.ndarray,
    target_height: int | None,
    target_width: int | None,
) -> tuple[np.ndarray, np.ndarray]:
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


def _write_csv_header(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "sequence",
                "feature_type",
                "status",
                "kitti_trans_percent",
                "kitti_rot_deg_per_m",
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


def _append_metrics_row(
    csv_path: Path,
    *,
    sequence: str,
    feature_type: str,
    status: str,
    metrics: dict | None,
) -> None:
    payload = dict(metrics or {})
    with csv_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                sequence,
                feature_type,
                status,
                _fmt_metric(payload.get("kitti_trans_percent")),
                _fmt_metric(payload.get("kitti_rot_deg_per_m")),
                _fmt_metric(payload.get("ate_rmse")),
                _fmt_metric(payload.get("ate_mean")),
                _fmt_metric(payload.get("ate_median")),
                _fmt_metric(payload.get("ate_rmse_associated")),
                _fmt_metric(payload.get("ate_mean_associated")),
                _fmt_metric(payload.get("ate_median_associated")),
                _fmt_metric(payload.get("rpe_trans_rmse")),
                _fmt_metric(payload.get("rpe_rot_rmse")),
                _fmt_metric(payload.get("scale_correction")),
                _fmt_metric(payload.get("scale_error_abs")),
                _fmt_metric(payload.get("scale_error_abs_log")),
                _fmt_metric(payload.get("coverage")),
            ]
        )


def _resolve_sequence(dataset_root: Path, sequence: str) -> tuple[Path | None, Path | None]:
    seq = str(sequence).strip()
    seq_dir = dataset_root / "dataset" / "sequences" / seq
    pose_path = dataset_root / "dataset" / "poses" / f"{seq}.txt"
    if not seq_dir.exists():
        return None, None
    return seq_dir, pose_path if pose_path.exists() else None


def _expected_frame_ids(image_files: list[Path]) -> np.ndarray:
    frame_ids = []
    for image_path in image_files:
        try:
            frame_ids.append(float(int(image_path.stem)))
        except Exception as exc:
            raise ValueError(f"Invalid KITTI image filename: {image_path.name}") from exc
    return np.asarray(frame_ids, dtype=np.float64)


def _pose_traj_from_matrices(poses: np.ndarray, timestamps: np.ndarray) -> PoseTrajectory3D:
    quats = R.from_matrix(np.asarray(poses[:, :3, :3], dtype=np.float64)).as_quat()
    quat_wxyz = np.column_stack([quats[:, 3], quats[:, 0], quats[:, 1], quats[:, 2]])
    return PoseTrajectory3D(
        positions_xyz=np.asarray(poses[:, :3, 3], dtype=np.float64),
        orientations_quat_wxyz=np.asarray(quat_wxyz, dtype=np.float64),
        timestamps=np.asarray(timestamps, dtype=np.float64),
    )


def _associate_pose_arrays(
    *,
    gt_poses: np.ndarray,
    gt_timestamps: np.ndarray,
    est_poses: np.ndarray,
    est_timestamps: np.ndarray,
    max_dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gt_traj = _pose_traj_from_matrices(gt_poses, gt_timestamps)
    est_traj = _pose_traj_from_matrices(est_poses, est_timestamps)
    gt_assoc, est_assoc = sync.associate_trajectories(gt_traj, est_traj, max_diff=float(max_dt))
    if gt_assoc.num_poses < 2 or est_assoc.num_poses < 2:
        raise RuntimeError("Not enough associated KITTI poses")
    return (
        np.asarray(gt_assoc.poses_se3, dtype=np.float64),
        np.asarray(est_assoc.poses_se3, dtype=np.float64),
        np.asarray(est_assoc.timestamps, dtype=np.float64),
    )


def _write_expected_timestamps(path: Path, frame_ids: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(f"{float(ts):.6f}" for ts in frame_ids) + ("\n" if frame_ids.size else ""), encoding="utf-8")


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
    image_height: int | None,
    image_width: int | None,
    frontend_mode: str = "dpvo_native",
    frontend_cfg=None,
    frontend=None,
    feature_type: str = "DPVO",
) -> tuple[str, dict | None]:
    seq_dir, pose_path = _resolve_sequence(dataset_root, sequence)
    if seq_dir is None:
        return "skipped_missing_sequence", None
    if pose_path is None:
        return "skipped_missing_groundtruth", None

    image_dir = seq_dir / "image_2"
    calib_path = seq_dir / "calib.txt"
    if not image_dir.exists():
        return "skipped_missing_sequence", None
    if not calib_path.exists():
        return "skipped_missing_groundtruth", None

    image_files = sorted(image_dir.glob("*.png"))[:: max(int(stride), 1)]
    if len(image_files) < 2:
        return "tracking_failed", None

    frame_ids = _expected_frame_ids(image_files)
    gt_all = np.loadtxt(str(pose_path), dtype=np.float64).reshape(-1, 3, 4)
    gt_all_h = np.tile(np.eye(4, dtype=np.float64), (gt_all.shape[0], 1, 1))
    gt_all_h[:, :3, :] = gt_all
    gt_selected = gt_all_h[frame_ids.astype(np.int64)]

    run_dir = output_dir / sequence
    traj_dir = output_dir / "trajectories"
    plot_dir = output_dir / "plots" / sequence
    run_dir.mkdir(parents=True, exist_ok=True)
    traj_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    gt_file = run_dir / "groundtruth_tum.txt"
    gt_assoc_kitti = run_dir / "groundtruth_kitti.txt"
    expected_ts = run_dir / "expected_timestamps.txt"
    traj_file = traj_dir / f"{sequence}_dpvo.txt"
    traj_kitti_file = traj_dir / f"{sequence}_dpvo_kitti.txt"
    error_file = run_dir / "error.txt"
    _write_expected_timestamps(expected_ts, frame_ids)
    write_tum_trajectory_from_kitti(gt_file, gt_all_h, timestamps=np.arange(gt_all_h.shape[0], dtype=np.float64))

    base_cfg, DPVO = _load_dpvo_modules(dpvo_root)
    cfg = base_cfg.clone()
    cfg.merge_from_file(str(config_path))
    cfg.BACKEND_THRESH = float(backend_thresh)
    if opts:
        cfg.merge_from_list(list(opts))

    intrinsics = _read_kitti_calib(calib_path)

    slam = None
    tracker = None
    try:
        with torch.no_grad():
            for image_path, frame_id in zip(image_files, frame_ids):
                bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if bgr is None:
                    raise FileNotFoundError(str(image_path))
                image_np, intrinsics_np = _prepare_image(
                    bgr,
                    intrinsics=intrinsics,
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
                tracker.step(float(frame_id), image, intrinsics_t)
    except Exception as exc:
        error_file.write_text(str(exc), encoding="utf-8")
        _cleanup_cuda(slam)
        return "tracking_failed", None

    if slam is None:
        return "tracking_failed", None

    try:
        poses_out, pose_timestamps = slam.terminate()
    except Exception as exc:
        error_file.write_text(str(exc), encoding="utf-8")
        _cleanup_cuda(slam)
        return "tracking_failed", None
    finally:
        _cleanup_cuda(slam)

    poses_out = np.asarray(poses_out, dtype=np.float64)
    pose_timestamps = np.asarray(pose_timestamps, dtype=np.float64)
    if poses_out.shape[0] < 2 or pose_timestamps.size < 2:
        return "tracking_failed", None

    try:
        n = min(poses_out.shape[0], pose_timestamps.size)
        est_pose_mats = poses_from_dpvo_output(poses_out[:n])
        est_timestamps = pose_timestamps[:n]
        write_tum_trajectory_from_kitti(traj_file, est_pose_mats, timestamps=est_timestamps)
        validate(traj_file, gt_file, max_dt)
        gt_assoc, est_assoc, assoc_timestamps = _associate_pose_arrays(
            gt_poses=gt_selected,
            gt_timestamps=frame_ids,
            est_poses=est_pose_mats,
            est_timestamps=est_timestamps,
            max_dt=max_dt,
        )
        write_kitti_poses(gt_assoc_kitti, gt_assoc)
        write_kitti_poses(traj_kitti_file, est_assoc)
        metrics = compute_metrics(
            str(gt_file),
            str(traj_file),
            max_dt=max_dt,
            missing_penalty=missing_penalty_m,
            expected_timestamps_file=str(expected_ts),
            correct_scale=True,
        )
        dpvo_style = compute_dpvo_style_metrics(gt_file, traj_file, max_dt=max_dt)
        kitti_metrics = compute_kitti_official_metrics(gt_assoc, est_assoc)
        metrics.update(
            {
                "dpvo_style_ate_rmse": float(dpvo_style.get("ate_rmse", math.nan)),
                "dpvo_style_ate_mean": float(dpvo_style.get("ate_mean", math.nan)),
                "dpvo_style_ate_median": float(dpvo_style.get("ate_median", math.nan)),
                "kitti_trans_percent": float(kitti_metrics.get("kitti_trans_percent", math.nan)),
                "kitti_rot_deg_per_m": float(kitti_metrics.get("kitti_rot_deg_per_m", math.nan)),
                "kitti_segment_count": float(kitti_metrics.get("kitti_segment_count", math.nan)),
                "num_associated_kitti_poses": int(est_assoc.shape[0]),
            }
        )
    except Exception as exc:
        error_file.write_text(str(exc), encoding="utf-8")
        return "invalid_trajectory", None

    coverage = float(metrics.get("coverage", float("nan")))
    if math.isfinite(coverage) and coverage < float(min_coverage_ok):
        return "partial_low_coverage", metrics
    return "ok", metrics


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


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate DINO-DPVO / DPVO on KITTI odometry sequences.")
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--dpvo-root", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--csv-path", default=None)
    ap.add_argument("--dpvo-style-csv-path", default=None)
    ap.add_argument("--sequences", default=",".join(KITTI_SEQUENCE_ORDER))
    ap.add_argument("--max-dt", type=float, default=0.01)
    ap.add_argument("--missing-penalty-m", type=float, default=3.0)
    ap.add_argument("--min-coverage-ok", type=float, default=0.95)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--backend-thresh", type=float, default=32.0)
    ap.add_argument("--image-height", type=int, default=None)
    ap.add_argument("--image-width", type=int, default=None)
    ap.add_argument("--viz", action="store_true")
    ap.add_argument("--opts", nargs="*", default=[])
    ap.add_argument("--frontend-mode", default="dpvo_native", choices=["dpvo_native", "dino_proposals", "dino_full", "dino_hybrid"])
    ap.add_argument("--frontend-config", default=None)
    ap.add_argument("--frontend-checkpoint", default=None)
    args = ap.parse_args()

    dataset_root = Path(args.dataset_root).expanduser().resolve()
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

    _write_csv_header(csv_path)
    _write_csv_header(dpvo_style_csv_path)

    sequences = [str(item).strip() for item in str(args.sequences).split(",") if str(item).strip()]
    for seq in sequences:
        status, metrics = evaluate_sequence(
            dataset_root=dataset_root,
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
        _append_metrics_row(
            csv_path,
            sequence=seq,
            feature_type=feature_type,
            status=status,
            metrics=metrics,
        )
        _append_metrics_row(
            dpvo_style_csv_path,
            sequence=seq,
            feature_type=feature_type,
            status=status,
            metrics=_dpvo_style_row_metrics(metrics),
        )
        print(f"[{seq}] method={str(args.frontend_mode)} status={status}")


if __name__ == "__main__":
    main()
