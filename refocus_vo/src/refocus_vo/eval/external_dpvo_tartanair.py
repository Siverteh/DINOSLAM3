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

from refocus_vo.data import (
    DPVO_VALIDATION_SPLIT,
    discover_tartanair_sequences,
    pose_vector_to_matrix,
    read_tartanair_depth,
    scale_intrinsics,
)
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
from refocus_vo.eval.dpvo_style_metrics import compute_dpvo_style_metrics
from refocus_vo.eval.metrics import compute_metrics
from refocus_vo.eval.plot_trajectory_3d import main as _plot_trajectory_main
from refocus_vo.eval.sparse_vo import _append_csv_row
from refocus_vo.eval.validate_trajectory import validate


def _select_eval_sequences(dataset_root: Path):
    available = discover_tartanair_sequences(dataset_root, include_validation=True, environments=())
    validation = [seq for seq in available if seq.key in DPVO_VALIDATION_SPLIT]
    if validation:
        return validation, "dpvo_validation"
    if available:
        return available, "available_subset"
    return [], "missing"


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


def _write_tum_pose_file(path: Path, timestamps: list[float], poses: list[np.ndarray]) -> None:
    lines = []
    for ts, pose in zip(timestamps, poses):
        tx, ty, tz = [float(v) for v in pose[:3, 3]]
        from scipy.spatial.transform import Rotation as R

        qx, qy, qz, qw = R.from_matrix(np.asarray(pose[:3, :3], dtype=np.float64)).as_quat()
        lines.append(f"{ts:.6f} {tx:.9f} {ty:.9f} {tz:.9f} {qx:.9f} {qy:.9f} {qz:.9f} {qw:.9f}")
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


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
    sequence,
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
    collect_diagnostics: bool = False,
    diagnostics_summary_path: Path | None = None,
    patch_diagnostics_path: Path | None = None,
    feature_type: str = "DPVO",
) -> tuple[str, dict | None]:
    from evo.core.trajectory import PoseTrajectory3D
    from evo.tools import file_interface

    seq_short = sequence.key.replace("/", "__")
    run_dir = output_dir / seq_short
    traj_dir = output_dir / "trajectories"
    plot_dir = output_dir / "plots" / seq_short
    run_dir.mkdir(parents=True, exist_ok=True)
    traj_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    gt_file = run_dir / "groundtruth_tum.txt"
    expected_ts = run_dir / "expected_timestamps.txt"
    traj_file = traj_dir / f"{seq_short}_dpvo.txt"
    plot_file = plot_dir / "trajectory_3d.png"
    error_file = run_dir / "error.txt"

    sampled_indices = list(range(0, sequence.num_frames, max(1, int(stride))))
    if len(sampled_indices) < 2:
        return "tracking_failed", None
    gt_timestamps = [float(i) for i in range(len(sampled_indices))]
    gt_poses = [pose_vector_to_matrix(sequence.poses[idx]) for idx in sampled_indices]
    _write_tum_pose_file(gt_file, gt_timestamps, gt_poses)
    expected_ts.write_text("\n".join(f"{ts:.6f}" for ts in gt_timestamps) + "\n", encoding="utf-8")

    base_cfg, DPVO = _load_dpvo_modules(dpvo_root)
    cfg = base_cfg.clone()
    cfg.merge_from_file(str(config_path))
    cfg.BACKEND_THRESH = float(backend_thresh)
    if opts:
        cfg.merge_from_list(list(opts))

    if frontend_cfg is not None and (image_height is None or image_width is None):
        image_size = frontend_cfg.model.get("image_size", [240, 320])
        if len(image_size) >= 2:
            image_height = int(image_size[0]) if image_height is None else int(image_height)
            image_width = int(image_size[1]) if image_width is None else int(image_width)

    diagnostics = PatchDiagnosticsRecorder() if collect_diagnostics else None
    if image_height is not None and image_width is not None:
        diag_intrinsics = scale_intrinsics(
            sequence.intrinsics,
            src_height=480,
            src_width=640,
            dst_height=int(image_height),
            dst_width=int(image_width),
        )
        diag_depths = [
            read_tartanair_depth(sequence.depth_files[idx], image_size=(int(image_height), int(image_width)))
            for idx in sampled_indices
        ]
    else:
        diag_intrinsics = np.asarray(sequence.intrinsics, dtype=np.float32)
        diag_depths = [read_tartanair_depth(sequence.depth_files[idx]) for idx in sampled_indices]
    gt_context = GroundTruthFrameContext(
        poses=[pose_vector_to_matrix(sequence.poses[idx]) for idx in sampled_indices],
        depths=diag_depths,
        intrinsics=np.asarray(diag_intrinsics, dtype=np.float32),
        image_size=(
            int(image_height if image_height is not None else diag_depths[0].shape[0]),
            int(image_width if image_width is not None else diag_depths[0].shape[1]),
        ),
    ) if collect_diagnostics else None

    slam = None
    tracker = None
    try:
        with torch.no_grad():
            for local_idx, frame_idx in enumerate(sampled_indices):
                image_path = sequence.image_files[frame_idx]
                bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if bgr is None:
                    raise FileNotFoundError(str(image_path))
                if image_height is not None and image_width is not None:
                    bgr = cv2.resize(bgr, (int(image_width), int(image_height)), interpolation=cv2.INTER_LINEAR)
                    intrinsics_np = scale_intrinsics(
                        sequence.intrinsics,
                        src_height=480,
                        src_width=640,
                        dst_height=int(image_height),
                        dst_width=int(image_width),
                    )
                else:
                    intrinsics_np = np.asarray(sequence.intrinsics, dtype=np.float32)

                image = torch.from_numpy(np.ascontiguousarray(bgr)).permute(2, 0, 1).to("cuda")
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
                tracker.step(float(local_idx), image, intrinsics)
                if diagnostics is not None:
                    diagnostics.observe_step(slam)
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
        dpvo_style = compute_dpvo_style_metrics(
            gt_file,
            traj_file,
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
        error_file.write_text(str(exc), encoding="utf-8")
        return "invalid_trajectory", None

    try:
        _run_plot(gt_file, traj_file, plot_file, f"{sequence.key} - dpvo", max_dt)
    except Exception:
        pass

    coverage = float(metrics.get("coverage", float("nan")))
    if diagnostics is not None and gt_context is not None and diagnostics_summary_path is not None and patch_diagnostics_path is not None:
        summary, patch_rows = diagnostics.summarize(
            sequence=sequence.key,
            feature_type=feature_type,
            status="partial_low_coverage" if math.isfinite(coverage) and coverage < float(min_coverage_ok) else "ok",
            metrics=metrics,
            gt_context=gt_context,
        )
        append_diagnostics_summary(diagnostics_summary_path, summary)
        append_patch_diagnostics(patch_diagnostics_path, patch_rows)
    if math.isfinite(coverage) and coverage < float(min_coverage_ok):
        return "partial_low_coverage", metrics
    return "ok", metrics


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate DPVO on TartanAir validation sequences.")
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--dpvo-root", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--csv-path", default=None)
    ap.add_argument("--dpvo-style-csv-path", default=None)
    ap.add_argument("--max-dt", type=float, default=0.02)
    ap.add_argument("--missing-penalty-m", type=float, default=3.0)
    ap.add_argument("--min-coverage-ok", type=float, default=0.95)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--backend-thresh", type=float, default=18.0)
    ap.add_argument("--image-height", type=int, default=None)
    ap.add_argument("--image-width", type=int, default=None)
    ap.add_argument("--viz", action="store_true")
    ap.add_argument("--opts", nargs="*", default=[])
    ap.add_argument("--frontend-mode", default="dpvo_native", choices=["dpvo_native", "dino_proposals", "dino_full", "dino_hybrid"])
    ap.add_argument("--frontend-config", default=None)
    ap.add_argument("--frontend-checkpoint", default=None)
    ap.add_argument("--collect-diagnostics", action="store_true")
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

    sequences, sequence_mode = _select_eval_sequences(dataset_root)
    if not sequences:
        raise FileNotFoundError(f"No TartanAir sequences found under {dataset_root}")
    if sequence_mode != "dpvo_validation":
        print(
            f"[external_dpvo_tartanair] WARNING: DPVO validation split not present under {dataset_root}; "
            f"falling back to {len(sequences)} available converted sequence(s)."
        )

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

    for seq in sequences:
        status, metrics = evaluate_sequence(
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
            collect_diagnostics=bool(args.collect_diagnostics),
            diagnostics_summary_path=diagnostics_summary_path,
            patch_diagnostics_path=patch_diagnostics_path,
            feature_type=feature_type,
        )
        _append_csv_row(
            csv_path,
            sequence=seq.key,
            feature_type=feature_type,
            status=status,
            metrics=metrics,
        )
        _append_csv_row(
            dpvo_style_csv_path,
            sequence=seq.key,
            feature_type=feature_type,
            status=status,
            metrics=_dpvo_style_row_metrics(metrics),
        )
        print(f"[{seq.key}] method={str(args.frontend_mode)} status={status}")

    print(f"DPVO TartanAir results written to {csv_path}")


if __name__ == "__main__":
    main()
