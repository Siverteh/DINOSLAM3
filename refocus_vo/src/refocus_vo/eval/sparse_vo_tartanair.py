from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import cv2
import numpy as np

from refocus_vo.data import DPVO_VALIDATION_SPLIT, discover_tartanair_sequences, pose_vector_to_matrix
from refocus_vo.eval.metrics import compute_metrics
from refocus_vo.eval.plot_trajectory_3d import main as _plot_trajectory_main
from refocus_vo.eval.sparse_vo import PYSLAM_ROOT, REPO_ROOT, _append_csv_row
from refocus_vo.eval.validate_trajectory import validate


def _select_eval_sequences(dataset_root: Path):
    available = discover_tartanair_sequences(dataset_root, include_validation=True, environments=())
    validation = [seq for seq in available if seq.key in DPVO_VALIDATION_SPLIT]
    if validation:
        return validation, "dpvo_validation"
    if available:
        return available, "available_subset"
    return [], "missing"


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
    from scipy.spatial.transform import Rotation as R

    lines = []
    for ts, pose in zip(timestamps, poses):
        tx, ty, tz = [float(v) for v in pose[:3, 3]]
        qx, qy, qz, qw = R.from_matrix(np.asarray(pose[:3, :3], dtype=np.float64)).as_quat()
        lines.append(f"{ts:.6f} {tx:.9f} {ty:.9f} {tz:.9f} {qx:.9f} {qy:.9f} {qz:.9f} {qw:.9f}")
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _load_pyslam_camera(width: int, height: int):
    from pyslam.slam.camera import PinholeCamera

    return PinholeCamera(
        {
            "cam_settings": {
                "Camera.width": int(width),
                "Camera.height": int(height),
                "Camera.fx": 320.0,
                "Camera.fy": 320.0,
                "Camera.cx": 320.0,
                "Camera.cy": 240.0,
                "Camera.fps": 10.0,
                "Camera.k1": 0.0,
                "Camera.k2": 0.0,
                "Camera.p1": 0.0,
                "Camera.p2": 0.0,
                "Camera.k3": 0.0,
            },
            "dataset_settings": {
                "sensor_type": "mono",
            },
        }
    )


def _pose_components(pose: np.ndarray) -> tuple[float, float, float, float, float, float, float]:
    pose = np.asarray(pose, dtype=np.float64).reshape(-1)
    return (
        float(pose[0]),
        float(pose[1]),
        float(pose[2]),
        float(pose[3]),
        float(pose[4]),
        float(pose[5]),
        float(pose[6]),
    )


def _make_groundtruth(sequence):
    from pyslam.io.ground_truth import GroundTruth, GroundTruthType

    class _LocalTartanAirGroundTruth(GroundTruth):
        def __init__(self):
            super().__init__(None, sequence.key, None, 0, type=GroundTruthType.TARTANAIR)
            self.timestamps = np.asarray([float(i) for i in range(sequence.num_frames)], dtype=np.float64)
            self.pose_vectors = np.asarray(sequence.poses, dtype=np.float64)

        def getTimestampPoseAndAbsoluteScale(self, frame_id):
            frame_id = int(frame_id)
            ts = float(self.timestamps[frame_id])
            pose = self.pose_vectors[frame_id]
            if frame_id == 0:
                scale = 1.0
            else:
                scale = float(np.linalg.norm(pose[:3] - self.pose_vectors[frame_id - 1, :3]))
            return (ts, *_pose_components(pose), scale)

    return _LocalTartanAirGroundTruth()


def evaluate_sequence(
    *,
    sequence,
    output_dir: Path,
    max_dt: float,
    missing_penalty_m: float,
    min_coverage_ok: float,
    stride: int,
) -> tuple[str, dict | None]:
    from pyslam.local_features.feature_tracker import feature_tracker_factory
    from pyslam.local_features.feature_tracker_configs import FeatureTrackerConfigs
    from pyslam.slam.visual_odometry import VisualOdometryEducational

    sampled_indices = list(range(0, sequence.num_frames, max(1, int(stride))))
    if len(sampled_indices) < 2:
        return "tracking_failed", None

    seq_short = sequence.key.replace("/", "__")
    run_dir = output_dir / seq_short
    traj_dir = output_dir / "trajectories"
    plot_dir = output_dir / "plots" / seq_short
    run_dir.mkdir(parents=True, exist_ok=True)
    traj_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    gt_file = run_dir / "groundtruth_tum.txt"
    expected_ts = run_dir / "expected_timestamps.txt"
    traj_file = traj_dir / f"{seq_short}_orb2_vo.txt"
    plot_file = plot_dir / "trajectory_3d.png"

    gt_timestamps = [float(i) for i in range(len(sampled_indices))]
    gt_poses = [pose_vector_to_matrix(sequence.poses[idx]) for idx in sampled_indices]
    _write_tum_pose_file(gt_file, gt_timestamps, gt_poses)
    expected_ts.write_text("\n".join(f"{ts:.6f}" for ts in gt_timestamps) + "\n", encoding="utf-8")

    gt = _make_groundtruth(sequence)
    camera = _load_pyslam_camera(width=640, height=480)
    tracker_cfg = dict(FeatureTrackerConfigs.ORB2)
    feature_tracker = feature_tracker_factory(**tracker_cfg)
    vo = VisualOdometryEducational(camera, gt, feature_tracker)

    timestamps = [gt_timestamps[0]]
    poses_w_c = [np.eye(4, dtype=np.float64)]
    try:
        for local_idx, frame_idx in enumerate(sampled_indices):
            bgr = cv2.imread(str(sequence.image_files[frame_idx]), cv2.IMREAD_COLOR)
            if bgr is None:
                raise FileNotFoundError(str(sequence.image_files[frame_idx]))
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            vo.track(rgb, None, None, local_idx, gt_timestamps[local_idx])
            if local_idx == 0:
                continue
            pose = np.eye(4, dtype=np.float64)
            pose[:3, :3] = np.asarray(vo.cur_R, dtype=np.float64)
            pose[:3, 3] = np.asarray(vo.cur_t, dtype=np.float64).reshape(3)
            timestamps.append(gt_timestamps[local_idx])
            poses_w_c.append(pose)
    except Exception as exc:
        (run_dir / "error.txt").write_text(str(exc), encoding="utf-8")
        if len(poses_w_c) < 2:
            return "tracking_failed", None
        _write_tum_pose_file(traj_file, timestamps, poses_w_c)
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

    _write_tum_pose_file(traj_file, timestamps, poses_w_c)
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
        _run_plot(gt_file, traj_file, plot_file, f"{sequence.key} - orb2_vo", max_dt)
    except Exception:
        pass

    coverage = float(metrics.get("coverage", float("nan")))
    if math.isfinite(coverage) and coverage < float(min_coverage_ok):
        return "partial_low_coverage", metrics
    return "ok", metrics


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate ORB2_VO on TartanAir validation sequences.")
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--csv-path", default=None)
    ap.add_argument("--max-dt", type=float, default=0.02)
    ap.add_argument("--missing-penalty-m", type=float, default=3.0)
    ap.add_argument("--min-coverage-ok", type=float, default=0.95)
    ap.add_argument("--stride", type=int, default=1)
    args = ap.parse_args()

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = Path(args.csv_path).expanduser().resolve() if args.csv_path else (output_dir / "metrics_summary.csv")

    sequences, sequence_mode = _select_eval_sequences(dataset_root)
    if not sequences:
        raise FileNotFoundError(f"No TartanAir sequences found under {dataset_root}")
    if sequence_mode != "dpvo_validation":
        print(
            f"[sparse_vo_tartanair] WARNING: DPVO validation split not present under {dataset_root}; "
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
            output_dir=output_dir,
            max_dt=float(args.max_dt),
            missing_penalty_m=float(args.missing_penalty_m),
            min_coverage_ok=float(args.min_coverage_ok),
            stride=int(args.stride),
        )
        _append_csv_row(
            csv_path,
            sequence=seq.key,
            feature_type="ORB2_VO",
            status=status,
            metrics=metrics,
        )
        print(f"[{seq.key}] method=orb2_vo status={status}")

    print(f"ORB2_VO TartanAir results written to {csv_path}")


if __name__ == "__main__":
    main()
