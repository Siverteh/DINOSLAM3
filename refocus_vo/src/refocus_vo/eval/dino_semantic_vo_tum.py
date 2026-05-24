from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from dpvo.lietorch import SE3

from refocus_vo.data import TUMRGBDSequence, matrix_to_pose_vector
from refocus_vo.dino_dpvo.semantic_vonet import load_dino_semantic_vonet_checkpoint
from refocus_vo.eval.metrics import compute_metrics
from refocus_vo.eval.plot_trajectory_3d import main as _plot_trajectory_main
from refocus_vo.eval.sparse_vo import _append_csv_row
from refocus_vo.eval.validate_trajectory import validate


DEFAULT_SEQUENCES = [
    "freiburg1_desk",
    "freiburg1_plant",
    "freiburg1_room",
    "freiburg2_desk_with_person",
    "freiburg3_large_cabinet",
    "freiburg3_walking_static",
]


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


def _normalize_sequence_name(name: str) -> str:
    return str(name).strip().replace("rgbd_dataset_", "", 1)


def _full_sequence_name(name: str) -> str:
    short = _normalize_sequence_name(name)
    return f"rgbd_dataset_{short}"


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
        dtype=np.float32,
    )
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return rgb, intrinsics


def _identity_pose_batch(batch: int, frames: int, device: torch.device) -> SE3:
    pose_vec = torch.zeros((int(batch), int(frames), 7), device=device, dtype=torch.float32)
    pose_vec[..., 6] = 1.0
    return SE3(pose_vec)


def _window_local_poses(traj, *, invert_output: bool) -> np.ndarray:
    final_gs = traj[-1][3]
    mats = final_gs.inv().matrix() if invert_output else final_gs.matrix()
    local = mats[0].detach().cpu().numpy().astype(np.float64)
    first_inv = np.linalg.inv(local[0])
    return np.stack([first_inv @ pose for pose in local], axis=0)


def _estimate_sequence_poses(
    *,
    model,
    sequence: str,
    frames,
    image_size: tuple[int, int],
    n_frames: int,
    dpvo_steps: int,
    invert_output: bool,
    device: torch.device,
) -> tuple[list[float], list[np.ndarray]]:
    prepared_images: list[np.ndarray] = []
    timestamps: list[float] = []
    intrinsics_np = None

    for frame in frames:
        bgr = cv2.imread(frame["rgb"], cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(frame["rgb"])
        rgb, intrinsics = _prepare_tum_image(
            bgr,
            sequence,
            target_height=int(image_size[0]),
            target_width=int(image_size[1]),
        )
        prepared_images.append(rgb)
        timestamps.append(float(frame["t_rgb"]))
        intrinsics_np = intrinsics

    assert intrinsics_np is not None
    total = len(prepared_images)
    if total < 2:
        raise RuntimeError("Need at least two frames for VO evaluation.")

    window = min(int(n_frames), total)
    intrinsics_t = torch.from_numpy(intrinsics_np).to(device=device, dtype=torch.float32).unsqueeze(0)
    global_poses: list[np.ndarray] = []
    global_ts: list[float] = []

    with torch.no_grad():
        for start in range(0, total - window + 1):
            batch_images = np.stack(prepared_images[start : start + window], axis=0)
            images_t = torch.from_numpy(batch_images).permute(0, 3, 1, 2).unsqueeze(0).to(device=device, dtype=torch.float32)
            poses_t = _identity_pose_batch(1, window, device)
            traj, _ = model(
                images_t,
                poses_t,
                None,
                intrinsics_t,
                STEPS=int(dpvo_steps),
                structure_only=False,
                frontend_mode="dino_hybrid",
                native_fraction=0.75,
                dino_fraction=0.25,
            )
            local_poses = _window_local_poses(traj, invert_output=invert_output)

            if start == 0:
                global_poses.extend([pose.copy() for pose in local_poses])
                global_ts.extend(timestamps[:window])
                continue

            rel = np.linalg.inv(local_poses[-2]) @ local_poses[-1]
            global_poses.append(global_poses[-1] @ rel)
            global_ts.append(timestamps[start + window - 1])

    return global_ts, global_poses


def evaluate_sequence(
    *,
    dataset_root: Path,
    sequence: str,
    model,
    image_size: tuple[int, int],
    n_frames: int,
    dpvo_steps: int,
    invert_output: bool,
    max_frames: int | None,
    output_dir: Path,
    max_dt: float,
    missing_penalty_m: float,
    min_coverage_ok: float,
    device: torch.device,
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
    frames = ds.frames[: int(max_frames)] if max_frames is not None else ds.frames
    if len(frames) < 2:
        return "tracking_failed", None

    run_dir = output_dir / seq_short
    traj_dir = output_dir / "trajectories"
    plot_dir = output_dir / "plots" / seq_short
    run_dir.mkdir(parents=True, exist_ok=True)
    traj_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    expected_ts = run_dir / "expected_timestamps.txt"
    traj_file = traj_dir / f"{_short_alias(seq_short)}_dino_semantic_vo.txt"
    plot_file = plot_dir / "trajectory_3d.png"
    _write_expected_timestamps(expected_ts, frames)

    try:
        timestamps, est_poses = _estimate_sequence_poses(
            model=model,
            sequence=seq_short,
            frames=frames,
            image_size=image_size,
            n_frames=n_frames,
            dpvo_steps=dpvo_steps,
            invert_output=invert_output,
            device=device,
        )
    except Exception as exc:
        (run_dir / "error.txt").write_text(str(exc), encoding="utf-8")
        return "tracking_failed", None

    if len(est_poses) < 2:
        return "tracking_failed", None

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
        _run_plot(gt_file, traj_file, plot_file, f"{seq_short} - dino_semantic_vo", max_dt)
    except Exception:
        pass

    coverage = float(metrics.get("coverage", float("nan")))
    if math.isfinite(coverage) and coverage < float(min_coverage_ok):
        return "partial_low_coverage", metrics
    return "ok", metrics


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate DinoSemanticVONet directly on TUM RGB-D.")
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
    ap.add_argument("--invert-output", action="store_true", default=False)
    ap.add_argument("--max-frames", type=int, default=None)
    args = ap.parse_args()

    cfg, model = load_dino_semantic_vonet_checkpoint(
        Path(args.checkpoint).expanduser().resolve(),
        config=Path(args.config).expanduser().resolve() if args.config else None,
        device=str(args.device),
    )
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = Path(args.csv_path).expanduser().resolve() if args.csv_path else (output_dir / "metrics_summary.csv")
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

    sequences = [s.strip() for s in str(args.sequences).split(",") if s.strip()]
    image_size = tuple(int(v) for v in cfg.model.get("image_size", [240, 320]))
    n_frames = int(cfg.training.get("n_frames", 15))
    dpvo_steps = int(cfg.training.get("dpvo_steps", 18))
    device = torch.device(str(args.device))

    for sequence in sequences:
        status, metrics = evaluate_sequence(
            dataset_root=dataset_root,
            sequence=sequence,
            model=model,
            image_size=image_size,
            n_frames=n_frames,
            dpvo_steps=dpvo_steps,
            invert_output=bool(args.invert_output),
            max_frames=int(args.max_frames) if args.max_frames is not None else None,
            output_dir=output_dir,
            max_dt=float(args.max_dt),
            missing_penalty_m=float(args.missing_penalty_m),
            min_coverage_ok=float(args.min_coverage_ok),
            device=device,
        )
        _append_csv_row(
            csv_path,
            sequence=_normalize_sequence_name(sequence),
            feature_type=str(cfg.feature_type),
            status=status,
            metrics=metrics,
        )
        print(f"[{_normalize_sequence_name(sequence)}] method=dino_semantic_vo status={status}")

    print(f"DinoSemanticVONet TUM results written to {csv_path}")


if __name__ == "__main__":
    main()
