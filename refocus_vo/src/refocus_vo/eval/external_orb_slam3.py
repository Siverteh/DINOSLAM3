from __future__ import annotations

import argparse
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

from refocus_vo.eval.metrics import compute_metrics
from refocus_vo.eval.plot_trajectory_3d import main as _plot_trajectory_main
from refocus_vo.eval.sparse_vo import (
    DEFAULT_SEQUENCES,
    _append_csv_row,
    _full_sequence_name,
    _normalize_sequence_name,
    _short_alias,
    _tum_settings_name,
    _write_expected_timestamps,
)
from refocus_vo.eval.validate_trajectory import validate


def _sequence_rgb_frames(seq_dir: Path) -> list[dict]:
    rgb_path = seq_dir / "rgb.txt"
    frames: list[dict] = []
    for raw in rgb_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        ts, rel = line.split()[:2]
        frames.append({"t_rgb": float(ts), "rgb": rel})
    return frames


def _load_assoc_frames(seq_dir: Path) -> list[dict]:
    assoc_path = seq_dir / "associations.txt"
    frames: list[dict] = []
    for raw in assoc_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        ts_rgb, rgb_rel, ts_depth, depth_rel = line.split()[:4]
        frames.append(
            {
                "t_rgb": float(ts_rgb),
                "rgb": rgb_rel,
                "t_depth": float(ts_depth),
                "depth": depth_rel,
            }
        )
    return frames


def _fmt_cmd(cmd: list[str]) -> str:
    return " ".join(cmd)


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


def _orbslam_env(orb_root: Path, env_dir: Path | None) -> dict[str, str]:
    env = os.environ.copy()
    lib_paths = [
        orb_root / "lib",
        orb_root / "Thirdparty" / "DBoW2" / "lib",
        orb_root / "Thirdparty" / "g2o" / "lib",
    ]
    if env_dir is not None:
        lib_paths.insert(0, env_dir / "lib")
    existing = env.get("LD_LIBRARY_PATH")
    env["LD_LIBRARY_PATH"] = ":".join(str(p) for p in lib_paths) + (
        f":{existing}" if existing else ""
    )
    return env


def evaluate_sequence(
    *,
    dataset_root: Path,
    sequence: str,
    orb_root: Path,
    env_dir: Path | None,
    output_dir: Path,
    max_dt: float,
    missing_penalty_m: float,
    min_coverage_ok: float,
    mode: str,
    timeout_s: int,
) -> tuple[str, dict | None]:
    seq_short = _normalize_sequence_name(sequence)
    seq_full = _full_sequence_name(sequence)
    seq_dir = dataset_root / seq_full
    if not seq_dir.exists():
        return "skipped_missing_sequence", None

    gt_file = seq_dir / "groundtruth.txt"
    if not gt_file.exists():
        return "skipped_missing_groundtruth", None

    settings = orb_root / "Examples" / ("Monocular" if mode == "mono" else "RGB-D") / _tum_settings_name(seq_short)
    vocabulary = orb_root / "Vocabulary" / "ORBvoc.txt"
    binary = orb_root / "Examples" / ("Monocular" if mode == "mono" else "RGB-D") / (
        "mono_tum" if mode == "mono" else "rgbd_tum"
    )
    if not binary.exists():
        return "missing_binary", None
    if not vocabulary.exists():
        return "missing_vocabulary", None

    frames = _sequence_rgb_frames(seq_dir) if mode == "mono" else _load_assoc_frames(seq_dir)
    if len(frames) < 2:
        return "tracking_failed", None

    run_dir = output_dir / seq_short
    traj_dir = output_dir / "trajectories"
    plot_dir = output_dir / "plots" / seq_short
    run_dir.mkdir(parents=True, exist_ok=True)
    traj_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    expected_ts = run_dir / "expected_timestamps.txt"
    _write_expected_timestamps(expected_ts, frames)
    raw_traj = run_dir / ("KeyFrameTrajectory.txt" if mode == "mono" else "CameraTrajectory.txt")
    traj_path = traj_dir / f"{_short_alias(seq_short)}_orb_slam3_{mode}.txt"
    plot_path = plot_dir / "trajectory_3d.png"
    log_path = run_dir / "orb_slam3.log"
    error_path = run_dir / "error.txt"

    for stale in (run_dir / "KeyFrameTrajectory.txt", run_dir / "CameraTrajectory.txt"):
        stale.unlink(missing_ok=True)

    cmd = [
        str(binary),
        str(vocabulary),
        str(settings),
        str(seq_dir),
    ]
    if mode == "rgbd":
        cmd.append(str(seq_dir / "associations.txt"))
    (run_dir / "command.txt").write_text(_fmt_cmd(cmd) + "\n", encoding="utf-8")

    try:
        completed = subprocess.run(
            cmd,
            cwd=run_dir,
            env=_orbslam_env(orb_root, env_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        log_path.write_text(exc.stdout or "", encoding="utf-8")
        error_path.write_text(f"timeout after {timeout_s}s\n", encoding="utf-8")
        return "timeout", None

    log_path.write_text(completed.stdout or "", encoding="utf-8")
    if completed.returncode != 0:
        error_path.write_text(f"return code {completed.returncode}\n", encoding="utf-8")
        return "tracking_failed", None
    if not raw_traj.exists() or raw_traj.stat().st_size == 0:
        error_path.write_text("trajectory file missing or empty\n", encoding="utf-8")
        return "tracking_failed", None

    shutil.copy2(raw_traj, traj_path)
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
        return "eval_failed", None

    try:
        _run_plot(
            gt_file,
            traj_path,
            plot_path,
            f"{seq_short} ORB-SLAM3 {mode}",
            max_dt,
        )
    except Exception as exc:
        (run_dir / "plot_error.txt").write_text(str(exc), encoding="utf-8")

    coverage = float(metrics.get("coverage", 0.0))
    status = "ok" if coverage >= min_coverage_ok else "partial_low_coverage"
    return status, metrics


def _parse_sequences(raw: str) -> list[str]:
    if raw.strip().lower() == "default":
        return list(DEFAULT_SEQUENCES)
    return [s.strip() for s in raw.split(",") if s.strip()]


def _write_header(csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if csv_path.exists():
        return
    csv_path.write_text(
        "sequence,feature_type,status,ate_rmse,ate_mean,ate_median,"
        "ate_rmse_associated,ate_mean_associated,ate_median_associated,"
        "rpe_trans_rmse,rpe_rot_rmse,scale_correction,scale_error_abs,"
        "scale_error_abs_log,coverage\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--orb-root", type=Path, required=True)
    parser.add_argument("--env-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--csv-path", type=Path, required=True)
    parser.add_argument("--sequences", default="default")
    parser.add_argument("--mode", choices=["mono", "rgbd"], default="mono")
    parser.add_argument("--max-dt", type=float, default=0.02)
    parser.add_argument("--missing-penalty-m", type=float, default=3.0)
    parser.add_argument("--min-coverage-ok", type=float, default=0.95)
    parser.add_argument("--timeout-s", type=int, default=900)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_header(args.csv_path)
    feature_type = f"ORB_SLAM3_{args.mode.upper()}"

    for sequence in _parse_sequences(args.sequences):
        status, metrics = evaluate_sequence(
            dataset_root=args.dataset_root,
            sequence=sequence,
            orb_root=args.orb_root,
            env_dir=args.env_dir,
            output_dir=args.output_dir,
            max_dt=args.max_dt,
            missing_penalty_m=args.missing_penalty_m,
            min_coverage_ok=args.min_coverage_ok,
            mode=args.mode,
            timeout_s=args.timeout_s,
        )
        _append_csv_row(
            args.csv_path,
            sequence=_normalize_sequence_name(sequence),
            feature_type=feature_type,
            status=status,
            metrics=metrics,
        )
        print(f"[{_normalize_sequence_name(sequence)}] method=orb_slam3_{args.mode} status={status}", flush=True)


if __name__ == "__main__":
    main()
