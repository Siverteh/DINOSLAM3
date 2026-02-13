#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_tum_xyz(path: Path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 8:
            continue
        vals = [float(x) for x in parts[:8]]
        rows.append(vals)
    if not rows:
        return np.zeros((0, 8), dtype=np.float64)
    return np.asarray(rows, dtype=np.float64)


def associate_rows_fallback(gt: np.ndarray, est: np.ndarray, max_dt: float):
    if gt.size == 0 or est.size == 0:
        return np.zeros((0, 3)), np.zeros((0, 3))
    gt_t = gt[:, 0]
    gt_xyz = gt[:, 1:4]
    out_gt = []
    out_est = []
    for r in est:
        t = r[0]
        j = int(np.argmin(np.abs(gt_t - t)))
        if abs(float(gt_t[j]) - float(t)) <= max_dt:
            out_gt.append(gt_xyz[j])
            out_est.append(r[1:4])
    if not out_gt:
        return np.zeros((0, 3)), np.zeros((0, 3))
    return np.asarray(out_gt), np.asarray(out_est)


def load_aligned_xyz(groundtruth: Path, trajectory: Path, max_dt: float):
    # Match compute_metrics.py behavior: associate with evo and align estimate to GT.
    from evo.core import sync
    from evo.tools import file_interface

    traj_ref = file_interface.read_tum_trajectory_file(str(groundtruth))
    traj_est = file_interface.read_tum_trajectory_file(str(trajectory))
    traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est, max_diff=max_dt)
    if traj_est.num_poses >= 2:
        traj_est.align(traj_ref, correct_scale=False)
    return traj_ref.positions_xyz.copy(), traj_est.positions_xyz.copy()


def set_axes_equal(ax, gt_xyz: np.ndarray, est_xyz: np.ndarray):
    pts = []
    if gt_xyz.shape[0] > 0:
        pts.append(gt_xyz)
    if est_xyz.shape[0] > 0:
        pts.append(est_xyz)
    if not pts:
        return
    all_pts = np.vstack(pts)
    mins = all_pts.min(axis=0)
    maxs = all_pts.max(axis=0)
    center = 0.5 * (mins + maxs)
    radius = 0.5 * np.max(maxs - mins)
    if not np.isfinite(radius) or radius <= 0:
        radius = 1.0
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groundtruth", required=True)
    ap.add_argument("--trajectory", required=True)
    ap.add_argument("--title", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--max-dt", type=float, default=0.02)
    args = ap.parse_args()

    gt_path = Path(args.groundtruth)
    traj_path = Path(args.trajectory)
    try:
        gt_xyz, est_xyz = load_aligned_xyz(gt_path, traj_path, args.max_dt)
    except Exception:
        gt = read_tum_xyz(gt_path)
        est = read_tum_xyz(traj_path)
        gt_xyz, est_xyz = associate_rows_fallback(gt, est, args.max_dt)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(12.8, 7.2), dpi=150)
    ax = fig.add_subplot(111, projection="3d")

    if est_xyz.shape[0] > 0:
        ax.plot(est_xyz[:, 0], est_xyz[:, 1], est_xyz[:, 2], color="tab:blue", linewidth=1.5, label="Estimated")
        ax.scatter(est_xyz[0, 0], est_xyz[0, 1], est_xyz[0, 2], color="tab:blue", marker="o", s=30, label="Estimated Start")
        ax.scatter(est_xyz[-1, 0], est_xyz[-1, 1], est_xyz[-1, 2], color="tab:blue", marker="x", s=40, label="Estimated End")

    if gt_xyz.shape[0] > 0:
        ax.plot(gt_xyz[:, 0], gt_xyz[:, 1], gt_xyz[:, 2], color="tab:green", linewidth=1.5, label="Ground Truth")
        ax.scatter(gt_xyz[0, 0], gt_xyz[0, 1], gt_xyz[0, 2], color="tab:green", marker="o", s=30, label="GT Start")
        ax.scatter(gt_xyz[-1, 0], gt_xyz[-1, 1], gt_xyz[-1, 2], color="tab:green", marker="x", s=40, label="GT End")

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(args.title)
    set_axes_equal(ax, gt_xyz, est_xyz)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


if __name__ == "__main__":
    main()
