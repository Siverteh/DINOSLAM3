#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def _read_tum(path: Path):
    rows = []
    for idx, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 8:
            raise ValueError(f"Invalid column count at line {idx}: expected 8, got {len(parts)}")
        try:
            vals = [float(x) for x in parts]
        except Exception as exc:
            raise ValueError(f"Non-numeric value at line {idx}") from exc
        if not all(math.isfinite(v) for v in vals):
            raise ValueError(f"NaN/Inf value at line {idx}")
        rows.append(vals)
    if not rows:
        raise ValueError("Trajectory is empty")
    return np.asarray(rows, dtype=np.float64)


def validate(traj_path: Path, gt_path: Path, max_dt: float):
    traj = _read_tum(traj_path)
    gt = _read_tum(gt_path)

    t_traj = traj[:, 0]
    t_gt = gt[:, 0]

    # Strict monotonic check
    if np.any(np.diff(t_traj) < 0):
        raise ValueError("Trajectory timestamps are not sorted ascending")

    # Nearest GT timestamp must be within threshold.
    for t in t_traj:
        j = int(np.argmin(np.abs(t_gt - t)))
        if abs(float(t_gt[j]) - float(t)) > max_dt:
            raise ValueError(f"Timestamp alignment failure: |{t_gt[j]} - {t}| > {max_dt}")

    return {
        "valid": True,
        "rows": int(traj.shape[0]),
        "max_dt": float(max_dt),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trajectory", required=True)
    ap.add_argument("--groundtruth", required=True)
    ap.add_argument("--max-dt", type=float, default=0.02)
    ap.add_argument("--output-json", default=None)
    args = ap.parse_args()

    out = {"valid": False}
    code = 1
    try:
        out = validate(Path(args.trajectory), Path(args.groundtruth), args.max_dt)
        code = 0
    except Exception as exc:
        out = {"valid": False, "error": str(exc)}

    if args.output_json:
        Path(args.output_json).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out))
    raise SystemExit(code)


if __name__ == "__main__":
    main()
