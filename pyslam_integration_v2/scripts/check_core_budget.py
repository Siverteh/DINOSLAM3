#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path


def dir_size_bytes(path: Path) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            fp = Path(root) / name
            try:
                total += fp.stat().st_size
            except OSError:
                pass
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None, help="pyslam_integration_v2 root")
    ap.add_argument("--max-core-py", type=int, default=49)
    ap.add_argument("--max-size-mb", type=int, default=500)
    args = ap.parse_args()

    root = Path(args.root) if args.root else Path(__file__).resolve().parents[1]
    core = root / "pyslam" / "pyslam"

    if not core.exists():
        print(f"ERROR: core package path missing: {core}")
        raise SystemExit(1)

    py_count = sum(1 for _ in core.rglob("*.py"))
    size_mb = dir_size_bytes(root) / (1024 * 1024)

    ok_core = py_count <= args.max_core_py
    ok_size = size_mb <= args.max_size_mb

    print(f"core_python_files={py_count} (limit={args.max_core_py})")
    print(f"integration_size_mb={size_mb:.2f} (limit={args.max_size_mb})")

    if not (ok_core and ok_size):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
