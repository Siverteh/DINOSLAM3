#!/usr/bin/env python3
"""Associate RGB and depth entries by nearest timestamps."""

from __future__ import annotations

import argparse
import sys


def read_file_list(filename):
    data = []
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                parts = line.split()
                if len(parts) >= 2:
                    data.append((float(parts[0]), parts[1]))
    return data


def associate(rgb_list, depth_list, max_difference=0.02):
    associations = []
    depth_index = 0
    for rgb_time, rgb_file in rgb_list:
        best_diff = float("inf")
        best_match = None
        for i in range(depth_index, len(depth_list)):
            depth_time, depth_file = depth_list[i]
            diff = abs(rgb_time - depth_time)
            if diff < best_diff:
                best_diff = diff
                best_match = (depth_time, depth_file, i)
            elif diff > best_diff:
                break
        if best_match and best_diff < max_difference:
            depth_time, depth_file, idx = best_match
            associations.append((rgb_time, rgb_file, depth_time, depth_file))
            depth_index = idx
    return associations


def main():
    parser = argparse.ArgumentParser(description="Associate RGB and depth files by timestamp.")
    parser.add_argument("rgb_file")
    parser.add_argument("depth_file")
    parser.add_argument("--max_difference", type=float, default=0.02)
    parser.add_argument("--output", "-o", required=True)
    args = parser.parse_args()

    rgb_list = read_file_list(args.rgb_file)
    depth_list = read_file_list(args.depth_file)
    associations = associate(rgb_list, depth_list, args.max_difference)

    with open(args.output, "w", encoding="utf-8") as f:
        for rgb_time, rgb_file, depth_time, depth_file in associations:
            f.write(f"{rgb_time} {rgb_file} {depth_time} {depth_file}\n")

    print(f"Associated {len(associations)} frame pairs -> {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
