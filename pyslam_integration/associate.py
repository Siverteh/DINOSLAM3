#!/usr/bin/env python3
"""
Associate RGB and depth images based on timestamps.
Based on the TUM RGB-D benchmark tools.
"""

import sys
import argparse


def read_file_list(filename):
    """
    Read list of timestamps and file names from TUM format file.

    Returns:
        List of (timestamp, filename) tuples
    """
    data = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                parts = line.split()
                timestamp = float(parts[0])
                filename = parts[1]
                data.append((timestamp, filename))
    return data


def associate(rgb_list, depth_list, max_difference=0.02):
    """
    Associate RGB and depth images based on closest timestamps.

    Args:
        rgb_list: List of (timestamp, filename) for RGB images
        depth_list: List of (timestamp, filename) for depth images
        max_difference: Maximum time difference in seconds (default 0.02)

    Returns:
        List of associations (rgb_time, rgb_file, depth_time, depth_file)
    """
    associations = []
    depth_index = 0

    for rgb_time, rgb_file in rgb_list:
        # Find closest depth image
        best_diff = float('inf')
        best_match = None

        # Search forward from last position
        for i in range(depth_index, len(depth_list)):
            depth_time, depth_file = depth_list[i]
            diff = abs(rgb_time - depth_time)

            if diff < best_diff:
                best_diff = diff
                best_match = (depth_time, depth_file, i)
            elif diff > best_diff:
                # Times are increasing, no point searching further
                break

        # Check if match is good enough
        if best_match and best_diff < max_difference:
            depth_time, depth_file, idx = best_match
            associations.append((rgb_time, rgb_file, depth_time, depth_file))
            depth_index = idx

    return associations


def main():
    parser = argparse.ArgumentParser(
        description='Associate RGB and depth images based on timestamps.'
    )
    parser.add_argument('rgb_file', help='File containing RGB timestamps and filenames')
    parser.add_argument('depth_file', help='File containing depth timestamps and filenames')
    parser.add_argument('--max_difference', type=float, default=0.02,
                       help='Maximum time difference in seconds (default: 0.02)')
    parser.add_argument('--output', '-o', help='Output file (default: stdout)')

    args = parser.parse_args()

    # Read input files
    print(f"Reading RGB list from {args.rgb_file}", file=sys.stderr)
    rgb_list = read_file_list(args.rgb_file)
    print(f"  Found {len(rgb_list)} RGB images", file=sys.stderr)

    print(f"Reading depth list from {args.depth_file}", file=sys.stderr)
    depth_list = read_file_list(args.depth_file)
    print(f"  Found {len(depth_list)} depth images", file=sys.stderr)

    # Associate
    print(f"Associating with max difference {args.max_difference}s", file=sys.stderr)
    associations = associate(rgb_list, depth_list, args.max_difference)
    print(f"  Found {len(associations)} associations", file=sys.stderr)

    # Write output
    if args.output:
        f = open(args.output, 'w')
    else:
        f = sys.stdout

    for rgb_time, rgb_file, depth_time, depth_file in associations:
        f.write(f"{rgb_time} {rgb_file} {depth_time} {depth_file}\n")

    if args.output:
        f.close()
        print(f"Associations written to {args.output}", file=sys.stderr)


if __name__ == '__main__':
    main()