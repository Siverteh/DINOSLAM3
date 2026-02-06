#!/usr/bin/env python3
"""
Download TUM RGB-D Dataset Sequences
Downloads all necessary sequences for training and evaluation.
"""

import os
import sys
import subprocess
from pathlib import Path
from tqdm import tqdm

# TUM RGB-D base URL
BASE_URL = "https://vision.in.tum.de/rgbd/dataset/{}/rgbd_dataset_{}_{}.tgz"

# All sequences to download
SEQUENCES = {
    # Freiburg 1 sequences (Kinect v1)
    'freiburg1': [
        'xyz',
        'rpy',
        'floor',
        'desk',
        'desk2',
        'room',
        '360',
        'plant',
        'teddy'
    ],
    # Freiburg 2 sequences (Kinect v1, different location)
    'freiburg2': [
        'xyz',
        'rpy',
        '360_hemisphere',
        '360_kidnap',
        'desk',
        'pioneer_360',
        'pioneer_slam',
        'pioneer_slam2',
        'pioneer_slam3',
        'dishes',
        'flowerbouquet',
        'flowerbouquet_brownbackground',
        'metallic_sphere',
        'metallic_sphere2',
        'coke',
        'desk_with_person'
    ],
    # Freiburg 3 sequences (Kinect v2)
    'freiburg3': [
        'long_office_household',
        'structure_texture_near',
        'structure_texture_far',
        'walking_xyz',
        'sitting_xyz',
        'walking_halfsphere',
        'sitting_halfsphere',
        'walking_rpy',
        'sitting_rpy',
        'teddy',
        'cabinet',
        'large_cabinet'
    ]
}

def download_sequence(freiburg_id: str, sequence_name: str, output_dir: Path) -> bool:
    """Download and extract a single sequence"""

    # Construct URL
    url = BASE_URL.format(freiburg_id, freiburg_id, sequence_name)

    # Output paths
    tgz_file = output_dir / f"rgbd_dataset_{freiburg_id}_{sequence_name}.tgz"
    extract_dir = output_dir / f"rgbd_dataset_{freiburg_id}_{sequence_name}"

    # Check if already exists
    if extract_dir.exists():
        print(f"✓ {extract_dir.name} already exists, skipping...")
        return True

    print(f"\n📥 Downloading {tgz_file.name}...")

    try:
        # Download with wget
        cmd = ["wget", "-c", url, "-O", str(tgz_file)]
        result = subprocess.run(cmd, check=True, capture_output=True)

        # Extract
        print(f"📦 Extracting {tgz_file.name}...")
        cmd = ["tar", "-xzf", str(tgz_file), "-C", str(output_dir)]
        subprocess.run(cmd, check=True, capture_output=True)

        # Remove tar file to save space
        tgz_file.unlink()

        print(f"✓ Successfully downloaded and extracted {extract_dir.name}")
        return True

    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to download {sequence_name}: {e}")
        if tgz_file.exists():
            tgz_file.unlink()
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def main():
    # Get output directory
    output_dir = Path(__file__).parent.parent / "tum_rgbd"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("TUM RGB-D DATASET DOWNLOADER")
    print("="*70)
    print(f"Output directory: {output_dir.resolve()}")
    print(f"Total sequences: {sum(len(seqs) for seqs in SEQUENCES.values())}")
    print("="*70)

    # Check for wget
    try:
        subprocess.run(["wget", "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("✗ wget not found. Please install wget:")
        print("  Ubuntu/Debian: sudo apt-get install wget")
        print("  macOS: brew install wget")
        sys.exit(1)

    # Download all sequences
    success_count = 0
    fail_count = 0

    for freiburg_id, sequences in SEQUENCES.items():
        print(f"\n{'='*70}")
        print(f"Downloading {freiburg_id} sequences ({len(sequences)} total)")
        print(f"{'='*70}")

        for seq_name in sequences:
            if download_sequence(freiburg_id, seq_name, output_dir):
                success_count += 1
            else:
                fail_count += 1

    # Summary
    print("\n" + "="*70)
    print("DOWNLOAD SUMMARY")
    print("="*70)
    print(f"✓ Successful: {success_count}")
    print(f"✗ Failed: {fail_count}")
    print(f"📁 Location: {output_dir.resolve()}")
    print("="*70)

    if fail_count > 0:
        print("\n⚠️  Some downloads failed. You can re-run this script to retry.")
        sys.exit(1)
    else:
        print("\n✓ All sequences downloaded successfully!")
        print("\nNext steps:")
        print("  1. Update configs/train_config.yaml with the dataset path")
        print("  2. Run: python train.py")


if __name__ == "__main__":
    main()