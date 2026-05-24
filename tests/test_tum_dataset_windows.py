from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dino_slam3.data.tum_rgbd import TUMRGBDDataset


def _write_minimal_tum_sequence(root: Path, sequence: str, n_frames: int = 20) -> Path:
    seq_dir = root / sequence
    seq_dir.mkdir(parents=True, exist_ok=True)
    (seq_dir / "rgb").mkdir(parents=True, exist_ok=True)
    (seq_dir / "depth").mkdir(parents=True, exist_ok=True)

    rgb_lines = ["# timestamp rgb_file"]
    depth_lines = ["# timestamp depth_file"]
    gt_lines = ["# timestamp tx ty tz qx qy qz qw"]
    for i in range(n_frames):
        ts = i * 0.033
        rgb_lines.append(f"{ts:.6f} rgb/{i:06d}.png")
        depth_lines.append(f"{ts:.6f} depth/{i:06d}.png")
        gt_lines.append(f"{ts:.6f} 0 0 0 0 0 0 1")
    (seq_dir / "rgb.txt").write_text("\n".join(rgb_lines) + "\n", encoding="utf-8")
    (seq_dir / "depth.txt").write_text("\n".join(depth_lines) + "\n", encoding="utf-8")
    (seq_dir / "groundtruth.txt").write_text("\n".join(gt_lines) + "\n", encoding="utf-8")
    return seq_dir


def _write_tum_sequence_with_images(root: Path, sequence: str, n_frames: int = 20) -> Path:
    seq_dir = _write_minimal_tum_sequence(root, sequence, n_frames=n_frames)
    for i in range(n_frames):
        rgb = np.full((32, 32, 3), fill_value=i, dtype=np.uint8)
        depth = np.full((32, 32), fill_value=1000, dtype=np.uint16)
        cv2.imwrite(str(seq_dir / f"rgb/{i:06d}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(seq_dir / f"depth/{i:06d}.png"), depth)
    return seq_dir


class TumDatasetWindowTests(unittest.TestCase):
    def test_frame_window_is_applied_before_pairing(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            seq = "rgbd_dataset_freiburg1_desk"
            _write_minimal_tum_sequence(root, seq, n_frames=20)

            ds = TUMRGBDDataset(
                dataset_root=root,
                sequence=seq,
                frame_spacing_min=1,
                frame_spacing_max=2,
                frame_start_idx=5,
                frame_end_idx=15,
                is_train=False,
            )
            self.assertEqual(len(ds.frames), 10)
            self.assertEqual(len(ds), 8)
            self.assertAlmostEqual(float(ds.frames[0]["t_rgb"]), 5 * 0.033, places=6)

    def test_too_small_window_raises(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            seq = "rgbd_dataset_freiburg1_desk"
            _write_minimal_tum_sequence(root, seq, n_frames=10)

            with self.assertRaises(RuntimeError):
                TUMRGBDDataset(
                    dataset_root=root,
                    sequence=seq,
                    frame_spacing_min=1,
                    frame_spacing_max=2,
                    frame_start_idx=0,
                    frame_end_idx=1,
                    is_train=False,
                )

    def test_signed_delta_sampling_can_return_backward_pairs(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            seq = "rgbd_dataset_freiburg1_desk"
            _write_tum_sequence_with_images(root, seq, n_frames=16)

            ds = TUMRGBDDataset(
                dataset_root=root,
                sequence=seq,
                frame_spacing_min=2,
                frame_spacing_max=2,
                is_train=True,
                pair_sampler={
                    "short_range": [2, 2],
                    "signed_deltas": True,
                },
            )

            saw_neg = False
            saw_pos = False
            for _ in range(40):
                item = ds[8]
                d = int(item["frame_delta"])
                saw_neg = saw_neg or (d < 0)
                saw_pos = saw_pos or (d > 0)
                if saw_neg and saw_pos:
                    break
            self.assertTrue(saw_neg and saw_pos)

    def test_pair_cache_build_and_load_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            seq = "rgbd_dataset_freiburg1_desk"
            _write_tum_sequence_with_images(root, seq, n_frames=12)
            cache_path = root / "pair_cache.npz"

            ds = TUMRGBDDataset(
                dataset_root=root,
                sequence=seq,
                frame_spacing_min=1,
                frame_spacing_max=4,
                is_train=True,
                pair_sampler={
                    "short_range": [1, 2],
                    "medium_range": [3, 4],
                    "signed_deltas": True,
                    "min_pair_valid_ratio": 0.0,
                    "cache_path": str(cache_path),
                },
            )
            self.assertTrue(cache_path.exists())
            self.assertTrue(len(ds._pair_candidates) > 0)

            ds2 = TUMRGBDDataset(
                dataset_root=root,
                sequence=seq,
                frame_spacing_min=1,
                frame_spacing_max=4,
                is_train=True,
                pair_sampler={
                    "short_range": [1, 2],
                    "medium_range": [3, 4],
                    "signed_deltas": True,
                    "min_pair_valid_ratio": 0.0,
                    "cache_path": str(cache_path),
                },
            )
            self.assertTrue(len(ds2._pair_candidates) > 0)
            self.assertGreaterEqual(ds2._pair_cache_quality_floor, 0.0)


if __name__ == "__main__":
    unittest.main()
