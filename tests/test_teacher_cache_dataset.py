from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dino_slam3.data.tum_rgbd import TUMRGBDDataset


def _write_tum_sequence_with_images(root: Path, sequence: str, n_frames: int = 8) -> Path:
    seq_dir = root / sequence
    rgb_dir = seq_dir / "rgb"
    depth_dir = seq_dir / "depth"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)

    rgb_lines = ["# timestamp rgb_file"]
    depth_lines = ["# timestamp depth_file"]
    gt_lines = ["# timestamp tx ty tz qx qy qz qw"]
    for i in range(n_frames):
        ts = i * 0.033
        rgb_rel = f"rgb/{i:06d}.png"
        depth_rel = f"depth/{i:06d}.png"
        rgb_lines.append(f"{ts:.6f} {rgb_rel}")
        depth_lines.append(f"{ts:.6f} {depth_rel}")
        gt_lines.append(f"{ts:.6f} 0 0 0 0 0 0 1")

        rgb = np.full((32, 32, 3), fill_value=i, dtype=np.uint8)
        depth = np.full((32, 32), fill_value=1000, dtype=np.uint16)
        cv2.imwrite(str(seq_dir / rgb_rel), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(seq_dir / depth_rel), depth)

    (seq_dir / "rgb.txt").write_text("\n".join(rgb_lines) + "\n", encoding="utf-8")
    (seq_dir / "depth.txt").write_text("\n".join(depth_lines) + "\n", encoding="utf-8")
    (seq_dir / "groundtruth.txt").write_text("\n".join(gt_lines) + "\n", encoding="utf-8")
    return seq_dir


class TeacherCacheDatasetTests(unittest.TestCase):
    def test_teacher_heatmaps_loaded_from_cache(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            seq = "rgbd_dataset_freiburg1_desk"
            seq_dir = _write_tum_sequence_with_images(root, seq, n_frames=8)

            keys = np.asarray([f"rgb/{i:06d}.png" for i in range(8)], dtype=np.str_)
            counts = np.asarray([1 for _ in range(8)], dtype=np.int32)
            pts = np.asarray([[8.0, 8.0] for _ in range(8)], dtype=np.float32)
            cache_path = root / "teacher_cache.npz"
            np.savez_compressed(
                str(cache_path),
                frame_keys=keys,
                counts=counts,
                points_xy=pts,
            )

            ds = TUMRGBDDataset(
                dataset_root=root,
                sequence=seq,
                frame_spacing_min=1,
                frame_spacing_max=1,
                is_train=False,
                teacher_enabled=True,
                teacher_type="orb2",
                teacher_cache_path=cache_path,
                teacher_stride=4,
                teacher_dilate_radius=0,
            )

            item = ds[0]
            self.assertIn("teacher_heatmap1", item)
            self.assertIn("teacher_heatmap2", item)
            self.assertEqual(tuple(item["teacher_heatmap1"].shape), (1, 8, 8))
            self.assertGreater(float(item["teacher_heatmap1"].sum().item()), 0.0)

            # Remove one key and ensure missing frame resolves to zero map.
            np.savez_compressed(
                str(cache_path),
                frame_keys=np.asarray(["rgb/000000.png"], dtype=np.str_),
                counts=np.asarray([1], dtype=np.int32),
                points_xy=np.asarray([[8.0, 8.0]], dtype=np.float32),
            )
            ds_missing = TUMRGBDDataset(
                dataset_root=root,
                sequence=seq,
                frame_spacing_min=1,
                frame_spacing_max=1,
                is_train=False,
                teacher_enabled=True,
                teacher_type="orb2",
                teacher_cache_path=cache_path,
                teacher_stride=4,
                teacher_dilate_radius=0,
            )
            item_missing = ds_missing[1]  # frame1 -> frame2
            self.assertEqual(float(item_missing["teacher_heatmap1"].sum().item()), 0.0)


if __name__ == "__main__":
    unittest.main()
