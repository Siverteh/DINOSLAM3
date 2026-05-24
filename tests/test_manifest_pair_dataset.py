from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
import sys

import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dino_slam3.data.manifest_rgbd_pairs import ManifestRGBDPairDataset


def _mk_rgb(path: Path) -> None:
    arr = np.zeros((16, 16, 3), dtype=np.uint8)
    arr[..., 0] = 127
    cv2.imwrite(str(path), arr)


def _mk_depth(path: Path) -> None:
    arr = np.full((16, 16), 1000, dtype=np.uint16)
    cv2.imwrite(str(path), arr)


class ManifestPairDatasetTests(unittest.TestCase):
    def test_quality_filter_and_hard_negative_injection(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            img1 = root / "a1.png"
            img2 = root / "a2.png"
            dep1 = root / "d1.png"
            dep2 = root / "d2.png"
            _mk_rgb(img1)
            _mk_rgb(img2)
            _mk_depth(dep1)
            _mk_depth(dep2)

            row_ok = {
                "img1": img1.as_posix(),
                "img2": img2.as_posix(),
                "depth1": dep1.as_posix(),
                "depth2": dep2.as_posix(),
                "K": [[500.0, 0.0, 8.0], [0.0, 500.0, 8.0], [0.0, 0.0, 1.0]],
                "relative_pose": np.eye(4, dtype=float).tolist(),
                "baseline_m": 0.05,
                "overlap_ratio": 0.8,
            }
            row_bad = {
                **row_ok,
                "baseline_m": 0.0,
                "overlap_ratio": 0.01,
            }

            main_manifest = root / "main.jsonl"
            hard_manifest = root / "hard.jsonl"
            with main_manifest.open("w", encoding="utf-8") as f:
                f.write(json.dumps(row_ok) + "\n")
                f.write(json.dumps(row_bad) + "\n")
            with hard_manifest.open("w", encoding="utf-8") as f:
                for _ in range(3):
                    f.write(json.dumps(row_ok) + "\n")

            ds = ManifestRGBDPairDataset(
                manifests=[main_manifest],
                dataset_root=None,
                is_train=True,
                pair_quality={
                    "min_overlap_ratio": 0.1,
                    "min_baseline_m": 0.01,
                    "max_baseline_m": 2.0,
                },
                pair_mining={
                    "hard_negative_fraction": 1.0,
                    "hard_negative_manifests": [hard_manifest.as_posix()],
                },
            )
            # 1 valid base sample + injected hard negatives
            self.assertGreaterEqual(len(ds), 2)

    def test_baseline_falls_back_to_relative_pose_when_missing(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            img1 = root / "a1.png"
            img2 = root / "a2.png"
            dep1 = root / "d1.png"
            dep2 = root / "d2.png"
            _mk_rgb(img1)
            _mk_rgb(img2)
            _mk_depth(dep1)
            _mk_depth(dep2)

            T21 = np.eye(4, dtype=float)
            T21[0, 3] = 0.05
            row = {
                "img1": img1.as_posix(),
                "img2": img2.as_posix(),
                "depth1": dep1.as_posix(),
                "depth2": dep2.as_posix(),
                "K": [[500.0, 0.0, 8.0], [0.0, 500.0, 8.0], [0.0, 0.0, 1.0]],
                "relative_pose": T21.tolist(),
                # baseline_m intentionally omitted
                "overlap_ratio": 0.8,
            }
            manifest = root / "m.jsonl"
            with manifest.open("w", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")

            ds = ManifestRGBDPairDataset(
                manifests=[manifest],
                dataset_root=None,
                is_train=True,
                pair_quality={
                    "min_overlap_ratio": 0.1,
                    "min_baseline_m": 0.01,
                    "max_baseline_m": 2.0,
                },
            )
            self.assertEqual(len(ds), 1)


if __name__ == "__main__":
    unittest.main()
