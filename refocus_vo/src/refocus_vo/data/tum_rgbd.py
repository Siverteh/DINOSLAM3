from __future__ import annotations

from pathlib import Path
from typing import Any, List, Tuple

import cv2
import numpy as np

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


def _read_assoc_file(path: Path) -> List[Tuple[float, str]]:
    items: List[Tuple[float, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        ts, rel = line.split()[:2]
        items.append((float(ts), rel))
    return items


def _associate_nearest(
    a: List[Tuple[float, Any]],
    b: List[Tuple[float, Any]],
    max_dt: float,
) -> List[Tuple[float, Any, float, Any]]:
    bt = np.array([t for t, _ in b], dtype=np.float64)
    out = []
    for ta, da in a:
        j = int(np.argmin(np.abs(bt - ta)))
        tb, db = b[j]
        if abs(tb - ta) <= max_dt:
            out.append((ta, da, tb, db))
    return out


class TUMRGBDSequence:
    def __init__(
        self,
        dataset_root: str | Path,
        sequence: str,
        *,
        max_rgb_depth_dt: float = 0.02,
    ):
        self.dataset_root = Path(dataset_root).expanduser().resolve()
        self.sequence = str(sequence)
        self.sequence_dir = self.dataset_root / self.sequence
        if not self.sequence_dir.exists():
            raise FileNotFoundError(f"Sequence folder not found: {self.sequence_dir}")

        rgb_txt = self.sequence_dir / "rgb.txt"
        depth_txt = self.sequence_dir / "depth.txt"
        if not rgb_txt.exists():
            raise FileNotFoundError(f"Missing {rgb_txt}")
        if not depth_txt.exists():
            raise FileNotFoundError(f"Missing {depth_txt}")

        rgb_list = _read_assoc_file(rgb_txt)
        depth_list = _read_assoc_file(depth_txt)
        rgb_depth = _associate_nearest(rgb_list, depth_list, max_dt=float(max_rgb_depth_dt))

        self.frames = [
            {
                "t_rgb": float(t_rgb),
                "rgb": str((self.sequence_dir / rgb_rel).resolve()),
                "t_depth": float(t_depth),
                "depth": str((self.sequence_dir / depth_rel).resolve()),
            }
            for t_rgb, rgb_rel, t_depth, depth_rel in rgb_depth
        ]

    def __len__(self) -> int:
        return len(self.frames)

    @staticmethod
    def read_rgb_np(path: str) -> np.ndarray:
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(path)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    @staticmethod
    def read_depth_np(path: str) -> np.ndarray:
        depth = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise FileNotFoundError(path)
        if depth.dtype != np.uint16:
            depth = depth.astype(np.uint16)
        return depth
