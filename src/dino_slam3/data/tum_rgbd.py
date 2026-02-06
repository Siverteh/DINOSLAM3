from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import random

import numpy as np
import torch
from torch.utils.data import Dataset
import cv2

@dataclass
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float

    def K(self) -> torch.Tensor:
        return torch.tensor(
            [[self.fx, 0.0, self.cx],
             [0.0, self.fy, self.cy],
             [0.0, 0.0, 1.0]],
            dtype=torch.float32,
        )

def tum_intrinsics_for_sequence(sequence: str) -> CameraIntrinsics:
    # Official intrinsics listed by the TUM benchmark documentation.
    # Freiburg 1 RGB: fx=517.3 fy=516.5 cx=318.6 cy=255.3
    # Freiburg 2 RGB: fx=520.9 fy=521.0 cx=325.1 cy=249.7
    # Freiburg 3 RGB: fx=535.4 fy=539.2 cx=320.1 cy=247.6
    s = sequence.lower()
    if "freiburg1" in s:
        return CameraIntrinsics(517.3, 516.5, 318.6, 255.3)
    if "freiburg2" in s:
        return CameraIntrinsics(520.9, 521.0, 325.1, 249.7)
    if "freiburg3" in s:
        return CameraIntrinsics(535.4, 539.2, 320.1, 247.6)
    # Fallback (ROS default)
    return CameraIntrinsics(525.0, 525.0, 319.5, 239.5)

def _read_assoc_file(path: Path) -> List[Tuple[float, str]]:
    items: List[Tuple[float, str]] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        ts, rel = line.split()[:2]
        items.append((float(ts), rel))
    return items

def _read_groundtruth(path: Path) -> List[Tuple[float, np.ndarray]]:
    """
    Each line: timestamp tx ty tz qx qy qz qw
    Returns T_w_c (4x4)
    """
    out: List[Tuple[float, np.ndarray]] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        t = float(parts[0])
        tx, ty, tz = map(float, parts[1:4])
        qx, qy, qz, qw = map(float, parts[4:8])

        # quaternion to rotation (x,y,z,w)
        x, y, z, w = qx, qy, qz, qw
        R = np.array([
            [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
            [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
            [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)],
        ], dtype=np.float32)

        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = R
        T[:3, 3] = np.array([tx, ty, tz], dtype=np.float32)
        out.append((t, T))
    return out

def _associate_nearest(
    a: List[Tuple[float, Any]],
    b: List[Tuple[float, Any]],
    max_dt: float
) -> List[Tuple[float, Any, float, Any]]:
    """
    For each a_i, pick nearest b_j within max_dt.
    Returns list of (ta, a_data, tb, b_data)
    """
    bt = np.array([t for t, _ in b], dtype=np.float64)
    out = []
    for ta, da in a:
        j = int(np.argmin(np.abs(bt - ta)))
        tb, db = b[j]
        if abs(tb - ta) <= max_dt:
            out.append((ta, da, tb, db))
    return out

def _extract_time_from_rgb_depth_item(item):
    # Handles:
    # 1) (t_rgb, rgb_path, depth_path) or (t_rgb, rgb_path, t_depth, depth_path)
    # 2) ((t_rgb, rgb_path), (t_depth, depth_path))
    # 3) (t_rgb, something)
    if isinstance(item, (list, tuple)):
        # nested pair format: ((t_rgb, rgb), (t_depth, depth))
        if len(item) == 2 and isinstance(item[0], (list, tuple)) and len(item[0]) >= 1:
            return float(item[0][0])
        # flat format: (t_rgb, ...)
        if len(item) >= 1:
            return float(item[0])
    # fallback
    return float(item)

class TUMRGBDDataset(Dataset):
    def __init__(
        self,
        dataset_root: str | Path,
        sequence: str,
        frame_spacing_min: int = 1,
        frame_spacing_max: int = 4,
        max_frames: int | None = None,
        pad_to: int = 16,
        is_train: bool = True,
        augmentation: dict | None = None,
        max_rgb_depth_dt: float = 0.02,
        max_rgb_gt_dt: float = 0.02,
        depth_scale: float = 5000.0,
    ):
        self.dataset_root = Path(dataset_root).expanduser()
        try:
            self.dataset_root = self.dataset_root.resolve()
        except Exception:
            pass

        self.sequence = sequence
        self.is_train = bool(is_train)

        self.frame_spacing_min = int(frame_spacing_min)
        self.frame_spacing_max = int(frame_spacing_max)
        assert self.frame_spacing_min >= 1
        assert self.frame_spacing_max >= self.frame_spacing_min

        self.max_frames = None if max_frames is None else int(max_frames)
        self.pad_to = int(pad_to)
        self.depth_scale = float(depth_scale)

        self.max_rgb_depth_dt = float(max_rgb_depth_dt)
        self.max_rgb_gt_dt = float(max_rgb_gt_dt)

        # Aug config (your _photometric_aug expects self.aug)
        self.aug = augmentation if (augmentation and self.is_train) else None

        # Sequence directory
        self.sequence_dir = self.dataset_root / sequence
        if not self.sequence_dir.exists():
            raise FileNotFoundError(f"Sequence folder not found: {self.sequence_dir}")

        rgb_txt = self.sequence_dir / "rgb.txt"
        depth_txt = self.sequence_dir / "depth.txt"
        gt_txt = self.sequence_dir / "groundtruth.txt"

        if not rgb_txt.exists():
            raise FileNotFoundError(f"Missing {rgb_txt}")
        if not depth_txt.exists():
            raise FileNotFoundError(f"Missing {depth_txt}")
        if not gt_txt.exists():
            raise FileNotFoundError(f"Missing {gt_txt}")

        # Read lists
        rgb_list = _read_assoc_file(rgb_txt)          # [(t, "rgb/..png"), ...]
        depth_list = _read_assoc_file(depth_txt)      # [(t, "depth/..png"), ...]
        gt_list = _read_groundtruth(gt_txt)           # [(t, 4x4), ...]

        if len(rgb_list) == 0 or len(depth_list) == 0:
            raise RuntimeError(f"Empty rgb/depth list in {self.sequence_dir}")
        if len(gt_list) == 0:
            raise RuntimeError(f"Empty groundtruth list in {self.sequence_dir}")

        # Associate RGB->Depth : returns list of (t_rgb, rgb_rel, t_d, depth_rel)
        rgb_depth = _associate_nearest(rgb_list, depth_list, max_dt=self.max_rgb_depth_dt)

        if self.max_frames is not None:
            rgb_depth = rgb_depth[: self.max_frames]

        if len(rgb_depth) < (1 + self.frame_spacing_min):
            raise RuntimeError(
                f"Not enough rgb-depth associations in {self.sequence_dir}. "
                f"Got {len(rgb_depth)}, need >= {1 + self.frame_spacing_min}."
            )

        # Prepare GT lookup (timestamps only in numpy)
        gt_ts = np.array([float(t) for (t, _) in gt_list], dtype=np.float64)
        gt_Ts = [T for (_, T) in gt_list]

        def nearest_gt_pose(t_rgb: float) -> Optional[np.ndarray]:
            j = int(np.argmin(np.abs(gt_ts - t_rgb)))
            if abs(float(gt_ts[j]) - float(t_rgb)) <= self.max_rgb_gt_dt:
                return gt_Ts[j].astype(np.float32)
            return None

        # Build frames: absolute paths + pose
        frames = []
        for t_rgb, rgb_rel, t_d, depth_rel in rgb_depth:
            T_w_c = nearest_gt_pose(float(t_rgb))
            if T_w_c is None:
                continue  # drop if no GT close enough (prevents None later)

            rgb_abs = (self.sequence_dir / rgb_rel).as_posix()
            depth_abs = (self.sequence_dir / depth_rel).as_posix()

            frames.append(
                {
                    "t_rgb": float(t_rgb),
                    "t_depth": float(t_d),
                    "rgb": rgb_abs,
                    "depth": depth_abs,
                    "T_w_c": T_w_c,
                }
            )

        if len(frames) < (1 + self.frame_spacing_min):
            raise RuntimeError(
                f"After GT association, not enough usable frames in {self.sequence_dir}. "
                f"Got {len(frames)}."
            )

        self.frames = frames
        self.intr = tum_intrinsics_for_sequence(sequence)

    def __len__(self) -> int:
        # Important: for train, delta can be up to frame_spacing_max
        max_delta = self.frame_spacing_max if self.is_train else self.frame_spacing_min
        return max(0, len(self.frames) - max_delta)

    def _read_rgb(self, path: str) -> torch.Tensor:
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(path)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        x = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        return x

    def _read_depth(self, path: str) -> torch.Tensor:
        d = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if d is None:
            raise FileNotFoundError(path)
        if d.dtype != np.uint16:
            d = d.astype(np.uint16)
        z = torch.from_numpy(d).float() / self.depth_scale
        return z.unsqueeze(0)

    def _pad(self, x: torch.Tensor) -> torch.Tensor:
        _, H, W = x.shape
        pad_b = (self.pad_to - (H % self.pad_to)) % self.pad_to
        pad_r = (self.pad_to - (W % self.pad_to)) % self.pad_to
        if pad_b == 0 and pad_r == 0:
            return x
        return torch.nn.functional.pad(x, (0, pad_r, 0, pad_b), value=0.0)

    def _photometric_aug(self, x: torch.Tensor) -> torch.Tensor:
        cfg = (self.aug or {}).get("photometric", {})
        if (not self.is_train) or (not cfg.get("enabled", False)):
            return x

        b = float(cfg.get("brightness", 0.0))
        c = float(cfg.get("contrast", 0.0))

        # brightness
        if b > 0:
            delta = (torch.rand(1).item() * 2 - 1) * b
            x = torch.clamp(x + delta, 0.0, 1.0)

        # contrast
        if c > 0:
            scale = 1.0 + (torch.rand(1).item() * 2 - 1) * c
            mean = x.mean(dim=(1, 2), keepdim=True)
            x = torch.clamp((x - mean) * scale + mean, 0.0, 1.0)

        return x

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self.is_train:
            delta = random.randint(self.frame_spacing_min, self.frame_spacing_max)
        else:
            delta = self.frame_spacing_min

        f1 = self.frames[idx]
        f2 = self.frames[idx + delta]

        rgb1 = self._pad(self._photometric_aug(self._read_rgb(f1["rgb"])))
        rgb2 = self._pad(self._photometric_aug(self._read_rgb(f2["rgb"])))

        depth1 = self._pad(self._read_depth(f1["depth"]))
        depth2 = self._pad(self._read_depth(f2["depth"]))

        valid1 = (depth1 > 0.0).float()
        valid2 = (depth2 > 0.0).float()

        T1 = torch.from_numpy(f1["T_w_c"]).float()
        T2 = torch.from_numpy(f2["T_w_c"]).float()

        # Relative pose cam1->cam2: inv(T_w_c2) @ T_w_c1
        T21 = torch.linalg.inv(T2) @ T1

        K = self.intr.K()

        return {
            "rgb1": rgb1,
            "rgb2": rgb2,
            "depth1": depth1,
            "depth2": depth2,
            "valid_depth1": valid1,
            "valid_depth2": valid2,
            "K": K,
            "relative_pose": T21,
            "sequence": self.sequence,
        }
