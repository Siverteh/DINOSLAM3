from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T


def _resolve_root(dataset_root: str | Path) -> Path:
    p = Path(dataset_root)
    if p.is_absolute():
        return p
    # resolve relative to CWD (user runs from repo root)
    return (Path.cwd() / p).resolve()


def _quat_to_rot(q: np.ndarray) -> np.ndarray:
    # q = [qx, qy, qz, qw]
    x, y, z, w = q
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    R = np.array([
        [1 - 2*(yy+zz),     2*(xy-wz),     2*(xz+wy)],
        [    2*(xy+wz), 1 - 2*(xx+zz),     2*(yz-wx)],
        [    2*(xz-wy),     2*(yz+wx), 1 - 2*(xx+yy)],
    ], dtype=np.float32)
    return R


def _pose_mat(tx, ty, tz, qx, qy, qz, qw) -> np.ndarray:
    R = _quat_to_rot(np.array([qx, qy, qz, qw], dtype=np.float32))
    Tm = np.eye(4, dtype=np.float32)
    Tm[:3, :3] = R
    Tm[:3, 3] = np.array([tx, ty, tz], dtype=np.float32)
    return Tm


def _load_groundtruth(path: Path, timestamps: List[float]) -> Optional[np.ndarray]:
    if not path.exists():
        return None
    # groundtruth.txt: timestamp tx ty tz qx qy qz qw
    gt = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 8:
                continue
            ts = float(parts[0])
            vals = list(map(float, parts[1:]))
            gt.append((ts, *vals))
    if not gt:
        return None
    gt = np.array(gt, dtype=np.float64)

    # nearest-neighbor timestamp association (good enough for training supervision)
    gt_ts = gt[:, 0]
    poses = []
    for t in timestamps:
        j = int(np.argmin(np.abs(gt_ts - t)))
        _, tx, ty, tz, qx, qy, qz, qw = gt[j]
        poses.append(_pose_mat(tx, ty, tz, qx, qy, qz, qw))
    return np.stack(poses, axis=0).astype(np.float32)


@dataclass
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float

    def scaled(self, sx: float, sy: float) -> "CameraIntrinsics":
        return CameraIntrinsics(
            fx=self.fx * sx,
            fy=self.fy * sy,
            cx=self.cx * sx,
            cy=self.cy * sy,
        )

    def K(self) -> torch.Tensor:
        return torch.tensor([
            [self.fx, 0.0, self.cx],
            [0.0, self.fy, self.cy],
            [0.0, 0.0, 1.0],
        ], dtype=torch.float32)


class TUMRGBDDataset(Dataset):
    """Returns paired frames with RGB, depth, relative pose and intrinsics.

    Keys:
      rgb1, rgb2: float tensors (3,H,W), ImageNet normalized
      depth1, depth2: float tensors (1,H,W) in meters
      K: float tensor (3,3) for resized resolution
      relative_pose: float tensor (4,4) mapping cam1 -> cam2 (T_21)
      valid_depth1, valid_depth2: bool tensors (1,H,W)
    """

    def __init__(
        self,
        dataset_root: str | Path,
        sequence: str,
        intrinsics: CameraIntrinsics,
        input_size: int = 448,
        frame_spacing: int = 1,
        max_frames: Optional[int] = None,
        augmentation: Optional[dict] = None,
        is_train: bool = True,
        depth_scale: float = 5000.0,
        original_resolution: Tuple[int, int] = (640, 480),
    ):
        self.dataset_root = _resolve_root(dataset_root)
        self.sequence = sequence
        self.input_size = int(input_size)
        self.frame_spacing = int(frame_spacing)
        self.is_train = bool(is_train)
        self.depth_scale = float(depth_scale)

        # Resolve sequence dir
        seq_dir = self.dataset_root / sequence
        if not seq_dir.exists():
            # allow dataset_root pointing directly to a sequence
            seq_dir = self.dataset_root
        self.sequence_dir = seq_dir
        self.rgb_dir = seq_dir / "rgb"
        self.depth_dir = seq_dir / "depth"
        self.gt_file = seq_dir / "groundtruth.txt"

        if not self.rgb_dir.exists() or not self.depth_dir.exists():
            raise FileNotFoundError(
                f"Could not find rgb/depth folders under {seq_dir}. "
                "Expected .../<sequence>/{rgb,depth}."
            )

        self.rgb_files = sorted([f for f in os.listdir(self.rgb_dir) if f.endswith('.png')])
        self.depth_files = sorted([f for f in os.listdir(self.depth_dir) if f.endswith('.png')])
        n = min(len(self.rgb_files), len(self.depth_files))
        self.rgb_files = self.rgb_files[:n]
        self.depth_files = self.depth_files[:n]
        self.timestamps = [float(f.split('.')[0]) for f in self.rgb_files]

        if max_frames is not None:
            n = min(n, int(max_frames))
            self.rgb_files = self.rgb_files[:n]
            self.depth_files = self.depth_files[:n]
            self.timestamps = self.timestamps[:n]

        self.poses = _load_groundtruth(self.gt_file, self.timestamps)

        # intrinsics scaling (original -> input_size square)
        ow, oh = original_resolution
        sx = self.input_size / float(ow)
        sy = self.input_size / float(oh)
        self.intr = intrinsics.scaled(sx, sy)

        self.rgb_tf = T.Compose([
            T.Resize((self.input_size, self.input_size), interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.augmentation = augmentation if (augmentation and is_train and augmentation.get("enabled", False)) else None
        if self.augmentation:
            self.color_jitter = T.ColorJitter(
                brightness=float(self.augmentation.get("brightness", 0.2)),
                contrast=float(self.augmentation.get("contrast", 0.2)),
                saturation=float(self.augmentation.get("saturation", 0.2)),
                hue=float(self.augmentation.get("hue", 0.05)),
            )
            self.blur_prob = float(self.augmentation.get("gaussian_blur", 0.2))
            self.blur = T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))

        # depth resize performed on tensor via interpolate

    def __len__(self) -> int:
        return max(0, len(self.rgb_files) - self.frame_spacing)

    def _aug(self, img: Image.Image, seed: int) -> Image.Image:
        random.seed(seed)
        img = self.color_jitter(img)
        if random.random() < self.blur_prob:
            img = self.blur(img)
        return img

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        i1 = idx
        i2 = idx + self.frame_spacing

        rgb1 = Image.open(self.rgb_dir / self.rgb_files[i1]).convert("RGB")
        rgb2 = Image.open(self.rgb_dir / self.rgb_files[i2]).convert("RGB")

        d1 = np.array(Image.open(self.depth_dir / self.depth_files[i1]), dtype=np.float32) / self.depth_scale
        d2 = np.array(Image.open(self.depth_dir / self.depth_files[i2]), dtype=np.float32) / self.depth_scale

        if self.augmentation:
            seed = random.randint(0, 2**31-1)
            rgb1 = self._aug(rgb1, seed)
            rgb2 = self._aug(rgb2, seed)

        rgb1_t = self.rgb_tf(rgb1)
        rgb2_t = self.rgb_tf(rgb2)

        depth1 = torch.from_numpy(d1).unsqueeze(0).unsqueeze(0)  # 1,1,H,W
        depth2 = torch.from_numpy(d2).unsqueeze(0).unsqueeze(0)

        depth1 = torch.nn.functional.interpolate(depth1, size=(self.input_size, self.input_size), mode="nearest").squeeze(0)
        depth2 = torch.nn.functional.interpolate(depth2, size=(self.input_size, self.input_size), mode="nearest").squeeze(0)

        valid1 = (depth1 > 0.0).float()
        valid2 = (depth2 > 0.0).float()

        out = {
            "rgb1": rgb1_t,
            "rgb2": rgb2_t,
            "depth1": depth1,
            "depth2": depth2,
            "valid_depth1": valid1,
            "valid_depth2": valid2,
            "K": self.intr.K(),
            "timestamp1": torch.tensor(self.timestamps[i1], dtype=torch.float32),
            "timestamp2": torch.tensor(self.timestamps[i2], dtype=torch.float32),
        }

        if self.poses is not None:
            T1 = self.poses[i1]
            T2 = self.poses[i2]
            T21 = T2 @ np.linalg.inv(T1)
            out["pose1"] = torch.from_numpy(T1)
            out["pose2"] = torch.from_numpy(T2)
            out["relative_pose"] = torch.from_numpy(T21)
        else:
            # training can still run with only depth if user provides external relative pose
            out["relative_pose"] = torch.eye(4, dtype=torch.float32)

        return out
