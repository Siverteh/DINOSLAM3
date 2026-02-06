from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T


def _resolve_root(dataset_root: str | Path) -> Path:
    p = Path(dataset_root)
    if p.is_absolute():
        return p
    return (Path.cwd() / p).resolve()


def _read_tum_list(txt_path: Path) -> List[Tuple[float, str]]:
    """
    Reads TUM rgb.txt / depth.txt style files:
      timestamp path
    Ignores comment lines starting with '#'.
    Returns sorted list by timestamp.
    """
    if not txt_path.exists():
        raise FileNotFoundError(f"Missing file: {txt_path}")
    items: List[Tuple[float, str]] = []
    for line in txt_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        ts = float(parts[0])
        rel = parts[1]
        items.append((ts, rel))
    items.sort(key=lambda x: x[0])
    return items


def _associate_by_timestamp(
    rgb_items: List[Tuple[float, str]],
    depth_items: List[Tuple[float, str]],
    max_dt: float = 0.02,
) -> List[Tuple[float, str, float, str]]:
    """
    Associates rgb and depth streams by nearest timestamp within max_dt.

    Returns list of (t_rgb, rgb_relpath, t_depth, depth_relpath).
    """
    if not rgb_items or not depth_items:
        return []

    depth_ts = np.array([t for t, _ in depth_items], dtype=np.float64)
    out: List[Tuple[float, str, float, str]] = []
    for t_rgb, rgb_rel in rgb_items:
        j = int(np.argmin(np.abs(depth_ts - t_rgb)))
        t_d, d_rel = depth_items[j]
        if abs(t_d - t_rgb) <= max_dt:
            out.append((t_rgb, rgb_rel, t_d, d_rel))
    return out


def _quat_to_rot(q: np.ndarray) -> np.ndarray:
    # q = [qx, qy, qz, qw]
    x, y, z, w = q
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    R = np.array(
        [
            [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
        ],
        dtype=np.float32,
    )
    return R


def _pose_mat(tx, ty, tz, qx, qy, qz, qw) -> np.ndarray:
    R = _quat_to_rot(np.array([qx, qy, qz, qw], dtype=np.float32))
    Tm = np.eye(4, dtype=np.float32)
    Tm[:3, :3] = R
    Tm[:3, 3] = np.array([tx, ty, tz], dtype=np.float32)
    return Tm


def _read_groundtruth(gt_path: Path) -> List[Tuple[float, np.ndarray]]:
    """
    groundtruth.txt:
      timestamp tx ty tz qx qy qz qw
    """
    if not gt_path.exists():
        return []
    out: List[Tuple[float, np.ndarray]] = []
    for line in gt_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 8:
            continue
        ts = float(parts[0])
        tx, ty, tz, qx, qy, qz, qw = map(float, parts[1:])
        out.append((ts, _pose_mat(tx, ty, tz, qx, qy, qz, qw)))
    out.sort(key=lambda x: x[0])
    return out


def _pose_nearest(gt: List[Tuple[float, np.ndarray]], ts: float) -> Optional[np.ndarray]:
    if not gt:
        return None
    ts_arr = np.array([t for t, _ in gt], dtype=np.float64)
    j = int(np.argmin(np.abs(ts_arr - ts)))
    return gt[j][1].astype(np.float32)


@dataclass
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float

    def K(self) -> torch.Tensor:
        return torch.tensor(
            [
                [self.fx, 0.0, self.cx],
                [0.0, self.fy, self.cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        )


@dataclass
class PreprocessInfo:
    H: int
    W: int
    pad_bottom: int
    pad_right: int


class TUMRGBDDataset(Dataset):
    """
    Returns paired frames with:
      rgb1, rgb2: float32 (3,H,W) ImageNet normalized
      depth1, depth2: float32 (1,H,W) in meters
      valid_depth1, valid_depth2: float32 (1,H,W) {0,1}
      K: float32 (3,3) intrinsics of returned tensors
      relative_pose: float32 (4,4) mapping cam1 -> cam2
    """

    def __init__(
        self,
        dataset_root: str | Path,
        sequence: str,
        intrinsics: CameraIntrinsics,
        frame_spacing: int = 1,
        max_frames: Optional[int] = None,
        is_train: bool = True,
        depth_scale: float = 5000.0,
        max_association_dt: float = 0.02,
        pad_to: int = 16,
        augmentation: Optional[dict] = None,
    ):
        self.dataset_root = _resolve_root(dataset_root)
        self.sequence = sequence
        self.frame_spacing = int(frame_spacing)
        self.max_frames = None if max_frames is None else int(max_frames)
        self.is_train = bool(is_train)
        self.depth_scale = float(depth_scale)
        self.max_association_dt = float(max_association_dt)
        self.pad_to = int(pad_to)

        seq_dir = self.dataset_root / sequence
        if not seq_dir.exists():
            seq_dir = self.dataset_root
        self.sequence_dir = seq_dir

        rgb_txt = seq_dir / "rgb.txt"
        depth_txt = seq_dir / "depth.txt"
        gt_txt = seq_dir / "groundtruth.txt"

        rgb_items = _read_tum_list(rgb_txt)
        depth_items = _read_tum_list(depth_txt)
        pairs = _associate_by_timestamp(rgb_items, depth_items, max_dt=self.max_association_dt)

        if self.max_frames is not None:
            pairs = pairs[: self.max_frames]

        if len(pairs) < (1 + self.frame_spacing):
            raise RuntimeError(f"Not enough associated frames in {seq_dir} with spacing={self.frame_spacing}")

        self.pairs = pairs
        self.gt = _read_groundtruth(gt_txt)
        self.intr = intrinsics

        self.augmentation = augmentation if (augmentation and self.is_train and augmentation.get("enabled", False)) else None
        if self.augmentation:
            self.color_jitter = T.ColorJitter(
                brightness=float(self.augmentation.get("brightness", 0.2)),
                contrast=float(self.augmentation.get("contrast", 0.2)),
                saturation=float(self.augmentation.get("saturation", 0.2)),
                hue=float(self.augmentation.get("hue", 0.05)),
            )
            self.blur_prob = float(self.augmentation.get("gaussian_blur", 0.2))
            self.blur = T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))

        self.to_tensor = T.ToTensor()
        self.norm = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __len__(self) -> int:
        return max(0, len(self.pairs) - self.frame_spacing)

    def _pad_to_multiple(self, x: torch.Tensor, pad_value: float = 0.0) -> Tuple[torch.Tensor, PreprocessInfo]:
        C, H, W = x.shape
        pad_bottom = (self.pad_to - (H % self.pad_to)) % self.pad_to
        pad_right = (self.pad_to - (W % self.pad_to)) % self.pad_to
        if pad_bottom == 0 and pad_right == 0:
            return x, PreprocessInfo(H=H, W=W, pad_bottom=0, pad_right=0)
        x_pad = torch.nn.functional.pad(
            x,
            pad=(0, pad_right, 0, pad_bottom),  # left,right,top,bottom
            mode="constant",
            value=float(pad_value),
        )
        return x_pad, PreprocessInfo(H=H, W=W, pad_bottom=pad_bottom, pad_right=pad_right)

    def _load_rgb(self, rel: str) -> Image.Image:
        p = self.sequence_dir / rel
        return Image.open(p).convert("RGB")

    def _load_depth_png(self, rel: str) -> np.ndarray:
        p = self.sequence_dir / rel
        d = np.array(Image.open(p), dtype=np.float32)
        d = d / self.depth_scale  # meters
        return d

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        i1 = idx
        i2 = idx + self.frame_spacing

        t1, rgb_rel1, _, depth_rel1 = self.pairs[i1]
        t2, rgb_rel2, _, depth_rel2 = self.pairs[i2]

        rgb1 = self._load_rgb(rgb_rel1)
        rgb2 = self._load_rgb(rgb_rel2)

        d1 = self._load_depth_png(depth_rel1)
        d2 = self._load_depth_png(depth_rel2)

        if self.augmentation:
            rgb1 = self.color_jitter(rgb1)
            rgb2 = self.color_jitter(rgb2)
            rng = np.random.RandomState(np.random.randint(0, 2**31 - 1))
            if rng.rand() < self.blur_prob:
                rgb1 = self.blur(rgb1)
            if rng.rand() < self.blur_prob:
                rgb2 = self.blur(rgb2)

        rgb1_t = self.norm(self.to_tensor(rgb1))  # (3,H,W)
        rgb2_t = self.norm(self.to_tensor(rgb2))

        depth1_t = torch.from_numpy(d1).unsqueeze(0)  # (1,H,W)
        depth2_t = torch.from_numpy(d2).unsqueeze(0)

        rgb1_t, info = self._pad_to_multiple(rgb1_t, pad_value=0.0)
        rgb2_t, _ = self._pad_to_multiple(rgb2_t, pad_value=0.0)
        depth1_t, _ = self._pad_to_multiple(depth1_t, pad_value=0.0)
        depth2_t, _ = self._pad_to_multiple(depth2_t, pad_value=0.0)

        valid1 = (depth1_t > 0.0).float()
        valid2 = (depth2_t > 0.0).float()

        # NOTE: We are NOT resizing anywhere here.
        # We only pad bottom/right to multiples of 16. That does NOT change cx/cy.
        K = self.intr.K()

        # Poses: TUM groundtruth is commonly T_w_c (camera in world).
        # We need relative pose mapping cam1 -> cam2:
        #   p_w  = T_w_c1 * p_c1
        #   p_c2 = inv(T_w_c2) * p_w = inv(T2) * T1 * p_c1
        T1 = _pose_nearest(self.gt, t1)
        T2 = _pose_nearest(self.gt, t2)
        if T1 is None or T2 is None:
            relative = np.eye(4, dtype=np.float32)
        else:
            relative = np.linalg.inv(T2) @ T1  # <-- FIXED DIRECTION

        out: Dict[str, torch.Tensor] = {
            "rgb1": rgb1_t,
            "rgb2": rgb2_t,
            "depth1": depth1_t,
            "depth2": depth2_t,
            "valid_depth1": valid1,
            "valid_depth2": valid2,
            "K": K,
            "relative_pose": torch.from_numpy(relative),
            "timestamp1": torch.tensor(float(t1), dtype=torch.float32),
            "timestamp2": torch.tensor(float(t2), dtype=torch.float32),
            "orig_hw": torch.tensor([info.H, info.W], dtype=torch.int32),
            "pad_br": torch.tensor([info.pad_bottom, info.pad_right], dtype=torch.int32),
        }
        return out
