from __future__ import annotations

import os
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import random

import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

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
    _GLOBAL_RGB_CACHE: Dict[str, np.ndarray] = {}
    _GLOBAL_DEPTH_CACHE: Dict[str, np.ndarray] = {}

    def __init__(
        self,
        dataset_root: str | Path,
        sequence: str,
        frame_spacing_min: int = 1,
        frame_spacing_max: int = 4,
        frame_start_idx: int = 0,
        frame_end_idx: int | None = None,
        max_frames: int | None = None,
        pad_to: int = 16,
        is_train: bool = True,
        augmentation: dict | None = None,
        max_rgb_depth_dt: float = 0.02,
        max_rgb_gt_dt: float = 0.02,
        depth_scale: float = 5000.0,
        cache_in_memory: bool = False,
        cache_to_disk: bool = False,
        cache_dir: str | Path | None = None,
        pair_sampler: Optional[Dict[str, Any]] = None,
        total_epochs: int = 1,
        teacher_enabled: bool = False,
        teacher_type: str = "orb2",
        teacher_cache_path: str | Path | None = None,
        teacher_max_features: int = 1000,
        teacher_stride: int = 4,
        teacher_dilate_radius: int = 1,
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
        self.frame_start_idx = max(0, int(frame_start_idx))
        self.frame_end_idx = None if frame_end_idx is None else int(frame_end_idx)

        self.max_frames = None if max_frames is None else int(max_frames)
        self.pad_to = int(pad_to)
        self.depth_scale = float(depth_scale)
        self.cache_in_memory = bool(cache_in_memory)
        self.cache_to_disk = bool(cache_to_disk)
        self.cache_dir = Path(cache_dir).expanduser() if cache_dir is not None else None
        self.pair_sampler = dict(pair_sampler or {})
        self.total_epochs = max(1, int(total_epochs))
        self.current_epoch = 1
        self.signed_deltas = bool(self.pair_sampler.get("signed_deltas", False))
        pair_distances_raw = self.pair_sampler.get("pair_distances", [])
        pair_distances: List[int] = []
        if isinstance(pair_distances_raw, (list, tuple)):
            for v in pair_distances_raw:
                try:
                    d = abs(int(v))
                except Exception:
                    continue
                if d >= 1:
                    pair_distances.append(int(d))
        self.pair_distances = sorted(set(pair_distances))
        self.pair_distance_probs_cfg = self.pair_sampler.get("pair_distance_probs", None)
        self.pair_distance_schedule_cfg = self.pair_sampler.get("pair_distance_schedule", None)
        self.min_pair_valid_ratio_cfg = self.pair_sampler.get("min_pair_valid_ratio", 0.0)
        self.hard_mining_start_epoch = max(1, int(self.pair_sampler.get("hard_mining_start_epoch", 1)))
        self.loop_min_gap = max(0, int(self.pair_sampler.get("loop_min_gap", 80)))
        self.loop_pose_dist_m = max(0.0, float(self.pair_sampler.get("loop_pose_dist_m", 0.40)))
        self.loop_yaw_deg = max(0.0, float(self.pair_sampler.get("loop_yaw_deg", 20.0)))
        cache_path_raw = self.pair_sampler.get("cache_path", None)
        self.pair_cache_path = (
            Path(cache_path_raw).expanduser().resolve()
            if cache_path_raw not in (None, "")
            else None
        )
        self._pair_quality_cache: Dict[Tuple[int, int], float] = {}
        self._pair_candidates: Dict[int, List[int]] = {}
        self._pair_cache_quality_floor: float = 0.0
        self._loop_candidates: Dict[int, List[int]] = {}
        self.teacher_enabled = bool(teacher_enabled)
        self.teacher_type = str(teacher_type).strip().lower()
        self.teacher_cache_path = (
            Path(teacher_cache_path).expanduser().resolve()
            if teacher_cache_path not in (None, "")
            else None
        )
        self.teacher_max_features = max(1, int(teacher_max_features))
        self.teacher_stride = max(1, int(teacher_stride))
        self.teacher_dilate_radius = max(0, int(teacher_dilate_radius))
        self._teacher_key_to_range: Dict[str, Tuple[int, int]] = {}
        self._teacher_points_xy: Optional[np.ndarray] = None

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

        # Optional frame-window slicing for single-sequence overfit/holdout lab mode.
        total_assoc = len(rgb_depth)
        start_idx = min(max(0, int(self.frame_start_idx)), total_assoc)
        end_idx = total_assoc if self.frame_end_idx is None else min(max(0, int(self.frame_end_idx)), total_assoc)
        if end_idx < start_idx:
            raise RuntimeError(
                f"Invalid frame window for {self.sequence_dir}: "
                f"frame_start_idx={start_idx}, frame_end_idx={end_idx}"
            )
        rgb_depth = rgb_depth[start_idx:end_idx]

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
        self._rgb_cache: Dict[str, np.ndarray] = {}
        self._depth_cache: Dict[str, np.ndarray] = {}
        if self.cache_to_disk and self.cache_dir is None:
            self.cache_dir = self.sequence_dir / ".dinoslam_cache"
        if self.cache_to_disk and self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        if self.cache_in_memory:
            self._build_memory_cache()

        if self.teacher_enabled:
            if self.teacher_type != "orb2":
                raise RuntimeError(f"Unsupported dataset.teacher.type='{self.teacher_type}' (expected 'orb2')")
            if self.teacher_cache_path is None:
                raise RuntimeError("dataset.teacher.enabled=true requires dataset.teacher.cache_path")
            self._load_teacher_cache(self.teacher_cache_path)

        if self.pair_cache_path is not None:
            try:
                if self.pair_cache_path.exists():
                    self._load_pair_cache(self.pair_cache_path)
                else:
                    self._build_pair_cache(self.pair_cache_path)
            except Exception as exc:
                print(f"[dataset:{self.sequence}] WARNING: pair cache unavailable ({exc}); using on-the-fly sampling")

    def __len__(self) -> int:
        max_delta = self.frame_spacing_max if self.is_train else self.frame_spacing_max
        if self.is_train and self.pair_sampler:
            if len(self.pair_distances) > 0:
                max_delta = max(max_delta, max(int(v) for v in self.pair_distances))
            ranges = []
            for key in ("short_range", "medium_range", "hard_range"):
                r = self.pair_sampler.get(key)
                if isinstance(r, (list, tuple)) and len(r) == 2:
                    lo, hi = int(r[0]), int(r[1])
                    if hi >= lo >= 1:
                        ranges.append((lo, hi))
            if ranges:
                max_delta = max(max_delta, max(hi for _, hi in ranges))
        if self.is_train and self.signed_deltas:
            return max(0, len(self.frames))
        return max(0, len(self.frames) - max_delta)

    def set_epoch(self, epoch: int, total_epochs: Optional[int] = None) -> None:
        self.current_epoch = max(1, int(epoch))
        if total_epochs is not None:
            self.total_epochs = max(1, int(total_epochs))

    def _resolve_min_pair_valid_ratio(self) -> float:
        cfg = self.min_pair_valid_ratio_cfg
        if isinstance(cfg, dict):
            start = cfg.get("start", 0.0)
            end = cfg.get("end", start)
            try:
                s = float(start)
            except Exception:
                s = 0.0
            try:
                e = float(end)
            except Exception:
                e = s
            t = 0.0 if self.total_epochs <= 1 else float(self.current_epoch - 1) / float(self.total_epochs - 1)
            return float((1.0 - t) * s + t * e)
        try:
            return float(cfg)
        except Exception:
            return 0.0

    def _resolve_pair_distance_probs(self) -> Optional[np.ndarray]:
        if len(self.pair_distances) == 0:
            return None
        n = len(self.pair_distances)
        probs = np.ones((n,), dtype=np.float64)

        raw_probs = self.pair_distance_probs_cfg
        if isinstance(raw_probs, (list, tuple)) and len(raw_probs) == n:
            try:
                probs = np.asarray([float(v) for v in raw_probs], dtype=np.float64).reshape(-1)
            except Exception:
                probs = np.ones((n,), dtype=np.float64)

        sched = self.pair_distance_schedule_cfg
        if isinstance(sched, dict):
            p0 = sched.get("start")
            p1 = sched.get("end")
            if isinstance(p0, (list, tuple)) and isinstance(p1, (list, tuple)) and len(p0) == n and len(p1) == n:
                try:
                    s = np.asarray([float(v) for v in p0], dtype=np.float64).reshape(-1)
                    e = np.asarray([float(v) for v in p1], dtype=np.float64).reshape(-1)
                    t = 0.0 if self.total_epochs <= 1 else float(self.current_epoch - 1) / float(self.total_epochs - 1)
                    t = float(np.clip(t, 0.0, 1.0))
                    probs = (1.0 - t) * s + t * e
                except Exception:
                    pass

        # Delay strong long-gap sampling until hard mining starts.
        if self.current_epoch < int(max(1, self.hard_mining_start_epoch)):
            dist = np.asarray(self.pair_distances, dtype=np.int32)
            hard_mask = dist >= 10
            if np.any(hard_mask):
                probs = probs.copy()
                probs[hard_mask] = 0.0

        probs = np.asarray(probs, dtype=np.float64).reshape(-1)
        probs[~np.isfinite(probs)] = 0.0
        probs = np.clip(probs, 0.0, None)
        s = float(probs.sum())
        if s <= 0.0:
            probs = np.ones((n,), dtype=np.float64) / float(max(1, n))
        else:
            probs = probs / s
        return probs

    def _pair_rng(self, idx1: int, idx2: int) -> np.random.Generator:
        seed = (int(idx1) * 1315423911 + int(idx2) * 2654435761 + 0x9E3779B9) & 0xFFFFFFFF
        return np.random.default_rng(seed=seed)

    def _estimate_pair_valid_ratio(self, idx1: int, idx2: int, samples: int = 256) -> float:
        key = (int(idx1), int(idx2))
        cached = self._pair_quality_cache.get(key)
        if cached is not None:
            return float(cached)

        try:
            f1 = self.frames[int(idx1)]
            f2 = self.frames[int(idx2)]
        except Exception:
            self._pair_quality_cache[key] = 0.0
            return 0.0

        d1 = self._read_depth(f1["depth"])[0].cpu().numpy()
        d2 = self._read_depth(f2["depth"])[0].cpu().numpy()
        H, W = d1.shape
        valid = np.argwhere(d1 > 0.0)
        if valid.size == 0:
            self._pair_quality_cache[key] = 0.0
            return 0.0

        rng = self._pair_rng(idx1, idx2)
        ns = min(int(samples), int(valid.shape[0]))
        sel = valid[rng.choice(valid.shape[0], size=ns, replace=False)]
        ys = sel[:, 0].astype(np.float32)
        xs = sel[:, 1].astype(np.float32)
        z1 = d1[sel[:, 0], sel[:, 1]].astype(np.float32)

        fx, fy, cx, cy = float(self.intr.fx), float(self.intr.fy), float(self.intr.cx), float(self.intr.cy)
        X1 = (xs - cx) * z1 / fx
        Y1 = (ys - cy) * z1 / fy
        P1 = np.stack([X1, Y1, z1, np.ones_like(z1)], axis=1)  # (N,4)

        T1 = f1["T_w_c"].astype(np.float32)
        T2 = f2["T_w_c"].astype(np.float32)
        T21 = np.linalg.inv(T2) @ T1
        P2 = (T21 @ P1.T).T[:, :3]
        Z2 = P2[:, 2]
        in_front = Z2 > 1e-3
        if not np.any(in_front):
            self._pair_quality_cache[key] = 0.0
            return 0.0

        X2 = P2[:, 0]
        Y2 = P2[:, 1]
        u2 = fx * X2 / np.maximum(Z2, 1e-6) + cx
        v2 = fy * Y2 / np.maximum(Z2, 1e-6) + cy
        inb = (u2 >= 0.0) & (u2 <= float(W - 1)) & (v2 >= 0.0) & (v2 <= float(H - 1)) & in_front
        if not np.any(inb):
            self._pair_quality_cache[key] = 0.0
            return 0.0

        u2i = np.clip(np.rint(u2).astype(np.int32), 0, W - 1)
        v2i = np.clip(np.rint(v2).astype(np.int32), 0, H - 1)
        z2_obs = d2[v2i, u2i].astype(np.float32)
        depth_valid = z2_obs > 0.0
        depth_bound = 0.05 + 0.03 * np.abs(Z2)
        depth_cons = np.abs(z2_obs - Z2.astype(np.float32)) < depth_bound.astype(np.float32)
        ok = inb & depth_valid & depth_cons
        ratio = float(ok.mean()) if ok.size > 0 else 0.0
        ratio = max(0.0, min(1.0, ratio))
        self._pair_quality_cache[key] = ratio
        return ratio

    def _pair_cache_deltas(self) -> List[int]:
        if len(self.pair_distances) > 0:
            deltas: List[int] = []
            for d in self.pair_distances:
                deltas.append(int(d))
                if self.signed_deltas:
                    deltas.append(int(-d))
            uniq = sorted(set(deltas), key=lambda v: (abs(int(v)), int(v)))
            return [int(v) for v in uniq if int(v) != 0]

        deltas: List[int] = []
        ranges = []
        for key_new, key_old in (
            ("local_range", "short_range"),
            ("mid_range", "medium_range"),
            ("long_range", "hard_range"),
        ):
            r = self.pair_sampler.get(key_new, self.pair_sampler.get(key_old))
            if isinstance(r, (list, tuple)) and len(r) == 2:
                lo, hi = int(r[0]), int(r[1])
                if hi >= lo >= 1:
                    ranges.append((lo, hi))
        if not ranges:
            ranges = [(int(self.frame_spacing_min), int(self.frame_spacing_max))]
        for lo, hi in ranges:
            for d in range(max(1, lo), max(1, hi) + 1):
                deltas.append(int(d))
                if self.signed_deltas:
                    deltas.append(int(-d))
        uniq = sorted(set(deltas), key=lambda v: (abs(int(v)), int(v)))
        return [int(v) for v in uniq if int(v) != 0]

    def _relative_pose_metrics(self, idx1: int, idx2: int) -> Tuple[float, float]:
        try:
            T1 = self.frames[int(idx1)]["T_w_c"].astype(np.float32)
            T2 = self.frames[int(idx2)]["T_w_c"].astype(np.float32)
        except Exception:
            return float("inf"), float("inf")
        t1 = T1[:3, 3]
        t2 = T2[:3, 3]
        dist = float(np.linalg.norm(t2 - t1))
        R1 = T1[:3, :3]
        R2 = T2[:3, :3]
        R = R2 @ R1.T
        yaw = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
        return dist, abs(yaw)

    def _loop_candidates_for_idx(self, idx: int) -> List[int]:
        if idx in self._loop_candidates:
            return self._loop_candidates[int(idx)]

        out: List[int] = []
        n = len(self.frames)
        if (
            self.loop_min_gap <= 0
            or self.loop_pose_dist_m <= 0.0
            or self.loop_yaw_deg <= 0.0
            or n <= 1
        ):
            self._loop_candidates[int(idx)] = out
            return out

        signed = bool(self.signed_deltas)
        for j in range(n):
            if j == idx:
                continue
            d = int(j - idx)
            if not signed and d < 0:
                continue
            if abs(d) < int(self.loop_min_gap):
                continue
            dist, yaw = self._relative_pose_metrics(idx, j)
            if not np.isfinite(dist) or not np.isfinite(yaw):
                continue
            if dist <= float(self.loop_pose_dist_m) and yaw <= float(self.loop_yaw_deg):
                out.append(int(d))
        out = sorted(set(out), key=lambda v: (abs(int(v)), int(v)))
        self._loop_candidates[int(idx)] = out
        return out

    def _load_pair_cache(self, cache_path: Path) -> None:
        with np.load(str(cache_path), allow_pickle=False) as data:
            idxs = np.asarray(data["idxs"], dtype=np.int32)
            deltas = np.asarray(data["deltas"], dtype=np.int16)
            ptr = np.asarray(data["ptr"], dtype=np.int64)
            quality_floor = float(np.asarray(data["quality_floor"]).reshape(-1)[0]) if "quality_floor" in data.files else 0.0
        if ptr.size != idxs.size + 1:
            raise RuntimeError("Invalid pair cache: ptr size mismatch")
        out: Dict[int, List[int]] = {}
        for i, idx in enumerate(idxs.tolist()):
            st = int(ptr[i])
            en = int(ptr[i + 1])
            out[int(idx)] = [int(v) for v in deltas[st:en].tolist()]
        self._pair_candidates = out
        self._pair_cache_quality_floor = float(quality_floor)
        print(
            f"[dataset:{self.sequence}] pair cache loaded "
            f"indices={len(out)} quality_floor={self._pair_cache_quality_floor:.3f} path={cache_path}"
        )

    def _build_pair_cache(self, cache_path: Path) -> None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        quality_floor = max(0.0, self._resolve_min_pair_valid_ratio())
        deltas = self._pair_cache_deltas()
        idxs: List[int] = []
        ptr: List[int] = [0]
        out_deltas: List[int] = []
        n = len(self.frames)
        for idx in range(n):
            accepted: List[int] = []
            for d in deltas:
                j = idx + int(d)
                if j < 0 or j >= n:
                    continue
                q = self._estimate_pair_valid_ratio(idx, j, samples=96)
                if q >= quality_floor:
                    accepted.append(int(d))
            if accepted:
                idxs.append(int(idx))
                out_deltas.extend(accepted)
                ptr.append(len(out_deltas))
        np.savez_compressed(
            str(cache_path),
            idxs=np.asarray(idxs, dtype=np.int32),
            deltas=np.asarray(out_deltas, dtype=np.int16),
            ptr=np.asarray(ptr, dtype=np.int64),
            quality_floor=np.asarray([quality_floor], dtype=np.float32),
        )
        self._pair_candidates = {int(i): [int(v) for v in out_deltas[ptr[k]:ptr[k + 1]]] for k, i in enumerate(idxs)}
        self._pair_cache_quality_floor = float(quality_floor)
        print(
            f"[dataset:{self.sequence}] pair cache built "
            f"indices={len(idxs)} quality_floor={quality_floor:.3f} path={cache_path}"
        )

    def _read_rgb_np(self, path: str) -> np.ndarray:
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(path)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def _read_rgb(self, path: str) -> torch.Tensor:
        rgb = self._rgb_cache.get(path)
        if rgb is None:
            rgb = self._GLOBAL_RGB_CACHE.get(path)
        if rgb is None:
            rgb = self._read_rgb_np(path)
            if self.cache_to_disk:
                self._save_disk_cache(path, rgb, kind="rgb")
        x = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        return x

    def _read_depth_np(self, path: str) -> np.ndarray:
        d = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if d is None:
            raise FileNotFoundError(path)
        if d.dtype != np.uint16:
            d = d.astype(np.uint16)
        return d

    def _read_depth(self, path: str) -> torch.Tensor:
        d = self._depth_cache.get(path)
        if d is None:
            d = self._GLOBAL_DEPTH_CACHE.get(path)
        if d is None:
            d = self._read_depth_np(path)
            if self.cache_to_disk:
                self._save_disk_cache(path, d, kind="depth")
        z = torch.from_numpy(d).float() / self.depth_scale
        return z.unsqueeze(0)

    def _cache_file(self, path: str, kind: str) -> Optional[Path]:
        if not self.cache_to_disk or self.cache_dir is None:
            return None
        key = hashlib.sha1(path.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{kind}_{key}.npy"

    def _load_disk_cache(self, path: str, kind: str) -> Optional[np.ndarray]:
        cp = self._cache_file(path, kind)
        if cp is None or not cp.exists():
            return None
        try:
            return np.load(cp, allow_pickle=False)
        except Exception:
            return None

    def _save_disk_cache(self, path: str, arr: np.ndarray, kind: str) -> None:
        cp = self._cache_file(path, kind)
        if cp is None:
            return
        if cp.exists():
            return
        try:
            np.save(cp, arr, allow_pickle=False)
        except Exception:
            pass

    def _build_memory_cache(self) -> None:
        rgb_paths = sorted({f["rgb"] for f in self.frames})
        depth_paths = sorted({f["depth"] for f in self.frames})
        print(
            f"[dataset:{self.sequence}] caching {len(rgb_paths)} RGB + {len(depth_paths)} depth frames in RAM..."
        )
        for p in rgb_paths:
            rgb = self._GLOBAL_RGB_CACHE.get(p)
            if rgb is None:
                rgb = self._load_disk_cache(p, kind="rgb")
            if rgb is None:
                rgb = self._read_rgb_np(p)
                if self.cache_to_disk:
                    self._save_disk_cache(p, rgb, kind="rgb")
            self._rgb_cache[p] = rgb
            self._GLOBAL_RGB_CACHE[p] = rgb
        for p in depth_paths:
            d = self._GLOBAL_DEPTH_CACHE.get(p)
            if d is None:
                d = self._load_disk_cache(p, kind="depth")
            if d is None:
                d = self._read_depth_np(p)
                if self.cache_to_disk:
                    self._save_disk_cache(p, d, kind="depth")
            self._depth_cache[p] = d
            self._GLOBAL_DEPTH_CACHE[p] = d
        print(f"[dataset:{self.sequence}] cache ready")

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

    def _frame_key_from_path(self, abs_path: str) -> str:
        p = Path(abs_path)
        try:
            return p.resolve().relative_to(self.sequence_dir.resolve()).as_posix()
        except Exception:
            return p.name

    def _load_teacher_cache(self, cache_path: Path) -> None:
        if not cache_path.exists():
            raise FileNotFoundError(f"Teacher cache not found: {cache_path}")

        with np.load(str(cache_path), allow_pickle=False) as data:
            keys = data["frame_keys"] if "frame_keys" in data.files else None
            counts = data["counts"] if "counts" in data.files else None
            points_xy = data["points_xy"] if "points_xy" in data.files else None
            cache_seq_short = str(data["sequence_short"]) if "sequence_short" in data.files else ""
            cache_seq_full = str(data["sequence_full"]) if "sequence_full" in data.files else ""
        if keys is None or counts is None or points_xy is None:
            raise RuntimeError(
                f"Invalid teacher cache format: {cache_path}. "
                "Expected frame_keys, counts, points_xy."
            )
        seq_short = str(self.sequence).replace("rgbd_dataset_", "", 1)
        if cache_seq_short and cache_seq_short != seq_short:
            print(
                f"[dataset:{self.sequence}] WARNING: teacher cache sequence_short mismatch "
                f"cache='{cache_seq_short}' dataset='{seq_short}'"
            )
        if cache_seq_full and cache_seq_full != str(self.sequence):
            print(
                f"[dataset:{self.sequence}] WARNING: teacher cache sequence_full mismatch "
                f"cache='{cache_seq_full}' dataset='{self.sequence}'"
            )

        frame_keys = [str(k) for k in keys.tolist()]
        counts = np.asarray(counts, dtype=np.int32).reshape(-1)
        points_xy = np.asarray(points_xy, dtype=np.float32).reshape(-1, 2)
        if len(frame_keys) != int(counts.shape[0]):
            raise RuntimeError(
                f"Teacher cache size mismatch: len(frame_keys)={len(frame_keys)} "
                f"!= len(counts)={int(counts.shape[0])}"
            )
        if int(points_xy.shape[0]) != int(counts.sum()):
            raise RuntimeError(
                f"Teacher cache size mismatch: points={int(points_xy.shape[0])} "
                f"!= sum(counts)={int(counts.sum())}"
            )

        key_to_range: Dict[str, Tuple[int, int]] = {}
        offset = 0
        for key, cnt in zip(frame_keys, counts.tolist()):
            n = max(0, int(cnt))
            key_to_range[str(key)] = (offset, offset + n)
            offset += n

        self._teacher_key_to_range = key_to_range
        self._teacher_points_xy = points_xy
        print(
            f"[dataset:{self.sequence}] teacher cache loaded "
            f"frames={len(frame_keys)} points={int(points_xy.shape[0])} from {cache_path}"
        )

    def _teacher_heatmap_for_frame(self, rgb_path: str, H: int, W: int) -> torch.Tensor:
        Hf = max(1, int(H // self.teacher_stride))
        Wf = max(1, int(W // self.teacher_stride))
        heat = torch.zeros((1, Hf, Wf), dtype=torch.float32)
        if self._teacher_points_xy is None or len(self._teacher_key_to_range) == 0:
            return heat

        key = self._frame_key_from_path(rgb_path)
        span = self._teacher_key_to_range.get(key)
        if span is None:
            return heat
        st, en = span
        if en <= st:
            return heat

        pts = self._teacher_points_xy[st:en]
        if pts.size == 0:
            return heat
        if pts.shape[0] > self.teacher_max_features:
            pts = pts[: self.teacher_max_features]

        xy = torch.from_numpy(pts).float()
        x = torch.clamp((xy[:, 0] / float(self.teacher_stride)).round().long(), 0, Wf - 1)
        y = torch.clamp((xy[:, 1] / float(self.teacher_stride)).round().long(), 0, Hf - 1)
        heat[0, y, x] = 1.0

        if self.teacher_dilate_radius > 0:
            k = 2 * int(self.teacher_dilate_radius) + 1
            heat = torch.nn.functional.max_pool2d(heat.unsqueeze(0), kernel_size=k, stride=1, padding=k // 2)[0]
            heat = torch.clamp(heat, 0.0, 1.0)
        return heat

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        n_frames = len(self.frames)
        idx = int(idx)
        if n_frames <= 1:
            raise RuntimeError(f"[dataset:{self.sequence}] not enough frames for pair sampling")
        idx = max(0, min(idx, n_frames - 1))

        use_pair_distance_mode = len(self.pair_distances) > 0

        def _clip_delta_mag(v: int) -> int:
            mag = int(abs(v))
            lo = max(1, int(self.frame_spacing_min))
            if use_pair_distance_mode:
                return max(lo, mag)
            hi = max(lo, int(self.frame_spacing_max))
            return max(lo, min(hi, mag))

        if use_pair_distance_mode:
            pair_distances = [int(v) for v in self.pair_distances]
            probs_np = self._resolve_pair_distance_probs()
            if probs_np is None or int(probs_np.size) != len(pair_distances):
                probs_np = np.ones((len(pair_distances),), dtype=np.float64) / float(max(1, len(pair_distances)))

            def _sample_delta_from_ranges() -> int:
                b = int(np.random.choice(len(pair_distances), p=probs_np))
                mag = _clip_delta_mag(int(pair_distances[b]))
                if self.signed_deltas:
                    sign = -1 if random.random() < 0.5 else 1
                else:
                    sign = 1
                return int(sign * mag)

        else:
            ranges: List[Tuple[int, int]] = []
            if self.pair_sampler:
                for key_new, key_old in (
                    ("local_range", "short_range"),
                    ("mid_range", "medium_range"),
                    ("long_range", "hard_range"),
                ):
                    r = self.pair_sampler.get(key_new, self.pair_sampler.get(key_old))
                    if isinstance(r, (list, tuple)) and len(r) == 2:
                        lo, hi = int(r[0]), int(r[1])
                        if hi >= lo >= 1:
                            ranges.append((lo, hi))
            if not ranges:
                ranges = [(int(self.frame_spacing_min), int(self.frame_spacing_max))]
            use_loop_bucket = bool(self.pair_sampler.get("loop_bucket", True))
            has_loop_bucket = bool(use_loop_bucket and self.loop_min_gap > 0 and self.loop_pose_dist_m > 0.0 and self.loop_yaw_deg > 0.0)

            default_probs = [1.0] * len(ranges)
            if len(ranges) == 3:
                default_probs = [0.70, 0.20, 0.10]
            if has_loop_bucket and len(ranges) == 3:
                default_probs = [0.60, 0.20, 0.15, 0.05]
            probs = self.pair_sampler.get("probs", default_probs)
            if (len(ranges) == 3) or has_loop_bucket:
                schedule = self.pair_sampler.get("schedule", {})
                if isinstance(schedule, dict):
                    p0 = schedule.get("start")
                    p1 = schedule.get("end")
                    expected = len(ranges) + (1 if has_loop_bucket else 0)
                    if isinstance(p0, (list, tuple)) and isinstance(p1, (list, tuple)) and len(p0) == expected and len(p1) == expected:
                        t = 0.0 if self.total_epochs <= 1 else float(self.current_epoch - 1) / float(self.total_epochs - 1)
                        probs = [(1.0 - t) * float(p0[i]) + t * float(p1[i]) for i in range(expected)]
                if self.current_epoch < int(max(1, self.hard_mining_start_epoch)) and len(probs) >= 3:
                    try:
                        probs = [float(x) for x in probs]
                        probs[2] = 0.0
                        if has_loop_bucket and len(probs) >= 4:
                            probs[3] = 0.0
                    except Exception:
                        probs = [0.8, 0.2, 0.0] + ([0.0] if has_loop_bucket else [])
            probs_np = np.asarray(probs, dtype=np.float64).reshape(-1)
            expected_bins = len(ranges) + (1 if has_loop_bucket else 0)
            if probs_np.size != expected_bins or np.any(probs_np < 0):
                probs_np = np.ones((expected_bins,), dtype=np.float64)
            s = float(probs_np.sum())
            probs_np = probs_np / s if s > 0 else np.ones((expected_bins,), dtype=np.float64) / float(max(1, expected_bins))

            def _sample_delta_from_ranges() -> int:
                b = int(np.random.choice(expected_bins, p=probs_np))
                if has_loop_bucket and b == len(ranges):
                    loop_deltas = self._loop_candidates_for_idx(idx)
                    if len(loop_deltas) > 0:
                        return int(random.choice(loop_deltas))
                    b = max(0, len(ranges) - 1)
                lo, hi = ranges[b]
                mag = _clip_delta_mag(random.randint(max(1, lo), max(1, hi)))
                if self.signed_deltas:
                    sign = -1 if random.random() < 0.5 else 1
                else:
                    sign = 1
                return int(sign * mag)

        min_quality = max(0.0, min(1.0, self._resolve_min_pair_valid_ratio()))
        cached_allowed = self._pair_candidates.get(idx, [])
        cached_allowed_set = set(int(v) for v in cached_allowed)

        def _quality_ok(i1: int, i2: int) -> bool:
            if min_quality <= 0.0:
                return True
            delta_local = int(i2 - i1)
            if cached_allowed and min_quality <= (self._pair_cache_quality_floor + 1e-6):
                if int(delta_local) in cached_allowed_set:
                    return True
            q = self._estimate_pair_valid_ratio(i1, i2)
            return bool(q >= min_quality)

        delta = None
        idx2 = None
        if self.is_train:
            if cached_allowed and min_quality <= (self._pair_cache_quality_floor + 1e-6):
                for _ in range(16):
                    d_try = _sample_delta_from_ranges()
                    if int(d_try) in cached_allowed_set:
                        j = idx + int(d_try)
                        if 0 <= j < n_frames:
                            delta = int(d_try)
                            idx2 = int(j)
                            break
            if delta is None:
                for _ in range(32):
                    d_try = _sample_delta_from_ranges()
                    j = idx + int(d_try)
                    if j < 0 or j >= n_frames:
                        continue
                    if not _quality_ok(idx, int(j)):
                        continue
                    delta = int(d_try)
                    idx2 = int(j)
                    break
            if delta is None:
                fallback = cached_allowed if cached_allowed else self._pair_cache_deltas()
                for d_try in sorted(fallback, key=lambda v: abs(int(v))):
                    j = idx + int(d_try)
                    if j < 0 or j >= n_frames:
                        continue
                    delta = int(d_try)
                    idx2 = int(j)
                    break
            if delta is None:
                mag = _clip_delta_mag(self.frame_spacing_min)
                delta = int(mag if (idx + mag) < n_frames else -mag)
                idx2 = max(0, min(n_frames - 1, idx + delta))
        else:
            if self.frame_spacing_max > self.frame_spacing_min:
                width = self.frame_spacing_max - self.frame_spacing_min + 1
                mag = self.frame_spacing_min + (int(idx) % width)
            else:
                mag = self.frame_spacing_min
            delta = int(_clip_delta_mag(mag))
            if self.signed_deltas and (idx % 2 == 1):
                delta = -delta
            idx2 = idx + delta
            if idx2 < 0 or idx2 >= n_frames:
                delta = -delta
                idx2 = idx + delta
            idx2 = max(0, min(n_frames - 1, int(idx2)))
            delta = int(idx2 - idx)
            if delta == 0:
                delta = 1 if idx + 1 < n_frames else -1
                idx2 = idx + delta

        f1 = self.frames[idx]
        f2 = self.frames[int(idx2)]

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

        out: Dict[str, Any] = {
            "rgb1": rgb1,
            "rgb2": rgb2,
            "depth1": depth1,
            "depth2": depth2,
            "valid_depth1": valid1,
            "valid_depth2": valid2,
            "K": K,
            "relative_pose": T21,
            "sequence": self.sequence,
            "frame_idx1": int(idx),
            "frame_idx2": int(idx2),
            "frame_delta": int(delta),
        }
        if self.teacher_enabled:
            out["teacher_heatmap1"] = self._teacher_heatmap_for_frame(f1["rgb"], H=rgb1.shape[-2], W=rgb1.shape[-1])
            out["teacher_heatmap2"] = self._teacher_heatmap_for_frame(f2["rgb"], H=rgb2.shape[-2], W=rgb2.shape[-1])
        return out
