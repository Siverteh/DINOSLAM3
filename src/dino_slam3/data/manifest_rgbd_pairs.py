from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


class ManifestRGBDPairDataset(Dataset):
    """
    RGB-D pair dataset driven by JSONL manifests.

    Each JSON line must contain:
      - img1, img2: image paths
      - depth1, depth2: depth paths
      - K: 3x3 matrix or [fx,fy,cx,cy]
      - relative_pose: 4x4 matrix (camera1 -> camera2)

    Optional:
      - depth_scale: per-sample depth scale divisor for uint16 depth
      - sequence: sequence tag for logging/analysis
    """

    def __init__(
        self,
        manifests: List[str] | List[Path],
        dataset_root: str | Path | None = None,
        pad_to: int = 16,
        depth_scale: float = 1000.0,
        is_train: bool = True,
        augmentation: Optional[Dict[str, Any]] = None,
        pair_quality: Optional[Dict[str, Any]] = None,
        pair_mining: Optional[Dict[str, Any]] = None,
        split_name: str = "train",
    ):
        self.manifests = [Path(p).expanduser().resolve() for p in manifests]
        self.dataset_root = Path(dataset_root).expanduser().resolve() if dataset_root is not None else None
        self.pad_to = int(pad_to)
        self.depth_scale = float(depth_scale)
        self.is_train = bool(is_train)
        self.aug = augmentation if (augmentation and self.is_train) else None
        self.pair_quality = pair_quality if isinstance(pair_quality, dict) else {}
        self.pair_mining = pair_mining if isinstance(pair_mining, dict) else {}
        self.split_name = str(split_name)

        samples: List[Dict[str, Any]] = []
        for mp in self.manifests:
            if not mp.exists():
                raise FileNotFoundError(f"Missing manifest: {mp}")
            base_dir = mp.parent
            with mp.open("r", encoding="utf-8") as f:
                for ln, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    sample = self._parse_row(row, base_dir)
                    if sample is not None:
                        samples.append(sample)

        if self.is_train:
            samples = self._inject_hard_negative_samples(samples)

        if not samples:
            raise RuntimeError("ManifestRGBDPairDataset: no valid samples found.")
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def _resolve_path(self, p: str, base_dir: Path) -> str:
        pp = Path(p)
        if pp.is_absolute():
            return pp.as_posix()
        if self.dataset_root is not None:
            cand = (self.dataset_root / pp).resolve()
            if cand.exists():
                return cand.as_posix()
        return (base_dir / pp).resolve().as_posix()

    @staticmethod
    def _to_mat3(v: Any) -> Optional[np.ndarray]:
        arr = np.asarray(v, dtype=np.float32)
        if arr.shape == (3, 3):
            return arr
        if arr.size == 4:
            fx, fy, cx, cy = arr.reshape(-1).tolist()
            return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
        return None

    @staticmethod
    def _to_mat4(v: Any) -> Optional[np.ndarray]:
        arr = np.asarray(v, dtype=np.float32)
        if arr.shape == (4, 4):
            return arr
        return None

    @staticmethod
    def _baseline_from_pose(T21: np.ndarray) -> float:
        try:
            return float(np.linalg.norm(T21[:3, 3]))
        except Exception:
            return 0.0

    def _parse_row(self, row: Dict[str, Any], base_dir: Path) -> Optional[Dict[str, Any]]:
        req = ["img1", "img2", "depth1", "depth2", "K", "relative_pose"]
        if any(k not in row for k in req):
            return None

        K = self._to_mat3(row["K"])
        T21 = self._to_mat4(row["relative_pose"])
        if K is None or T21 is None:
            return None

        if not self._passes_quality_filters(row, T21):
            return None

        baseline_m = float(row.get("baseline_m", self._baseline_from_pose(T21)))
        pair_delta = int(row.get("pair_delta", 0))
        if "pair_type" in row:
            pair_type = str(row.get("pair_type", "unknown"))
        else:
            ad = abs(pair_delta)
            if ad <= 2:
                pair_type = "short"
            elif ad <= 4:
                pair_type = "medium"
            else:
                pair_type = "hard"

        return {
            "rgb1": self._resolve_path(str(row["img1"]), base_dir),
            "rgb2": self._resolve_path(str(row["img2"]), base_dir),
            "depth1": self._resolve_path(str(row["depth1"]), base_dir),
            "depth2": self._resolve_path(str(row["depth2"]), base_dir),
            "K": K,
            "relative_pose": T21,
            "depth_scale": float(row.get("depth_scale", self.depth_scale)),
            "sequence": str(row.get("sequence", "manifest_pair")),
            "dataset": str(row.get("dataset", "external")),
            "scene_id": str(row.get("scene_id", "unknown")),
            "pair_delta": pair_delta,
            "pair_type": pair_type,
            "baseline_m": baseline_m,
            "overlap_ratio": float(row.get("overlap_ratio", 1.0)),
        }

    def _passes_quality_filters(self, row: Dict[str, Any], T21: Optional[np.ndarray] = None) -> bool:
        cfg = self.pair_quality
        if not cfg:
            return True
        try:
            overlap = float(row.get("overlap_ratio", 1.0))
            if "baseline_m" in row:
                baseline = float(row.get("baseline_m", 0.0))
            elif T21 is not None:
                baseline = self._baseline_from_pose(T21)
            else:
                baseline = 0.0
        except Exception:
            return False
        min_overlap = float(cfg.get("min_overlap_ratio", 0.0))
        min_baseline = float(cfg.get("min_baseline_m", 0.0))
        max_baseline = float(cfg.get("max_baseline_m", 1e9))
        if overlap < min_overlap:
            return False
        if baseline < min_baseline:
            return False
        if baseline > max_baseline:
            return False
        return True

    def _inject_hard_negative_samples(self, base_samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        frac = float(self.pair_mining.get("hard_negative_fraction", 0.0) or 0.0)
        if frac <= 0.0:
            return base_samples

        hard_paths = []
        raw = self.pair_mining.get("hard_negative_manifests", self.pair_mining.get("failed_pairs_jsonl", []))
        if isinstance(raw, (str, Path)):
            hard_paths = [raw]
        elif isinstance(raw, list):
            hard_paths = raw
        hard_paths = [Path(str(p)).expanduser().resolve() for p in hard_paths]
        hard_paths = [p for p in hard_paths if p.exists()]
        if not hard_paths:
            return base_samples

        hard_samples: List[Dict[str, Any]] = []
        for hp in hard_paths:
            with hp.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except Exception:
                        continue
                    sample = self._parse_row(row, hp.parent)
                    if sample is not None:
                        hard_samples.append(sample)

        if not hard_samples:
            return base_samples

        target_add = int(max(1, round(len(base_samples) * frac)))
        out = list(base_samples)
        if len(hard_samples) <= target_add:
            out.extend(hard_samples)
        else:
            out.extend(random.sample(hard_samples, target_add))
        return out

    def _read_rgb(self, p: str) -> torch.Tensor:
        bgr = cv2.imread(p, cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(p)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0

    def _read_depth(self, p: str, depth_scale: float) -> torch.Tensor:
        d = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        if d is None:
            raise FileNotFoundError(p)
        if d.dtype == np.uint16:
            z = torch.from_numpy(d.astype(np.float32)) / float(depth_scale)
        else:
            z = torch.from_numpy(d.astype(np.float32))
        return z.unsqueeze(0)

    def _pad(self, x: torch.Tensor) -> torch.Tensor:
        _, H, W = x.shape
        pb = (self.pad_to - (H % self.pad_to)) % self.pad_to
        pr = (self.pad_to - (W % self.pad_to)) % self.pad_to
        if pb == 0 and pr == 0:
            return x
        return torch.nn.functional.pad(x, (0, pr, 0, pb), value=0.0)

    def _photometric_aug(self, x: torch.Tensor) -> torch.Tensor:
        cfg = (self.aug or {}).get("photometric", {})
        if (not self.is_train) or (not cfg.get("enabled", False)):
            return x
        b = float(cfg.get("brightness", 0.0))
        c = float(cfg.get("contrast", 0.0))
        if b > 0:
            delta = (torch.rand(1).item() * 2 - 1) * b
            x = torch.clamp(x + delta, 0.0, 1.0)
        if c > 0:
            scale = 1.0 + (torch.rand(1).item() * 2 - 1) * c
            mean = x.mean(dim=(1, 2), keepdim=True)
            x = torch.clamp((x - mean) * scale + mean, 0.0, 1.0)
        return x

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        s = self.samples[int(idx)]
        rgb1 = self._pad(self._photometric_aug(self._read_rgb(s["rgb1"])))
        rgb2 = self._pad(self._photometric_aug(self._read_rgb(s["rgb2"])))
        depth1 = self._pad(self._read_depth(s["depth1"], s["depth_scale"]))
        depth2 = self._pad(self._read_depth(s["depth2"], s["depth_scale"]))
        valid1 = (depth1 > 0.0).float()
        valid2 = (depth2 > 0.0).float()
        return {
            "rgb1": rgb1,
            "rgb2": rgb2,
            "depth1": depth1,
            "depth2": depth2,
            "valid_depth1": valid1,
            "valid_depth2": valid2,
            "K": torch.from_numpy(s["K"]).float(),
            "relative_pose": torch.from_numpy(s["relative_pose"]).float(),
            "sequence": s["sequence"],
        }
