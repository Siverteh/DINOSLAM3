"""
DINOSLAM3 feature wrapper for pySLAM.

This module is used by a SUPERPOINT shim (feature_superpoint.py) so that
pySLAM can be forced to instantiate "SuperPointFeature2D" while actually
running your LocalFeatureNet.

Your model output names (confirmed):
  FeatureOutputs.{desc, heatmap, offset, reliability}
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from pyslam.local_features.feature_base import BaseFeature2D
from pyslam.utilities.logging import Printer


def _ensure_dinoslam3_on_path() -> None:
    root = os.environ.get("DINOSLAM3_ROOT", "/workspace")
    src = Path(root) / "src"
    if src.exists():
        p = str(src)
        if p not in sys.path:
            sys.path.insert(0, p)


def _np_image_to_torch_rgb01(img: np.ndarray, device: torch.device) -> torch.Tensor:
    """
    Accepts:
      - HxWx3 (BGR or RGB, uint8/float)
      - HxW (grayscale, uint8/float)
    Returns:
      - (1,3,H,W) float32 in [0,1]
    """
    if img is None:
        raise ValueError("Image is None")

    if img.ndim == 2:
        img = np.repeat(img[..., None], 3, axis=2)

    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"Expected image HxWx3 or HxW. Got {img.shape}")

    if img.dtype == np.uint8:
        x = torch.from_numpy(img).to(device=device, dtype=torch.float32) / 255.0
    else:
        x = torch.from_numpy(img).to(device=device, dtype=torch.float32)
        if x.max() > 1.5:
            x = x / 255.0

    # pySLAM images are BGR; convert to RGB.
    x = x[..., [2, 1, 0]]
    x = x.permute(2, 0, 1).unsqueeze(0).contiguous()
    return x


def _read_env_positive_int(name: str) -> Optional[int]:
    raw = os.environ.get(name)
    if raw is None:
        return None
    raw = raw.strip()
    if raw == "":
        return None
    try:
        v = int(raw)
    except Exception:
        Printer.orange(f"[DINOSLAM3] ignoring invalid {name}='{raw}' (expected int > 0)")
        return None
    if v <= 0:
        Printer.orange(f"[DINOSLAM3] ignoring invalid {name}='{raw}' (expected int > 0)")
        return None
    return v


def _build_model_and_load_ckpt(
    checkpoint_path: str,
    device: str,
) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    """
    ckpt format:
      {epoch, model, optimizer, config}
    """
    _ensure_dinoslam3_on_path()

    from dino_slam3.models.network import LocalFeatureNet

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model_cfg = (ckpt.get("config", {}) or {}).get("model", {}) or {}

    dinov3_cfg = model_cfg["dinov3"]
    heads_cfg = model_cfg["heads"]
    fine_cfg = model_cfg["fine_cnn"]

    model = LocalFeatureNet(
        dinov3_name=dinov3_cfg["name_or_path"],
        patch_size=int(model_cfg.get("patch_size", 16)),
        descriptor_dim=int(heads_cfg["descriptor_dim"]),
        fine_channels=int(fine_cfg["channels"]),
        fine_blocks=int(fine_cfg["num_blocks"]),
        freeze_backbone=bool(model_cfg.get("freeze_backbone", True)),
        use_offset=bool((heads_cfg.get("offset", {}) or {}).get("enabled", True)),
        use_reliability=bool((heads_cfg.get("reliability", {}) or {}).get("enabled", True)),
        dinov3_dtype=str(dinov3_cfg.get("dtype", "bf16")),
    ).eval()

    sd = ckpt.get("model", ckpt)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing_backbone = [k for k in missing if k.startswith("backbone.")]
    unexpected_backbone = [k for k in unexpected if k.startswith("backbone.")]
    allow_partial_ckpt = os.environ.get("DINOSLAM3_ALLOW_PARTIAL_CKPT", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    if (missing_backbone or unexpected_backbone) and not allow_partial_ckpt:
        raise RuntimeError(
            "[DINOSLAM3] Backbone checkpoint mismatch detected. "
            f"missing_backbone={len(missing_backbone)}, unexpected_backbone={len(unexpected_backbone)}. "
            "This usually means the runtime backbone differs from the training backbone "
            "(e.g., fallback from gated model to timm). "
            "Fix HF access / backbone selection, or set DINOSLAM3_ALLOW_PARTIAL_CKPT=1 to override."
        )

    if unexpected:
        Printer.yellow(f"[DINOSLAM3] unexpected keys (first 20): {unexpected[:20]}")
    if missing:
        Printer.yellow(f"[DINOSLAM3] missing keys (first 20): {missing[:20]}")

    dev = torch.device(device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(dev)
    return model, model_cfg


class DinoSlam3Feature2D(BaseFeature2D):
    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda",
        num_features: int = 1000,
        normalize_descriptors: bool = True,
        nms_radius: int = 6,
    ):
        super().__init__(num_features=int(num_features), device=device)

        if not checkpoint_path or not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"DINOSLAM3 checkpoint not found: {checkpoint_path}")

        self.checkpoint_path = checkpoint_path
        self.normalize_descriptors = bool(normalize_descriptors)
        self.nms_radius = int(nms_radius)
        self.dev = torch.device(device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))

        Printer.green(f"  [DINOSLAM3] CKPT: {checkpoint_path}")
        Printer.green(f"  [DINOSLAM3] device={self.dev} num_features={int(num_features)}")

        self.model, model_cfg = _build_model_and_load_ckpt(checkpoint_path, device=str(self.dev))

        _ensure_dinoslam3_on_path()
        from dino_slam3.slam.keypoints_torch import extract_keypoints_torch

        self._extract_keypoints_torch = extract_keypoints_torch

        det_cfg = (model_cfg.get("heads", {}) or {}).get("detector", {}) or {}
        self.stride = int(model_cfg.get("stride", 4))

        self.kp_nms_radius = int(
            _read_env_positive_int("DINOSLAM3_KP_NMS_RADIUS")
            or det_cfg.get("nms_radius", self.nms_radius)
        )
        self.kp_tile_size = int(
            _read_env_positive_int("DINOSLAM3_KP_TILE_SIZE")
            or det_cfg.get("tile_size", 8)
        )
        self.kp_per_tile = int(
            _read_env_positive_int("DINOSLAM3_KP_PER_TILE")
            or det_cfg.get("k_per_tile", 2)
        )
        self.kp_max_override = _read_env_positive_int("DINOSLAM3_KP_MAX")

        Printer.green(
            "  [DINOSLAM3] kp cfg: "
            f"stride={self.stride} nms={self.kp_nms_radius} "
            f"tile={self.kp_tile_size} k/tile={self.kp_per_tile} "
            f"max_override={self.kp_max_override if self.kp_max_override is not None else 'none'}"
        )

    def setMaxFeatures(self, num_features: int):
        self.num_features = int(num_features)

    def _optional_mask_to_tensor(
        self,
        mask: Optional[np.ndarray],
        height: int,
        width: int,
    ) -> Optional[torch.Tensor]:
        if mask is None:
            return None
        if torch.is_tensor(mask):
            m = mask.to(device=self.dev, dtype=torch.float32)
        else:
            m = torch.from_numpy(np.asarray(mask)).to(device=self.dev, dtype=torch.float32)

        if m.ndim == 2:
            m = m.unsqueeze(0).unsqueeze(0)
        elif m.ndim == 3:
            if m.shape[0] in (1, 3):
                m = m[:1].unsqueeze(0)
            else:
                m = m[..., :1].permute(2, 0, 1).unsqueeze(0)
        elif m.ndim == 4:
            pass
        else:
            return None

        if m.shape[-2:] != (height, width):
            m = F.interpolate(m, size=(height, width), mode="nearest")

        return (m > 0).float()

    @torch.no_grad()
    def detectAndCompute(self, frame, mask=None):
        """
        Returns (kps, desc) where:
          - kps: list of cv2.KeyPoint
          - desc: (N,D) float32
        """
        import cv2

        x = _np_image_to_torch_rgb01(frame, device=self.dev)
        out = self.model(x)

        desc_map = out.desc
        heatmap = out.heatmap
        offset = out.offset
        reliability = out.reliability

        if desc_map is None or heatmap is None:
            raise RuntimeError("DINOSLAM3 model output missing desc or heatmap.")

        max_k = int(self.num_features)
        if self.kp_max_override is not None:
            max_k = min(max_k, self.kp_max_override)
        if max_k <= 0:
            return [], None

        valid_mask = self._optional_mask_to_tensor(mask, x.shape[-2], x.shape[-1])

        k = self._extract_keypoints_torch(
            heat_logits=heatmap,
            desc_map=desc_map,
            offset_map=offset,
            reliability_logits=reliability,
            stride=int(self.stride),
            nms_radius=int(self.kp_nms_radius),
            tile_size=int(self.kp_tile_size),
            k_per_tile=int(self.kp_per_tile),
            max_keypoints=int(max_k),
            valid_mask_img=valid_mask,
        )

        xy_img = k.xy_img[0]
        des = k.desc[0]

        if xy_img.numel() == 0 or des.numel() == 0:
            return [], None

        if self.normalize_descriptors:
            des = F.normalize(des, p=2, dim=1, eps=1e-6)

        des_np = des.detach().cpu().numpy().astype(np.float32, copy=False)
        xy_np = xy_img.detach().cpu().numpy()

        kps = [cv2.KeyPoint(float(pt[0]), float(pt[1]), 1.0) for pt in xy_np]
        return kps, des_np

    def detect(self, frame, mask=None):
        kps, _ = self.detectAndCompute(frame, mask=mask)
        return kps

    def compute(self, frame, kps=None, mask=None):
        return self.detectAndCompute(frame, mask=mask)
