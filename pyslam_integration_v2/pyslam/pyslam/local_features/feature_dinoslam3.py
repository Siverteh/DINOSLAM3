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
from typing import Optional, Tuple, List

import numpy as np
import torch
import torch.nn.functional as F

from pyslam.utilities.logging import Printer
from pyslam.local_features.feature_base import BaseFeature2D


# -------------------------
# Path helpers
# -------------------------
def _ensure_dinoslam3_on_path() -> None:
    root = os.environ.get("DINOSLAM3_ROOT", "/workspace")
    src = Path(root) / "src"
    if src.exists():
        p = str(src)
        if p not in sys.path:
            sys.path.insert(0, p)


# -------------------------
# Image conversion
# -------------------------
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
        # grayscale -> 3ch
        img = np.repeat(img[..., None], 3, axis=2)

    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"Expected image HxWx3 or HxW. Got {img.shape}")

    if img.dtype == np.uint8:
        x = torch.from_numpy(img).to(device=device, dtype=torch.float32) / 255.0
    else:
        x = torch.from_numpy(img).to(device=device, dtype=torch.float32)
        # if user already normalized, keep; else clamp-ish
        if x.max() > 1.5:
            x = x / 255.0

    # pySLAM images are BGR; our network doesn't really care, but keep consistent:
    # Convert BGR->RGB
    x = x[..., [2, 1, 0]]

    x = x.permute(2, 0, 1).unsqueeze(0).contiguous()  # (1,3,H,W)
    return x


# -------------------------
# Model loading
# -------------------------
def _build_model_and_load_ckpt(checkpoint_path: str, device: str) -> torch.nn.Module:
    """
    Your ckpt is a dict with keys: {epoch, model, optimizer, config}
    We instantiate LocalFeatureNet from dino_slam3.models.network and load ckpt["model"].
    """
    _ensure_dinoslam3_on_path()

    import torch
    from dino_slam3.models.network import LocalFeatureNet

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    cfg = ckpt.get("config", {}).get("model", {})

    # Pull model kwargs from ckpt config
    dinov3_name = cfg["dinov3"]["name_or_path"]
    patch_size = int(cfg.get("patch_size", 16))
    descriptor_dim = int(cfg["heads"]["descriptor_dim"])
    fine_channels = int(cfg["fine_cnn"]["channels"])
    fine_blocks = int(cfg["fine_cnn"]["num_blocks"])
    freeze_backbone = bool(cfg.get("freeze_backbone", True))
    use_offset = bool(cfg["heads"]["offset"]["enabled"])
    use_reliability = bool(cfg["heads"]["reliability"]["enabled"])
    dinov3_dtype = str(cfg["dinov3"].get("dtype", "bf16"))

    model = LocalFeatureNet(
        dinov3_name=dinov3_name,
        patch_size=patch_size,
        descriptor_dim=descriptor_dim,
        fine_channels=fine_channels,
        fine_blocks=fine_blocks,
        freeze_backbone=freeze_backbone,
        use_offset=use_offset,
        use_reliability=use_reliability,
        dinov3_dtype=dinov3_dtype,
    ).eval()

    # load weights
    sd = ckpt.get("model", ckpt)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if len(unexpected) > 0:
        Printer.yellow(f"[DINOSLAM3] unexpected keys (first 20): {unexpected[:20]}")
    if len(missing) > 0:
        Printer.yellow(f"[DINOSLAM3] missing keys (first 20): {missing[:20]}")

    dev = torch.device(device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(dev)
    return model


# -------------------------
# Keypoint extraction
# -------------------------
def _simple_nms(scores: torch.Tensor, radius: int) -> torch.Tensor:
    # scores: (1,1,H,W)
    if radius <= 0:
        return scores
    maxpool = F.max_pool2d(scores, kernel_size=2 * radius + 1, stride=1, padding=radius)
    keep = (scores == maxpool).to(scores.dtype)
    return scores * keep


def _topk_keypoints_from_heatmap(
    heatmap: torch.Tensor,
    max_k: int,
    nms_radius: int = 6,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    heatmap: (1,1,H,W) logits or scores
    Returns:
      xy: (N,2) in heatmap coords (float)
      score: (N,)
    """
    hm = heatmap
    if hm.dtype != torch.float32:
        hm = hm.float()

    # Treat as logits; apply sigmoid so values in (0,1)
    hm = torch.sigmoid(hm)

    hm = _simple_nms(hm, nms_radius)

    H, W = hm.shape[-2], hm.shape[-1]
    flat = hm.view(-1)
    k = int(min(max_k, flat.numel()))
    vals, idx = torch.topk(flat, k=k, largest=True, sorted=True)
    ys = (idx // W).float()
    xs = (idx % W).float()
    xy = torch.stack([xs, ys], dim=1)  # (k,2)

    return xy.detach().cpu().numpy(), vals.detach().cpu().numpy()


def _sample_desc(desc: torch.Tensor, xy: torch.Tensor) -> torch.Tensor:
    """
    desc: (1,C,H,W)
    xy: (N,2) in desc coords (x,y)
    returns: (N,C)
    """
    # grid_sample expects normalized coords in [-1,1] as (x,y)
    H, W = desc.shape[-2], desc.shape[-1]
    x = xy[:, 0]
    y = xy[:, 1]
    gx = (x / (W - 1)) * 2 - 1
    gy = (y / (H - 1)) * 2 - 1
    grid = torch.stack([gx, gy], dim=1).view(1, -1, 1, 2)  # (1,N,1,2)
    samp = F.grid_sample(desc, grid, mode="bilinear", align_corners=True)  # (1,C,N,1)
    samp = samp.squeeze(0).squeeze(-1).transpose(0, 1).contiguous()  # (N,C)
    return samp


# -------------------------
# Feature class
# -------------------------
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

        self.model = _build_model_and_load_ckpt(checkpoint_path, device=str(self.dev))

    def setMaxFeatures(self, num_features: int):
        self.num_features = int(num_features)

    @torch.no_grad()
    def detectAndCompute(self, frame, mask=None):
        """
        Returns (kps, desc) where:
          - kps: list of cv2.KeyPoint
          - desc: (N,D) float32
        """
        import cv2

        x = _np_image_to_torch_rgb01(frame, device=self.dev)  # (1,3,H,W)

        out = self.model(x)
        # confirmed names
        desc_map = out.desc          # (1,C,H/4,W/4)
        heatmap = out.heatmap        # (1,1,H/4,W/4)
        offset = out.offset          # (1,2,H/4,W/4)
        # reliability exists but not required for SLAM right now
        # reliability = out.reliability

        if desc_map is None or heatmap is None:
            raise RuntimeError("DINOSLAM3 model output missing desc or heatmap.")

        # pick keypoints in stride-4 space
        xy_hm_np, score_np = _topk_keypoints_from_heatmap(
            heatmap, max_k=int(self.num_features), nms_radius=int(self.nms_radius)
        )

        if xy_hm_np.shape[0] == 0:
            return [], None

        xy = torch.from_numpy(xy_hm_np).to(device=self.dev, dtype=torch.float32)  # (N,2) in hm coords

        # apply learned offset (subpixel in hm coords)
        # sample offset at xy
        off = _sample_desc(offset, xy)  # (N,2)
        xy_ref = xy + off  # refined in hm coords

        # sample descriptors at refined coords
        des = _sample_desc(desc_map, xy_ref)  # (N,C)

        if self.normalize_descriptors:
            des = F.normalize(des, p=2, dim=1)

        des_np = des.detach().cpu().numpy().astype(np.float32)

        # map hm coords (stride=4) back to image pixels
        stride = 4.0
        xs = (xy_ref[:, 0] * stride).detach().cpu().numpy()
        ys = (xy_ref[:, 1] * stride).detach().cpu().numpy()

        # build cv2.KeyPoint list
        kps = [cv2.KeyPoint(float(xp), float(yp), 1.0) for xp, yp in zip(xs, ys)]

        return kps, des_np

    def detect(self, frame, mask=None):
        kps, _ = self.detectAndCompute(frame, mask=mask)
        return kps

    def compute(self, frame, kps=None, mask=None):
        return self.detectAndCompute(frame, mask=mask)
