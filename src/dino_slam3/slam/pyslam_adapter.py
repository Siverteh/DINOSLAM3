from __future__ import annotations
from typing import Optional, Tuple, List
import numpy as np
import torch
import cv2

from dino_slam3.models.network import LocalFeatureNet
from dino_slam3.slam.keypoints_torch import extract_keypoints_torch

class DinoSLAM3FeatureExtractor:
    """
    OpenCV-like feature extractor: detectAndCompute(image, mask=None) -> (keypoints, descriptors)
    Returns float descriptors (L2) suitable for BFMatcher(NORM_L2).
    """
    def __init__(
        self,
        model: LocalFeatureNet,
        checkpoint_path: Optional[str] = None,
        device: str = "cuda",
        stride: int = 4,
        nms_radius: int = 4,
        max_keypoints: int = 1024,
        tile_size: int = 16,
        k_per_tile: int = 8,
        pad_to: int = 16,
        use_reliability_in_score: bool = False,
    ):
        self.device = torch.device(device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = model.to(self.device).eval()
        self.stride = int(stride)
        self.nms_radius = int(nms_radius)
        self.max_keypoints = int(max_keypoints)
        self.tile_size = int(tile_size)
        self.k_per_tile = int(k_per_tile)
        self.pad_to = int(pad_to)
        self.use_reliability_in_score = bool(use_reliability_in_score)
        self.adaptive_tiling = bool(getattr(model, "adaptive_tiling", False))
        self.adaptive_k_min = int(getattr(model, "adaptive_k_min", 1))
        self.adaptive_k_max = getattr(model, "adaptive_k_max", None)

        if checkpoint_path:
            ckpt = torch.load(checkpoint_path, map_location="cpu")
            self.model.load_state_dict(ckpt["model"], strict=False)

        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    def _pad_to_multiple(self, img: np.ndarray) -> Tuple[np.ndarray, int, int]:
        H, W = img.shape[:2]
        pad_bottom = (self.pad_to - (H % self.pad_to)) % self.pad_to
        pad_right = (self.pad_to - (W % self.pad_to)) % self.pad_to
        if pad_bottom == 0 and pad_right == 0:
            return img, 0, 0
        img_pad = cv2.copyMakeBorder(img, 0, pad_bottom, 0, pad_right, borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))
        return img_pad, pad_bottom, pad_right

    @torch.no_grad()
    def detectAndCompute(self, image: np.ndarray, mask=None):
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        if image.shape[2] != 3:
            raise ValueError("Expected HxWx3 image")

        H0, W0 = image.shape[:2]
        img_rgb = image[..., ::-1].copy()  # BGR->RGB
        img_rgb, pad_bottom, pad_right = self._pad_to_multiple(img_rgb)

        img_t = torch.from_numpy(img_rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0

        # DINO normalization (ImageNet mean/std)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        img_t = (img_t - mean) / std
        img_t = img_t.to(self.device)

        out = self.model(img_t)
        k = extract_keypoints_torch(
            out.heatmap, out.desc, out.offset, out.reliability,
            stride=self.stride,
            nms_radius=self.nms_radius,
            tile_size=self.tile_size,
            k_per_tile=self.k_per_tile,
            max_keypoints=self.max_keypoints,
            valid_mask_img=None,
            use_reliability_in_score=self.use_reliability_in_score,
            adaptive_tiling=self.adaptive_tiling,
            adaptive_k_min=self.adaptive_k_min,
            adaptive_k_max=self.adaptive_k_max,
        )
        xy = k.xy_img[0].detach().cpu().numpy().astype(np.float32)
        desc = k.desc[0].detach().cpu().numpy().astype(np.float32)
        score = k.score[0].detach().cpu().numpy().astype(np.float32)

        # drop padded area
        keep = (xy[:, 0] < W0) & (xy[:, 1] < H0) & (score > 0)
        xy, desc, score = xy[keep], desc[keep], score[keep]

        kps = [cv2.KeyPoint(float(x), float(y), 1.0, _response=float(s)) for (x, y), s in zip(xy, score)]
        return kps, desc
