from __future__ import annotations
from typing import Optional, List, Tuple

import numpy as np
import torch
import cv2

from dino_slam3.models.network import LocalFeatureNet
from dino_slam3.slam.keypoints import extract_keypoints_and_descriptors, to_cv2


class DinoSLAM3FeatureExtractor:
    """
    OpenCV-like feature extractor with correct coordinate handling.
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
    ):
        self.device = torch.device(device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = model.to(self.device).eval()
        self.stride = int(stride)
        self.nms_radius = int(nms_radius)
        self.max_keypoints = int(max_keypoints)
        self.tile_size = int(tile_size)
        self.k_per_tile = int(k_per_tile)
        self.pad_to = int(pad_to)

        if checkpoint_path:
            ckpt = torch.load(checkpoint_path, map_location="cpu")
            self.model.load_state_dict(ckpt["model"], strict=False)

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
        # Accept grayscale or BGR uint8
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        if image.shape[2] != 3:
            raise ValueError("Expected HxWx3 image")

        H0, W0 = image.shape[:2]
        img_rgb = image[..., ::-1].copy()  # BGR->RGB

        img_rgb, pad_bottom, pad_right = self._pad_to_multiple(img_rgb)
        Hp, Wp = img_rgb.shape[:2]

        img_t = torch.from_numpy(img_rgb).float() / 255.0
        img_t = img_t.permute(2, 0, 1).unsqueeze(0)  # 1,3,H,W

        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        img_t = (img_t - mean) / std
        img_t = img_t.to(self.device)

        out = self.model(img_t)
        kp_batch = extract_keypoints_and_descriptors(
            out.heatmap, out.desc, out.offset, out.reliability,
            stride=self.stride,
            nms_radius=self.nms_radius,
            max_keypoints=self.max_keypoints,
            tile_size=self.tile_size,
            k_per_tile=self.k_per_tile,
        )[0]

        # Map keypoints from padded coords back to original
        kpts = kp_batch.kpts.copy()
        kpts[:, 0] = np.clip(kpts[:, 0], 0, Wp - 1)
        kpts[:, 1] = np.clip(kpts[:, 1], 0, Hp - 1)
        # Since we padded only bottom/right, original top-left aligns.
        # Just drop points that fall in padded area.
        keep = (kpts[:, 0] < W0) & (kpts[:, 1] < H0)
        kpts = kpts[keep]
        desc = kp_batch.desc[keep]
        scores = kp_batch.scores[keep]

        kp_batch.kpts = kpts
        kp_batch.desc = desc
        kp_batch.scores = scores

        kps, desc_cv = to_cv2(kp_batch)
        return kps, desc_cv
