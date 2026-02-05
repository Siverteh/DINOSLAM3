from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np
import torch
import cv2

from dino_slam3.models.network import LocalFeatureNet
from dino_slam3.slam.keypoints import extract_keypoints_and_descriptors, to_cv2

class DinoSLAM3FeatureExtractor:
    """OpenCV-like feature extractor wrapper.

    Usage (conceptually):
      extractor = DinoSLAM3FeatureExtractor(ckpt_path, device="cuda")
      kps, desc = extractor.detectAndCompute(gray_or_rgb, None)

    pySLAM can be adapted to call detectAndCompute on this object.
    """

    def __init__(
        self,
        model: LocalFeatureNet,
        checkpoint_path: Optional[str] = None,
        device: str = "cuda",
        input_size: int = 448,
        stride: int = 4,
        nms_radius: int = 4,
        max_keypoints: int = 1024,
    ):
        self.device = torch.device(device if (device != "auto") else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = model.to(self.device).eval()
        self.input_size = int(input_size)
        self.stride = int(stride)
        self.nms_radius = int(nms_radius)
        self.max_keypoints = int(max_keypoints)

        if checkpoint_path:
            ckpt = torch.load(checkpoint_path, map_location="cpu")
            self.model.load_state_dict(ckpt["model"], strict=False)

    @torch.no_grad()
    def detectAndCompute(self, image: np.ndarray, mask=None):
        # Accept BGR or RGB uint8
        if image.ndim == 2:
            image = np.stack([image]*3, axis=-1)
        if image.shape[2] == 3:
            # assume BGR (opencv) -> RGB
            img_rgb = image[..., ::-1].copy()
        else:
            raise ValueError("Expected HxWx3 image")

        img_rgb = cv2.resize(img_rgb, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR)
        img_t = torch.from_numpy(img_rgb).float() / 255.0
        img_t = img_t.permute(2,0,1).unsqueeze(0)  # 1,3,H,W
        # ImageNet normalize
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)
        img_t = (img_t - mean) / std
        img_t = img_t.to(self.device)

        out = self.model(img_t)
        batches = extract_keypoints_and_descriptors(
            out.heatmap, out.desc, out.offset, out.reliability,
            stride=self.stride,
            nms_radius=self.nms_radius,
            max_keypoints=self.max_keypoints,
        )
        kp_batch = batches[0]
        kps, desc = to_cv2(kp_batch)

        return kps, desc
