# dino_slam3/inference/extractor_api.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
import torch

@dataclass
class ExtractorOutput:
    kpts_xy: np.ndarray      # (N, 2) float32, pixel coords in ORIGINAL image (u,v)
    desc: np.ndarray         # (N, D) float32
    scores: np.ndarray       # (N,) float32

class DinoSlam3Extractor:
    """
    Single source of truth for preprocessing + inference.
    Both eval_inliers and pySLAM must call this.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: str = "cuda",
        patch_size: int = 16,
        max_kpts: int = 1000,
        score_thresh: float = 0.0,
        l2_normalize_desc: bool = True,
    ) -> None:
        self.model = model.to(device).eval()
        self.device = device
        self.patch_size = patch_size
        self.max_kpts = max_kpts
        self.score_thresh = score_thresh
        self.l2_normalize_desc = l2_normalize_desc

    @torch.inference_mode()
    def extract_from_bgr(self, img_bgr: np.ndarray) -> ExtractorOutput:
        """
        img_bgr: OpenCV BGR uint8 (H,W,3) from pySLAM.
        Returns kpts in ORIGINAL pixel coordinates (no resize mismatch).
        """
        assert img_bgr.ndim == 3 and img_bgr.shape[2] == 3

        H0, W0 = img_bgr.shape[:2]

        # --- MUST MATCH TRAIN/EVAL PREPROCESSING ---
        # 1) BGR -> RGB
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # 2) Convert to float in [0,1]
        img = img_rgb.astype(np.float32) / 255.0

        # 3) Pad to multiple of patch size (16) WITHOUT changing content
        #    (Do NOT resize unless train/eval resized; if you did resize there, do the exact same resize here.)
        pad_h = (self.patch_size - (H0 % self.patch_size)) % self.patch_size
        pad_w = (self.patch_size - (W0 % self.patch_size)) % self.patch_size
        if pad_h or pad_w:
            img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant", constant_values=0)

        Hp, Wp = img.shape[:2]

        # 4) Normalize (PUT YOUR TRAIN/EVAL MEAN/STD HERE EXACTLY)
        # If your training used ImageNet mean/std, keep it.
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std

        # 5) To tensor BCHW
        t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(self.device)

        # 6) Model forward: MUST be the same call you use in eval_inliers
        # Expected to return: kpts in padded image coords, descriptors, scores
        # Replace the next line with your model’s actual inference call.
        pred = self.model.infer(t)  # <-- you will wire this to your existing inference

        kpts = pred["kpts_xy"].detach().float().cpu().numpy()    # (N,2) in padded coords
        desc = pred["desc"].detach().float().cpu().numpy()       # (N,D)
        scores = pred["scores"].detach().float().cpu().numpy()   # (N,)

        # 7) Remove padding region keypoints
        keep = (kpts[:, 0] >= 0) & (kpts[:, 0] < W0) & (kpts[:, 1] >= 0) & (kpts[:, 1] < H0)
        if self.score_thresh > 0:
            keep &= (scores >= self.score_thresh)
        kpts, desc, scores = kpts[keep], desc[keep], scores[keep]

        # 8) Top-K by score (stable)
        if kpts.shape[0] > self.max_kpts:
            idx = np.argpartition(-scores, self.max_kpts - 1)[: self.max_kpts]
            idx = idx[np.argsort(-scores[idx])]
            kpts, desc, scores = kpts[idx], desc[idx], scores[idx]

        # 9) Descriptor normalization MUST match matcher metric used in pySLAM
        # If pySLAM uses L2 distance, unit-normalize so L2 ~ cosine.
        if self.l2_normalize_desc:
            n = np.linalg.norm(desc, axis=1, keepdims=True) + 1e-12
            desc = desc / n

        return ExtractorOutput(
            kpts_xy=kpts.astype(np.float32),
            desc=desc.astype(np.float32),
            scores=scores.astype(np.float32),
        )
