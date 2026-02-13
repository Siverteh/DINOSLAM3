from __future__ import annotations

import cv2
import numpy as np

from .feature_base import BaseFeature2D


class OrbFeature2D(BaseFeature2D):
    def __init__(self, num_features: int = 1000):
        super().__init__(num_features=num_features, device="cpu")
        self.orb = cv2.ORB_create(nfeatures=int(num_features))

    def setMaxFeatures(self, num_features: int):
        super().setMaxFeatures(num_features)
        self.orb.setMaxFeatures(int(num_features))

    def detectAndCompute(self, frame, mask=None):
        if frame is None:
            return [], np.zeros((0, 32), dtype=np.uint8)
        if frame.ndim == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        kps, des = self.orb.detectAndCompute(gray, mask)
        if des is None:
            des = np.zeros((0, 32), dtype=np.uint8)
        return kps, des
