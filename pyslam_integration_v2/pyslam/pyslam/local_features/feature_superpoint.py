"""SUPERPOINT shim redirected to DINOSLAM3 (lazy import)."""

import os

DEFAULT_DINOSLAM3_CHECKPOINT = os.environ.get("DINOSLAM3_CKPT", "")

class SuperPointFeature2D:
    def __init__(self, *args, **kwargs):
        ckpt = kwargs.pop("checkpoint_path", None) or DEFAULT_DINOSLAM3_CHECKPOINT
        device = kwargs.pop("device", "cuda")
        num_features = kwargs.pop("num_features", kwargs.pop("num_keypoints", 1000))
        normalize = kwargs.pop("normalize_descriptors", True)

        if not ckpt:
            raise RuntimeError("DINOSLAM3_CKPT env var not set and no checkpoint_path provided.")

        from pyslam.local_features.feature_dinoslam3 import DinoSlam3Feature2D
        self._impl = DinoSlam3Feature2D(
            checkpoint_path=ckpt,
            device=device,
            num_features=int(num_features),
            normalize_descriptors=bool(normalize),
        )

    def setMaxFeatures(self, n):
        if hasattr(self._impl, "setMaxFeatures"):
            self._impl.setMaxFeatures(n)

    def detectAndCompute(self, frame, mask=None):
        return self._impl.detectAndCompute(frame, mask=mask)

    def detect(self, frame, mask=None):
        return self._impl.detect(frame, mask=mask)

    def compute(self, frame, kps=None, mask=None):
        return self._impl.compute(frame, kps=kps, mask=mask)
