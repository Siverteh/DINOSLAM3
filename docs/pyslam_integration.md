# pySLAM integration notes

pySLAM expects a local feature extractor that behaves like OpenCV `Feature2D`:

- `detectAndCompute(image, mask=None) -> (keypoints, descriptors)`

This repo provides `dino_slam3.slam.pyslam_adapter.DinoSLAM3FeatureExtractor`.

## Typical wiring

1. Train your model and obtain a checkpoint:
   `runs/<run_name>/checkpoints/epoch_XXX.pt`

2. Build the model and extractor:

```python
from dino_slam3.models.network import LocalFeatureNet
from dino_slam3.slam.pyslam_adapter import DinoSLAM3FeatureExtractor

model = LocalFeatureNet(patch_size=16, descriptor_dim=128)
extractor = DinoSLAM3FeatureExtractor(
    model=model,
    checkpoint_path="runs/.../checkpoints/epoch_020.pt",
    device="cuda",
)
```

3. In pySLAM, replace the ORB/SIFT extractor with this `extractor`.

## Important
- DINOv3 must be available in `DinoV3Backbone.load()`
- The extractor resizes inputs to the model's `input_size` (default 448). If your SLAM pipeline
  assumes original resolution, you should rescale keypoints back (or change the adapter to keep scale).
