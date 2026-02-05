# DINO-SLAM3 (clean rewrite)

This repository is a thesis-ready, **clean** implementation of a hybrid local-feature pipeline designed
to work with **DINOv3** (coarse 16×16 tokens) by adding the missing pieces needed for SLAM:

- a **fine refinement CNN** to recover local detail
- a **keypoint heatmap head** to avoid “points collapse on one side”
- a **sub-pixel offset head** to beat the 16×16 quantization
- a **reliability / uncertainty head** to downweight unstable matches
- **depth+pose supervised geometry** on TUM RGB‑D (no handcrafted edge losses)

It is designed to plug into pySLAM / OpenCV-style pipelines via an adapter.

## Why this is different from your previous repo
Your previous code trained on handcrafted losses (e.g. edge losses) and was vulnerable to degenerate
solutions (all keypoints at one region, trivial activations, unstable descriptors). This rewrite uses:

- **Depth-projected correspondences** from TUM RGB-D (geometry supervision)
- **Balanced keypoint sampling** + NMS + spatial regularization **without edge heuristics**
- **Coarse-to-fine**: DINOv3 tokens provide semantics; a fine CNN recovers precise localization

## Quickstart

### 1) Install
```bash
pip install -e .
```

### 2) Point to TUM RGB‑D
Set in a config file (see `configs/train/tum_stage1.yaml`):
- `dataset.root`: path to the TUM root containing sequences (e.g. `tum_rgbd/rgbd_dataset_freiburg1_xyz`)

### 3) Provide a DINOv3 backbone
This repo **does not vendor** Meta's DINOv3 code/weights.
You must implement `DinoV3Backbone.load()` in:
`src/dino_slam3/models/backbones/dinov3.py`

The wrapper expects:
- input: normalized RGB tensor `B×3×H×W`
- output: token grid features `B×C×H/16×W/16` (patch size 16)

If you already have DINOv3 integrated, you can drop it in quickly.

### 4) Train (recommended)
Stage 1: depth+pose supervised (strong, stable)
```bash
python -m dino_slam3.scripts.train --config configs/train/tum_stage1.yaml
```

Stage 2: optional fine-tuning with harder negatives / longer baselines
```bash
python -m dino_slam3.scripts.train --config configs/train/tum_stage2.yaml
```

### 5) Visualize matches
```bash
python -m dino_slam3.scripts.viz_matches --config configs/eval/viz.yaml
```

## Output
Checkpoints and logs go under `runs/<run_name>/`.

## Thesis-ready structure
- `src/dino_slam3/` : library
- `configs/` : YAML configs
- `docs/` : integration notes
- `runs/` : outputs (gitignored)

# DINOSLAM3
