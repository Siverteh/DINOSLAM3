#!/bin/bash
# Run pySLAM using your DINOSLAM3 learned features by hijacking SUPERPOINT,
# with semantics enabled (this script is safe to re-run; it always rewrites the shim/wrapper).
#
# Fixes included:
#  - Uses your actual model output names: FeatureOutputs.{desc, heatmap, offset, reliability}
#  - Accepts either BGR (HxWx3) or grayscale (HxW) frames from pySLAM
#  - Forces a loop detector compatible with SUPERPOINT front-end by using ORB2 in an INDEPENDENT loop manager:
#      LoopDetectionConfig.name = DBOW3_INDEPENDENT
#
# Expected ckpt:
#   /workspace/runs/tum_stage1_dinov3_refine_v1/checkpoints/best.pt

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PYSLAM_DIR="$SCRIPT_DIR/pyslam"

DINOSLAM3_ROOT="${DINOSLAM3_ROOT:-/workspace}"
CKPT="${CKPT:-$DINOSLAM3_ROOT/runs/tum_stage1_dinov3_refine_v1/checkpoints/best.pt}"

DATA_PATH="${DATA_PATH:-$DINOSLAM3_ROOT/src/dino_slam3/data/tum_rgbd}"
OUTPUT_PATH="${OUTPUT_PATH:-$SCRIPT_DIR/results/semantic}"

ASSOC_SCRIPT="${ASSOC_SCRIPT:-$SCRIPT_DIR/associate.py}"
GTSAM_LIB_DIR="$PYSLAM_DIR/thirdparty/gtsam_local/install/lib"

PYTHON_CANDIDATES=(
  "$SCRIPT_DIR/.venv/bin/python"
  "$DINOSLAM3_ROOT/.venv/bin/python"
  "python3"
)

PYTHON_BIN=""
for candidate in "${PYTHON_CANDIDATES[@]}"; do
  if [ -x "$candidate" ] || [ "$candidate" = "python3" ]; then
    if "$candidate" - <<'PY' >/dev/null 2>&1
try:
    import yaml  # noqa
except Exception:
    raise SystemExit(1)
PY
    then
      PYTHON_BIN="$candidate"
      break
    fi
  fi
done

if [ -z "$PYTHON_BIN" ]; then
  echo "✗ No Python interpreter with PyYAML found." >&2
  exit 1
fi

if [ ! -d "$PYSLAM_DIR" ]; then
  echo "✗ PYSLAM_DIR not found: $PYSLAM_DIR" >&2
  echo "  Run ./setup.sh first (the one that clones pySLAM into ./pyslam)." >&2
  exit 1
fi

if [ ! -f "$CKPT" ]; then
  echo "✗ Checkpoint not found: $CKPT" >&2
  exit 1
fi

mkdir -p "$OUTPUT_PATH/trajectories" "$OUTPUT_PATH/logs" "$OUTPUT_PATH/plots"

# Ensure GTSAM shared libs visible
if [ -d "$GTSAM_LIB_DIR" ]; then
  export LD_LIBRARY_PATH="$GTSAM_LIB_DIR:${LD_LIBRARY_PATH:-}"
fi

# Ensure cpp libs symlink for pySLAM Python package
CPP_BUILT_LIB="$PYSLAM_DIR/cpp/lib"
CPP_PKG_LIB="$PYSLAM_DIR/pyslam/slam/cpp/lib"
mkdir -p "$(dirname "$CPP_PKG_LIB")"
if [ -d "$CPP_PKG_LIB" ] && [ ! -L "$CPP_PKG_LIB" ]; then
  rm -rf "$CPP_PKG_LIB"
fi
ln -sfn "$CPP_BUILT_LIB" "$CPP_PKG_LIB"

# Make your repo importable
export PYTHONPATH="$DINOSLAM3_ROOT/src:${PYTHONPATH:-}"
export DINOSLAM3_ROOT="$DINOSLAM3_ROOT"
export DINOSLAM3_CKPT="$CKPT"

EXECUTABLE="$PYTHON_BIN $PYSLAM_DIR/main_slam.py"

echo "================================================"
echo "Running pySLAM + DINOSLAM3 learned features (SUPERPOINT hijack)"
echo "================================================"
echo "DINOSLAM3_ROOT: $DINOSLAM3_ROOT"
echo "PYSLAM_DIR:     $PYSLAM_DIR"
echo "Checkpoint:     $CKPT"
echo "Dataset:        $DATA_PATH"
echo "Output:         $OUTPUT_PATH"
echo "Python:         $PYTHON_BIN"
echo "PYTHONPATH:     $PYTHONPATH"
echo "================================================"
echo ""

# Headless
if ! command -v xvfb-run &> /dev/null; then
  echo "Installing xvfb for headless operation..."
  apt-get update -qq && apt-get install -y xvfb
fi

# ----------------------------------------------------------------------
# Write feature_dinoslam3.py (wrapper)
# ----------------------------------------------------------------------
FEATURE_DINOSLAM3="$PYSLAM_DIR/pyslam/local_features/feature_dinoslam3.py"
FEATURE_SUPERPOINT="$PYSLAM_DIR/pyslam/local_features/feature_superpoint.py"

cat > "$FEATURE_DINOSLAM3" <<'PY'
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
PY

# ----------------------------------------------------------------------
# Overwrite feature_superpoint.py with the lazy shim
# ----------------------------------------------------------------------
cat > "$FEATURE_SUPERPOINT" <<'PY'
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
PY

echo "[OK] Wrote feature_dinoslam3.py and SUPERPOINT shim"
echo ""

SEQUENCES=(
  "rgbd_dataset_freiburg1_desk"
  "rgbd_dataset_freiburg1_plant"
  "rgbd_dataset_freiburg1_room"
  "rgbd_dataset_freiburg3_long_office_household"
  "rgbd_dataset_freiburg3_walking_static"
  "rgbd_dataset_freiburg3_walking_xyz"
)

for SEQ in "${SEQUENCES[@]}"; do
  echo "================================================"
  echo "Processing: $SEQ"
  echo "================================================"

  ASSOC_FILE="$DATA_PATH/$SEQ/associations.txt"
  if [ ! -f "$ASSOC_FILE" ]; then
    echo "⚠ Association file not found, generating..."
    if [ -f "$DATA_PATH/$SEQ/rgb.txt" ] && [ -f "$DATA_PATH/$SEQ/depth.txt" ]; then
      "$PYTHON_BIN" "$ASSOC_SCRIPT" \
        "$DATA_PATH/$SEQ/rgb.txt" \
        "$DATA_PATH/$SEQ/depth.txt" \
        --output "$ASSOC_FILE"
      echo "✓ Generated association file"
    else
      echo "✗ Cannot generate associations for $SEQ (missing rgb.txt/depth.txt)" >&2
      continue
    fi
  else
    echo "✓ Using existing association file"
  fi

  if [[ $SEQ == *"freiburg1"* ]]; then
    BASE_SETTINGS="settings/TUM1.yaml"
  elif [[ $SEQ == *"freiburg2"* ]]; then
    BASE_SETTINGS="settings/TUM2.yaml"
  else
    BASE_SETTINGS="settings/TUM3.yaml"
  fi

  # Patch the *settings* file (this is what config.py reads for FeatureTrackerConfig.name)
  PATCHED_SETTINGS="$PYSLAM_DIR/settings/_DINOSLAM3_${SEQ}.yaml"
  cp "$PYSLAM_DIR/$BASE_SETTINGS" "$PATCHED_SETTINGS"

  "$PYTHON_BIN" - <<EOF
import yaml
p="$PATCHED_SETTINGS"
with open(p,"r") as f:
    y=yaml.safe_load(f) or {}

# Force SUPERPOINT path (which we hijacked)
y["FeatureTrackerConfig.name"]="SUPERPOINT"
y["FeatureTrackerConfig.nFeatures"]=1000

# IMPORTANT: loop closing compatibility
# Use an INDEPENDENT ORB2 local_feature_manager for DBOW3 vocabulary
y["LoopDetectionConfig.name"]="DBOW3_INDEPENDENT"

with open(p,"w") as f:
    yaml.safe_dump(y,f,sort_keys=False)
print("Wrote patched settings:", p)
print("FeatureTrackerConfig.name =>", y.get("FeatureTrackerConfig.name"))
print("FeatureTrackerConfig.nFeatures =>", y.get("FeatureTrackerConfig.nFeatures"))
print("Loop detector =>", y.get("LoopDetectionConfig.name"))
EOF

  # Create a per-sequence config.yaml pointing to the patched settings
  CONFIG_FILE="/tmp/pyslam_config_${SEQ}_DINOSLAM3.yaml"
  cp "$PYSLAM_DIR/config.yaml" "$CONFIG_FILE"

  "$PYTHON_BIN" - <<EOF
import yaml
cfg_path="$CONFIG_FILE"
with open(cfg_path,"r") as f:
    cfg=yaml.safe_load(f) or {}

def ensure(cfg,k):
    if not isinstance(cfg.get(k),dict):
        cfg[k]={}
    return cfg[k]

ensure(cfg,"DATASET")["type"]="TUM_DATASET"
ensure(cfg,"TUM_DATASET")
cfg["TUM_DATASET"]["type"]="tum"
cfg["TUM_DATASET"]["sensor_type"]="rgbd"
cfg["TUM_DATASET"]["base_path"]="$DATA_PATH"
cfg["TUM_DATASET"]["name"]="$SEQ"
cfg["TUM_DATASET"]["settings"]=f"settings/_DINOSLAM3_{'$SEQ'}.yaml"
cfg["TUM_DATASET"]["associations"]="associations.txt"
cfg["TUM_DATASET"]["groundtruth_file"]="auto"

ensure(cfg,"SAVE_TRAJECTORY")
cfg["SAVE_TRAJECTORY"]["save_trajectory"]=True
cfg["SAVE_TRAJECTORY"]["format_type"]="tum"
cfg["SAVE_TRAJECTORY"]["output_folder"]="$OUTPUT_PATH"
cfg["SAVE_TRAJECTORY"]["basename"]="${SEQ}_DINOSLAM3"

with open(cfg_path,"w") as f:
    yaml.safe_dump(cfg,f,sort_keys=False)
EOF

  LOG_FILE="$OUTPUT_PATH/logs/${SEQ}_DINOSLAM3.log"
  TRAJ_FILE="$OUTPUT_PATH/trajectories/${SEQ}_DINOSLAM3_trajectory.txt"

  echo "Running pySLAM..."
  echo "  Using settings: settings/_DINOSLAM3_${SEQ}.yaml"
  echo "  Loop detector forced to DBOW3_INDEPENDENT (ORB2 vocab)"
  echo "  Expect: SUPERPOINT + [DINOSLAM3] CKPT banner"

  cd "$PYSLAM_DIR"
  xvfb-run -a -s "-screen 0 640x480x24" \
    $EXECUTABLE --config_path "$CONFIG_FILE" --headless --no_output_date \
    2>&1 | tee "$LOG_FILE"

  # Move trajectory outputs into trajectories/
  TRAJ_GEN="$OUTPUT_PATH/${SEQ}_DINOSLAM3_final.txt"
  TRAJ_ONL="$OUTPUT_PATH/${SEQ}_DINOSLAM3_online.txt"
  if [ -f "$TRAJ_GEN" ]; then
    mv "$TRAJ_GEN" "$TRAJ_FILE"
    [ -f "$TRAJ_ONL" ] && rm "$TRAJ_ONL" || true
    echo "✓ Final trajectory saved: $TRAJ_FILE"
  elif [ -f "$TRAJ_ONL" ]; then
    mv "$TRAJ_ONL" "$TRAJ_FILE"
    echo "✓ Online trajectory saved: $TRAJ_FILE"
  else
    echo "⚠ No trajectory produced for $SEQ (check log: $LOG_FILE)"
  fi

  # Move plots into plots/<SEQ> like baseline does
  if [ -d "$OUTPUT_PATH/plot" ]; then
    mkdir -p "$OUTPUT_PATH/plots/${SEQ}"
    mv "$OUTPUT_PATH/plot/"* "$OUTPUT_PATH/plots/${SEQ}/" 2>/dev/null || true
    rmdir "$OUTPUT_PATH/plot" || true
    echo "✓ Plots moved to: $OUTPUT_PATH/plots/${SEQ}/"
  fi

  mkdir -p "$OUTPUT_PATH/plots/${SEQ}"
  mv "$OUTPUT_PATH/"*.png "$OUTPUT_PATH/plots/${SEQ}/" 2>/dev/null || true


  rm -f "$CONFIG_FILE"
  echo ""
done

echo "================================================"
echo "✓ DINOSLAM3 semantic runs complete!"
echo "================================================"
echo "Trajectories: $OUTPUT_PATH/trajectories/"
echo "Logs:         $OUTPUT_PATH/logs/"
