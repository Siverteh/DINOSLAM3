# Recommended Training Regime (DINO-SLAM3)

This regime is designed for robust full-sequence tracking first, then better trajectory accuracy.

## 1) Train Stage 1 (from scratch)

```bash
cd /home/sivert/DINOSLAM3
source .venv/bin/activate
export PYTHONPATH=$PWD/src
python -m dino_slam3.scripts.train --config configs/train/tum_regime_stage1.yaml
```

Expected best checkpoint:
`runs/tum_regime_stage1_semantic_v1/checkpoints/best.pt`

## 2) Train Stage 2 (fine-tune from Stage 1)

```bash
cd /home/sivert/DINOSLAM3
source .venv/bin/activate
export PYTHONPATH=$PWD/src
python -m dino_slam3.scripts.train \
  --config configs/train/tum_regime_stage2.yaml \
  --init-ckpt runs/tum_regime_stage1_semantic_v1/checkpoints/best.pt
```

Expected best checkpoint:
`runs/tum_regime_stage2_hardpairs_v1/checkpoints/best.pt`

## 3) One-command run (Stage 1 + Stage 2)

```bash
cd /home/sivert/DINOSLAM3
./scripts/train_regime.sh
```

## 4) Resume interrupted training

```bash
python -m dino_slam3.scripts.train \
  --config configs/train/tum_regime_stage1.yaml \
  --resume-ckpt runs/tum_regime_stage1_semantic_v1/checkpoints/epoch_010.pt
```

## 5) Evaluate in pySLAM

```bash
cd /home/sivert/DINOSLAM3/pyslam_integration
CKPT=/home/sivert/DINOSLAM3/runs/tum_regime_stage2_hardpairs_v1/checkpoints/best.pt \
USE_LOOP_CLOSING=1 \
SEQUENCES=freiburg1_desk,freiburg1_plant,freiburg1_room \
./run_semantic.sh
```
