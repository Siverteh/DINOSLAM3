#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
STAGE1_CFG="${STAGE1_CFG:-$ROOT/configs/train/tum_regime_stage1.yaml}"
STAGE2_CFG="${STAGE2_CFG:-$ROOT/configs/train/tum_regime_stage2.yaml}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "ERROR: python not found/executable: $PYTHON_BIN" >&2
  exit 1
fi
if [[ ! -f "$STAGE1_CFG" ]]; then
  echo "ERROR: stage1 config not found: $STAGE1_CFG" >&2
  exit 1
fi
if [[ ! -f "$STAGE2_CFG" ]]; then
  echo "ERROR: stage2 config not found: $STAGE2_CFG" >&2
  exit 1
fi

export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

stage1_run_name="$("$PYTHON_BIN" - "$STAGE1_CFG" <<'PY'
import sys
from dino_slam3.utils.config import load_config
cfg = load_config(sys.argv[1])
print(cfg["run"]["name"])
PY
)"
stage1_best="$ROOT/runs/$stage1_run_name/checkpoints/best.pt"

echo "[regime] stage1 config: $STAGE1_CFG"
echo "[regime] stage2 config: $STAGE2_CFG"
echo "[regime] stage1 run: $stage1_run_name"
echo "[regime] python: $PYTHON_BIN"

echo "[regime] training stage1..."
"$PYTHON_BIN" -m dino_slam3.scripts.train --config "$STAGE1_CFG"

if [[ ! -f "$stage1_best" ]]; then
  echo "ERROR: stage1 best checkpoint not found: $stage1_best" >&2
  exit 1
fi
echo "[regime] stage1 best: $stage1_best"

echo "[regime] training stage2 (init from stage1 best)..."
"$PYTHON_BIN" -m dino_slam3.scripts.train --config "$STAGE2_CFG" --init-ckpt "$stage1_best"

stage2_run_name="$("$PYTHON_BIN" - "$STAGE2_CFG" <<'PY'
import sys
from dino_slam3.utils.config import load_config
cfg = load_config(sys.argv[1])
print(cfg["run"]["name"])
PY
)"
stage2_best="$ROOT/runs/$stage2_run_name/checkpoints/best.pt"
echo "[regime] stage2 best: $stage2_best"
echo "[regime] done"
