#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBTREE_ROOT="$(cd "$ROOT/.." && pwd)"
REPO_ROOT="$(cd "$SUBTREE_ROOT/.." && pwd)"
RUNS_ROOT="${RUNS_ROOT:-$SUBTREE_ROOT/runs}"
DATA_PATH="${DATA_PATH:-$SUBTREE_ROOT/data/tartanair_v2_converted}"
EVAL_DATA_PATH="${EVAL_DATA_PATH:-$DATA_PATH}"
TUM_EVAL_DATA_PATH="${TUM_EVAL_DATA_PATH:-$REPO_ROOT/src/dino_slam3/data/tum_rgbd}"
EUROC_EVAL_DATA_PATH="${EUROC_EVAL_DATA_PATH:-$SUBTREE_ROOT/data/euroc_asl}"
KITTI_EVAL_DATA_PATH="${KITTI_EVAL_DATA_PATH:-$SUBTREE_ROOT/data/kitti_odometry}"
TRAIN_RUN_ID="${TRAIN_RUN_ID:-dino_dpvo_proposals_v1}"
DINO_DPVO_CONFIG="${DINO_DPVO_CONFIG:-$SUBTREE_ROOT/configs/dino_dpvo_proposals_v1.yaml}"
SUBSET_CONFIG="${SUBSET_CONFIG:-$SUBTREE_ROOT/configs/tartanair_subset_v1.yaml}"
OUTPUT_DIR="${OUTPUT_DIR_OVERRIDE:-$RUNS_ROOT/train/${TRAIN_RUN_ID}}"
DPVO_REPO_DIR="${DPVO_REPO_DIR:-$SUBTREE_ROOT/external/repos/DPVO}"
DPVO_WEIGHTS_PATH="${DPVO_WEIGHTS_PATH:-$DPVO_REPO_DIR/dpvo.pth}"
DPVO_CONFIG_PATH="${DPVO_CONFIG_PATH:-$DPVO_REPO_DIR/config/default.yaml}"
MAX_TRAIN_WINDOWS="${MAX_TRAIN_WINDOWS:-}"
MAX_DEV_WINDOWS="${MAX_DEV_WINDOWS:-}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-}"
INIT_MODE="${INIT_MODE:-strict}"
SEED="${SEED:-13}"
TRAIN_DEVICE="${TRAIN_DEVICE:-cuda}"
DETERMINISTIC="${DETERMINISTIC:-0}"
LEGACY_REPRO="${LEGACY_REPRO:-0}"

python_has_modules() {
  local py="$1"
  local modlist="$2"
  "$py" - "$modlist" <<'PY' >/dev/null 2>&1
import importlib
import sys
mods = [m.strip() for m in sys.argv[1].split(',') if m.strip()]
for name in mods:
    importlib.import_module(name)
PY
}

resolve_python() {
  local required_mods="$1"
  local candidates=(
    "${PYTHON_BIN:-}"
    "$SUBTREE_ROOT/.micromamba/envs/dpvo/bin/python"
    "$REPO_ROOT/.venv_pyslam_integration_v2/bin/python"
    "$REPO_ROOT/.venv/bin/python"
    "python3"
  )
  local cand=""
  for cand in "${candidates[@]}"; do
    [[ -n "$cand" ]] || continue
    if [[ "$cand" == */* ]]; then
      [[ -e "$cand" && -x "$cand" ]] || continue
    else
      command -v "$cand" >/dev/null 2>&1 || continue
      cand="$(command -v "$cand")"
    fi
    if python_has_modules "$cand" "$required_mods"; then
      echo "$cand"
      return
    fi
  done
  echo "ERROR: Could not find Python with modules: $required_mods" >&2
  exit 1
}

PYTHON_BIN="$(resolve_python "yaml,cv2,torch,scipy,matplotlib,transformers,evo,torch_scatter")"

if [[ ! -f "$DINO_DPVO_CONFIG" ]]; then
  echo "ERROR: DINO-DPVO config not found at $DINO_DPVO_CONFIG." >&2
  exit 1
fi
if [[ ! -d "$DATA_PATH" ]]; then
  echo "ERROR: TartanAir data root not found at $DATA_PATH." >&2
  exit 1
fi
if [[ ! -d "$EVAL_DATA_PATH" ]]; then
  echo "ERROR: TartanAir eval data root not found at $EVAL_DATA_PATH." >&2
  exit 1
fi
if [[ ! -d "$DPVO_REPO_DIR" ]]; then
  echo "ERROR: DPVO repo not found at $DPVO_REPO_DIR." >&2
  exit 1
fi
if [[ ! -f "$DPVO_WEIGHTS_PATH" ]]; then
  echo "ERROR: DPVO weights not found at $DPVO_WEIGHTS_PATH." >&2
  exit 1
fi
if [[ ! -f "$DPVO_CONFIG_PATH" ]]; then
  echo "ERROR: DPVO config not found at $DPVO_CONFIG_PATH." >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"
export PYTHONPATH="$SUBTREE_ROOT/src:$DPVO_REPO_DIR:$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

CMD=(
  "$PYTHON_BIN" -m refocus_vo.train_dino_dpvo_frontend
  --dataset-root "$DATA_PATH"
  --eval-dataset-root "$EVAL_DATA_PATH"
  --config "$DINO_DPVO_CONFIG"
  --subset-config "$SUBSET_CONFIG"
  --output-dir "$OUTPUT_DIR"
  --dpvo-root "$DPVO_REPO_DIR"
  --dpvo-weights "$DPVO_WEIGHTS_PATH"
  --dpvo-config "$DPVO_CONFIG_PATH"
  --tum-dataset-root "$TUM_EVAL_DATA_PATH"
  --seed "$SEED"
  --device "$TRAIN_DEVICE"
)

if [[ -d "$EUROC_EVAL_DATA_PATH" ]]; then
  CMD+=( --euroc-dataset-root "$EUROC_EVAL_DATA_PATH" )
fi
if [[ -d "$KITTI_EVAL_DATA_PATH" ]]; then
  CMD+=( --kitti-dataset-root "$KITTI_EVAL_DATA_PATH" )
fi

if [[ -n "$MAX_TRAIN_WINDOWS" ]]; then
  CMD+=( --max-train-windows "$MAX_TRAIN_WINDOWS" )
fi
if [[ -n "$MAX_DEV_WINDOWS" ]]; then
  CMD+=( --max-dev-windows "$MAX_DEV_WINDOWS" )
fi
if [[ -n "$INIT_CHECKPOINT" ]]; then
  CMD+=( --init-checkpoint "$INIT_CHECKPOINT" )
fi
if [[ -n "$INIT_MODE" ]]; then
  CMD+=( --init-mode "$INIT_MODE" )
fi
if [[ "$DETERMINISTIC" == "1" ]]; then
  CMD+=( --deterministic )
fi
if [[ "$LEGACY_REPRO" == "1" ]]; then
  CMD+=( --legacy-repro )
fi

"${CMD[@]}"

echo "DINO-DPVO frontend training outputs written to $OUTPUT_DIR"
