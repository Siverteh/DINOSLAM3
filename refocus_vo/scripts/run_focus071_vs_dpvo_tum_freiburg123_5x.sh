#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBTREE_ROOT="$(cd "$ROOT/.." && pwd)"
REPO_ROOT="$(cd "$SUBTREE_ROOT/.." && pwd)"

RUNS_ROOT="${RUNS_ROOT:-$SUBTREE_ROOT/runs}"
DATASET_ROOT="${DATASET_ROOT:-$REPO_ROOT/src/dino_slam3/data/tum_rgbd}"
DPVO_REPO_DIR="${DPVO_REPO_DIR:-$SUBTREE_ROOT/external/repos/DPVO}"
DPVO_WEIGHTS_PATH="${DPVO_WEIGHTS_PATH:-$DPVO_REPO_DIR/dpvo.pth}"
DPVO_CONFIG_PATH="${DPVO_CONFIG_PATH:-$DPVO_REPO_DIR/config/default.yaml}"
FOCUS071_CHECKPOINT="${FOCUS071_CHECKPOINT:-$SUBTREE_ROOT/runs/sweeps/dino_dpvo_focus071_lr_only_sweep_v1/train/focus071lr_orig_const3e6_v1/best_pure100.pt}"
FOCUS071_SOURCE_CONFIG="${FOCUS071_SOURCE_CONFIG:-$SUBTREE_ROOT/runs/sweeps/dino_dpvo_focus071_lr_only_sweep_v1/generated_train_configs/focus071lr_orig_const3e6_v1.yaml}"
REPEATS="${REPEATS:-5}"
SEQUENCES="${SEQUENCES:-}"
DRY_RUN="${DRY_RUN:-0}"

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

next_output_root() {
  local base="$1"
  if [[ ! -e "$base" ]]; then
    echo "$base"
    return
  fi
  local idx=1
  while true; do
    local candidate="${base}_rerun${idx}"
    if [[ ! -e "$candidate" ]]; then
      echo "$candidate"
      return
    fi
    idx=$((idx + 1))
  done
}

if [[ ! -d "$DATASET_ROOT" ]]; then
  echo "ERROR: TUM dataset root not found at $DATASET_ROOT" >&2
  exit 1
fi
if [[ ! -d "$DPVO_REPO_DIR" ]]; then
  echo "ERROR: DPVO repo not found at $DPVO_REPO_DIR" >&2
  exit 1
fi
if [[ ! -f "$DPVO_WEIGHTS_PATH" ]]; then
  echo "ERROR: DPVO weights not found at $DPVO_WEIGHTS_PATH" >&2
  exit 1
fi
if [[ ! -f "$DPVO_CONFIG_PATH" ]]; then
  echo "ERROR: DPVO config not found at $DPVO_CONFIG_PATH" >&2
  exit 1
fi
if [[ ! -f "$FOCUS071_CHECKPOINT" ]]; then
  echo "ERROR: Focus071 checkpoint not found at $FOCUS071_CHECKPOINT" >&2
  exit 1
fi
if [[ ! -f "$FOCUS071_SOURCE_CONFIG" ]]; then
  echo "ERROR: Focus071 source config not found at $FOCUS071_SOURCE_CONFIG" >&2
  exit 1
fi

PYTHON_BIN="$(resolve_python "yaml,cv2,torch,scipy,matplotlib,evo,torch_scatter")"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$SUBTREE_ROOT/src:$DPVO_REPO_DIR:$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

DEFAULT_OUTPUT_ROOT="$RUNS_ROOT/eval/tum_rgbd_freiburg123_dpvo_vs_focus071_v1"
OUTPUT_ROOT="${OUTPUT_ROOT:-$(next_output_root "$DEFAULT_OUTPUT_ROOT")}"
mkdir -p "$OUTPUT_ROOT"

CMD=(
  "$PYTHON_BIN" -m refocus_vo.eval.focus071_vs_dpvo_tum_freiburg123_5x
  --dataset-root "$DATASET_ROOT"
  --dpvo-root "$DPVO_REPO_DIR"
  --dpvo-weights "$DPVO_WEIGHTS_PATH"
  --dpvo-config "$DPVO_CONFIG_PATH"
  --focus071-checkpoint "$FOCUS071_CHECKPOINT"
  --focus071-source-config "$FOCUS071_SOURCE_CONFIG"
  --output-root "$OUTPUT_ROOT"
  --repeats "$REPEATS"
)

if [[ -n "$SEQUENCES" ]]; then
  CMD+=( --sequences "$SEQUENCES" )
fi
if [[ "$DRY_RUN" == "1" ]]; then
  CMD+=( --dry-run )
fi

echo "[tum_freiburg123_5x] output_root=$OUTPUT_ROOT"
"${CMD[@]}"
