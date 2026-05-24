#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBTREE_ROOT="$(cd "$ROOT/.." && pwd)"
REPO_ROOT="$(cd "$SUBTREE_ROOT/.." && pwd)"
RUNS_ROOT="${RUNS_ROOT:-$SUBTREE_ROOT/runs}"
DATA_PATH="${DATA_PATH:-$REPO_ROOT/src/dino_slam3/data/tum_rgbd}"
DPVO_REPO_DIR="${DPVO_REPO_DIR:-$SUBTREE_ROOT/external/repos/DPVO}"
DPVO_WEIGHTS_PATH="${DPVO_WEIGHTS_PATH:-$DPVO_REPO_DIR/dpvo.pth}"
DPVO_CONFIG_PATH="${DPVO_CONFIG_PATH:-$DPVO_REPO_DIR/config/fast.yaml}"
FRONTEND_MODE="${FRONTEND_MODE:-dino_proposals}"
FRONTEND_CONFIG="${FRONTEND_CONFIG:-$SUBTREE_ROOT/configs/dino_dpvo_proposals_v1.yaml}"
CHECKPOINT="${CHECKPOINT:-}"
DINO_DPVO_RUN_ID="${DINO_DPVO_RUN_ID:-dino_dpvo_tum_v1}"
OUTPUT_DIR="${OUTPUT_DIR_OVERRIDE:-$RUNS_ROOT/eval/${DINO_DPVO_RUN_ID}}"
CSV_PATH="${CSV_PATH_OVERRIDE:-$OUTPUT_DIR/metrics_summary.csv}"
SEQUENCES="${SEQUENCES:-freiburg1_desk,freiburg1_plant,freiburg1_room,freiburg2_desk_with_person,freiburg3_large_cabinet,freiburg3_walking_static}"
COLLECT_DIAGNOSTICS="${COLLECT_DIAGNOSTICS:-1}"
MAX_DT="${MAX_DT:-0.02}"
MISSING_PENALTY_METERS="${MISSING_PENALTY_METERS:-3.0}"
MIN_COVERAGE_OK="${MIN_COVERAGE_OK:-0.95}"
STRIDE="${DPVO_STRIDE:-4}"
BACKEND_THRESH="${DPVO_BACKEND_THRESH:-32.0}"
IMAGE_HEIGHT="${DPVO_IMAGE_HEIGHT:-240}"
IMAGE_WIDTH="${DPVO_IMAGE_WIDTH:-320}"
DPVO_OPTS="${DPVO_OPTS:-}"

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

REQUIRED_MODS="yaml,cv2,torch,scipy,matplotlib,evo,torch_scatter"
if [[ "$FRONTEND_MODE" != "dpvo_native" ]]; then
  REQUIRED_MODS="$REQUIRED_MODS,transformers"
fi
PYTHON_BIN="$(resolve_python "$REQUIRED_MODS")"

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
if [[ "$FRONTEND_MODE" != "dpvo_native" && ! -f "$FRONTEND_CONFIG" && -z "$CHECKPOINT" ]]; then
  echo "ERROR: FRONTEND_CONFIG is required for non-native DINO-DPVO eval." >&2
  exit 1
fi
if [[ -n "$CHECKPOINT" && ! -f "$CHECKPOINT" ]]; then
  echo "ERROR: checkpoint not found at $CHECKPOINT." >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"
export PYTHONPATH="$SUBTREE_ROOT/src:$DPVO_REPO_DIR:$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

CMD=(
  "$PYTHON_BIN" -m refocus_vo.eval.external_dpvo
  --dataset-root "$DATA_PATH"
  --dpvo-root "$DPVO_REPO_DIR"
  --weights "$DPVO_WEIGHTS_PATH"
  --config "$DPVO_CONFIG_PATH"
  --output-dir "$OUTPUT_DIR"
  --csv-path "$CSV_PATH"
  --sequences "$SEQUENCES"
  --max-dt "$MAX_DT"
  --missing-penalty-m "$MISSING_PENALTY_METERS"
  --min-coverage-ok "$MIN_COVERAGE_OK"
  --stride "$STRIDE"
  --backend-thresh "$BACKEND_THRESH"
  --image-height "$IMAGE_HEIGHT"
  --image-width "$IMAGE_WIDTH"
  --frontend-mode "$FRONTEND_MODE"
)

if [[ -n "$FRONTEND_CONFIG" && -f "$FRONTEND_CONFIG" ]]; then
  CMD+=( --frontend-config "$FRONTEND_CONFIG" )
fi
if [[ -n "$CHECKPOINT" ]]; then
  CMD+=( --frontend-checkpoint "$CHECKPOINT" )
fi
if [[ "$COLLECT_DIAGNOSTICS" == "1" ]]; then
  CMD+=( --collect-diagnostics )
fi
if [[ -n "$DPVO_OPTS" ]]; then
  # shellcheck disable=SC2206
  RAW_OPTS=( $DPVO_OPTS )
  EXTRA_OPTS=()
  for opt in "${RAW_OPTS[@]}"; do
    if [[ "$opt" == *=* ]]; then
      EXTRA_OPTS+=( "${opt%%=*}" "${opt#*=}" )
    else
      EXTRA_OPTS+=( "$opt" )
    fi
  done
  if [[ "${#EXTRA_OPTS[@]}" -gt 0 ]]; then
    CMD+=( --opts "${EXTRA_OPTS[@]}" )
  fi
fi

"${CMD[@]}"

echo "DINO-DPVO TUM results written to $CSV_PATH"
