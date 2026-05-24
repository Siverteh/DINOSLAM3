#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT/common.sh"

REPO_DIR="${DPVO_REPO_DIR:-$REPOS_ROOT/DPVO}"
ENV_DIR="${DPVO_ENV_DIR:-$MAMBA_ROOT/envs/dpvo}"
WEIGHTS_PATH="${DPVO_WEIGHTS_PATH:-$REPO_DIR/dpvo.pth}"
CONFIG_PATH="${DPVO_CONFIG_PATH:-$REPO_DIR/config/default.yaml}"
RUNS_ROOT="${RUNS_ROOT:-$RUNS_ROOT_DEFAULT}"
RUN_ID="${DPVO_RUN_ID:-dpvo_tartanair_v1}"
RESULTS_DIR="${RESULTS_DIR_OVERRIDE:-$RUNS_ROOT/external/${RUN_ID}}"
CSV_PATH="${CSV_PATH_OVERRIDE:-$RESULTS_DIR/metrics_summary.csv}"
DATA_PATH="${DATA_PATH:-$SUBTREE_ROOT/data/tartanair_v2_converted}"
MAX_DT="${MAX_DT:-0.02}"
MISSING_PENALTY_METERS="${MISSING_PENALTY_METERS:-3.0}"
MIN_COVERAGE_OK="${MIN_COVERAGE_OK:-0.95}"
STRIDE="${DPVO_STRIDE:-1}"
BACKEND_THRESH="${DPVO_BACKEND_THRESH:-18.0}"
IMAGE_HEIGHT="${DPVO_IMAGE_HEIGHT:-}"
IMAGE_WIDTH="${DPVO_IMAGE_WIDTH:-}"

if [[ ! -d "$REPO_DIR" ]]; then
  echo "ERROR: DPVO repo not found at $REPO_DIR. Run refocus_vo/external/install_dpvo.sh first." >&2
  exit 1
fi
ensure_env_dir "$ENV_DIR"
if [[ ! -f "$WEIGHTS_PATH" ]]; then
  echo "ERROR: DPVO weights not found at $WEIGHTS_PATH. Run refocus_vo/external/install_dpvo.sh first." >&2
  exit 1
fi
if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "ERROR: DPVO config not found at $CONFIG_PATH." >&2
  exit 1
fi

mkdir -p "$RESULTS_DIR"

CMD=( python -m refocus_vo.eval.external_dpvo_tartanair
  --dataset-root "$DATA_PATH"
  --dpvo-root "$REPO_DIR"
  --weights "$WEIGHTS_PATH"
  --config "$CONFIG_PATH"
  --output-dir "$RESULTS_DIR"
  --csv-path "$CSV_PATH"
  --max-dt "$MAX_DT"
  --missing-penalty-m "$MISSING_PENALTY_METERS"
  --min-coverage-ok "$MIN_COVERAGE_OK"
  --stride "$STRIDE"
  --backend-thresh "$BACKEND_THRESH"
)

if [[ -n "$IMAGE_HEIGHT" ]]; then
  CMD+=( --image-height "$IMAGE_HEIGHT" )
fi
if [[ -n "$IMAGE_WIDTH" ]]; then
  CMD+=( --image-width "$IMAGE_WIDTH" )
fi
if [[ "${DPVO_VIZ:-0}" == "1" ]]; then
  CMD+=( --viz )
fi

EXTRA_OPTS=()
if [[ -n "${DPVO_OPTS:-}" ]]; then
  # shellcheck disable=SC2206
  RAW_OPTS=( $DPVO_OPTS )
else
  RAW_OPTS=()
fi

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

PYTHONPATH="$SUBTREE_ROOT/src:$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}" \
run_in_env "$ENV_DIR" "${CMD[@]}"

echo "DPVO TartanAir results written to $CSV_PATH"
