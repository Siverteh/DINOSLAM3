#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT/common.sh"

REPO_DIR="${ORB_SLAM3_REPO_DIR:-$REPOS_ROOT/ORB_SLAM3}"
ENV_DIR="${ORB_SLAM3_ENV_DIR:-$MAMBA_ROOT/envs/orbslam3-build}"
RUNS_ROOT="${RUNS_ROOT:-$RUNS_ROOT_DEFAULT}"
MODE="${ORB_SLAM3_MODE:-mono}"
RUN_ID="${ORB_SLAM3_RUN_ID:-orb_slam3_tum_${MODE}_v1}"
RESULTS_DIR="${RESULTS_DIR_OVERRIDE:-$RUNS_ROOT/external/${RUN_ID}}"
CSV_PATH="${CSV_PATH_OVERRIDE:-$RESULTS_DIR/metrics_summary.csv}"
DATA_PATH="${DATA_PATH:-$REPO_ROOT/src/dino_slam3/data/tum_rgbd}"
SEQUENCES="${SEQUENCES:-freiburg1_desk}"
MAX_DT="${MAX_DT:-0.02}"
MISSING_PENALTY_METERS="${MISSING_PENALTY_METERS:-3.0}"
MIN_COVERAGE_OK="${MIN_COVERAGE_OK:-0.95}"
TIMEOUT_S="${ORB_SLAM3_TIMEOUT_S:-900}"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="${PYTHON_BIN_FALLBACK:-python3}"
fi

if [[ ! -d "$REPO_DIR" ]]; then
  echo "ERROR: ORB-SLAM3 repo not found at $REPO_DIR." >&2
  exit 1
fi
ensure_env_dir "$ENV_DIR"
if [[ ! -f "$REPO_DIR/Vocabulary/ORBvoc.txt" && -f "$REPO_DIR/Vocabulary/ORBvoc.txt.tar.gz" ]]; then
  tar -xf "$REPO_DIR/Vocabulary/ORBvoc.txt.tar.gz" -C "$REPO_DIR/Vocabulary"
fi
if [[ ! -x "$REPO_DIR/Examples/Monocular/mono_tum" && "$MODE" == "mono" ]]; then
  echo "ERROR: ORB-SLAM3 mono_tum binary not found. Build ORB-SLAM3 first." >&2
  exit 1
fi
if [[ ! -x "$REPO_DIR/Examples/RGB-D/rgbd_tum" && "$MODE" == "rgbd" ]]; then
  echo "ERROR: ORB-SLAM3 rgbd_tum binary not found. Build ORB-SLAM3 first." >&2
  exit 1
fi

mkdir -p "$RESULTS_DIR"

PYTHONPATH="$SUBTREE_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
"$PYTHON_BIN" -m refocus_vo.eval.external_orb_slam3 \
  --dataset-root "$DATA_PATH" \
  --orb-root "$REPO_DIR" \
  --env-dir "$ENV_DIR" \
  --output-dir "$RESULTS_DIR" \
  --csv-path "$CSV_PATH" \
  --sequences "$SEQUENCES" \
  --mode "$MODE" \
  --max-dt "$MAX_DT" \
  --missing-penalty-m "$MISSING_PENALTY_METERS" \
  --min-coverage-ok "$MIN_COVERAGE_OK" \
  --timeout-s "$TIMEOUT_S"

echo "ORB-SLAM3 results written to $CSV_PATH"
