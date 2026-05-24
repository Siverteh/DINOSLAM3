#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT/common.sh"

REPO_DIR="${DROID_REPO_DIR:-$REPOS_ROOT/DROID-SLAM}"
ENV_DIR="${DROID_ENV_DIR:-$MAMBA_ROOT/envs/droid-slam}"
WEIGHTS_PATH="${DROID_WEIGHTS_PATH:-$REPO_DIR/droid.pth}"
RUNS_ROOT="${RUNS_ROOT:-$RUNS_ROOT_DEFAULT}"
RUN_ID="${DROID_RUN_ID:-droid_slam_tum_pack_v1}"
RESULTS_DIR="${RESULTS_DIR_OVERRIDE:-$RUNS_ROOT/external/${RUN_ID}}"
CSV_PATH="${CSV_PATH_OVERRIDE:-$RESULTS_DIR/metrics_summary.csv}"
DATA_PATH="${DATA_PATH:-$REPO_ROOT/src/dino_slam3/data/tum_rgbd}"
SEQUENCES="${SEQUENCES:-freiburg1_desk,freiburg1_plant,freiburg1_room,freiburg2_desk_with_person,freiburg3_large_cabinet,freiburg3_walking_static}"
MAX_DT="${MAX_DT:-0.02}"
MISSING_PENALTY_METERS="${MISSING_PENALTY_METERS:-3.0}"
MIN_COVERAGE_OK="${MIN_COVERAGE_OK:-0.95}"
IMAGE_STRIDE="${DROID_IMAGE_STRIDE:-2}"
BUFFER_SIZE="${DROID_BUFFER_SIZE:-512}"
FRONTEND_DEVICE="${DROID_FRONTEND_DEVICE:-cuda:0}"
BACKEND_DEVICE="${DROID_BACKEND_DEVICE:-cuda:0}"
VO_MODE="${DROID_VO_MODE:-0}"

if [[ ! -d "$REPO_DIR" ]]; then
  echo "ERROR: DROID-SLAM repo not found at $REPO_DIR. Run refocus_vo/external/install_droid_slam.sh first." >&2
  exit 1
fi
ensure_env_dir "$ENV_DIR"
if [[ ! -f "$WEIGHTS_PATH" ]]; then
  echo "ERROR: DROID-SLAM weights not found at $WEIGHTS_PATH. Run refocus_vo/external/install_droid_slam.sh first." >&2
  exit 1
fi

mkdir -p "$RESULTS_DIR"

CMD=( python -m refocus_vo.eval.external_droid_slam
  --dataset-root "$DATA_PATH"
  --droid-root "$REPO_DIR"
  --weights "$WEIGHTS_PATH"
  --output-dir "$RESULTS_DIR"
  --csv-path "$CSV_PATH"
  --sequences "$SEQUENCES"
  --max-dt "$MAX_DT"
  --missing-penalty-m "$MISSING_PENALTY_METERS"
  --min-coverage-ok "$MIN_COVERAGE_OK"
  --image-stride "$IMAGE_STRIDE"
  --buffer "$BUFFER_SIZE"
  --frontend-device "$FRONTEND_DEVICE"
  --backend-device "$BACKEND_DEVICE"
  --disable-vis
)

if [[ "${DROID_ASYNCHRONOUS:-0}" == "1" ]]; then
  CMD+=( --asynchronous )
fi

if [[ "$VO_MODE" == "1" ]]; then
  CMD+=( --vo-mode )
fi

PYTHONPATH="$SUBTREE_ROOT/src:$REPO_DIR:$REPO_DIR/droid_slam${PYTHONPATH:+:$PYTHONPATH}" \
run_in_env "$ENV_DIR" "${CMD[@]}"

echo "DROID-SLAM results written to $CSV_PATH"
