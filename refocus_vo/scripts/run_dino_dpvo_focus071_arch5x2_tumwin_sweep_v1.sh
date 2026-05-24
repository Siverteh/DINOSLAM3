#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBTREE_ROOT="$(cd "$ROOT/.." && pwd)"
REPO_ROOT="$(cd "$SUBTREE_ROOT/.." && pwd)"
MANIFEST_PATH="${MANIFEST_PATH:-$SUBTREE_ROOT/configs/dino_dpvo_focus071_arch5x2_tumwin_sweep_v1.yaml}"
BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-$SUBTREE_ROOT/runs/sweeps/dino_dpvo_focus071_arch5x2_tumwin_sweep_v1}"
PYTHON_BIN="${PYTHON_BIN:-$SUBTREE_ROOT/.micromamba/envs/dpvo/bin/python}"
DETACH=0
ARGS=()

for arg in "$@"; do
  if [[ "$arg" == "--detach" ]]; then
    DETACH=1
  else
    ARGS+=("$arg")
  fi
done

mkdir -p "$BASE_OUTPUT_DIR"
LOG_PATH="$BASE_OUTPUT_DIR/sweep.log"
PID_PATH="$BASE_OUTPUT_DIR/runner.pid"

if [[ "$DETACH" -eq 1 && "${RUN_DETACHED_CHILD:-0}" != "1" ]]; then
  if [[ -f "$PID_PATH" ]]; then
    EXISTING_PID="$(cat "$PID_PATH" 2>/dev/null || true)"
    if [[ -n "$EXISTING_PID" ]] && kill -0 "$EXISTING_PID" 2>/dev/null; then
      echo "Sweep runner already active: PID $EXISTING_PID"
      echo "Log: $LOG_PATH"
      echo "Leaderboard: $BASE_OUTPUT_DIR/leaderboard_dev.csv"
      exit 0
    fi
  fi
  (
    cd "$REPO_ROOT"
    nohup setsid env \
      RUN_DETACHED_CHILD=1 \
      MANIFEST_PATH="$MANIFEST_PATH" \
      BASE_OUTPUT_DIR="$BASE_OUTPUT_DIR" \
      PYTHON_BIN="$PYTHON_BIN" \
      bash "$0" "${ARGS[@]}" >"$LOG_PATH" 2>&1 < /dev/null &
    echo $! >"$PID_PATH"
  )
  NEW_PID="$(cat "$PID_PATH")"
  echo "Detached Focus071 architecture sweep started."
  echo "Sweep root: $BASE_OUTPUT_DIR"
  echo "Runner PID: $NEW_PID"
  echo "Watch log:"
  echo "tail -f $LOG_PATH"
  echo "Watch leaderboard:"
  echo "tail -f $BASE_OUTPUT_DIR/leaderboard_dev.csv"
  exit 0
fi

CMD=(
  "$PYTHON_BIN" -m refocus_vo.sweeps.run_assoc9_sweep
  --manifest "$MANIFEST_PATH"
  --base-output-dir "$BASE_OUTPUT_DIR"
  "${ARGS[@]}"
)

cd "$REPO_ROOT"
export PYTHONPATH="$SUBTREE_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
"${CMD[@]}"
