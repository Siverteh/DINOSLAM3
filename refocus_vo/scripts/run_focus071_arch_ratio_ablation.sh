#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBTREE_ROOT="$(cd "$ROOT/.." && pwd)"
REPO_ROOT="$(cd "$SUBTREE_ROOT/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$SUBTREE_ROOT/.micromamba/envs/dpvo/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$SUBTREE_ROOT/runs/eval/tum_rgbd_freiburg123_arch_ratio_ablation_v1}"
DETACH=0
ARGS=()

for arg in "$@"; do
  if [[ "$arg" == "--detach" ]]; then
    DETACH=1
  else
    ARGS+=("$arg")
  fi
done

mkdir -p "$OUTPUT_ROOT"
LOG_PATH="$OUTPUT_ROOT/ratio_ablation.log"
PID_PATH="$OUTPUT_ROOT/ratio_ablation.pid"

if [[ "$DETACH" -eq 1 && "${RUN_DETACHED_CHILD:-0}" != "1" ]]; then
  if [[ -f "$PID_PATH" ]]; then
    EXISTING_PID="$(cat "$PID_PATH" 2>/dev/null || true)"
    if [[ -n "$EXISTING_PID" ]] && kill -0 "$EXISTING_PID" 2>/dev/null; then
      echo "Ratio ablation runner already active: PID $EXISTING_PID"
      echo "Log: $LOG_PATH"
      exit 0
    fi
  fi
  (
    cd "$REPO_ROOT"
    nohup setsid env \
      RUN_DETACHED_CHILD=1 \
      PYTHON_BIN="$PYTHON_BIN" \
      OUTPUT_ROOT="$OUTPUT_ROOT" \
      bash "$0" "${ARGS[@]}" >"$LOG_PATH" 2>&1 < /dev/null &
    echo $! >"$PID_PATH"
  )
  NEW_PID="$(cat "$PID_PATH")"
  echo "Detached Focus071 ratio ablation started."
  echo "Output root: $OUTPUT_ROOT"
  echo "Runner PID: $NEW_PID"
  echo "Watch log:"
  echo "tail -f $LOG_PATH"
  exit 0
fi

CMD=(
  "$PYTHON_BIN" -m refocus_vo.eval.focus071_arch_ratio_ablation
  --output-root "$OUTPUT_ROOT"
  "${ARGS[@]}"
)

cd "$REPO_ROOT"
export PYTHONPATH="$SUBTREE_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
"${CMD[@]}"
