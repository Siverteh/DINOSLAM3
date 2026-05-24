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
ABLATION_ROOT="${ABLATION_ROOT:-$SUBTREE_ROOT/runs/eval/tum_rgbd_freiburg123_arch_ratio_ablation_v1}"
SCREENING_SUMMARY_PATH="${SCREENING_SUMMARY_PATH:-$ABLATION_ROOT/screening_summary.csv}"
BASELINE_PER_SEQUENCE_PATH="${BASELINE_PER_SEQUENCE_PATH:-$SUBTREE_ROOT/runs/eval/tum_rgbd_freiburg123_dpvo_vs_focus071_v1_rerun6/summary/per_sequence_median.csv}"
HISTORICAL_METHOD_COMPARISON_PATH="${HISTORICAL_METHOD_COMPARISON_PATH:-$SUBTREE_ROOT/runs/eval/tum_rgbd_freiburg123_dpvo_vs_focus071_v1_rerun6/summary/method_comparison.csv}"
HISTORICAL_REPEAT_SUMMARY_PATH="${HISTORICAL_REPEAT_SUMMARY_PATH:-$SUBTREE_ROOT/runs/eval/tum_rgbd_freiburg123_dpvo_vs_focus071_v1_rerun6/summary/repeat_summary.csv}"
REPEATS="${REPEATS:-5}"
DRY_RUN="${DRY_RUN:-0}"
DETACH=0
ARGS=()

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

for arg in "$@"; do
  if [[ "$arg" == "--detach" ]]; then
    DETACH=1
  else
    ARGS+=("$arg")
  fi
done

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
if [[ ! -f "$SCREENING_SUMMARY_PATH" ]]; then
  echo "ERROR: Screening summary not found at $SCREENING_SUMMARY_PATH" >&2
  exit 1
fi
if [[ ! -f "$BASELINE_PER_SEQUENCE_PATH" ]]; then
  echo "ERROR: Baseline per-sequence summary not found at $BASELINE_PER_SEQUENCE_PATH" >&2
  exit 1
fi
if [[ ! -f "$HISTORICAL_METHOD_COMPARISON_PATH" ]]; then
  echo "ERROR: Historical method comparison not found at $HISTORICAL_METHOD_COMPARISON_PATH" >&2
  exit 1
fi
if [[ ! -f "$HISTORICAL_REPEAT_SUMMARY_PATH" ]]; then
  echo "ERROR: Historical repeat summary not found at $HISTORICAL_REPEAT_SUMMARY_PATH" >&2
  exit 1
fi

PYTHON_BIN="$(resolve_python "yaml,cv2,torch,scipy,matplotlib,evo,torch_scatter")"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$SUBTREE_ROOT/src:$DPVO_REPO_DIR:$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

OUTPUT_ROOT="${OUTPUT_ROOT:-$RUNS_ROOT/eval/tum_rgbd_freiburg123_arch_dual_finalists_5x_v1}"
mkdir -p "$OUTPUT_ROOT"
LOG_PATH="$OUTPUT_ROOT/dual_finalists_5x.log"
PID_PATH="$OUTPUT_ROOT/dual_finalists_5x.pid"

if [[ "$DETACH" -eq 1 && "${RUN_DETACHED_CHILD:-0}" != "1" ]]; then
  if [[ -f "$PID_PATH" ]]; then
    EXISTING_PID="$(cat "$PID_PATH" 2>/dev/null || true)"
    if [[ -n "$EXISTING_PID" ]] && kill -0 "$EXISTING_PID" 2>/dev/null; then
      echo "Dual-finalists 5x runner already active: PID $EXISTING_PID"
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
  echo "Detached dual-finalists Freiburg 5x started."
  echo "Output root: $OUTPUT_ROOT"
  echo "Runner PID: $NEW_PID"
  echo "Watch log:"
  echo "tail -f $LOG_PATH"
  exit 0
fi

CMD=(
  "$PYTHON_BIN" -m refocus_vo.eval.focus071_arch_dual_finalists_5x
  --dataset-root "$DATASET_ROOT"
  --dpvo-root "$DPVO_REPO_DIR"
  --dpvo-weights "$DPVO_WEIGHTS_PATH"
  --dpvo-config "$DPVO_CONFIG_PATH"
  --ablation-root "$ABLATION_ROOT"
  --screening-summary "$SCREENING_SUMMARY_PATH"
  --baseline-per-sequence "$BASELINE_PER_SEQUENCE_PATH"
  --historical-method-comparison "$HISTORICAL_METHOD_COMPARISON_PATH"
  --historical-repeat-summary "$HISTORICAL_REPEAT_SUMMARY_PATH"
  --output-root "$OUTPUT_ROOT"
  --repeats "$REPEATS"
)

if [[ "$DRY_RUN" == "1" ]]; then
  CMD+=( --dry-run )
fi
if [[ "${#ARGS[@]}" -gt 0 ]]; then
  CMD+=( "${ARGS[@]}" )
fi

cd "$REPO_ROOT"
echo "[arch_dual_finalists_5x] output_root=$OUTPUT_ROOT"
"${CMD[@]}"
