#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBTREE_ROOT="$(cd "$ROOT/.." && pwd)"
REPO_ROOT="$(cd "$SUBTREE_ROOT/.." && pwd)"
RAW_ROOT="${RAW_ROOT:-$SUBTREE_ROOT/data/tartanair_v2_raw}"
CONVERTED_ROOT="${CONVERTED_ROOT:-$SUBTREE_ROOT/data/tartanair_v2_converted}"
SUBSET_CONFIG="${SUBSET_CONFIG:-$SUBTREE_ROOT/configs/tartanair_subset_v1.yaml}"
SOURCE_API="${SOURCE_API:-auto}"

resolve_python() {
  local candidates=(
    "${PYTHON_BIN:-}"
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
    echo "$cand"
    return
  done
  echo "ERROR: Could not find a usable Python interpreter." >&2
  exit 1
}

PYTHON_BIN="$(resolve_python)"
mkdir -p "$RAW_ROOT" "$CONVERTED_ROOT"

export PYTHONPATH="$SUBTREE_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

"$PYTHON_BIN" -m refocus_vo.bootstrap_tartanair_subset \
  --raw-root "$RAW_ROOT" \
  --converted-root "$CONVERTED_ROOT" \
  --subset-config "$SUBSET_CONFIG" \
  --source-api "$SOURCE_API"

echo "TartanAir subset bootstrap complete under $CONVERTED_ROOT"
