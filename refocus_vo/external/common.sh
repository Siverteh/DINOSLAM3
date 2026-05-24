#!/usr/bin/env bash
set -euo pipefail

EXTERNAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBTREE_ROOT="$(cd "$EXTERNAL_ROOT/.." && pwd)"
REPO_ROOT="$(cd "$SUBTREE_ROOT/.." && pwd)"
REPOS_ROOT="${REPOS_ROOT:-$EXTERNAL_ROOT/repos}"
RUNS_ROOT_DEFAULT="${RUNS_ROOT:-$SUBTREE_ROOT/runs}"
MAMBA_ROOT="${MAMBA_ROOT:-$SUBTREE_ROOT/.micromamba}"
MAMBA_BIN="${MAMBA_BIN:-$MAMBA_ROOT/bin/micromamba}"

export MAMBA_ROOT_PREFIX="$MAMBA_ROOT"

ensure_micromamba() {
  if [[ ! -x "$MAMBA_BIN" ]]; then
    echo "ERROR: micromamba not found at $MAMBA_BIN. Run refocus_vo/external/bootstrap_micromamba.sh first." >&2
    exit 1
  fi
}

ensure_repo() {
  local name="$1"
  local url="$2"
  local dir="$REPOS_ROOT/$name"
  mkdir -p "$REPOS_ROOT"
  if [[ ! -d "$dir/.git" ]]; then
    git clone --depth 1 "$url" "$dir"
  fi
  printf '%s\n' "$dir"
}

ensure_env_dir() {
  local dir="$1"
  if [[ ! -d "$dir" ]]; then
    echo "ERROR: environment not found at $dir. Run the matching install script first." >&2
    exit 1
  fi
}

run_mamba() {
  ensure_micromamba
  "$MAMBA_BIN" "$@"
}

run_in_env() {
  local env_dir="$1"
  shift
  ensure_env_dir "$env_dir"
  run_mamba run -r "$MAMBA_ROOT" -p "$env_dir" "$@"
}
