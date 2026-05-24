#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBTREE_ROOT="$(cd "$ROOT/.." && pwd)"

export DATA_PATH="${DATA_PATH:-$SUBTREE_ROOT/data/tartanair_v2_raw}"
export EVAL_DATA_PATH="${EVAL_DATA_PATH:-$SUBTREE_ROOT/data/tartanair_v2_converted}"
export SUBSET_CONFIG="${SUBSET_CONFIG:-$SUBTREE_ROOT/configs/tartanair_subset_raw_8env_x4_v1.yaml}"
export DINO_DPVO_CONFIG="${DINO_DPVO_CONFIG:-$SUBTREE_ROOT/configs/dino_dpvo_final_frontend_100cover_ft_v1.yaml}"
export DPVO_CONFIG_PATH="${DPVO_CONFIG_PATH:-$SUBTREE_ROOT/external/repos/DPVO/config/default.yaml}"
export TRAIN_RUN_ID="${TRAIN_RUN_ID:-dino_dpvo_final_frontend_100cover_ft_v1}"
export INIT_CHECKPOINT="${INIT_CHECKPOINT:-$SUBTREE_ROOT/runs/train/dino_dpvo_final_frontend_raw_v1/best.pt}"
export INIT_MODE="${INIT_MODE:-partial}"

"$SUBTREE_ROOT/scripts/train_dino_dpvo_frontend.sh"
