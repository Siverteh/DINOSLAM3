#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT/common.sh"

if [[ ! -x "$MAMBA_BIN" ]]; then
  "$ROOT/bootstrap_micromamba.sh"
fi

REPO_DIR="$(ensure_repo "DROID-SLAM" "https://github.com/princeton-vl/DROID-SLAM.git")"
ENV_DIR="${DROID_ENV_DIR:-$MAMBA_ROOT/envs/droid-slam}"
TORCH_ARCH_LIST="${DROID_TORCH_CUDA_ARCH_LIST:-9.0}"
LIETORCH_ARCHS="${DROID_LIETORCH_CUDA_ARCHS:-90}"
MAX_JOBS="${DROID_MAX_JOBS:-8}"

git -C "$REPO_DIR" submodule update --init --recursive

if [[ ! -d "$ENV_DIR" ]]; then
  echo "[droid-slam] creating env at $ENV_DIR"
  run_mamba create -y -r "$MAMBA_ROOT" -p "$ENV_DIR" python=3.10 pip
fi

echo "[droid-slam] installing python dependencies"
run_in_env "$ENV_DIR" bash -lc "cd '$REPO_DIR' && pip install -r requirements.txt"
run_in_env "$ENV_DIR" bash -lc "cd '$REPO_DIR' && rm -rf build thirdparty/lietorch/build thirdparty/lietorch/*.egg-info && find . -maxdepth 3 -name '*.so' -delete"
run_in_env "$ENV_DIR" bash -lc "cd '$REPO_DIR' && TORCH_CUDA_ARCH_LIST='$TORCH_ARCH_LIST' LIETORCH_CUDA_ARCHS='$LIETORCH_ARCHS' MAX_JOBS='$MAX_JOBS' pip install --force-reinstall --no-deps --no-build-isolation thirdparty/lietorch"
run_in_env "$ENV_DIR" bash -lc "cd '$REPO_DIR' && TORCH_CUDA_ARCH_LIST='$TORCH_ARCH_LIST' MAX_JOBS='$MAX_JOBS' pip install --force-reinstall --no-deps --no-build-isolation thirdparty/pytorch_scatter"
run_in_env "$ENV_DIR" bash -lc "cd '$REPO_DIR' && TORCH_CUDA_ARCH_LIST='$TORCH_ARCH_LIST' MAX_JOBS='$MAX_JOBS' pip install --force-reinstall --no-deps --no-build-isolation -e ."

if [[ "${INSTALL_VIS_DEPS:-0}" == "1" ]]; then
  run_in_env "$ENV_DIR" bash -lc "cd '$REPO_DIR' && pip install moderngl moderngl-window"
fi

if [[ ! -f "$REPO_DIR/droid.pth" ]]; then
  echo "[droid-slam] downloading official weights"
  run_in_env "$ENV_DIR" bash -lc "cd '$REPO_DIR' && bash tools/download_model.sh"
fi

echo "DROID-SLAM installed."
echo "Repo: $REPO_DIR"
echo "Env:  $ENV_DIR"
