#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
INSTALL_SYS_DEPS="${INSTALL_SYS_DEPS:-0}"
USE_LOCAL_VENV="${USE_LOCAL_VENV:-1}"
VENV_DIR="${VENV_DIR:-$ROOT/../.venv_pyslam_integration_v2}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "ERROR: PYTHON_BIN not found: $PYTHON_BIN" >&2
  exit 1
fi

if [[ "$USE_LOCAL_VENV" == "1" ]]; then
  if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    "$PYTHON_BIN" -m venv "$VENV_DIR"
  fi
  PYTHON_BIN="$VENV_DIR/bin/python"
fi

if [[ "$INSTALL_SYS_DEPS" == "1" ]] && command -v apt-get >/dev/null 2>&1; then
  SUDO=""
  if [[ "$(id -u)" -ne 0 ]]; then
    if command -v sudo >/dev/null 2>&1 && sudo -n true >/dev/null 2>&1; then
      SUDO="sudo"
    else
      echo "WARN: sudo without password is unavailable; skipping apt system dependency install"
    fi
  fi

  if [[ "$(id -u)" -eq 0 || -n "$SUDO" ]]; then
    echo "Installing minimal system dependencies"
    $SUDO apt-get update
    $SUDO DEBIAN_FRONTEND=noninteractive apt-get install -y \
      build-essential cmake pkg-config \
      libeigen3-dev libopencv-dev libgl1 libglib2.0-0 xvfb
  fi
fi

echo "Installing Python dependencies"
"$PYTHON_BIN" -m pip install --upgrade pip setuptools wheel
"$PYTHON_BIN" -m pip install \
  numpy scipy pyyaml matplotlib evo opencv-python-headless transformers timm

# Torch is required for DINOSLAM3 features.
if ! "$PYTHON_BIN" -c "import torch" >/dev/null 2>&1; then
  "$PYTHON_BIN" -m pip install torch
fi

echo "Installing minimal pyslam package"
"$PYTHON_BIN" -m pip install -e "$ROOT/pyslam"

echo "Building optional C++ components if present"
if [[ -x "$ROOT/pyslam/build_cpp_core.sh" ]]; then
  (cd "$ROOT/pyslam" && bash build_cpp_core.sh)
else
  echo "INFO: build_cpp_core.sh not present in minimal fork; skipping"
fi

if [[ -d "$ROOT/pyslam/cpp/ORBextractor" ]]; then
  if [[ -x "$ROOT/pyslam/cpp/ORBextractor/build.sh" ]]; then
    (cd "$ROOT/pyslam/cpp/ORBextractor" && bash build.sh)
  else
    echo "INFO: ORBextractor build script missing; skipping"
  fi
else
  echo "INFO: ORBextractor source not present; skipping"
fi

if [[ -d "$ROOT/pyslam/thirdparty/pangolin" ]]; then
  if [[ -x "$ROOT/pyslam/thirdparty/pangolin/build.sh" ]]; then
    (cd "$ROOT/pyslam/thirdparty/pangolin" && bash build.sh)
  else
    echo "INFO: Pangolin build script missing; skipping"
  fi
else
  echo "INFO: Pangolin source not present; skipping"
fi

if [[ -d "$ROOT/pyslam/thirdparty/gtsam" ]]; then
  if [[ -x "$ROOT/pyslam/thirdparty/gtsam/build.sh" ]]; then
    (cd "$ROOT/pyslam/thirdparty/gtsam" && bash build.sh)
  else
    echo "INFO: GTSAM build script missing; skipping"
  fi
else
  echo "INFO: GTSAM source not present; skipping"
fi

echo "Verifying import"
"$PYTHON_BIN" -c "import pyslam; print('import pyslam OK')"

echo "Checking core budgets"
"$PYTHON_BIN" "$ROOT/scripts/check_core_budget.py" --root "$ROOT" --max-core-py 49 --max-size-mb 500

echo "Setup completed"
