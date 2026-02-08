#!/bin/bash
# Setup script for pySLAM baseline (ORB) for DINOSLAM3 workspace
# Adds:
#  - Feature types repair (indent corruption + stray DINOSLAM3 lines)
#  - AVX helper fix for cpp_core build
#  - cpp_core import fix via site-packages .pth

set -euo pipefail

THIS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# If you *really* want /workspace/pyslam_integration no matter where you run from:
if [ "${FORCE_WORKSPACE_ROOT:-0}" = "1" ]; then
  BASE_DIR="/workspace/pyslam_integration"
else
  BASE_DIR="$THIS_DIR"
fi

PYSLAM_DIR="$BASE_DIR/pyslam"

echo "================================================"
echo "Setting up pySLAM baseline (ORB) + DINOSLAM3 integration hooks"
echo "================================================"
echo "setup.sh dir: $THIS_DIR"
echo "base dir:     $BASE_DIR"
echo "pySLAM dir:   $PYSLAM_DIR"
echo ""

mkdir -p "$BASE_DIR"

# 1) Clone/update pySLAM
if [ ! -d "$PYSLAM_DIR/.git" ]; then
  if [ -d "$PYSLAM_DIR" ] && [ "$(ls -A "$PYSLAM_DIR" 2>/dev/null | wc -l)" -gt 0 ]; then
    echo "✗ $PYSLAM_DIR exists but is not a git repo and not empty." >&2
    echo "  Move it away or delete it, then re-run." >&2
    exit 1
  fi
  echo "Cloning pySLAM into $PYSLAM_DIR ..."
  git clone --recursive https://github.com/luigifreda/pyslam.git "$PYSLAM_DIR"
else
  echo "pySLAM already exists, updating..."
  cd "$PYSLAM_DIR"
  git pull
  git submodule update --init --recursive
fi

cd "$PYSLAM_DIR"

# 2) Patch pyproject.toml (optional)
if [ -f "pyproject.toml" ]; then
  echo ""
  echo "Patching pyproject.toml..."
  sed -i 's/requires-python = ">=3\.11\.9"/requires-python = ">=3.10.0"/' pyproject.toml || true
  sed -i 's/requires-python = ">=3\.11"/requires-python = ">=3.10.0"/' pyproject.toml || true
  sed -i '/"onnxruntime>=1\.22\.0"/d' pyproject.toml || true
  sed -i '/"open3d"/d' pyproject.toml || true
  sed -i '/"pyqt5"/d' pyproject.toml || true
fi

# 3) System deps
echo ""
echo "Installing system dependencies..."
if command -v apt-get >/dev/null 2>&1; then
  apt-get update -y
  apt-get install -y \
    build-essential cmake git pkg-config \
    libeigen3-dev libopencv-dev \
    nlohmann-json3-dev \
    libgl1-mesa-dev libglu1-mesa-dev \
    libx11-dev libxi-dev libxmu-dev libxrandr-dev \
    libglew-dev
fi

# Some builds expect /usr/bin/python
if [ ! -e /usr/bin/python ] && [ -x /usr/bin/python3 ]; then
  ln -s /usr/bin/python3 /usr/bin/python || true
fi

# 4) Python deps
echo ""
echo "Installing Python dependencies..."
python3 -m pip install --upgrade pip setuptools wheel build
python3 -m pip uninstall -y pyflann || true

python3 -m pip install \
  "numpy<2" opencv-python matplotlib scipy pyyaml pillow tqdm \
  kornia==0.7.3 gdown hjson ujson timm evo trimesh munch plyfile \
  glfw PyOpenGL PyGLM rich ruff configargparse numba \
  scikit-learn scikit-image rerun-sdk pyflann-py3 faiss-cpu \
  ordered-set

echo ""
echo "Installing pySLAM in editable mode..."
python3 -m pip install --no-deps -e .

# ============================================================
# FIX 1: Patch AVX helper for cpp_core build (hsum256_ps)
# ============================================================
echo ""
echo "Patching AVX helper for cpp_core build (hsum256_ps)..."
python3 - <<'PY'
from pathlib import Path

path = Path("pyslam/slam/cpp/utils/descriptor_helpers.h")
if not path.exists():
    print("[WARN] descriptor_helpers.h not found at", path)
    raise SystemExit(0)

text = path.read_text()
marker = "#elif defined(__AVX2__)"

if "#if defined(__AVX512F__)" in text and marker in text:
    avx512_block = text.split(marker)[0]
    if "hsum256_ps(__m256 v)" not in avx512_block:
        insertion = """
static inline float hsum256_ps(__m256 v) noexcept {
    __m128 low = _mm256_castps256_ps128(v);
    __m128 high = _mm256_extractf128_ps(v, 1);
    __m128 sum = _mm_add_ps(low, high);
    __m128 shuf = _mm_movehdup_ps(sum);  // (sum3,sum3,sum1,sum1)
    __m128 sums = _mm_add_ps(sum, shuf); // (s3+s2, s3+s2, s1+s0, s1+s0)
    shuf = _mm_movehl_ps(shuf, sums);    // (   ,    , s3+s2,   )
    sums = _mm_add_ss(sums, shuf);       // s3+s2+s1+s0
    return _mm_cvtss_f32(sums);
}
"""
        text = text.replace(marker, insertion + marker, 1)
        path.write_text(text)
        print("[OK] Inserted hsum256_ps into", path)
    else:
        print("[OK] hsum256_ps already present in AVX512 block")
else:
    print("[OK] No AVX512/AVX2 marker pattern found (nothing to patch)")
PY

# ============================================================
# Build dependencies: GTSAM + cpp_core + other thirdparty
# ============================================================
echo ""
echo "Building GTSAM..."
./scripts/install_gtsam.sh

echo ""
echo "Building pySLAM C++ core (cpp_core)..."
./build_cpp_core.sh

echo ""
echo "Building pySLAM cpp module..."
mkdir -p "$PYSLAM_DIR/cpp/build"
cd "$PYSLAM_DIR/cpp/build"
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j"$(nproc)"

echo ""
echo "Building thirdparty components..."
cd "$PYSLAM_DIR/thirdparty/orbslam2_features" && ./build.sh
cd "$PYSLAM_DIR/thirdparty/pangolin" && ./build.sh
cd "$PYSLAM_DIR/thirdparty/g2opy" && sed -i 's/sudo //g' build.sh 2>/dev/null || true && ./build.sh
cd "$PYSLAM_DIR/thirdparty/pydbow3" && ./build.sh
cd "$PYSLAM_DIR/thirdparty/pyibow" && ./build.sh

# ============================================================
# FIX 2: Link cpp libs into python package (existing behavior)
# ============================================================
echo ""
echo "Linking cpp libs into python package..."
CPP_BUILT_LIB="$PYSLAM_DIR/cpp/lib"
CPP_PKG_LIB="$PYSLAM_DIR/pyslam/slam/cpp/lib"
mkdir -p "$(dirname "$CPP_PKG_LIB")"
if [ -d "$CPP_PKG_LIB" ] && [ ! -L "$CPP_PKG_LIB" ]; then
  rm -rf "$CPP_PKG_LIB"
fi
ln -sfn "$CPP_BUILT_LIB" "$CPP_PKG_LIB"

# ============================================================
# FIX 3: Make `import cpp_core` work everywhere (.pth file)
# ============================================================
echo ""
echo "Ensuring cpp_core is importable (writing .pth into site-packages)..."
python3 - <<'PY'
import site
from pathlib import Path

libdir = "/workspace/pyslam_integration/pyslam/pyslam/slam/cpp/lib"
# If you didn't force workspace root, resolve libdir dynamically:
# but your run scripts are using /workspace/pyslam_integration anyway.
try:
    sp = site.getsitepackages()[0]
except Exception:
    # fallback for some envs
    sp = site.getusersitepackages()

pth = Path(sp) / "pyslam_cpp_core.pth"
pth.write_text(libdir + "\n")
print("[OK] wrote", str(pth), "->", libdir)
PY

# ============================================================
# FIX 4: Repair feature_types.py if it got indentation-corrupted
# ============================================================
echo ""
echo "Repairing pyslam/local_features/feature_types.py indentation (safety fix)..."
python3 - <<'PY'
from pathlib import Path
import re

p = Path("/workspace/pyslam_integration/pyslam/pyslam/local_features/feature_types.py")
if not p.exists():
    # if not forced workspace root, try local relative
    p = Path("pyslam/local_features/feature_types.py")

txt = p.read_text()
bak = p.with_suffix(".py.bak_feature_types_repair")
bak.write_text(txt)

lines = txt.splitlines(True)

# Remove any stray DINOSLAM3 mapping lines anywhere (these are frequently inserted wrongly by patchers).
out = []
for ln in lines:
    s = ln.strip()
    if "FeatureDescriptorTypes.DINOSLAM3" in s and ("norm_type[" in s or "max_descriptor_distance[" in s):
        continue
    if s in {"# DINOSLAM3", "#DINOSLAM3"}:
        continue
    out.append(ln)
txt2 = "".join(out)

m = re.search(r'^\s*class\s+FeatureInfo\b.*?:\s*$', txt2, flags=re.M)
if not m:
    print("[WARN] Could not find class FeatureInfo; leaving file unchanged.")
    raise SystemExit(0)

# FeatureInfo usually runs to EOF; still handle "next top-level class" if present.
m_next = re.search(r'^\S.*\bclass\s+\w+.*?:\s*$', txt2[m.end():], flags=re.M)
end = len(txt2) if not m_next else m.end() + m_next.start()

before = txt2[:m.end()]
block  = txt2[m.end():end]
after  = txt2[end:]

fixed = []
for ln in block.splitlines(True):
    if ln.strip() == "":
        fixed.append(ln)
        continue
    # If accidentally at col 0 but should be inside FeatureInfo:
    if not ln.startswith("    "):
        stripped = ln.lstrip()
        if stripped.startswith(("norm_type[", "max_descriptor_distance[", "#")):
            fixed.append("    " + stripped)
            continue
    fixed.append(ln)

txt3 = before + "".join(fixed) + after

# Normalize weird 2-space comment lines that sometimes end up inside enums.
txt3 = re.sub(r'^(  #)', r'    #', txt3, flags=re.M)

p.write_text(txt3)
print("[OK] Repaired:", p)
print("[OK] Backup:  ", bak)

# Quick sanity import (will throw if still broken)
import importlib.util
spec = importlib.util.spec_from_file_location("feature_types", str(p))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
print("[OK] feature_types parses")
PY

echo ""
echo "================================================"
echo "✓ pySLAM Setup Complete!"
echo "================================================"
echo "base dir:   $BASE_DIR"
echo "pySLAM dir: $PYSLAM_DIR"
echo ""
echo "Notes:"
echo " - cpp_core should now be importable from anywhere (via pyslam_cpp_core.pth)."
echo " - feature_types.py will be auto-repaired if a patcher messes indentation."
