#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT/common.sh"

if [[ ! -x "$MAMBA_BIN" ]]; then
  "$ROOT/bootstrap_micromamba.sh"
fi

REPO_DIR="$(ensure_repo "DPVO" "https://github.com/princeton-vl/DPVO.git")"
ENV_DIR="${DPVO_ENV_DIR:-$MAMBA_ROOT/envs/dpvo}"

git -C "$REPO_DIR" submodule update --init --recursive

if [[ ! -d "$ENV_DIR" ]]; then
  echo "[dpvo] creating env at $ENV_DIR"
  run_mamba env create -y -r "$MAMBA_ROOT" -p "$ENV_DIR" -f "$REPO_DIR/environment.yml"
fi

if [[ ! -d "$REPO_DIR/thirdparty/eigen-3.4.0" ]]; then
  echo "[dpvo] installing Eigen dependency"
  run_in_env "$ENV_DIR" python - <<PY
from pathlib import Path
from urllib.request import urlretrieve
import zipfile

repo = Path(r"$REPO_DIR")
archive = repo / "eigen-3.4.0.zip"
target = repo / "thirdparty"
target.mkdir(parents=True, exist_ok=True)
if not archive.exists():
    urlretrieve("https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip", archive)
with zipfile.ZipFile(archive, "r") as zf:
    zf.extractall(target)
PY
fi

echo "[dpvo] installing python package"
run_in_env "$ENV_DIR" bash -lc "cd '$REPO_DIR' && pip install --no-build-isolation ."

if [[ ! -f "$REPO_DIR/dpvo.pth" ]]; then
  echo "[dpvo] downloading official weights"
  run_in_env "$ENV_DIR" python - <<PY
from pathlib import Path
from urllib.request import Request, urlopen
import zipfile
import shutil

repo = Path(r"$REPO_DIR")
archive = repo / "models.zip"
url = "https://www.dropbox.com/s/nap0u8zslspdwm4/models.zip?dl=1"

def download_zip(dst: Path) -> None:
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req) as resp, dst.open("wb") as f:
        shutil.copyfileobj(resp, f)

if (not archive.exists()) or (not zipfile.is_zipfile(archive)):
    if archive.exists():
        archive.unlink()
    download_zip(archive)

if not zipfile.is_zipfile(archive):
    raise RuntimeError(f"Downloaded DPVO weights archive is not a valid zip: {archive}")

with zipfile.ZipFile(archive, "r") as zf:
    zf.extractall(repo)
PY
fi

echo "DPVO installed."
echo "Repo: $REPO_DIR"
echo "Env:  $ENV_DIR"
