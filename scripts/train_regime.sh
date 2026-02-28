#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
STAGEA_CFG="${STAGEA_CFG:-$ROOT/configs/train/tum_regime_stageA_external.yaml}"
STAGEB_CFG="${STAGEB_CFG:-$ROOT/configs/train/tum_regime_stageB_tum_short.yaml}"
STAGEC_CFG="${STAGEC_CFG:-$ROOT/configs/train/tum_regime_stageC_tum_mixed.yaml}"
STAGED_CFG="${STAGED_CFG:-$ROOT/configs/train/tum_regime_stageD_reliability.yaml}"
STAGEA_ENABLE="${STAGEA_ENABLE:-auto}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "ERROR: python not found/executable: $PYTHON_BIN" >&2
  exit 1
fi
if [[ ! -f "$STAGEA_CFG" ]]; then
  echo "ERROR: stageA config not found: $STAGEA_CFG" >&2
  exit 1
fi
if [[ ! -f "$STAGEB_CFG" ]]; then
  echo "ERROR: stageB config not found: $STAGEB_CFG" >&2
  exit 1
fi
if [[ ! -f "$STAGEC_CFG" ]]; then
  echo "ERROR: stageC config not found: $STAGEC_CFG" >&2
  exit 1
fi
if [[ ! -f "$STAGED_CFG" ]]; then
  echo "ERROR: stageD config not found: $STAGED_CFG" >&2
  exit 1
fi

export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

cfg_run_name() {
  "$PYTHON_BIN" - "$1" <<'PY'
import sys
from dino_slam3.utils.config import load_config
cfg = load_config(sys.argv[1])
print(cfg["run"]["name"])
PY
}

cfg_has_all_manifests() {
  "$PYTHON_BIN" - "$1" <<'PY'
import sys
from pathlib import Path
from dino_slam3.utils.config import load_config
cfg = load_config(sys.argv[1])
manifests = cfg.get("dataset", {}).get("train_manifests", []) or []
ok = bool(manifests) and all(Path(m).expanduser().exists() for m in manifests)
print("1" if ok else "0")
PY
}

echo "[regime] stageA config: $STAGEA_CFG"
echo "[regime] stageB config: $STAGEB_CFG"
echo "[regime] stageC config: $STAGEC_CFG"
echo "[regime] stageD config: $STAGED_CFG"
echo "[regime] python: $PYTHON_BIN"
echo "[regime] stageA enable: $STAGEA_ENABLE"

stagea_run_name="$(cfg_run_name "$STAGEA_CFG")"
stagea_best="$ROOT/runs/$stagea_run_name/checkpoints/best.pt"

start_ckpt=""
do_stagea=0
if [[ "$STAGEA_ENABLE" == "1" || "$STAGEA_ENABLE" == "true" || "$STAGEA_ENABLE" == "yes" ]]; then
  do_stagea=1
elif [[ "$STAGEA_ENABLE" == "auto" ]]; then
  if [[ "$(cfg_has_all_manifests "$STAGEA_CFG")" == "1" ]]; then
    do_stagea=1
  fi
fi

if [[ "$do_stagea" == "1" ]]; then
  echo "[regime] training stageA external pretrain..."
  "$PYTHON_BIN" -m dino_slam3.scripts.train --config "$STAGEA_CFG"
  if [[ -f "$stagea_best" ]]; then
    start_ckpt="$stagea_best"
    echo "[regime] stageA best: $stagea_best"
  else
    # fallback to last epoch checkpoint when selection metric is unavailable
    stagea_last="$(ls -1 "$ROOT/runs/$stagea_run_name/checkpoints"/epoch_*.pt 2>/dev/null | sort | tail -n 1 || true)"
    if [[ -n "$stagea_last" ]]; then
      start_ckpt="$stagea_last"
      echo "[regime] stageA fallback ckpt: $stagea_last"
    else
      echo "ERROR: stageA produced no checkpoint in runs/$stagea_run_name/checkpoints" >&2
      exit 1
    fi
  fi
else
  echo "[regime] skipping stageA (missing manifests or disabled)"
fi

stageb_run_name="$(cfg_run_name "$STAGEB_CFG")"
stageb_best="$ROOT/runs/$stageb_run_name/checkpoints/best.pt"
echo "[regime] training stageB TUM short..."
if [[ -n "$start_ckpt" ]]; then
  "$PYTHON_BIN" -m dino_slam3.scripts.train --config "$STAGEB_CFG" --init-ckpt "$start_ckpt"
else
  "$PYTHON_BIN" -m dino_slam3.scripts.train --config "$STAGEB_CFG"
fi
if [[ ! -f "$stageb_best" ]]; then
  stageb_last="$(ls -1 "$ROOT/runs/$stageb_run_name/checkpoints"/epoch_*.pt 2>/dev/null | sort | tail -n 1 || true)"
  if [[ -n "$stageb_last" ]]; then
    stageb_best="$stageb_last"
    echo "[regime] stageB fallback ckpt: $stageb_last"
  else
    echo "ERROR: stageB best checkpoint not found and no epoch fallback: $stageb_best" >&2
    exit 1
  fi
fi
echo "[regime] stageB best: $stageb_best"

stagec_run_name="$(cfg_run_name "$STAGEC_CFG")"
stagec_best="$ROOT/runs/$stagec_run_name/checkpoints/best.pt"
echo "[regime] training stageC TUM mixed curriculum..."
"$PYTHON_BIN" -m dino_slam3.scripts.train --config "$STAGEC_CFG" --init-ckpt "$stageb_best"
if [[ ! -f "$stagec_best" ]]; then
  stagec_last="$(ls -1 "$ROOT/runs/$stagec_run_name/checkpoints"/epoch_*.pt 2>/dev/null | sort | tail -n 1 || true)"
  if [[ -n "$stagec_last" ]]; then
    stagec_best="$stagec_last"
    echo "[regime] stageC fallback ckpt: $stagec_last"
  else
    echo "ERROR: stageC best checkpoint not found and no epoch fallback: $stagec_best" >&2
    exit 1
  fi
fi
echo "[regime] stageC best: $stagec_best"

staged_run_name="$(cfg_run_name "$STAGED_CFG")"
staged_best="$ROOT/runs/$staged_run_name/checkpoints/best.pt"
staged_rel_best="$ROOT/runs/$staged_run_name/checkpoints/rel_best.pt"
echo "[regime] training stageD reliability..."
"$PYTHON_BIN" -m dino_slam3.scripts.train --config "$STAGED_CFG" --init-ckpt "$stagec_best"

if [[ ! -f "$staged_best" && ! -f "$staged_rel_best" ]]; then
  echo "ERROR: stageD did not produce best.pt or rel_best.pt" >&2
  exit 1
fi
echo "[regime] stageD best: $staged_best"
if [[ -f "$staged_rel_best" ]]; then
  echo "[regime] stageD rel_best: $staged_rel_best"
fi
echo "[regime] stageC geom_best: $ROOT/runs/$stagec_run_name/checkpoints/geom_best.pt"
echo "[regime] done"
