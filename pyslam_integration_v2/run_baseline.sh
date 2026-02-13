#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_PATH="${DATA_PATH:-$ROOT/../src/dino_slam3/data/tum_rgbd}"
USE_XVFB="${USE_XVFB:-1}"
MAX_DT="${MAX_DT:-0.02}"
NFEATURES="${NFEATURES:-1000}"
RUN_TIMEOUT_SECONDS="${RUN_TIMEOUT_SECONDS:-60}"

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
  if [[ -n "${PYTHON_BIN:-}" ]]; then
    local py="$PYTHON_BIN"
    if [[ "$py" == */* ]]; then
      py="$(cd "$(dirname "$py")" && pwd)/$(basename "$py")"
    else
      py="$(command -v "$py")"
    fi
    if [[ ! -x "$py" ]]; then
      echo "ERROR: PYTHON_BIN is not executable: $PYTHON_BIN" >&2
      exit 1
    fi
    if ! python_has_modules "$py" "$required_mods"; then
      echo "ERROR: PYTHON_BIN missing required modules: $required_mods" >&2
      exit 1
    fi
    echo "$py"
    return
  fi

  local candidates=(
    "$ROOT/../.venv/bin/python"
    "$ROOT/../.venv_pyslam_integration_v2/bin/python"
    "python3"
  )

  local cand=""
  for cand in "${candidates[@]}"; do
    if [[ "$cand" == */* ]]; then
      [[ -x "$cand" ]] || continue
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

PYTHON_BIN="$(resolve_python "yaml,ujson")"
RUN_PYSLAM_DIR="$ROOT/pyslam"
SETTINGS_DIR="$ROOT/pyslam/settings"
MAIN_ENTRY="$RUN_PYSLAM_DIR/main_slam.py"

if [[ ! -d "$RUN_PYSLAM_DIR" ]]; then
  echo "ERROR: v2 pySLAM dir not found: $RUN_PYSLAM_DIR" >&2
  exit 1
fi
if [[ ! -f "$MAIN_ENTRY" ]]; then
  echo "ERROR: v2 main_slam.py not found: $MAIN_ENTRY" >&2
  exit 1
fi

DATA_PATH="$(realpath "$DATA_PATH")"
if [[ ! -d "$DATA_PATH" ]]; then
  echo "ERROR: DATA_PATH not found: $DATA_PATH" >&2
  exit 1
fi

RESULTS_DIR="$ROOT/results/baseline"
TRAJ_DIR="$RESULTS_DIR/trajectories"
PLOTS_DIR="$RESULTS_DIR/plots"
LOG_DIR="$RESULTS_DIR/logs"
TMP_DIR="$RESULTS_DIR/tmp"
CSV_PATH="$RESULTS_DIR/metrics_summary.csv"

mkdir -p "$TRAJ_DIR" "$PLOTS_DIR" "$LOG_DIR" "$TMP_DIR"

TARGET_SEQUENCES=(
  "freiburg1_desk"
  "freiburg1_plant"
  "freiburg1_room"
  "freiburg3_long_office_household"
  "freiburg3_walking_static"
  "freiburg3_walking_xyz"
)

short_name() {
  case "$1" in
    freiburg1_desk) echo "f1_desk" ;;
    freiburg1_plant) echo "f1_plant" ;;
    freiburg1_room) echo "f1_room" ;;
    freiburg3_long_office_household) echo "f3_loh" ;;
    freiburg3_walking_static) echo "f3_wstatic" ;;
    freiburg3_walking_xyz) echo "f3_wxyz" ;;
    *) echo "$1" ;;
  esac
}

settings_file() {
  case "$1" in
    freiburg1_*) echo "TUM1.yaml" ;;
    freiburg2_*) echo "TUM2.yaml" ;;
    freiburg3_*) echo "TUM3.yaml" ;;
    *) echo "TUM1.yaml" ;;
  esac
}

normalize_seq() {
  local seq="$1"
  seq="${seq#rgbd_dataset_}"
  echo "$seq"
}

apply_sequence_filter() {
  local filtered=()
  local requested_raw=""
  if [[ -n "${SEQUENCE:-}" ]]; then
    requested_raw="$SEQUENCE"
  elif [[ -n "${SEQUENCES:-}" ]]; then
    requested_raw="$SEQUENCES"
  fi

  if [[ -z "$requested_raw" ]]; then
    printf '%s\n' "${TARGET_SEQUENCES[@]}"
    return
  fi

  local req=()
  IFS=',' read -r -a req <<< "$requested_raw"
  for item in "${req[@]}"; do
    item="$(normalize_seq "$item")"
    for seq in "${TARGET_SEQUENCES[@]}"; do
      if [[ "$seq" == "$item" ]]; then
        filtered+=("$seq")
      fi
    done
  done

  if [[ "${#filtered[@]}" -eq 0 ]]; then
    echo "ERROR: No valid sequences selected via SEQUENCE/SEQUENCES" >&2
    exit 1
  fi

  printf '%s\n' "${filtered[@]}"
}

run_headless() {
  if [[ "$USE_XVFB" != "0" ]] && command -v xvfb-run >/dev/null 2>&1; then
    xvfb-run -a "$@"
  else
    "$@"
  fi
}

append_row() {
  local sequence="$1"
  local feature_type="$2"
  local status="$3"
  local ate_rmse="$4"
  local ate_mean="$5"
  local ate_median="$6"
  local rpe_trans_rmse="$7"
  local rpe_rot_rmse="$8"
  printf "%s,%s,%s,%s,%s,%s,%s,%s\n" \
    "$sequence" "$feature_type" "$status" \
    "$ate_rmse" "$ate_mean" "$ate_median" "$rpe_trans_rmse" "$rpe_rot_rmse" >> "$CSV_PATH"
}

append_nan_row() {
  local sequence="$1"
  local feature_type="$2"
  local status="$3"
  append_row "$sequence" "$feature_type" "$status" "NaN" "NaN" "NaN" "NaN" "NaN"
}

echo "sequence,feature_type,status,ate_rmse,ate_mean,ate_median,rpe_trans_rmse,rpe_rot_rmse" > "$CSV_PATH"

mapfile -t ACTIVE_SEQUENCES < <(apply_sequence_filter)

for sequence in "${ACTIVE_SEQUENCES[@]}"; do
  short="$(short_name "$sequence")"
  full_name="rgbd_dataset_${sequence}"
  seq_folder="$DATA_PATH/$full_name"
  gt_file="$seq_folder/groundtruth.txt"
  log_file="$LOG_DIR/${short}.log"
  run_tmp="$TMP_DIR/$short"
  run_output="$run_tmp/run_output"
  run_cfg="$run_tmp/config.yaml"
  run_settings="$run_tmp/settings.yaml"
  out_traj="$TRAJ_DIR/${short}_trajectory.txt"
  plot_png="$PLOTS_DIR/$short/trajectory_3d.png"

  mkdir -p "$run_tmp" "$run_output" "$(dirname "$plot_png")"
  : > "$log_file"
  rm -f "$out_traj" "$plot_png"
  rm -f "$run_output"/*_online.txt "$run_output"/*_final.txt

  echo "[$sequence] start" | tee -a "$log_file"

  if [[ ! -d "$seq_folder" ]]; then
    echo "[$sequence] missing dataset folder -> skipped_missing_sequence" | tee -a "$log_file"
    append_nan_row "$sequence" "ORB" "skipped_missing_sequence"
    continue
  fi

  if [[ ! -f "$gt_file" ]]; then
    echo "[$sequence] missing groundtruth -> skipped_missing_groundtruth" | tee -a "$log_file"
    append_nan_row "$sequence" "ORB" "skipped_missing_groundtruth"
    continue
  fi

  assoc_file="$seq_folder/associations.txt"
  if [[ ! -f "$assoc_file" ]]; then
    rgb_file="$seq_folder/rgb.txt"
    depth_file="$seq_folder/depth.txt"
    if [[ ! -f "$rgb_file" || ! -f "$depth_file" ]]; then
      echo "[$sequence] missing rgb.txt/depth.txt -> tracking_failed" | tee -a "$log_file"
      append_nan_row "$sequence" "ORB" "tracking_failed"
      continue
    fi
    echo "[$sequence] generating associations.txt" | tee -a "$log_file"
    if ! "$PYTHON_BIN" "$ROOT/scripts/associate.py" "$rgb_file" "$depth_file" --output "$assoc_file" >> "$log_file" 2>&1; then
      echo "[$sequence] association generation failed" | tee -a "$log_file"
      append_nan_row "$sequence" "ORB" "tracking_failed"
      continue
    fi
  fi

  setting_template="$SETTINGS_DIR/$(settings_file "$sequence")"
  if [[ ! -f "$setting_template" ]]; then
    echo "[$sequence] missing settings template -> tracking_failed" | tee -a "$log_file"
    append_nan_row "$sequence" "ORB" "tracking_failed"
    continue
  fi

  "$PYTHON_BIN" - "$setting_template" "$run_settings" "$NFEATURES" <<'PY' >> "$log_file" 2>&1
import sys
import yaml

base_settings, out_settings, nfeatures = sys.argv[1], sys.argv[2], int(sys.argv[3])
with open(base_settings, "r", encoding="utf-8") as f:
    data = yaml.safe_load(f) or {}
data["FeatureTrackerConfig.name"] = "ORB2"
data["FeatureTrackerConfig.nFeatures"] = int(nfeatures)
with open(out_settings, "w", encoding="utf-8") as f:
    yaml.safe_dump(data, f, sort_keys=False)
PY

  cp "$RUN_PYSLAM_DIR/config.yaml" "$run_cfg"
  "$PYTHON_BIN" - "$run_cfg" "$DATA_PATH" "$full_name" "$run_settings" "$run_output" <<'PY' >> "$log_file" 2>&1
import sys
import yaml

cfg_path, data_path, sequence_name, settings_path, output_path = sys.argv[1:6]
with open(cfg_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}

def ensure_dict(d, key):
    if not isinstance(d.get(key), dict):
        d[key] = {}
    return d[key]

ensure_dict(cfg, "DATASET")["type"] = "TUM_DATASET"
tum = ensure_dict(cfg, "TUM_DATASET")
tum["type"] = "tum"
tum["sensor_type"] = "rgbd"
tum["base_path"] = data_path
tum["name"] = sequence_name
tum["settings"] = settings_path
tum["associations"] = "associations.txt"
tum["groundtruth_file"] = "auto"

save = ensure_dict(cfg, "SAVE_TRAJECTORY")
save["save_trajectory"] = True
save["format_type"] = "tum"
save["output_folder"] = output_path
save["basename"] = sequence_name

glob = ensure_dict(cfg, "GLOBAL_PARAMETERS")
glob["show_viewer"] = False

with open(cfg_path, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
PY

  run_cmd=("$PYTHON_BIN" "$MAIN_ENTRY" --config_path "$run_cfg" --headless --no_output_date)
  if [[ "$USE_XVFB" != "0" ]] && command -v xvfb-run >/dev/null 2>&1; then
    run_cmd=(xvfb-run -a "${run_cmd[@]}")
  fi

  set +e
  (
    cd "$RUN_PYSLAM_DIR"
    if [[ "$RUN_TIMEOUT_SECONDS" =~ ^[0-9]+$ ]] && [[ "$RUN_TIMEOUT_SECONDS" -gt 0 ]]; then
      timeout --signal=TERM --kill-after=5s "${RUN_TIMEOUT_SECONDS}s" "${run_cmd[@]}"
    else
      "${run_cmd[@]}"
    fi
  ) >> "$log_file" 2>&1
  run_rc=$?
  set -e

  raw_final="$run_output/${full_name}_final.txt"
  raw_online="$run_output/${full_name}_online.txt"
  if [[ -s "$raw_final" ]]; then
    cp "$raw_final" "$out_traj"
  elif [[ -s "$raw_online" ]]; then
    cp "$raw_online" "$out_traj"
  fi

  if [[ -f "$out_traj" ]]; then
    "$PYTHON_BIN" "$ROOT/scripts/plot_trajectory_3d.py" \
      --groundtruth "$gt_file" \
      --trajectory "$out_traj" \
      --title "$sequence - ORB" \
      --output "$plot_png" \
      --max-dt "$MAX_DT" >> "$log_file" 2>&1 || true
  fi

  if [[ $run_rc -eq 124 ]]; then
    echo "[$sequence] SLAM run timed out after ${RUN_TIMEOUT_SECONDS}s -> tracking_failed" | tee -a "$log_file"
    append_nan_row "$sequence" "ORB" "tracking_failed"
    continue
  fi

  if [[ $run_rc -ne 0 ]]; then
    echo "[$sequence] SLAM run failed (rc=$run_rc) -> tracking_failed" | tee -a "$log_file"
    append_nan_row "$sequence" "ORB" "tracking_failed"
    continue
  fi

  if [[ ! -f "$out_traj" ]]; then
    echo "[$sequence] trajectory missing -> tracking_failed" | tee -a "$log_file"
    append_nan_row "$sequence" "ORB" "tracking_failed"
    continue
  fi

  set +e
  "$PYTHON_BIN" "$ROOT/scripts/validate_trajectory.py" \
    --trajectory "$out_traj" \
    --groundtruth "$gt_file" \
    --max-dt "$MAX_DT" >> "$log_file" 2>&1
  val_rc=$?
  set -e

  if [[ $val_rc -ne 0 ]]; then
    echo "[$sequence] invalid trajectory" | tee -a "$log_file"
    append_nan_row "$sequence" "ORB" "invalid_trajectory"
    continue
  fi

  metrics_json="$run_tmp/metrics.json"
  set +e
  "$PYTHON_BIN" "$ROOT/scripts/compute_metrics.py" \
    --groundtruth "$gt_file" \
    --trajectory "$out_traj" \
    --max-dt "$MAX_DT" \
    --output-json "$metrics_json" >> "$log_file" 2>&1
  met_rc=$?
  set -e

  if [[ $met_rc -ne 0 || ! -f "$metrics_json" ]]; then
    echo "[$sequence] metrics failed" | tee -a "$log_file"
    append_nan_row "$sequence" "ORB" "invalid_trajectory"
    continue
  fi

  read -r ate_rmse ate_mean ate_median rpe_trans_rmse rpe_rot_rmse < <(
    "$PYTHON_BIN" - "$metrics_json" <<'PY'
import json
import math
import sys

with open(sys.argv[1], "r", encoding="utf-8") as f:
    d = json.load(f)
vals = [
    d.get("ate_rmse", math.nan),
    d.get("ate_mean", math.nan),
    d.get("ate_median", math.nan),
    d.get("rpe_trans_rmse", math.nan),
    d.get("rpe_rot_rmse", math.nan),
]
out = []
for v in vals:
    try:
        fv = float(v)
    except Exception:
        fv = math.nan
    out.append("NaN" if not math.isfinite(fv) else f"{fv:.6f}")
print(" ".join(out))
PY
  )

  append_row "$sequence" "ORB" "ok" "$ate_rmse" "$ate_mean" "$ate_median" "$rpe_trans_rmse" "$rpe_rot_rmse"
  echo "[$sequence] done status=ok" | tee -a "$log_file"
done

echo "Baseline evaluation complete"
echo "backend: v2_standalone"
echo "pyslam_dir: $RUN_PYSLAM_DIR"
echo "python: $PYTHON_BIN"
echo "metrics_summary: $CSV_PATH"
