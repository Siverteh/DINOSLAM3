#!/bin/bash
# Run pySLAM with built-in (ORB) features on TUM RGB-D sequences

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PYSLAM_DIR="$SCRIPT_DIR/pyslam"
DATA_PATH="/workspace/src/dino_slam3/data/tum_rgbd"
OUTPUT_PATH="$SCRIPT_DIR/results/baseline"
GTSAM_LIB_DIR="$PYSLAM_DIR/thirdparty/gtsam_local/install/lib"
EXECUTABLE="$PYTHON_BIN $PYSLAM_DIR/main_slam.py"
ASSOC_SCRIPT="$SCRIPT_DIR/associate.py"

PYTHON_CANDIDATES=(
    "/workspace/DINOSLAM3/.venv/bin/python"
    "/workspace/.venv/bin/python"
    "python3"
)

PYTHON_BIN=""
for candidate in "${PYTHON_CANDIDATES[@]}"; do
    if [ -x "$candidate" ] || [ "$candidate" = "python3" ]; then
        if "$candidate" - <<'PY' >/dev/null 2>&1
try:
    import yaml  # noqa: F401
    print("ok")
except Exception:
    raise SystemExit(1)
PY
        then
            PYTHON_BIN="$candidate"
            break
        fi
    fi
done

if [ -z "$PYTHON_BIN" ]; then
    echo "✗ No Python interpreter with PyYAML found. Install pyyaml in your active env." >&2
    exit 1
fi

# Ensure C++ libs are discoverable by the Python package
CPP_BUILT_LIB="$PYSLAM_DIR/cpp/lib"
CPP_PKG_LIB="$PYSLAM_DIR/pyslam/slam/cpp/lib"
mkdir -p "$(dirname "$CPP_PKG_LIB")"
if [ -d "$CPP_PKG_LIB" ] && [ ! -L "$CPP_PKG_LIB" ]; then
    rm -rf "$CPP_PKG_LIB"
fi
ln -sfn "$CPP_BUILT_LIB" "$CPP_PKG_LIB"

# Ensure GTSAM shared libraries are discoverable (libmetis-gtsam.so, libgtsam.so)
if [ -d "$GTSAM_LIB_DIR" ]; then
    export LD_LIBRARY_PATH="$GTSAM_LIB_DIR:${LD_LIBRARY_PATH:-}"
fi

mkdir -p "$OUTPUT_PATH/trajectories" "$OUTPUT_PATH/logs" "$OUTPUT_PATH/plots"

SEQUENCES=(
    "rgbd_dataset_freiburg1_desk"
    "rgbd_dataset_freiburg1_plant"
    "rgbd_dataset_freiburg1_room"
    "rgbd_dataset_freiburg3_long_office_household"
    "rgbd_dataset_freiburg3_walking_static"
    "rgbd_dataset_freiburg3_walking_xyz"
)

echo "================================================"
echo "Running pySLAM with Built-in Features (ORB)"
echo "================================================"
echo "Dataset:     $DATA_PATH"
echo "Output:      $OUTPUT_PATH"
echo ""

if ! command -v xvfb-run &> /dev/null; then
    echo "Installing xvfb for headless operation..."
    apt-get update -qq && apt-get install -y xvfb
fi

for SEQ in "${SEQUENCES[@]}"; do
    echo ""
    echo "Processing: $SEQ"
    echo "----------------------------------------"

    ASSOC_FILE="$DATA_PATH/$SEQ/associations.txt"
    if [ ! -f "$ASSOC_FILE" ]; then
        echo "⚠ Association file not found, generating..."
        if [ -f "$DATA_PATH/$SEQ/rgb.txt" ] && [ -f "$DATA_PATH/$SEQ/depth.txt" ]; then
            "$PYTHON_BIN" "$ASSOC_SCRIPT" \
                "$DATA_PATH/$SEQ/rgb.txt" \
                "$DATA_PATH/$SEQ/depth.txt" \
                --output "$ASSOC_FILE"
            echo "✓ Generated association file"
        else
            echo "✗ Cannot find rgb.txt or depth.txt in $DATA_PATH/$SEQ"
            continue
        fi
    else
        echo "✓ Using existing association file"
    fi

    if [[ $SEQ == *"freiburg1"* ]]; then
        SETTINGS="settings/TUM1.yaml"
    elif [[ $SEQ == *"freiburg2"* ]]; then
        SETTINGS="settings/TUM2.yaml"
    else
        SETTINGS="settings/TUM3.yaml"
    fi

    CONFIG_FILE="/tmp/pyslam_config_${SEQ}.yaml"
    cp "$PYSLAM_DIR/config.yaml" "$CONFIG_FILE"

    "$PYTHON_BIN" - <<EOF
import yaml
with open("$CONFIG_FILE", "r") as f:
    config = yaml.safe_load(f) or {}

def ensure_dict(cfg, key):
    if not isinstance(cfg.get(key), dict):
        cfg[key] = {}
    return cfg[key]

ensure_dict(config, "DATASET")["type"] = "TUM_DATASET"
ensure_dict(config, "TUM_DATASET")
config["TUM_DATASET"]["type"] = "tum"
config["TUM_DATASET"]["sensor_type"] = "rgbd"
config["TUM_DATASET"]["base_path"] = "$DATA_PATH"
config["TUM_DATASET"]["name"] = "$SEQ"
config["TUM_DATASET"]["settings"] = "$SETTINGS"
config["TUM_DATASET"]["associations"] = "associations.txt"
config["TUM_DATASET"]["groundtruth_file"] = "auto"
ensure_dict(config, "SAVE_TRAJECTORY")
config["SAVE_TRAJECTORY"]["save_trajectory"] = True
config["SAVE_TRAJECTORY"]["format_type"] = "tum"
config["SAVE_TRAJECTORY"]["output_folder"] = "$OUTPUT_PATH"
config["SAVE_TRAJECTORY"]["basename"] = "$SEQ"
ensure_dict(config, "GLOBAL_PARAMETERS")
config["GLOBAL_PARAMETERS"]["show_viewer"] = False
with open("$CONFIG_FILE", "w") as f:
    yaml.safe_dump(config, f)
EOF

    TRAJ_FILE="$OUTPUT_PATH/trajectories/${SEQ}_trajectory.txt"
    LOG_FILE="$OUTPUT_PATH/logs/${SEQ}.log"

    echo "Running pySLAM..."
    cd "$PYSLAM_DIR"

    xvfb-run -a -s "-screen 0 640x480x24" \
        $EXECUTABLE --config_path "$CONFIG_FILE" --headless --no_output_date \
        2>&1 | tee "$LOG_FILE"

    TRAJ_GEN="$OUTPUT_PATH/${SEQ}_final.txt"
    TRAJ_ONLINE="$OUTPUT_PATH/${SEQ}_online.txt"

    if [ -f "$TRAJ_GEN" ]; then
        mv "$TRAJ_GEN" "$TRAJ_FILE"
        echo "✓ Final trajectory saved: $TRAJ_FILE"
        [ -f "$TRAJ_ONLINE" ] && rm "$TRAJ_ONLINE"
    elif [ -f "$TRAJ_ONLINE" ]; then
        mv "$TRAJ_ONLINE" "$TRAJ_FILE"
        echo "✓ Online trajectory saved (final not found): $TRAJ_FILE"
    else
        TRAJ_SAVED=$(ls "$OUTPUT_PATH/${SEQ}"* 2>/dev/null | grep -v "_trajectory.txt" | head -n 1)
        if [ -f "$TRAJ_SAVED" ]; then
            mv "$TRAJ_SAVED" "$TRAJ_FILE"
            echo "✓ Found and moved trajectory: $TRAJ_FILE"
        else
            echo "✗ Failed to generate trajectory for $SEQ"
        fi
    fi

    if [ -d "$OUTPUT_PATH/plot" ]; then
        mkdir -p "$OUTPUT_PATH/plots/${SEQ}"
        mv "$OUTPUT_PATH/plot/"* "$OUTPUT_PATH/plots/${SEQ}/" 2>/dev/null || true
        rmdir "$OUTPUT_PATH/plot" || true
        echo "✓ Plots moved to: $OUTPUT_PATH/plots/${SEQ}/"
    fi

    mv "$OUTPUT_PATH/"*.png "$OUTPUT_PATH/plots/${SEQ}/" 2>/dev/null || true

    rm -f "$CONFIG_FILE"
done

echo ""
echo "================================================"
echo "✓ Baseline runs complete!"
echo "================================================"
echo "Trajectories saved in: $OUTPUT_PATH/trajectories/"