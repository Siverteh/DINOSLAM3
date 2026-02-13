# pySLAM Integration v2 (Minimal ORB vs DINOSLAM3 Evaluation)

This folder contains a strict evaluation-only fork for TUM RGB-D comparison between `ORB` and `DINOSLAM3`.

## Layout

- `pyslam/`: minimal pySLAM v2 fork used for runs
- `scripts/`: helper tools for associations, validation, metrics, plotting, and budget checks
- `run_baseline.sh`: ORB batch runner
- `run_semantic.sh`: DINOSLAM3 batch runner
- `setup.sh`: environment setup + package install + budget checks
- `results/`: generated outputs

## Setup

From repository root:

```bash
cd pyslam_integration_v2
./setup.sh
```

Optional setup variables:

- `PYTHON_BIN` (default: `python3`)
- `INSTALL_SYS_DEPS=1` to install apt packages when available
- `USE_LOCAL_VENV=1` (default) to create/use `../.venv_pyslam_integration_v2`
- `VENV_DIR` to override local venv path

Example:

```bash
PYTHON_BIN=.venv/bin/python INSTALL_SYS_DEPS=1 ./setup.sh
```

## Baseline Evaluation (ORB)

```bash
cd pyslam_integration_v2
./run_baseline.sh
```

Optional variables:

- `DATA_PATH` (default: `../src/dino_slam3/data/tum_rgbd`)
- `PYTHON_BIN` (default: `python3`)
- `USE_XVFB` (default: `1`)
- `SEQUENCE` or `SEQUENCES` to run subset (`freiburg1_desk`, etc.)

Smoke example:

```bash
SEQUENCE=freiburg1_desk ./run_baseline.sh
```

## Semantic Evaluation (DINOSLAM3)

```bash
cd pyslam_integration_v2
./run_semantic.sh
```

Checkpoint behavior:

- Default checkpoint: `runs/tum_stage1_dinov3_refine_v1/checkpoints/best.pt`
- Override with `CKPT=/path/to/checkpoint.pt`
- Missing checkpoint exits immediately with code `1`

Optional variables:

- `CKPT`
- `DATA_PATH`
- `PYTHON_BIN`
- `USE_XVFB`
- `SEQUENCE` / `SEQUENCES`
- `DINOSLAM3_BACKBONE_NAME` (optional backbone override if default one is gated)
- `DINOSLAM3_PATCH_SIZE` (optional patch size override for alternate backbones)

Smoke example:

```bash
CKPT=runs/tum_stage1_dinov3_refine_v1/checkpoints/best.pt SEQUENCE=freiburg1_desk ./run_semantic.sh
```

## Output Artifacts

Baseline:

- `results/baseline/trajectories/<short>_trajectory.txt`
- `results/baseline/plots/<short>/trajectory_3d.png`
- `results/baseline/logs/<short>.log`
- `results/baseline/metrics_summary.csv`

Semantic:

- `results/semantic/trajectories/<short>_DINOSLAM3_trajectory.txt`
- `results/semantic/plots/<short>/trajectory_3d.png`
- `results/semantic/logs/<short>_DINOSLAM3.log`
- `results/semantic/metrics_summary.csv`

CSV schema:

```text
sequence,feature_type,status,ate_rmse,ate_mean,ate_median,rpe_trans_rmse,rpe_rot_rmse
```

`status` values include:

- `ok`
- `skipped_missing_sequence`
- `skipped_missing_groundtruth`
- `tracking_failed`
- `invalid_trajectory`

Non-`ok` rows are recorded with `NaN` metrics.

## Sequence Policy

Target set:

- `freiburg1_desk`
- `freiburg1_plant`
- `freiburg1_room`
- `freiburg3_long_office_household`
- `freiburg3_walking_static`
- `freiburg3_walking_xyz`

If `rgbd_dataset_freiburg3_walking_static` is missing, it is skipped and explicitly recorded as `skipped_missing_sequence`.

## Troubleshooting

- Missing checkpoint: set `CKPT` to a valid `.pt` file path.
- Missing dataset: set `DATA_PATH` to your TUM root folder.
- Missing associations: runners auto-generate `associations.txt` from `rgb.txt` and `depth.txt`.
- Import issues: rerun `./setup.sh` and verify `python -c "import pyslam"`.
- Budget failure: run `python scripts/check_core_budget.py --root .` to inspect file and size limits.
- Gated Hugging Face backbone: authenticate (`huggingface-cli login`) or set `DINOSLAM3_BACKBONE_NAME` to an accessible model.
