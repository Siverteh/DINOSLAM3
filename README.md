# DINO-DPVO Thesis Code

This folder contains the code-only release bundle for the thesis implementation. It is meant to be copied into a GitHub repository and cited from the thesis as the reproducibility/code artifact.

No generated result files, plots, datasets, model checkpoints, trajectory dumps, logs, or external repository checkouts are included here.

## What Is Included

- `refocus_vo/src/refocus_vo/dino_dpvo/`: the DINO-DPVO frontend, DPVO adapter, tracker wrapper, diagnostics, and configuration dataclasses.
- `refocus_vo/src/refocus_vo/backbones/`: DINOv3 backbone loading utilities.
- `refocus_vo/src/refocus_vo/data/`: TUM RGB-D and TartanAir dataset/window helpers.
- `refocus_vo/src/refocus_vo/eval/`: evaluation code for native DPVO, DINO-DPVO, ratio ablations, final 5-repeat benchmarking, external baselines, trajectory validation, and aggregation.
- `refocus_vo/src/refocus_vo/sweeps/`: sweep orchestration used for the architecture-family search.
- `refocus_vo/src/refocus_vo/train_dino_dpvo_frontend.py`: frontend training entrypoint.
- `src/dino_slam3/`: legacy DINO-SLAM3 training, feature, geometry, tracking, and dataset helper code that some tests and scripts still reference.
- `refocus_vo/configs/`: workspace-compatible YAML configs used by the scripts.
- `refocus_vo/configs/architecture_sweep/`: generated architecture-family sweep configurations.
- `refocus_vo/configs/ratio_ablation/`: generated native/DINO ratio ablation configurations, including the final `multiscale_32x4` 50/50 setting.
- `refocus_vo/configs/final_method/`: curated final method, native DPVO, pure-DINO, runtime, and final-sweep configs.
- `refocus_vo/configs/data/`: TartanAir subset manifests used for training/validation setup.
- `refocus_vo/scripts/`: shell entrypoints for training, architecture sweep, ratio ablation, final 5-repeat benchmark, and dataset download.
- `refocus_vo/external/`: helper scripts for installing/running external DPVO, DROID-SLAM, and ORB-SLAM3 baselines.
- `analysis/`: code used for runtime sanity checks and figure regeneration. The CSV/image outputs are intentionally omitted.
- `tests/`: unit/smoke tests for the thesis code paths.

## Main Thesis Entry Points

Architecture-family sweep:

```bash
bash refocus_vo/scripts/run_dino_dpvo_focus071_arch5x2_tumwin_sweep_v1.sh
```

Ratio ablation over native/DINO patch-source mixtures:

```bash
bash refocus_vo/scripts/run_focus071_arch_ratio_ablation.sh
```

Final native DPVO vs. DINO-DPVO 5-repeat benchmark:

```bash
bash refocus_vo/scripts/run_focus071_vs_dpvo_tum_freiburg123_5x.sh
```

Frontend training:

```bash
bash refocus_vo/scripts/train_dino_dpvo_frontend.sh
```

Runtime/FPS sanity check:

```bash
python analysis/measure_runtime_sanity.py --help
```

## External Files Not Included

To run the full pipeline, provide these separately:

- TUM RGB-D sequences.
- TartanAir training/validation images.
- DINOv3 weights, downloaded through the model loader/Hugging Face stack.
- Native DPVO checkout and weights, normally under `refocus_vo/external/repos/DPVO/`.
- DROID-SLAM and ORB-SLAM3 checkouts if external baseline scripts are used.
- Trained DINO-DPVO checkpoints.

The paths in the configs and shell scripts preserve the original thesis workspace layout. When moving this folder into a clean repository, update dataset, checkpoint, and external-repository paths before running experiments.

## Suggested Citation Note

In the thesis, this can be referenced as the accompanying code artifact for DINO-DPVO: frozen DINOv3-guided patch selection for DPVO-style monocular visual odometry.
