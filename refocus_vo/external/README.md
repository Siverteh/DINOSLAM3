# External VO Context Baselines

This directory isolates heavier third-party comparison methods from the main repo environments.

Canonical setup flow:

```bash
refocus_vo/external/bootstrap_micromamba.sh
refocus_vo/external/install_droid_slam.sh
refocus_vo/external/install_dpvo.sh
```

Canonical run commands:

```bash
refocus_vo/external/run_droid_slam_tum.sh
refocus_vo/external/run_dpvo_tum.sh
refocus_vo/external/run_dpvo_tartanair.sh
```

Defaults:

- micromamba root: `refocus_vo/.micromamba`
- DROID-SLAM env: `refocus_vo/.micromamba/envs/droid-slam`
- DPVO env: `refocus_vo/.micromamba/envs/dpvo`
- TUM pack output root: `refocus_vo/runs/external/`

Notes:

- These are context baselines, not main apples-to-apples RGB-D comparators.
- Both wrappers currently evaluate on the fixed six-sequence TUM pack by default.
- `run_dpvo_tartanair.sh` evaluates DPVO on the official DPVO TartanAir validation split.
- Both wrappers require CUDA.
- The external runners fail clearly if the env, repo, or weights are missing.
