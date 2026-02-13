# pySLAM v2 Fork & Evaluation Scripts - Specification

**Status**: Draft  
**Last Updated**: 2025-02-13  
**Project**: DINOSLAM3 Thesis - Baseline & Semantic Feature Evaluation

---

## 1. Introduction

### What This Is
A minimal, thesis-focused fork of pySLAM in `pyslam_integration_v2/` that:
- Runs baseline ORB-SLAM3 features on TUM RGB-D sequences
- Runs DINOSLAM3 learned semantic features on the same sequences
- Produces trajectory files and 3D visualizations for comparison
- Generates ATE/RPE metrics for thesis results

### The Problem It Solves
The original pySLAM repository contains extensive features (real-time visualization, loop closure, multiple backends, debugging tools) that are unnecessary for thesis evaluation. This creates:
- Bloated setup with unused dependencies
- Confusing file structure with irrelevant code
- Slower iteration when modifying feature extractors
- Risk of breaking unrelated components during integration

This spec defines a **stripped-down, evaluation-only fork** with two clear execution paths:
1. **Baseline**: Run pySLAM with built-in ORB features → trajectory files + metrics
2. **Semantic**: Run pySLAM with DINOSLAM3 features → trajectory files + metrics

### Who It's For
- **Primary user**: You (thesis student) running experiments for ATE/RPE comparison
- **Secondary user**: Thesis committee/readers reproducing results

### Success Criteria
✅ Clean fork runs both baseline and semantic evaluations headlessly  
✅ Produces TUM-format trajectory files compatible with `evo` tool  
✅ Generates 3D trajectory vs ground-truth PNG visualizations  
✅ Outputs ATE/RPE metrics to console/log files  
✅ All TUM sequences complete without crashes  
✅ Setup takes <10 minutes on a fresh Ubuntu container

---

## 2. Requirements (Functional)

### R1: Minimal pySLAM Fork Structure

**User Story**: As a thesis student, I want a clean pySLAM fork containing only evaluation-critical code, so I can quickly understand, modify, and maintain the baseline system.

**Acceptance Criteria**:
- **GIVEN** the original pySLAM repository with 100+ files
- **WHEN** I run the cleanup process
- **THEN** the `pyslam_integration_v2/` fork contains:
  - Core SLAM components (tracking, pose estimation, bundle adjustment)
  - TUM RGB-D dataset loader
  - ORB feature detector/descriptor
  - Trajectory saving (TUM format)
  - Visualization (trajectory plotting to PNG)
  - Build scripts for C++ dependencies (Pangolin, GTSAM, ORB extractors)
  
- **AND** the fork does NOT contain:
  - Real-time GUI viewers (OpenGL windows)
  - Loop closure detection modules
  - Vocabulary training tools
  - Example datasets or sequences
  - Development/debugging notebooks
  - Alternative backends (DSO, SVO, etc.)
  - Network training code
  - Benchmark scripts unrelated to TUM

**Success Metrics**:
- Fork directory is <500MB (vs original ~2GB+ with submodules)
- Fewer than 50 Python files remain in core library
- No unused third-party dependencies in build scripts

---

### R2: Baseline ORB Evaluation Script

**User Story**: As a researcher, I want to run ORB-SLAM3 features through pySLAM on all TUM test sequences, so I can establish baseline ATE/RPE metrics.

**Acceptance Criteria**:
- **GIVEN** TUM RGB-D sequences in `src/dino_slam3/data/tum_rgbd/`
- **AND** a built pySLAM fork with ORB features
- **WHEN** I run `./run_baseline.sh`
- **THEN** the script:
  1. Processes each sequence: `freiburg1_desk`, `freiburg1_plant`, `freiburg1_room`, `freiburg3_long_office_household`, `freiburg3_walking_static`, `freiburg3_walking_xyz`
  2. Generates trajectory files: `results/baseline/trajectories/<sequence>_trajectory.txt` (TUM format)
  3. Saves 3D visualizations: `results/baseline/plots/<sequence>/trajectory_3d.png`
  4. Logs console output: `results/baseline/logs/<sequence>.log`
  5. Completes without crashes (exit code 0)

- **AND** trajectory files are valid:
  - Format: `timestamp tx ty tz qx qy qz qw` (space-separated, 8 columns)
  - Timestamps align with TUM ground truth (within 20ms)
  - No NaN or Inf values

- **AND** metrics are computed:
  - ATE RMSE (meters) printed to console
  - RPE RMSE (meters, degrees) printed to console
  - Values match hand-computed `evo_ape` / `evo_rpe` results

**Success Metrics**:
- Baseline script runs all 6 sequences in <30 minutes on RTX 5070
- ATE on `freiburg1_desk` is within 5% of published ORB-SLAM3 results
- 100% trajectory completion rate (no sequence failures)

---

### R3: Semantic DINOSLAM3 Evaluation Script

**User Story**: As a thesis student, I want to run my learned DINOSLAM3 features through pySLAM on TUM sequences, so I can compare semantic features against the ORB baseline.

**Acceptance Criteria**:
- **GIVEN** a trained DINOSLAM3 checkpoint at `runs/tum_stage1_dinov3_refine_v1/checkpoints/best.pt`
- **AND** TUM RGB-D sequences in `src/dino_slam3/data/tum_rgbd/`
- **WHEN** I run `./run_semantic.sh`
- **THEN** the script:
  1. Loads the DINOSLAM3 model from checkpoint
  2. Integrates semantic features into pySLAM (via SUPERPOINT hijack or custom extractor)
  3. Processes the same 6 sequences as baseline
  4. Generates trajectory files: `results/semantic/trajectories/<sequence>_DINOSLAM3_trajectory.txt`
  5. Saves 3D visualizations: `results/semantic/plots/<sequence>/trajectory_3d.png`
  6. Logs output: `results/semantic/logs/<sequence>_DINOSLAM3.log`

- **AND** the feature integration:
  - Accepts RGB images (640×480, uint8 or normalized float)
  - Returns keypoints with (x, y) pixel coordinates
  - Returns descriptors (128-dim or 256-dim float32, L2-normalized)
  - Maintains real-time performance (20+ FPS)

- **AND** trajectory files are valid:
  - Same TUM format as baseline
  - Timestamps aligned with ground truth
  - No missing frames due to tracking failure

**Success Metrics**:
- Semantic script runs all 6 sequences in <45 minutes on RTX 5070
- At least 90% tracking success on `freiburg1_plant` (vs ORB's expected failure)
- ATE improvement of ≥15% on low-texture sequences (plant, desk)

---

### R4: Trajectory Visualization

**User Story**: As a researcher, I want to see 3D plots of estimated trajectories vs ground truth, so I can visually confirm SLAM performance and identify failure modes.

**Acceptance Criteria**:
- **GIVEN** a completed SLAM run with trajectory file and ground truth
- **WHEN** the visualization step executes
- **THEN** a PNG file is generated with:
  - Estimated trajectory (colored line, e.g., blue)
  - Ground truth trajectory (reference line, e.g., green)
  - Start/end markers
  - 3D axes labeled (X, Y, Z in meters)
  - Title: `<sequence_name> - <feature_type>`

- **AND** the plot clearly shows:
  - Trajectory shape and scale
  - Drift (divergence between estimated and GT)
  - Loop closures (if any, though not required in this fork)

**Success Metrics**:
- All plots render correctly (no blank images)
- Visual inspection confirms ATE metrics (large drift = high ATE)
- Plots are readable at 1920×1080 resolution

---

### R5: Automatic ATE/RPE Computation

**User Story**: As a thesis student, I want ATE and RPE metrics automatically computed and displayed, so I don't have to manually run `evo` commands.

**Acceptance Criteria**:
- **GIVEN** trajectory files and ground truth for a sequence
- **WHEN** a SLAM run completes
- **THEN** the script:
  1. Calls `evo_ape tum <groundtruth> <trajectory>` (or equivalent)
  2. Calls `evo_rpe tum <groundtruth> <trajectory>` (or equivalent)
  3. Extracts key metrics:
     - ATE RMSE (m)
     - ATE Mean (m)
     - ATE Median (m)
     - RPE Translation RMSE (m)
     - RPE Rotation RMSE (deg)
  4. Prints metrics to console
  5. Appends metrics to a summary file: `results/<baseline|semantic>/metrics_summary.csv`

- **AND** the CSV format is:
  ```
  sequence,feature_type,ate_rmse,ate_mean,ate_median,rpe_trans_rmse,rpe_rot_rmse
  freiburg1_desk,ORB,0.023,0.019,0.015,0.012,0.34
  freiburg1_desk,DINOSLAM3,0.018,0.015,0.012,0.010,0.28
  ```

**Success Metrics**:
- Metrics match manual `evo` execution (within 0.001m tolerance)
- CSV is valid for import into Python/Excel
- Summary file enables easy baseline vs semantic comparison

---

## 3. Key Entities and Concepts

### pySLAM Fork Structure
The minimal fork contains:

```
pyslam_integration_v2/
├── pyslam/                      # Forked pySLAM repository (cleaned)
│   ├── pyslam/                  # Python package
│   │   ├── slam/                # Core SLAM logic
│   │   │   ├── tracking.py      # Frame-to-frame tracking
│   │   │   ├── pose_estimation.py
│   │   │   └── bundle_adjustment.py
│   │   ├── local_features/      # Feature extractors
│   │   │   ├── feature_orb.py   # ORB baseline
│   │   │   └── feature_dinoslam3.py  # Semantic features
│   │   ├── datasets/            # Dataset loaders
│   │   │   └── tum_rgbd.py      # TUM RGB-D loader
│   │   └── utils/
│   │       ├── trajectory.py    # TUM format saving
│   │       └── visualization.py # 3D plotting
│   ├── cpp/                     # C++ components
│   │   ├── ORBextractor/        # ORB feature extraction
│   │   └── lib/                 # Compiled libraries
│   ├── thirdparty/              # Dependencies
│   │   ├── pangolin/            # Visualization (for plots only)
│   │   ├── gtsam/               # Pose graph optimization
│   │   └── pydbow3/             # (optional, for loop closure if needed)
│   ├── settings/                # Camera calibration files
│   │   ├── TUM1.yaml
│   │   ├── TUM2.yaml
│   │   └── TUM3.yaml
│   └── config.yaml              # SLAM configuration
├── scripts/
│   ├── setup.sh                 # Build pySLAM and dependencies
│   ├── run_baseline.sh          # ORB evaluation script
│   └── run_semantic.sh          # DINOSLAM3 evaluation script
├── results/                     # Auto-generated outputs
│   ├── baseline/
│   │   ├── trajectories/
│   │   ├── plots/
│   │   ├── logs/
│   │   └── metrics_summary.csv
│   └── semantic/
│       ├── trajectories/
│       ├── plots/
│       ├── logs/
│       └── metrics_summary.csv
└── README.md                    # Setup and usage instructions
```

### Feature Extractor Contract
Both ORB and DINOSLAM3 must implement:
```python
def detectAndCompute(image: np.ndarray, mask=None) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
    """
    Args:
        image: (H, W, 3) BGR uint8 or (H, W) grayscale
        mask: Optional (H, W) uint8 mask
    
    Returns:
        keypoints: List of cv2.KeyPoint objects
        descriptors: (N, D) float32 array, L2-normalized
    """
```

### TUM Trajectory Format
Space-separated values (8 columns):
```
timestamp tx ty tz qx qy qz qw
```
- `timestamp`: Unix timestamp (float, seconds)
- `tx, ty, tz`: Translation (meters)
- `qx, qy, qz, qw`: Quaternion rotation (Hamilton convention)

---

## 4. External Behavior

### Inputs
1. **TUM RGB-D Sequences**:
   - Location: `src/dino_slam3/data/tum_rgbd/<sequence>/`
   - Required files: `rgb.txt`, `depth.txt`, `groundtruth.txt`, `rgb/`, `depth/`

2. **DINOSLAM3 Checkpoint**:
   - Location: `runs/tum_stage1_dinov3_refine_v1/checkpoints/best.pt`
   - Format: PyTorch checkpoint with `model` state dict

3. **Camera Intrinsics**:
   - Embedded in TUM sequence folders or settings YAML
   - Format: `fx, fy, cx, cy`

### Outputs
1. **Trajectory Files** (per sequence):
   - Format: TUM (see above)
   - Example: `results/baseline/trajectories/freiburg1_desk_trajectory.txt`

2. **Visualization PNGs** (per sequence):
   - 3D plot: estimated trajectory (blue) vs ground truth (green)
   - Example: `results/baseline/plots/freiburg1_desk/trajectory_3d.png`

3. **Metrics Summary CSV**:
   - Aggregated ATE/RPE for all sequences
   - Example: `results/baseline/metrics_summary.csv`

4. **Log Files** (per sequence):
   - Console output from SLAM run
   - Example: `results/baseline/logs/freiburg1_desk.log`

### State Transitions
```
[Setup] → [Build pySLAM] → [Ready]
[Ready] → [Run Baseline] → [Trajectories Generated]
[Ready] → [Run Semantic] → [Trajectories Generated]
[Trajectories Generated] → [Compute Metrics] → [Results Saved]
```

---

## 5. Constraints and Rules

### Business Rules
1. **Trajectory Validity**: All saved trajectories must have no NaN/Inf values
2. **Timestamp Alignment**: Trajectory timestamps must align with ground truth (±20ms)
3. **Completeness**: A sequence run is only "successful" if it produces a valid trajectory file and PNG plot

### Technical Constraints
1. **Headless Execution**: Scripts must run without GUI (use `xvfb` for Pangolin plotting)
2. **CUDA Requirement**: DINOSLAM3 requires GPU (RTX 5070 or better)
3. **Python 3.10+**: pySLAM and DINOSLAM3 compatible
4. **OpenCV Version**: Must support both ORB and SIFT/SURF (OpenCV 4.x recommended)

### Performance Requirements
1. **Real-time Semantic Features**: DINOSLAM3 forward pass <50ms per frame (20 FPS minimum)
2. **Batch Processing**: All 6 sequences complete in <1 hour total (baseline + semantic)
3. **Memory Usage**: <8GB VRAM during SLAM runs

---

## 6. Edge Cases and Error Handling

### Tracking Failure
- **GIVEN** ORB or DINOSLAM3 loses tracking (too few features matched)
- **WHEN** consecutive frames fail to estimate pose
- **THEN**:
  - Attempt re-initialization for 5 frames
  - If recovery fails, terminate the run
  - Mark sequence as "tracking_failed" in metrics summary
  - Save partial trajectory up to failure point

### Missing Ground Truth
- **GIVEN** a TUM sequence without `groundtruth.txt`
- **WHEN** attempting to run evaluation
- **THEN**:
  - Skip that sequence
  - Print warning: "Skipping <sequence>: no ground truth"
  - Continue with remaining sequences

### Checkpoint Not Found
- **GIVEN** DINOSLAM3 checkpoint path is invalid
- **WHEN** running `run_semantic.sh`
- **THEN**:
  - Print error: "Checkpoint not found: <path>"
  - Suggest running training first
  - Exit with code 1

### Corrupted Trajectory File
- **GIVEN** a trajectory file with malformed lines (wrong column count, non-numeric values)
- **WHEN** computing metrics
- **THEN**:
  - Print error: "Invalid trajectory format in <file>"
  - Skip metric computation for that sequence
  - Continue with other sequences

---

## 7. Out of Scope

### Explicitly NOT Included
❌ Real-time GUI visualization (Pangolin viewer running during SLAM)  
❌ Loop closure detection and optimization  
❌ Map saving/loading (sparse point cloud, keyframes)  
❌ Multi-session mapping  
❌ Integration with ROS  
❌ Support for other datasets (KITTI, EuRoC, etc.)  
❌ Training DINOSLAM3 from scratch (handled by separate training pipeline)  
❌ Feature extractor comparisons beyond ORB vs DINOSLAM3 (no SIFT, SuperPoint, etc.)  
❌ Online trajectory correction or smoothing  
❌ Relocalization after tracking loss  

### Future Considerations (Post-Thesis)
- Add more baseline features (SIFT, SuperPoint) for comparison
- Support for dynamic object masking (if needed)
- Real-time performance profiling per component

---

## 8. Implementation Plan Overview

### Phase 1: Fork Cleanup (Week 1)
1. Clone fresh pySLAM repository to `pyslam_integration_v2/pyslam/`
2. Remove unused files (loop closure, alternative backends, examples)
3. Strip GUI viewer components (keep only headless Pangolin for plotting)
4. Remove development/debugging tools

### Phase 2: Build System (Week 1)
1. Create `setup.sh` script:
   - Build GTSAM
   - Build Pangolin (headless mode)
   - Build ORB extractor
   - Install Python dependencies
2. Test on clean Ubuntu 22.04 container

### Phase 3: Baseline Script (Week 2)
1. Create `run_baseline.sh`:
   - Iterate over TUM sequences
   - Configure pySLAM for ORB features
   - Run headless (xvfb)
   - Save trajectories
   - Generate plots
   - Compute metrics
2. Validate against published ORB-SLAM3 results

### Phase 4: Semantic Integration (Week 2)
1. Create `feature_dinoslam3.py` adapter (or hijack SUPERPOINT)
2. Create `run_semantic.sh`:
   - Load DINOSLAM3 checkpoint
   - Run same sequences as baseline
   - Save trajectories and plots
   - Compute metrics
3. Validate tracking success on low-texture sequences

### Phase 5: Validation & Documentation (Week 3)
1. Run both scripts on all 6 sequences
2. Compare metrics with hand-computed `evo` results
3. Write `README.md` with setup and usage instructions
4. Create troubleshooting guide

---

## 9. Acceptance Testing Checklist

### Setup Tests
- [ ] `setup.sh` completes without errors on Ubuntu 22.04
- [ ] All C++ dependencies build successfully
- [ ] Python imports work: `import pyslam`

### Baseline Tests
- [ ] `run_baseline.sh` runs all 6 sequences without crashes
- [ ] Trajectory files exist for all sequences
- [ ] Trajectory files are valid TUM format
- [ ] PNG plots exist and show trajectories
- [ ] ATE RMSE on `freiburg1_desk` < 0.03m (matches literature)
- [ ] Metrics summary CSV is generated and valid

### Semantic Tests
- [ ] DINOSLAM3 checkpoint loads successfully
- [ ] `run_semantic.sh` runs all 6 sequences without crashes
- [ ] Feature extraction maintains 20+ FPS
- [ ] Trajectory files exist for all sequences
- [ ] Tracking succeeds on `freiburg1_plant` (low-texture)
- [ ] ATE on `freiburg1_plant` is ≥15% better than ORB
- [ ] Metrics summary CSV is generated

### Metric Validation
- [ ] ATE values match `evo_ape` manual execution (±0.001m)
- [ ] RPE values match `evo_rpe` manual execution (±0.001m)
- [ ] CSV is importable into Pandas without errors

### Documentation
- [ ] README includes setup instructions
- [ ] README includes usage examples
- [ ] Troubleshooting section exists

---

## 10. Open Questions

1. **Loop Closure**: Do we need loop closure for any of the 6 test sequences? (Likely not, but confirm)
2. **ORB Parameters**: Should we use default ORB parameters or tune them for indoor scenes?
3. **Failure Handling**: If a sequence fails (e.g., tracking lost), should we report it in metrics as NaN or omit it entirely?
4. **Checkpoint Path**: Should the checkpoint path be configurable via environment variable or hardcoded?

---

**Next Steps After Approval**:
1. Confirm open questions above
2. Begin Phase 1 (fork cleanup)
3. Set up basic build system
4. Iterate on baseline script until metrics match 