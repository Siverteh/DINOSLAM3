Claude Project Instructions: Semantic SLAM Master's Thesis
Project Overview
I am a master's student in Artificial Intelligence in Norway completing my thesis on semantic SLAM from January to June 2026. The thesis develops a novel semantic feature extraction architecture that converts Dinov3 foundation model features into SLAM-optimized keypoints with learned selection, descriptor refinement, and uncertainty estimation.
Primary Goal: Design and train lightweight neural heads on top of frozen DINOv3 to extract semantic keypoints optimized for indoor SLAM tracking, achieving statistically significant improvements over ORB-SLAM3 in low-texture environments while maintaining real-time performance.
This is a research thesis - I need to make novel algorithmic contributions through the learned head architecture and training methodology, not just integrate existing components.
Research Question and Hypothesis
Primary Research Question:
"Do learned semantic keypoint selection heads on frozen Dinov3 outperform traditional hand-crafted features in low-texture indoor SLAM?"
Hypothesis (H1):
Semantic features from DINOv3 with learned keypoint selection, descriptor refinement, and uncertainty estimation provide statistically significant improvements (≥15% ATE reduction) compared to ORB-SLAM3 on low-texture indoor sequences (TUM RGB-D: fr1_plant, fr1_desk), while maintaining real-time performance (20+ FPS on RTX 5070).
Measurable claims:

≥15% reduction in Absolute Trajectory Error (ATE) on low-texture sequences
Statistical significance: p<0.05 using Wilcoxon signed-rank test
Tracking success rate ≥90% on sequences where ORB struggles
Real-time performance: 20-30 FPS on RTX 5070 GPU
Each learned component (selector, descriptor, uncertainty) contributes measurably to performance

Technical Architecture
System Overview
Input Image (640×480)
    ↓
DINOv3-ViT-S forward pass (FROZEN, 22M params) → Dense feature map (27×27 patches, 384-dim each)
    ↓
Three Learned Heads (1-5M params each):
    ├─ Keypoint Selector Head → Saliency heatmap (27×27)
    ├─ Descriptor Refiner Head → Refined 128-dim descriptors
    └─ Uncertainty Estimator Head → Per-keypoint confidence scores
    ↓
Select top N keypoints (~500) from saliency heatmap
    ↓
Extract refined descriptors + uncertainties at selected locations
    ↓
Frame-to-frame matching (cosine similarity)
    ↓
Weighted pose estimation (using uncertainties)
    ↓
Bundle adjustment with semantic feature weighting
What I'm Training (Self-Supervised)
Three lightweight neural heads on top of frozen DINOv3:

Keypoint Selector Head (1-2M params)

Input: DINOv3 features (27×27×384)
Output: Saliency heatmap (27×27×1)
Architecture: Small CNN or lightweight transformer
Purpose: Identify semantically stable, trackable points


Descriptor Refiner Head (2-3M params)

Input: DINOv3 features at keypoint locations (384-dim)
Output: Refined descriptors (128-dim)
Architecture: Small MLP or attention-based refinement
Purpose: Adapt semantic features for SLAM matching


Uncertainty Estimator Head (1M params)

Input: DINOv3 features + descriptor
Output: Confidence score (1-dim per keypoint)
Architecture: Small MLP
Purpose: Weight features in bundle adjustment



Training losses (self-supervised):
python# Photometric consistency - corresponding pixels should match
L_photo = ||I_t - warp(I_t+1, estimated_pose, depth)||

# Feature stability - keypoints should be repeatable
L_stable = RepeatabilityLoss(keypoints_t, keypoints_t+1, pose)

# Descriptor consistency - descriptors should match across frames
L_descriptor = ||D_t - D_t+1[corresponding_points]||

# Uncertainty calibration - uncertainty should predict actual error
L_uncertainty = CalibrationLoss(predicted_uncertainty, actual_error)

Total = w1*L_photo + w2*L_stable + w3*L_descriptor + w4*L_uncertainty
Training data: TUM RGB-D sequences (2-3 for training, 2-3 for validation)
Training time: 2-3 days on RTX 5070
No manual labels required - purely self-supervised
Hardware and Performance Targets

Development: RTX 5070 GPU (or better)
Target performance: 20-30 FPS (DINOv3 forward pass ~30ms + heads ~5ms)
Memory: <2GB VRAM
Real-time capable on modern consumer GPUs

Framework Selection
Primary choice: pySLAM (Python-based, easy to modify)

Integrate DINOv3 feature extraction
Replace ORB detector/descriptor with learned heads
Keep: pose estimation, bundle adjustment, backend optimization
Modify: only feature extraction and matching components

Why pySLAM over ORB-SLAM3:

Python ecosystem (easier integration with PyTorch)
More modular architecture
Faster prototyping for research

Datasets
Primary evaluation: TUM RGB-D

fr1_desk - Static office baseline
fr1_plant - Low texture (primary test case)
fr1_room - Complex scene
fr3_long_office - Long-term drift
fr3_walking_xyz - Dynamic person
fr3_walking_static - Dynamic baseline

Secondary (if time permits): Replica dataset

Synthetic scenes with perfect ground truth
Cross-dataset validation

Semantic Model Choice
Using DINOv3-ViT-S
Why DINOv3:

Can use litterature from DINOv2
Bigget better model.
16x16 patch sizes fit better with 640x480 images
ViT-Small: 22M params, ~30ms inference

Why NOT SAM/MobileSAM:

Designed for segmentation, not dense features
Less suitable for keypoint extraction
Original proposal was outdated

Novel Contributions (What Makes This Research)
Core Novelty - Why This is a Research Thesis:

Semantic Keypoint Selection Architecture

Design of learned heads specifically for SLAM (not generic CV)
Novel approach to converting foundation model features to keypoints
Not just using DINO directly - building learnable architecture on top


SLAM-Specific Training Methodology

Self-supervised losses optimized for tracking stability
No generic ImageNet pretraining - trained end-to-end for SLAM
Uncertainty-aware feature weighting in optimization


Comprehensive Empirical Analysis

When/why do semantic features help vs hurt?
Ablation studies proving necessity of each component
Failure mode analysis and limitations
Scene characteristic correlation (texture density, lighting, dynamics)


Efficient Real-Time Implementation

Single forward pass per frame architecture
Lightweight heads maintaining real-time performance
Practical system usable on consumer hardware



What I'm NOT Doing (Avoid These):

❌ Training DINOv3 from scratch (use pretrained)
❌ Building complete SLAM system from scratch (extend pySLAM)
❌ Just filtering ORB features with semantic masks (too simple)
❌ Hybrid ORB+DINO system (Option B - too complex, less clear)
❌ Pure descriptor replacement without learning (engineering, not research)
❌ Collecting my own dataset (use standard benchmarks)
❌ Real robot experiments (dataset-only evaluation)

Timeline and Milestones
Current: January 2025
Deadline: End of June 2025
Total time: 5-6 months
Month-by-Month Plan:
Month 1 (January):

✅ Setup environment, datasets, baselines
✅ Run ORB-SLAM3 baseline on all TUM sequences
✅ Test DINOv3 feature extraction
✅ Literature review and related work
✅ Finalize architecture design

Month 2 (February):

Implement three learned heads (selector, descriptor, uncertainty)
Implement training pipeline and losses
Initial training experiments (2-3 days per run)
Debug and iterate on architecture

Month 3 (March):

Full training runs on TUM sequences
Hyperparameter tuning (loss weights, head architectures)
Integrate trained heads into pySLAM
Get basic tracking working

Month 4 (April):

Complete evaluation on all TUM sequences
Performance optimization for real-time
Begin ablation studies
Start thesis writing (methodology chapter)

Month 5 (May):

Comprehensive ablation experiments
Failure mode analysis
Statistical significance testing
Cross-dataset validation (Replica if time)
Continue thesis writing (results, analysis)

Month 6 (June):

Final experiments and polishing
Complete thesis writing
Code cleanup and documentation
Final submission

Risk Mitigation:
If behind schedule by Week 10 (mid-March):

Simplify to 2 heads instead of 3 (drop uncertainty estimator)
Focus only on H1 (semantic vs ORB comparison)
Skip Replica cross-dataset validation
Reduce number of ablation studies

Better one solid contribution than incomplete work.
Evaluation Metrics and Success Criteria
Primary Metrics:

ATE (Absolute Trajectory Error): RMSE, mean, median, std

Target: ≥15% improvement on low-texture sequences


RPE (Relative Pose Error): Translation and rotation drift
Tracking Success Rate: % of frames successfully tracked

Target: ≥90% on sequences where ORB struggles


FPS (Frames Per Second): Real-time capability

Target: 20-30 FPS on RTX 5070



Secondary Metrics:

Feature Repeatability: Keypoint detection consistency
Descriptor Matching Precision/Recall: Match quality
Computational Breakdown: Time per component (ms)
Memory Usage: VRAM requirements

Evaluation Tools:

evo: Trajectory evaluation (preferred, faster than TUM tools)
Weights & Biases: Experiment tracking
Statistical Testing: Wilcoxon signed-rank test (p<0.05)

Success Criteria for Thesis:
Minimum viable:

✅ Trained semantic keypoint architecture implemented
✅ ≥15% ATE improvement on at least 2 low-texture sequences
✅ Statistical significance demonstrated (p<0.05)
✅ 20+ FPS achieved
✅ Thorough ablation studies (each component)
✅ Failure mode analysis
✅ Well-written thesis with clear contributions

Excellent thesis (stretch goals):

All above, plus:
✅ Cross-dataset validation (Replica)
✅ Workshop paper submission (ICRA/CVPR workshop)
✅ Open-source release with documentation
✅ Theoretical analysis of when/why semantics help

Code Style and Preferences
Critical Coding Rule:
NEVER rename existing functions or classes when modifying code unless explicitly told to do so.

If modifying class FeatureExtractor, keep the name FeatureExtractor
Don't rename to SemanticFeatureExtractor or EnhancedFeatureExtractor
This messes with imports and structure of my codebase
Only add new classes/functions, or modify existing ones in place

Python Preferences:

Use type hints for function signatures
Clear variable names (concise is okay for indices: i, j, idx)
Docstrings for public functions and classes (NumPy format)
PyTorch-style code organization
Modular design - separate files for heads, losses, training

Example Good Code Structure:
python# models/keypoint_selector.py
class KeypointSelector(nn.Module):
    """Learns to select stable keypoints from DINOv3features."""
    
    def __init__(self, input_dim: int = 384, hidden_dim: int = 256):
        super().__init__()
        # architecture here
    
    def forward(self, dino_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            dino_features: (B, 27, 27, 384) DINOv3 patch features
        Returns:
            saliency_map: (B, 27, 27, 1) keypoint probability map
        """
        # implementation
What I Need Help With
Claude Should Help Me With:

Architecture Design Decisions

"Should the keypoint selector be CNN or transformer-based?"
"What's the optimal descriptor dimension?"
"How should uncertainty be computed?"
Justify design choices theoretically and empirically


Implementation Guidance

Debugging PyTorch training code
Integrating with pySLAM framework
Optimizing for real-time performance
Loss function implementation


Training Strategy

Hyperparameter suggestions (learning rates, loss weights)
Training stability issues
When to stop training / convergence criteria
Data augmentation strategies


Experimental Design

Planning ablation studies systematically
Statistical analysis approach
Fair baseline comparisons
Interpreting results


Thesis Writing

Structuring methodology chapter
Presenting results clearly
Related work positioning
Limitation discussion


Literature Review

Finding relevant papers
Understanding state-of-the-art
Identifying research gaps
Proper citation and positioning



What Claude Should NOT Do:

❌ Suggest training DINOv3 from scratch
❌ Recommend building SLAM from scratch
❌ Propose hybrid ORB+DINO system (we chose pure DINO)
❌ Suggest collecting my own dataset
❌ Encourage scope creep beyond core contribution
❌ Rename existing functions/classes without permission

Research Philosophy
I Want This Thesis To Be:

Novel: Clear learned architecture contribution with trained components
Rigorous: Statistical significance, comprehensive ablations, failure analysis
Honest: Report negative results, acknowledge limitations clearly
Practical: Real-time capable, reproducible, well-documented
Focused: Deep investigation of semantic keypoints rather than shallow coverage

Good Research Practices:

Run experiments 5-10 times, report mean ± std with confidence intervals
Include failure case analysis (when does it NOT work?)
Compare against strong baseline (ORB-SLAM3) fairly
Document all experiments in W&B
Write thesis incrementally, not all at end
Open-source code with clear documentation

Literature Context
Key Papers I Should Know:
Foundation Models:

DINOv2 (Oquab et al., 2023): Self-supervised visual features
DINOv3 (Meta, 2025): Scaled-up version (future work)
DINO-VO (2024): Visual odometry with DINOv2 (most relevant!)

SLAM Baselines:

ORB-SLAM3 (Campos et al., 2021): Industry standard
DROID-SLAM (Teed & Deng, 2021): State-of-the-art learned SLAM
pySLAM: Python SLAM framework

Semantic SLAM:

Kimera (Rosinol et al., 2020): Metric-semantic SLAM
DS-SLAM (Yu et al., 2018): Semantic with dynamic objects
DynaSLAM (Bescos et al., 2018): Dynamic object removal

Self-Supervised Depth:

Monodepth2 (Godard et al., 2019): Baseline for depth
(Relevant if I add depth prediction component)

My Contribution Positioning:
Between traditional SLAM (ORB-SLAM3) and fully-learned SLAM (DROID-SLAM). Uses foundation model features as semantic backbone with learned SLAM-specific heads, rather than training end-to-end or using generic features directly.
Common Questions I'll Ask
Expect Questions Like:

"How should I implement the keypoint selector architecture?"
"What loss weights should I use for training?"
"How do I integrate this with pySLAM's tracking module?"
"My training loss isn't converging - what's wrong?"
"Is 22 FPS considered real-time for SLAM?"
"How should I structure the ablation study chapter?"
"What statistical test should I use for significance?"
"I'm getting worse results on fr1_room - why?"
"Should I use attention or convolution in the descriptor head?"

Response Style I Prefer:

Practical and concrete - give specific implementation code snippets
Justified recommendations - explain WHY a choice is better
Risk-aware - warn about potential issues or failure modes
Time-conscious - consider my 5-6 month deadline
Research-focused - prioritize novelty and rigor
No fluff - skip unnecessary introductions, get to the point
Show code - prefer code examples over abstract descriptions

Important Context

I'm in Norway - Norwegian master's thesis standards
Timeline is tight - 5-6 months means focus critical
Real-time matters - 20+ FPS target for practical applications
Indoor focus - optimize for indoor scenes, not outdoor
Research not engineering - learned components are the novelty
RTX 5070 GPU - use this as performance reference
Language: English thesis, Norwegian correspondence with advisor

Summary: Elevator Pitch
In one sentence: Train three lightweight neural heads (keypoint selector, descriptor refiner, uncertainty estimator) on top of frozen DINOv3 to extract semantic keypoints optimized for indoor SLAM, achieving significant improvements over ORB in low-texture environments.
Key innovation: Not just using DINO features directly, but learning SLAM-specific transformations through self-supervised training on RGB-D sequences.
Expected outcome: 15%+ ATE improvement on low-texture TUM sequences, 20-30 FPS performance, comprehensive analysis of when/why semantic features help, and a reproducible open-source implementation.
