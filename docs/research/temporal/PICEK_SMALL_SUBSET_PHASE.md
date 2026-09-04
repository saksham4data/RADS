# Small PICEK Temporal Phase & Experiment Readiness (P01)

## 1. Project Research Progression

The RADS research progression has advanced systematically through the following verified milestones:

```text
TUDAT v2 Benchmark (93 videos, deduplicated)
    │
    ├── E07 Spatial Baseline (ResNet18 2D frame-level, Test Acc: 33.3%, Accident Recall: 42.9%, F1: 0.375) [FROZEN]
    │
    └── T01 Temporal Proof-of-Concept (ResNet18 + GRU, Test Acc: 46.7%, Accident Recall: 100%, F1: 0.636) [FROZEN]
            │
            ├── Lineage Verified (W&B: q8o10v54, checkpoint: 2026-08-24_20-12-10/checkpoints/best.pt)
            │
            ├── T01 Reference Frozen (training_config_t01_frozen.yaml)
            │
            └── Transition to Small PICEK Subset Phase
                    │
                    ├── PICEK Data Audit (2,027 real videos across 5 collision types)
                    │
                    ├── Stratified Subset Extraction (v1_small: 100 videos, 20 per class, 70/15/15 split)
                    │
                    ├── Pipeline Scaffolding (training_config_picek_small.yaml)
                    │
                    └── Validation Gates (5/5 tests passed, 49/49 repository tests passed)
```

---

## 2. PICEK Data Findings & Structure

### Raw Dataset Audit
- **Location**: [`Datasets/raw/picekl`](file:///e:/Rads/Datasets/raw/picekl)
- **Available Real Videos**: `2,027` valid `.mp4` video files
- **Available Synthetic Videos**: None present on disk (raw directory contains only real videos)
- **Collision Categories (5 Classes)**:
  1. `single`: 680 videos (single-vehicle accidents / off-road collisions)
  2. `t-bone`: 657 videos (side-impact intersection collisions)
  3. `rear-end`: 328 videos (following vehicle collisions)
  4. `sideswipe`: 245 videos (lateral impact collisions)
  5. `head-on`: 117 videos (frontal impact collisions)
- **Temporal Event Annotations**:
  - `100%` of real videos contain verified `accident_time` (seconds) and `accident_frame` indices.
  - Video durations range from $1.3\text{s}$ to $115.0\text{s}$ (median $26.8\text{s}$, mean frame count $377.4$).
  - Video frame rates average $17.2\text{ fps}$ (mostly $15\text{–}30\text{ fps}$).

---

## 3. Small PICEK Subset Design (`v1_small`)

To enable rapid iteration and validate temporal sequence modeling before scaling to full PICEK, we extracted a controlled, versioned subset:

- **Subset Artifact**: [`Datasets/processed/picek/v1_small/picek_v1_small_metadata.csv`](file:///e:/Rads/Datasets/processed/picek/v1_small/picek_v1_small_metadata.csv)
- **Integrity Report**: [`Datasets/processed/picek/v1_small/picek_v1_small_integrity_report.md`](file:///e:/Rads/Datasets/processed/picek/v1_small/picek_v1_small_integrity_report.md)
- **Sampling Strategy**:
  - Exactly 20 real videos sampled per collision category ($20 \times 5 = 100$ videos total).
  - Deterministic random sampling with fixed seed (`seed=42`).
  - Pre-filtered to confirm physical `.mp4` file existence and OpenCV decodability.
- **Split Distribution**:
  - **Train**: 70 videos (14 head-on, 14 rear-end, 14 sideswipe, 14 single, 14 t-bone)
  - **Val**: 15 videos (3 head-on, 3 rear-end, 3 sideswipe, 3 single, 3 t-bone)
  - **Test**: 15 videos (3 head-on, 3 rear-end, 3 sideswipe, 3 single, 3 t-bone)
- **Zero Leakage**: Strict video ID partitioning ensures zero cross-split overlap.

---

## 4. Pipeline Scaffolding & Configuration

- **Configuration File**: [`training/config/training_config_picek_small.yaml`](file:///e:/Rads/training/config/training_config_picek_small.yaml)
- **Architecture**:
  - Spatial Backbone: `resnet18` (pretrained on ImageNet, fine-tuned)
  - Temporal Encoder: `gru` (`hidden_dim: 256`, `num_layers: 1`, `classifier_dropout: 0.3`)
  - Classification Head: `Linear(256, 5)` for 5-class collision categorization
- **Sequence Dataloader**:
  - Uniform sampling of 8 frames per video ($T=8$, $H=224, W=224$).
  - Transforms: `conservative_v1` spatial and color augmentations during training.
- **Checkpoint Metric**: `val/macro_f1` (mitigating the single-class recall bias identified during T01).

---

## 5. Validation Gate Results

The automated validation suite ([`training/tests/test_picek_subset_validation.py`](file:///e:/Rads/training/tests/test_picek_subset_validation.py)) executed 5 strict gates:

| Gate | Verification Check | Result |
|---|---|---|
| **Gate 1** | Subset Manifest Integrity (100 records, schema valid) | **PASSED** |
| **Gate 2** | Zero Cross-Split Leakage & 70/15/15 Stratified Balance | **PASSED** |
| **Gate 3** | Physical Video File Existence on Disk (100/100 verified) | **PASSED** |
| **Gate 4** | `VideoSequenceDataset` Sequence Tensor Generation `[8, 3, 224, 224]` | **PASSED** |
| **Gate 5** | Model Forward Pass `[B, 5]`, Cross-Entropy Loss & Gradient Flow | **PASSED** |

**Full Regression Test Suite**: `49/49 passed` in 44.4s.

---

## 6. Readiness & Next Steps

The repository is now fully prepared for **Experiment P01** (Small PICEK Temporal GRU training). No further dataset or infrastructure modifications are needed.
