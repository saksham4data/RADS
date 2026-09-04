# Dataset Lineage

This file records what the repository actually shows about RADS dataset evolution.

## D00 - Global Merged Metadata

- Tracked file: `Datasets/processed/global_master_metadata.csv`
- Documented in: `docs/datset_version.md`
- Records: 5338
- Datasets represented: Kaggle 989, PICEK 4238, TUDAT 111
- Type values observed: accident, challenging, non-accident, head-on, rear-end, sideswipe, single, t-bone
- Historical EDA target: yes
- Current P02 target: no

The older EDA reports and notebooks belong to this broader merged metadata stage. They should not be described as statistics for P02.

## D01 - TUDAT v1 Baseline Stage

- Tracked config: `training/config/training_config_v1.yaml`
- Approximate records from training docs: 111 TUDAT records
- Experiments: E01-E06
- Purpose: initial image/video-classification baseline work
- Important decision: challenging clips were merged into accident for binary experiments after inspection and documentation review.

## D02 - TUDAT v2 Deduplicated Benchmark

- Tracked files:
  - `Datasets/processed/tudat/v2/global_master_metadata_v2.csv`
  - `Datasets/processed/tudat/v2/tudat_v2_manifest.json`
  - `Datasets/processed/tudat/v2/tudat_v2_integrity_report.md`
- Records verified from CSV: 93
- Classes: 44 accident, 49 non-accident
- Split: 65 train, 13 validation, 15 test
- Split by class:
  - Train: 31 accident, 34 non-accident
  - Validation: 6 accident, 7 non-accident
  - Test: 7 accident, 8 non-accident
- Experiments: E07, T01
- Purpose: deduplicated frozen benchmark for spatial baseline and temporal proof of concept.

## D03 - P01 Small PICEK Binary Dataset

- Metadata file present locally: `Datasets/processed/picek/p01_small_binary/picek_p01_small_binary_metadata.csv`
- Builder: `scripts/build_p01_small_binary_dataset.py`
- Records verified from CSV: 200
- Composition: 100 accident, 100 normal
- Source videos: 100 paired sources, one positive and one pre-accident negative clip per source
- Collision categories: 40 clips each for head-on, rear-end, sideswipe, single, t-bone
- Split: 140 train, 30 validation, 30 test
- Split by class:
  - Train: 70 accident, 70 normal
  - Validation: 15 accident, 15 normal
  - Test: 15 accident, 15 normal
- Source-video leakage check from metadata: 0 overlap across train/validation/test.
- Experiment: P01

P01 was a controlled small PICEK binary temporal experiment using the ResNet18 + GRU baseline.

## D04 - P02 Current Working Dataset

- Metadata file present locally: `Datasets/processed/picek/p02_binary/picek_p02_binary_metadata.csv`
- Builder: `scripts/build_p02_binary_dataset.py`
- Records verified from CSV: 500
- Composition: 250 accident, 250 normal
- Source videos: 250 paired sources, one positive and one pre-accident negative clip per source
- Collision categories: 100 clips each for head-on, rear-end, sideswipe, single, t-bone
- Split: 350 train, 76 validation, 74 test
- Split by class:
  - Train: 175 accident, 175 normal
  - Validation: 38 accident, 38 normal
  - Test: 37 accident, 37 normal
- Source-video leakage check from metadata: 0 overlap across train/validation/test.
- Experiment: P02
- Current baseline dataset: yes

## Missing For P02

The repository does not currently contain a current EDA report for P02 with distributions such as:

- duration distribution
- FPS/resolution distribution
- weather/day-time distribution
- quality-score distribution
- object counts or detection statistics
- event timing distribution
- hard-negative categorization beyond paired pre-accident negatives

These should be computed in a future EDA pass. They should not be inferred from the older global EDA.
