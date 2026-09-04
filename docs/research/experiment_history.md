# Experiment History

This file summarizes the research path from early EDA and ResNet18 baselines to the current object-centric RADS direction.

## Earlier EDA

The EDA module targeted `Datasets/processed/global_master_metadata.csv`, a 5338-record merged metadata file covering PICEK, TUDAT, and Kaggle image data.

Tracked EDA docs include:

- `docs/eda/dataset_readiness.md`
- `docs/eda/dataset_schema_comparison.md`
- `docs/eda/feature_classification.md`
- `docs/eda/metadata_dictionary.md`
- `docs/eda/missing_value_audit.md`

Generated EDA outputs exist locally under `eda/reports/`, `eda/figures/`, `eda/exports/`, and W&B folders. These are historical generated artifacts and should stay out of Git unless a specific report is promoted into `docs/research`.

Important limitation: this EDA predates the P02 500-video balanced dataset. It is valid as historical analysis of earlier metadata, not as current P02 EDA.

## E01-E06 - TUDAT Classification Baselines

Source: `docs/research/experiments_log.md`, `docs/research/decisions.md`, tracked training configs, and archived local outputs.

- E01: initial TUDAT 3-class ResNet18 baseline. Result showed strong challenging-class bias and poor validation performance.
- E02: binary TUDAT after merging challenging into accident. Training improved, but overfitting remained severe.
- E03: binary ResNet18 with early stopping and more epochs. Training reached stronger metrics, but validation/test remained poor; test accident recall was very low.
- E04: increased sampled frames per video from 5 to 8. Test accuracy improved to about 36.4%, but generalization remained poor.
- E05: conservative augmentation. Mild regularization improvement, but accident recall remained weak.
- E06: checkpoint selection changed to monitor accident F1. This improved metric alignment.

Important caveat: the original six-configuration P01-style hyperparameter comparison later had an early-stopping direction issue and must not be used as reliable evidence for optimizer superiority.

## E07 - TUDAT v2 Frozen Spatial Baseline

Dataset: TUDAT v2, 93 deduplicated canonical records.

Result recorded in `docs/research/experiments_log.md`:

- Validation accident F1: 0.3119
- Test accuracy: 33.33%
- Test accident recall: 42.86%
- Test accident F1: 0.3750

Purpose: establish a cleaner frozen spatial baseline without duplicate leakage.

## T01 - TUDAT v2 Temporal Proof Of Concept

Dataset: TUDAT v2.

Model: ResNet18 + GRU, controlled comparison against E07.

Result recorded in `docs/research/experiments_log.md` and `docs/research/temporal/T01_TEMPORAL_PROOF_OF_CONCEPT.md`:

- Validation accident F1: 0.6316
- Test accuracy: 46.67%
- Test accident recall: 100.0%
- Test accident F1: 0.6364

Interpretation: adding temporal sequence modeling helped accident recall on the small TUDAT v2 benchmark. It did not solve the larger RADS goal because the system still produced clip-level classifications rather than object-level event reasoning.

## P01 - Small PICEK Binary Temporal Experiment

Dataset: P01 small PICEK binary dataset.

Composition verified from local metadata:

- 200 clips
- 100 accident, 100 normal
- 140 train, 30 validation, 30 test
- 0 source-video overlap across splits

Model/config:

- ResNet18 + GRU
- 8 frames per video
- GRU hidden size 128
- Adam, learning rate 1e-4, weight decay 1e-4
- Cosine scheduler
- Early stopping and checkpointing on `val/f1_accident` with `mode: max`

Recorded P01 test results from `training/outputs/2026-08-31_16-20-59/experiment_report.md`:

- Accuracy: 0.4000
- Balanced accuracy: 0.4000
- AUROC: 0.4844
- Macro F1: 0.3541
- Normal F1: 0.1818
- Accident F1: 0.5263
- Confusion matrix, class order `normal`, `accident`: `[[2, 13], [5, 10]]`
- Normal false-positive rate: 13 / 15 = 86.7%

Interpretation: the model strongly overpredicted accident and had poor normal-class recall.

## P01 Hyperparameter Study Caveat

The P01 hyperparameter study artifacts are preserved locally under `training/outputs/archive/p01_hyperopt_study/`, but the study has a known training-control issue: early stopping monitored accident F1 with the wrong optimization direction in earlier runs.

Preserve these artifacts as historical evidence, but do not present their optimizer rankings as reliable.

## P02 - Current Baseline On 500 Balanced PICEK Clips

Dataset: P02 current working dataset.

Composition verified from local metadata:

- 500 clips
- 250 accident, 250 normal
- 350 train, 76 validation, 74 test
- 0 source-video overlap across splits

Model/config:

- ResNet18 + GRU
- 8 frames per video
- GRU hidden size 128
- Adam, learning rate 1e-4, weight decay 1e-4
- Cosine scheduler
- Early stopping and checkpointing on `val/f1_accident` with `mode: max`

Recorded P02 test results from `training/outputs/2026-08-31_19-55-16/experiment_report.md`:

- Accuracy: 0.4595
- Balanced accuracy: 0.4595
- AUROC: 0.5347
- Macro F1: 0.4071
- Normal F1: 0.2308
- Accident F1: 0.5833
- Confusion matrix, class order `normal`, `accident`: `[[6, 31], [9, 28]]`
- Normal false-positive rate: 31 / 37 = 83.8%

Interpretation: scaling from P01 to P02 did not fix the core failure. The baseline still overpredicts accident and generates too many false positives on normal clips.

## Architecture Transition

Decision D007 records the pivot from ResNet18 + GRU video classification to YOLO-based object-centric temporal event reasoning.

Reason:

- T01 showed temporal information matters.
- P01 and P02 showed clip-level RGB classification remains weak for RADS.
- The P02 baseline has an 83.8% false-positive rate on normal clips.
- The five active architecture specs require detection, tracking, trajectories, interactions, event localization, severity, and structured output.

The ResNet18 + GRU work remains the baseline. It should not be deleted or rewritten. Future RADS phases should compare against P02 where a clip-level comparison is meaningful.
