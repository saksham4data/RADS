# Repository Audit After Phase 0

Date: 2026-09-05

Scope: final repository and research-history audit after Phase 0 scaffolding. No later RADS phases were implemented.

## Current Source Structure

```text
RADS/
  README.md
  .gitignore
  requirements.txt
  requirements-training.txt
  Datasets/
  docs/
  eda/
  rads/
  scripts/
  training/
```

## What Should Remain Tracked

- Source code:
  - `rads/`
  - `training/`
  - `Datasets/metadata_gen/`
  - `Datasets/pipeline/`
  - `Datasets/reporting/`
  - `scripts/`
- Configuration:
  - `training/config/*.yaml`
  - `rads/config/pipeline_config.yaml`
  - requirements files
- Lightweight dataset definitions:
  - `Datasets/processed/global_master_metadata.csv`
  - `Datasets/processed/tudat/v2/*`
  - P01/P02 metadata should be considered for tracking if size is acceptable and the dataset is meant to be reproducible from Git without video files.
- Research docs:
  - `docs/architecture/`
  - `docs/research/`
  - `docs/eda/`

## Should Stay Local / Ignored

- `.venv/`
- `.pytest_cache/`
- `__pycache__/`
- `logs/`
- `wandb/`
- `training/wandb/`
- `eda/notebooks/wandb/`
- `Datasets/raw/`
- `Datasets/processed/picek_sorted/`
- generated videos and image outputs
- model checkpoints: `*.pt`, `*.pth`, `*.ckpt`, etc.
- generated training outputs under `training/outputs/`
- generated EDA outputs under `eda/reports/`, `eda/figures/`, and `eda/exports/`
- `scratch/`

## Outdated Or Duplicated Outputs

- Root-level generated prediction files under `training/p02_prediction.json` and `training/prediction_result.json` are local generated outputs. Important conclusions have been summarized in `experiment_history.md`.
- Timestamped training outputs contain useful historical artifacts but should not be committed wholesale because they include generated files and large checkpoints.
- W&B run folders duplicate experiment tracking data and should remain local or in W&B, not Git.
- Older EDA generated outputs reflect historical global metadata EDA, not P02.

## Historical Artifacts To Preserve Outside Main Git Tracking

- `training/outputs/archive/tudat_e01_e06/`
- `training/outputs/archive/tudat_e07_t01/`
- `training/outputs/archive/p01_hyperopt_study/`
- generated EDA reports/figures/exports
- W&B run folders

These should be preserved locally or moved to external artifact storage. They explain the research history, but they are too generated/heavy for the main Git repository.

## Safely Disposable Generated Material

- Python bytecode caches
- pytest cache
- empty or transient logs
- local W&B debug logs if the corresponding W&B run is synced or no longer needed locally

These do not carry unique research value once important findings are summarized.

## Items That Could Break Phase 0 If Moved Or Removed

- `rads/` directory and `__init__.py` files
- `rads/config/pipeline_config.yaml`
- `requirements.txt`
- `docs/architecture/IMPLEMENTATION_TRACKER.md`
- existing training configs and code used as baseline references
- dataset metadata referenced by configs:
  - `Datasets/processed/tudat/v2/global_master_metadata_v2.csv`
  - `Datasets/processed/picek/p01_small_binary/picek_p01_small_binary_metadata.csv`
  - `Datasets/processed/picek/p02_binary/picek_p02_binary_metadata.csv`

Large video files can remain untracked, but paths in metadata/configs must continue to resolve locally for training/evaluation.

## Audit Verdict

The repository is understandable for a new contributor after adding the top-level README and research history documents.

The requirement files were consolidated after this audit: `requirements.txt` is the single dependency source of truth, while `requirements-training.txt` remains only as a compatibility wrapper. The former `requirements-pipeline.txt` was removed because its unique Phase 0 dependency (`ultralytics`) now lives in `requirements.txt`.

The remaining important GitHub-readiness issue is deciding whether to commit the currently untracked Phase 0 and P02/P01-related source/config files. The heavy generated artifacts are now covered by ignore policy and summarized in documentation.
