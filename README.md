# RADS

RADS is a research-oriented road accident detection and severity analysis project.

The current direction is an object-centric temporal pipeline:

```text
video -> YOLO detection -> tracking -> trajectories -> motion features
-> object interactions -> accident event reasoning -> localization
-> severity estimate -> structured output / visualization
```

The repository also preserves the earlier ResNet18 + GRU video-classification baseline and the research path that led to the current architecture. Weak or failed experiments are intentionally documented because they explain why RADS moved from clip-level visual classification toward object and event understanding.

## Current Status

Phase 0 is complete. It added only scaffolding for the future RADS pipeline:

- `rads/` package skeleton
- `rads/config/pipeline_config.yaml`
- `requirements.txt`
- `docs/architecture/IMPLEMENTATION_TRACKER.md`

No Phase 1 implementation has started yet.

## Main Areas

```text
rads/                  Future object-centric RADS pipeline scaffold
training/              Existing ResNet18 + GRU baseline and training code
Datasets/              Dataset pipeline code plus lightweight tracked metadata
eda/                   Older EDA tooling and notebooks
docs/architecture/     Active RADS specifications and implementation plan
docs/research/         Dataset lineage, experiments, decisions, audits
scripts/               Dataset construction and utility scripts
```

Large datasets, videos, checkpoints, W&B runs, logs, caches, virtual environments, and generated outputs are intentionally kept out of Git.

## Reproducibility

Install the project environment:

```bash
pip install -r requirements.txt
```

For compatibility, `requirements-training.txt` remains as a thin pointer to `requirements.txt`.


Important tracked references:

- Architecture source of truth: `docs/architecture/`
- Research history: `docs/research/README.md`
- Dataset lineage: `docs/research/dataset_lineage.md`
- Baseline experiments: `docs/research/experiment_history.md`
- Repository hygiene audit: `docs/research/repository_audit_phase0.md`

## Baseline

The current baseline to surpass is P02:

- Dataset: 500 PICEK clips, balanced 250 accident / 250 normal
- Split: 350 train / 76 validation / 74 test
- Model: ResNet18 + GRU
- Test accuracy: 45.9%
- Test AUROC: 0.535
- Normal false-positive rate: 83.8%

See `docs/research/experiment_history.md` for context and caveats.
