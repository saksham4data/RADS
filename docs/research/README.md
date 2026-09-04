# RADS Research History

This directory records how RADS was developed, not only the latest design.

## Read First

- `dataset_lineage.md` - dataset versions, splits, known statistics, and what is missing.
- `experiment_history.md` - EDA, E01-E07, T01, P01, P02, and the architecture transition.
- `repository_audit_phase0.md` - final repository hygiene audit after Phase 0.
- `decisions.md` - decision log, including the transition to the YOLO/object-centric architecture.

## Existing Research Areas

- `investigations/` - focused audits such as frame decoding reliability and metadata leakage.
- `temporal/` - the ResNet18 + GRU temporal proof-of-concept phase and PICEK transition planning.

## Important Boundary

The older EDA was performed against earlier dataset metadata, especially `Datasets/processed/global_master_metadata.csv`.

It was not performed against the current P02 500-video balanced dataset. Any statistics from older EDA must be interpreted as historical context, not as current P02 dataset statistics.
