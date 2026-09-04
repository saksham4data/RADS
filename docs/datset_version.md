# Dataset Version History

This file is retained for historical continuity. For the fuller reconstructed lineage, see `docs/research/dataset_lineage.md`.

## Version 1.0

- **Release Date**: 2026-08-05

- **Description**: Initial merged dataset after preprocessing and EDA

- **Datasets**:
    - Picek
    - TUDAT
    - KAGGLE IMAGES

- **Records**: 5338

- **Features**: 51

- **Features count by dataset**:
    - Picek: 51
    - TUDAT: 51
    - KAGGLE IMAGES: 51

- **Status**: Frozen

- **EDA**: Completed for this historical merged metadata version

- **Training**: Started for expiremental model using only TUDAT Dataset

## Later Derived Datasets

- **TUDAT v2**: 93-record deduplicated TUDAT benchmark used by E07 and T01.
- **P01 PICEK binary**: 200 clips, 100 accident and 100 normal, used by the P01 ResNet18 + GRU experiment.
- **P02 PICEK binary**: current working 500-video balanced dataset, 250 accident and 250 normal, with a 350 / 76 / 74 train/validation/test split.

The older EDA in `eda/` and `docs/eda/` should not be treated as P02 EDA.
