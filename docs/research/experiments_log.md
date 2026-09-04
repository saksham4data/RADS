# Experiments log 

## This is a place for logging experiments and decision made by the author of the project

- For each experiment I will keep a record of 
    - date
    - config used 
    - training details 
    - results
    
    
- W&B tracking and artifact management will be used to track experiments and manage artifacts.


## Log Table 
| ID | Date | Dataset | Labels | Model | Frames | Epochs | Early Stopping | Batch | Main Change | Hypothesis | Result | Status |
|---|---|---|---|---|---:|---:|---:|---:|---|---|---|---|
| E01 | 06/08/2026 | TUDAT | 3 | ResNet18 | 5 | 20 | Enabled | 32 | Initial baseline | Establish multiclass baseline | Strong `challenging` class bias and poor validation performance | Completed |
| E02 | 08/08/2026 | TUDAT | Binary | ResNet18 | 5 | 20 | Disabled | 32 | Merged `challenging → accident` | Focus the model on accident vs non-accident | Training performance improved, but severe overfitting was observed | Completed |
| E03 | 08/08/2026 | TUDAT | Binary | ResNet18 | 5 | 50 | Enabled, patience 5 | 32 | Enable early stopping and increase maximum epochs | Allow more training while stopping when validation performance stops improving | Training reached 80% accuracy / 0.798 macro-F1, but validation and test performance remained poor. Test confusion matrix showed 5/49 accidents detected. Large train-validation gap indicates severe overfitting remains. | Completed|
| E04 | 10/08/2026 | TUDAT | Binary | ResNet18 | 8 | 50 | Enabled, patience 5 | 32 | Increased frames/video from 5 → 8 | Test accuracy improved to ~36.4% and accident recall to ~21.5%, but validation performance remained poor and severe overfitting persisted. | Completed |
| E05 | 10/08/2026 | TUDAT | Binary | ResNet18 | 8 | 50 | Enabled, patience 5 | 32 | Enable `conservative_v1` data augmentation | Reduce severe overfitting by applying mild spatial and photometric augmentations during training | Mild improvement in regularization, but frame-level accident recall remained low | Completed |
| E06 | 13/08/2026 | TUDAT | Binary | ResNet18 | 8 | 50 | Enabled, patience 5 | 32 | Checkpoint selection on `val/f1_accident` | Align checkpoint saving directly with minority accident detection | Saved checkpoint prioritizing accident F1; established clean metric monitoring | Completed |
| E07 | 24/08/2026 | TUDAT v2 | Binary | ResNet18 | 8 | 50 | Enabled, patience 5 | 32 | Deduplicated TUDAT v2 dataset (93 records, frozen split) | Clean benchmark without duplicate video leakage across train/val/test splits | Val F1 accident = 0.3119, Test Acc = 33.33%, Test Recall (Accident) = 42.86%, Test F1 = 0.3750. Establishes frozen spatial baseline. | Completed (Frozen Baseline) |
| T01 | 24/08/2026 | TUDAT v2 | Binary | ResNet18 + GRU (dim=256) | 8 | 50 | Enabled, patience 5 | 32 | Add 1-layer GRU temporal encoder over ResNet18 features | Temporal sequence modeling across video clips captures dynamic accident progression | Val F1 accident = 0.6316 (+102.5% vs E07), Test Acc = 46.67% (+40.0% vs E07), Test Recall (Accident) = 100.0% (7/7 caught, 0 missed), Test F1 = 0.6364 (+69.7% vs E07). | Completed (Frozen Temporal POC) |