# Phase 1: RADS P01 Diagnostic Rerun Decisions

## Background
An independent investigation of the RADS P01 repository uncovered multiple issues affecting the training pipeline:
1. **Critical Bug**: The early stopping logic defaulted to `mode="min"`, but the monitored metric `val/f1_accident` needs to be maximized. This resulted in premature termination after 7-10 epochs for all previous experiments, leaving models untrained (selecting epoch-0 checkpoints).
2. **Aggressive Scheduler**: `StepLR(step_size=5, gamma=0.1)` was too aggressive, dropping learning rate by 10x too early in the training process.
3. **Overparameterization**: An unfrozen `ResNet18` backbone and a large `GRU` (`hidden_dim=256`) were prone to overfitting on the small 140-video training set.

## Changes Made
To address these issues and run the Phase 1 diagnostic (P01-R01), the following changes were applied to `training/config/training_config_p01_small.yaml` and `training/train.py`:

1. **Fixed Early Stopping Bug**: 
   - Updated `train.py` to use `config.monitor_mode` (which is `"max"`) instead of the hardcoded `"min"` default for Early Stopping.
   - Explicitly added `mode: "max"` to the `early_stopping` section in `training_config_p01_small.yaml`.
   - Increased early stopping `patience` from 5 to 10 epochs.

2. **Replaced Learning Rate Scheduler**:
   - Replaced `step_lr` with `cosine` (`CosineAnnealingLR`) scheduler for smoother convergence across the 50-epoch budget.

3. **Adjusted Model Architecture to Prevent Overfitting**:
   - Set `freeze_backbone: true` to lock the ImageNet-pretrained weights of ResNet18.
   - Reduced GRU `hidden_dim` from 256 to 128.
   - Increased classifier `dropout` from 0.3 to 0.5.

4. **Tuned Optimizer**:
   - Switched from learning rate `0.001` to `0.0001` for the Adam optimizer to provide stable learning with the frozen backbone.

## Objective
By applying these fixes, the goal of Phase 1 is to determine if the temporal model can learn accident detection features effectively on the 200-video dataset (Target: Validation AUROC > 0.60, Test AUROC > 0.55). If successful, this validates the architecture before scaling to the 500-video dataset.
