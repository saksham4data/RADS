# RADS — Training Infrastructure (Pipeline 4)

Modular, reusable training infrastructure for the RADS project.
Validates the complete ML workflow using a lightweight baseline model
before moving to object detection.

## Quick Start

```bash
# 1. Install training dependencies
pip install -r requirements-training.txt

# 2. Run training (1-epoch sanity check by default)
python training/train.py

# 3. Evaluate on validation set
python training/validate.py --checkpoint training/outputs/latest/checkpoints/best.pt

# 4. Run inference on a video
python training/predict.py --checkpoint training/outputs/latest/checkpoints/best.pt --input path/to/video.mp4
```

## Architecture

```
training/
├── config/                     # YAML configuration files
│   └── training_config.yaml    # Central configuration
├── configs/                    # Python config loader
│   └── config.py               # TrainingConfig dataclass
├── datasets/                   # Data pipeline
│   ├── frame_dataset.py        # Generic VideoFrameDataset
│   ├── transforms.py           # Train/val augmentation
│   └── dataloader.py           # DataLoader factory
├── models/                     # Model definitions
│   ├── classification_model.py # Multi-backbone classifier
│   └── model_factory.py        # Registry (resnet18, efficientnet_b0)
├── losses/                     # Loss functions
│   └── classification_loss.py  # CrossEntropyLoss + class weights
├── metrics/                    # Evaluation metrics
│   └── classification_metrics.py # Top-1 acc, F1, confusion matrix
├── callbacks/                  # Training callbacks
│   ├── checkpoint.py           # Save/resume checkpoints
│   ├── early_stopping.py       # Patience-based stopping
│   └── lr_monitor.py           # Learning rate logging
├── engine/                     # Training loops
│   ├── trainer.py              # Core training engine
│   └── evaluator.py            # Standalone evaluation
├── utils/                      # Shared utilities
│   ├── config.py               # Config re-export
│   ├── logger.py               # Dual-sink JSON + console logger
│   ├── wandb_manager.py        # W&B lifecycle manager
│   ├── system_monitor.py       # SystemMonitor (from eda)
│   ├── output_manager.py       # Timestamped output directories
│   ├── seed.py                 # Global seed setter
│   └── device.py               # GPU/CPU detection
├── notebooks/                  # Verification notebooks
├── outputs/                    # Timestamped run outputs
│   ├── checkpoints/
│   ├── metrics/
│   ├── predictions/
│   └── logs/
├── train.py                    # Training entry point
├── validate.py                 # Validation entry point
├── test.py                     # Test entry point
└── predict.py                  # Inference entry point
```

## Available Models

| Name | Backbone | Feature Dim | Config Key |
|---|---|---|---|
| ResNet-18 | `torchvision.models.resnet18` | 512 | `model.name: "resnet18"` |
| EfficientNet-B0 | `torchvision.models.efficientnet_b0` | 1280 | `model.name: "efficientnet_b0"` |

## Configuration

All settings are controlled via `training/config/training_config.yaml`.
Key options:

| Section | Key | Default | Description |
|---|---|---|---|
| `data` | `dataset_name` | `"tudat"` | Dataset to train on |
| `data` | `frames_per_video` | `5` | Frames sampled per video |
| `model` | `name` | `"resnet18"` | Model backbone |
| `training` | `epochs` | `1` | Number of training epochs |
| `training` | `batch_size` | `16` | Batch size |
| `optimizer` | `lr` | `0.001` | Learning rate |

## CLI Usage

```bash
# Training with custom config
python training/train.py --config path/to/custom.yaml

# Resume training
python training/train.py --resume training/outputs/latest/checkpoints/last.pt

# Validate
python training/validate.py --checkpoint path/to/best.pt

# Test (final evaluation)
python training/test.py --checkpoint path/to/best.pt

# Predict on video(s)
python training/predict.py --checkpoint best.pt --input video.mp4
python training/predict.py --checkpoint best.pt --input videos/
```

## Output Structure

Each training run creates a timestamped directory:

```
training/outputs/2026-08-05_10-00-00/
├── checkpoints/
│   ├── best.pt                # Best model (by val_loss)
│   └── last.pt                # Latest model
├── metrics/
│   ├── metrics.json           # Final metrics
│   └── training_history.json  # Loss/metric curves
├── predictions/
│   ├── val_predictions.json   # Prediction results
│   └── val_confusion_matrix.json
├── logs/
│   └── training_*.log         # Run-level log
└── manifest.json              # Run metadata
```

## Design Principles

- **Reuse, don't duplicate** — SystemMonitor imported from EDA, patterns mirrored
- **Dataset-agnostic** — Switch datasets via YAML (`dataset_name: "picek"`)
- **Backbone-agnostic** — Switch models via YAML (`model.name: "efficientnet_b0"`)
- **Reproducible** — Global seeding, config logging, W&B tracking
- **Production-ready** — CLI entry points, checkpoint resume, manifest/history
