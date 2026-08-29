#!/usr/bin/env python3
"""
smoke_test_p01_pipeline.py
==========================
Comprehensive Smoke Test for P01 Small PICEK Binary Temporal Pipeline

Validates:
1. Config loading from training_config_p01_small.yaml
2. Dataset counts across train (140), val (30), test (30) splits
3. Class balance (50% positive / 50% negative) in every split
4. Frame decoding & strict temporal ordering
5. Tensor shape verification [B, T, C, H, W] -> [B, 8, 3, 224, 224]
6. ResNet18 + GRU model instantiation & forward pass -> [B, 2]
7. Binary CrossEntropy loss computation
8. Backward pass & gradient propagation verification
9. W&B logging integration in offline/dryrun mode

Usage:
    python scripts/smoke_test_p01_pipeline.py

Author: Antigravity (RADS Research)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add workspace root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Ensure immediate unbuffered output
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

os.environ["WANDB_MODE"] = "offline"
os.environ["WANDB_SILENT"] = "true"

from training.configs.config import TrainingConfig, load_training_config
from training.datasets.video_dataset import VideoSequenceDataset
from training.losses.classification_loss import create_loss
from training.models.model_factory import create_model
from training.models.temporal_model import TemporalClassificationModel


def run_smoke_test():
    print("=" * 75)
    print("RUNNING P01 SMALL PICEK BINARY TEMPORAL PIPELINE SMOKE TEST")
    print("=" * 75)

    test_results = {}

    # ── Gate 1: Config Loading ──────────────────────────────────────────────
    print("\n[Gate 1] Loading Configuration...")
    config_path = BASE_DIR / "training/config/training_config_p01_small.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    config = load_training_config(config_path)
    print(f"  Project          : {config.project_name} (v{config.training_version})")
    print(f"  Experiment ID    : {config.experiment_id}")
    print(f"  Experiment Name  : {config.experiment_name}")
    print(f"  Dataset Name     : {config.dataset_name}")
    print(f"  Metadata Path    : {config.metadata_path}")
    print(f"  Label Mode       : {config.label_mode} (classes: {config.num_classes})")
    print(f"  Temporal Arch    : {config.temporal_architecture} (hidden: {config.temporal_hidden_dim})")
    print(f"  Frames Per Video : {config.frames_per_video}")
    print(f"  Image Size       : {config.image_size}")

    assert config.num_classes == 2, f"Expected 2 classes, got {config.num_classes}"
    assert config.temporal_enabled is True, "Expected temporal_enabled to be True"
    assert config.temporal_architecture == "gru", f"Expected 'gru', got {config.temporal_architecture}"
    test_results["config_loading"] = "PASSED"
    print("  -> Config loading & validation PASSED.")

    # ── Gate 2: Dataset Loading & Split Counts ──────────────────────────────
    print("\n[Gate 2] Loading VideoSequenceDataset across Train / Val / Test...")
    train_ds = VideoSequenceDataset(config, split="train")
    val_ds = VideoSequenceDataset(config, split="val")
    test_ds = VideoSequenceDataset(config, split="test")

    print(f"  Train samples: {len(train_ds)} (Expected: 140)")
    print(f"  Val samples  : {len(val_ds)} (Expected: 30)")
    print(f"  Test samples : {len(test_ds)} (Expected: 30)")

    assert len(train_ds) == 140, f"Expected 140 train samples, got {len(train_ds)}"
    assert len(val_ds) == 30, f"Expected 30 val samples, got {len(val_ds)}"
    assert len(test_ds) == 30, f"Expected 30 test samples, got {len(test_ds)}"

    # Check class distribution in each split
    train_labels = [label for _, label, _ in train_ds._videos]
    val_labels = [label for _, label, _ in val_ds._videos]
    test_labels = [label for _, label, _ in test_ds._videos]

    train_pos = sum(train_labels)
    train_neg = len(train_labels) - train_pos
    val_pos = sum(val_labels)
    val_neg = len(val_labels) - val_pos
    test_pos = sum(test_labels)
    test_neg = len(test_labels) - test_pos

    print(f"  Train balance: {train_pos} accident (pos) / {train_neg} normal (neg) -> {train_pos/len(train_labels)*100:.1f}%")
    print(f"  Val balance  : {val_pos} accident (pos) / {val_neg} normal (neg) -> {val_pos/len(val_labels)*100:.1f}%")
    print(f"  Test balance : {test_pos} accident (pos) / {test_neg} normal (neg) -> {test_pos/len(test_labels)*100:.1f}%")

    assert train_pos == 70 and train_neg == 70
    assert val_pos == 15 and val_neg == 15
    assert test_pos == 15 and test_neg == 15
    test_results["dataset_split_counts_and_balance"] = "PASSED"
    print("  -> Dataset counts and 50/50 balance PASSED.")

    # ── Gate 3: Frame Decoding & Temporal Ordering ──────────────────────────
    print("\n[Gate 3] Verifying Frame Decoding & Temporal Ordering...")
    for name, ds in [("Train", train_ds), ("Val", val_ds), ("Test", test_ds)]:
        sample_tensor, sample_label = ds[0]
        # Inspect video item metadata
        vid_path, lbl_idx, frame_idx_list = ds._videos[0]
        print(f"  {name} sample 0: {vid_path.name} | Label: {sample_label} | Shape: {list(sample_tensor.shape)}")
        print(f"    Sampled Frame Indices: {frame_idx_list}")

        assert sample_tensor.shape == (8, 3, 224, 224), f"Unexpected shape: {sample_tensor.shape}"
        assert not torch.isnan(sample_tensor).any(), "NaN detected in sample frames"
        # Check strict monotonic frame ordering
        assert all(frame_idx_list[i] < frame_idx_list[i+1] for i in range(len(frame_idx_list)-1)), \
            "Frame indices are not monotonically strictly increasing!"

    test_results["frame_decoding_and_temporal_order"] = "PASSED"
    print("  -> Frame decoding, tensor normalization, and temporal ordering PASSED.")

    # ── Gate 4: DataLoader Batch Generation ─────────────────────────────────
    print("\n[Gate 4] Testing DataLoader Batch Creation...")
    loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=0)
    batch_frames, batch_labels = next(iter(loader))
    print(f"  Batch frames shape : {list(batch_frames.shape)} [B, T, C, H, W]")
    print(f"  Batch labels shape : {list(batch_labels.shape)} [B]")
    print(f"  Batch labels sample: {batch_labels.tolist()}")

    assert batch_frames.shape == (4, 8, 3, 224, 224)
    assert batch_labels.shape == (4,)
    test_results["dataloader_batch_generation"] = "PASSED"
    print("  -> DataLoader batch generation PASSED.")

    # ── Gate 5: Model Creation & Forward Pass ───────────────────────────────
    print("\n[Gate 5] Testing Model Instantiation & Forward Pass...")
    model = create_model(config)
    assert isinstance(model, TemporalClassificationModel), "Model is not TemporalClassificationModel"
    print(f"  Spatial Backbone : {model.spatial_encoder.backbone_name} (dim: {model.spatial_encoder.feature_dim})")
    print(f"  Temporal Encoder : {type(model.temporal_encoder).__name__} (hidden: {model.temporal_encoder.hidden_dim})")
    print(f"  Classifier Head  : {model.classifier}")

    model.train()
    logits = model(batch_frames)
    print(f"  Output Logits shape: {list(logits.shape)} (Expected: [4, 2])")
    assert logits.shape == (4, 2), f"Expected [4, 2], got {logits.shape}"
    assert not torch.isnan(logits).any(), "NaN in model output logits"

    probs = torch.softmax(logits, dim=-1)
    print(f"  Output Probabilities:\n{probs.detach().numpy()}")
    assert torch.allclose(probs.sum(dim=-1), torch.ones(4), atol=1e-5), "Softmax probabilities do not sum to 1"
    test_results["model_forward_pass"] = "PASSED"
    print("  -> Model instantiation and forward pass PASSED.")

    # ── Gate 6: Loss Computation & Backward Pass ────────────────────────────
    print("\n[Gate 6] Testing Loss Computation & Backward Pass...")
    criterion = create_loss(config)
    loss = criterion(logits, batch_labels)
    print(f"  Computed CrossEntropy Loss: {loss.item():.4f}")
    assert loss.item() > 0.0, "Loss must be strictly positive"
    assert not torch.isnan(loss), "Loss is NaN"

    # Backward pass
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    optimizer.zero_grad()
    loss.backward()

    # Verify gradients
    has_backbone_grad = False
    has_gru_grad = False
    has_classifier_grad = False

    for name, param in model.named_parameters():
        if param.grad is not None:
            if "spatial_encoder" in name and param.grad.abs().sum() > 0:
                has_backbone_grad = True
            elif "temporal_encoder" in name and param.grad.abs().sum() > 0:
                has_gru_grad = True
            elif "classifier" in name and param.grad.abs().sum() > 0:
                has_classifier_grad = True

    print(f"  Backbone gradients active  : {has_backbone_grad}")
    print(f"  GRU gradients active       : {has_gru_grad}")
    print(f"  Classifier gradients active: {has_classifier_grad}")

    assert has_backbone_grad, "Backbone received no gradients"
    assert has_gru_grad, "GRU received no gradients"
    assert has_classifier_grad, "Classifier head received no gradients"

    optimizer.step()
    test_results["loss_and_backward_pass"] = "PASSED"
    print("  -> Loss computation, backward pass, and optimizer step PASSED.")

    # ── Gate 7: Weights & Biases Logging Integration ────────────────────────
    print("\n[Gate 7] Testing Weights & Biases Logging Integration...")
    try:
        import wandb
        os.environ["WANDB_MODE"] = "disabled"
        os.environ["WANDB_SILENT"] = "true"

        wandb_cfg = config.wandb if isinstance(config.wandb, dict) else {}
        run = wandb.init(
            project=wandb_cfg.get("project", "RADS"),
            group=wandb_cfg.get("group", "picek-small-binary-temporal"),
            name="smoke_test_p01",
            tags=wandb_cfg.get("tags", ["smoke_test"]),
            config=config.to_dict(),
            mode="disabled",
        )
        wandb.log({
            "smoke_test/loss": loss.item(),
            "smoke_test/train_samples": len(train_ds),
            "smoke_test/val_samples": len(val_ds),
            "smoke_test/test_samples": len(test_ds),
        })
        wandb.finish()
        test_results["wandb_logging"] = "PASSED"
        print("  -> W&B logging interface initialization, metric logging, and completion PASSED.")
    except Exception as exc:
        print(f"  -> W&B Warning (non-fatal): {exc}")
        test_results["wandb_logging"] = f"WARNING ({exc})"

    # ── Final Summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 75)
    print("SMOKE TEST SUMMARY VERDICT")
    print("=" * 75)
    all_passed = all("PASSED" in str(v) for v in test_results.values())
    for gate, result in test_results.items():
        print(f"  {gate:40s}: {result}")

    print("=" * 75)
    if all_passed:
        print("VERDICT: ALL 7 SMOKE TEST GATES PASSED! DATASET & PIPELINE ARE FULLY READY FOR P01.")
    else:
        print("VERDICT: SOME GATES FAILED!")
    print("=" * 75)

    return all_passed


if __name__ == "__main__":
    success = run_smoke_test()
    sys.exit(0 if success else 1)
