"""
Validation gates for the Small PICEK Temporal Subset (v1_small).

Validates:
  1. Subset manifest integrity and physical video existence.
  2. Zero cross-split leakage and stratified class balance.
  3. Correct label and class mapping for multi-class collision classification.
  4. VideoSequenceDataset produces [T, C, H, W] = [8, 3, 224, 224] sequence tensors.
  5. Model forward pass through ResNet18 + GRU produces [B, 5] logits.
  6. Loss computation and gradient flow through a minimal batch.
"""

from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn

from training.configs.config import load_training_config
from training.datasets.dataloader import create_temporal_dataloaders
from training.datasets.video_dataset import VideoSequenceDataset
from training.losses.classification_loss import create_loss
from training.models.model_factory import create_model


class TestPicekSubsetValidation(unittest.TestCase):
    """Validation gates for the small PICEK subset and temporal pipeline."""

    @classmethod
    def setUpClass(cls):
        cls.config_path = Path("training/config/training_config_picek_small.yaml")
        cls.config = load_training_config(cls.config_path)
        cls.metadata_path = Path("Datasets/processed/picek/v1_small/picek_v1_small_metadata.csv")

    def test_01_metadata_manifest_exists_and_valid(self):
        """Gate 1: Metadata file exists, has exactly 100 records and expected columns."""
        self.assertTrue(self.metadata_path.is_file(), "PICEK v1_small metadata CSV missing.")
        df = pd.read_csv(self.metadata_path)
        self.assertEqual(len(df), 100, "Subset must contain exactly 100 records.")
        required_cols = {"video_id", "original_path", "type", "split", "split_in_distribution", "accident_frame"}
        self.assertTrue(required_cols.issubset(df.columns), f"Missing required columns in metadata: {required_cols - set(df.columns)}")

    def test_02_zero_leakage_and_balanced_splits(self):
        """Gate 2: Verify zero cross-split video leakage and exact 70/15/15 balanced distribution."""
        df = pd.read_csv(self.metadata_path)
        train_videos = set(df[df["split_in_distribution"] == "train"]["video_id"])
        val_videos = set(df[df["split_in_distribution"] == "val"]["video_id"])
        test_videos = set(df[df["split_in_distribution"] == "test"]["video_id"])

        self.assertEqual(len(train_videos), 70, "Train split must have 70 unique videos.")
        self.assertEqual(len(val_videos), 15, "Val split must have 15 unique videos.")
        self.assertEqual(len(test_videos), 15, "Test split must have 15 unique videos.")

        # Zero leakage
        self.assertEqual(len(train_videos & val_videos), 0, "Leakage detected between train and val.")
        self.assertEqual(len(train_videos & test_videos), 0, "Leakage detected between train and test.")
        self.assertEqual(len(val_videos & test_videos), 0, "Leakage detected between val and test.")

        # Check per-class balance
        class_counts = df.groupby(["split_in_distribution", "type"]).size()
        for cls_name in ["head-on", "rear-end", "sideswipe", "single", "t-bone"]:
            self.assertEqual(class_counts.get(("train", cls_name), 0), 14)
            self.assertEqual(class_counts.get(("val", cls_name), 0), 3)
            self.assertEqual(class_counts.get(("test", cls_name), 0), 3)

    def test_03_physical_video_files_exist_on_disk(self):
        """Gate 3: All 100 video files must physically exist and be readable."""
        df = pd.read_csv(self.metadata_path)
        raw_base = Path("Datasets/raw")
        for orig_path in df["original_path"]:
            p1 = raw_base / "picekl" / orig_path
            p2 = raw_base / "picek" / orig_path
            p3 = raw_base / orig_path
            exists = p1.is_file() or p2.is_file() or p3.is_file()
            self.assertTrue(exists, f"Physical video file not found for: {orig_path}")

    def test_04_dataset_sequence_loading(self):
        """Gate 4: VideoSequenceDataset returns [T, C, H, W] = [8, 3, 224, 224] tensors."""
        dataset = VideoSequenceDataset(self.config, split="val")
        self.assertEqual(len(dataset), 15)
        self.assertEqual(dataset.num_classes, 5)

        sample, label = dataset[0]
        self.assertIsInstance(sample, torch.Tensor)
        self.assertEqual(sample.shape, (8, 3, 224, 224), f"Unexpected sequence shape: {sample.shape}")
        self.assertIsInstance(label, int)
        self.assertTrue(0 <= label < 5, f"Label {label} outside valid range [0, 4]")

    def test_05_model_forward_loss_and_backward(self):
        """Gate 5: Minimal batch passes through model forward, loss, and backward pass."""
        device = torch.device("cpu")
        model = create_model(self.config)
        model.to(device)
        model.train()

        loss_fn = create_loss(self.config, device=device)

        # Batch of 2 video sequences
        dummy_batch = torch.randn(2, 8, 3, 224, 224, device=device)
        dummy_targets = torch.tensor([0, 4], dtype=torch.long, device=device)

        logits = model(dummy_batch)
        self.assertEqual(logits.shape, (2, 5), f"Logits shape mismatch: {logits.shape}")

        loss = loss_fn(logits, dummy_targets)
        self.assertFalse(torch.isnan(loss), "Computed loss is NaN.")
        self.assertGreater(loss.item(), 0.0, "Computed loss must be positive.")

        # Test backward pass
        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"Gradient missing for {name}")


if __name__ == "__main__":
    unittest.main()
