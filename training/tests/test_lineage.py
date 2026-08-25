"""
Tests for checkpoint lineage tracking, evaluation manifest recording, and experiment reporting.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from training.callbacks.checkpoint import CheckpointManager
from training.configs.config import TrainingConfig, load_training_config
from training.engine.evaluator import Evaluator
from training.utils.experiment_report import generate_experiment_report
from training.utils.output_manager import TrainingOutputManager


class DummyClassificationModel(nn.Module):
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.fc = nn.Linear(4, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class TestCheckpointManagerDescribe(unittest.TestCase):
    """Tests for CheckpointManager.describe lineage extraction."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_describe_with_source_manifest(self):
        run_dir = self.root / "outputs" / "2026-08-24_20-12-10"
        ckpts_dir = run_dir / "checkpoints"
        ckpts_dir.mkdir(parents=True, exist_ok=True)

        source_manifest = {
            "timestamp": "2026-08-24T20:12:10.119156+00:00",
            "run_name": "T01 Temporal GRU on TUDAT v2",
            "git_commit": "fe1a7f22bc27cac2c430d7f07ae3facd9c09778c",
            "wandb_run_id": "q8o10v54",
            "wandb_run_url": "https://wandb.ai/saksham4data-vinkura/RADS/runs/q8o10v54",
            "execution": {
                "duration_seconds": 3442.91,
                "status": "success",
                "total_epochs": 6,
                "best_metric": 0.631578947368421,
                "best_epoch": 0,
            },
        }
        (run_dir / "manifest.json").write_text(json.dumps(source_manifest), encoding="utf-8")

        ckpt_path = ckpts_dir / "best.pt"
        state = {
            "epoch": 0,
            "model_state_dict": {},
            "optimizer_state_dict": {},
            "scheduler_state_dict": None,
            "metrics": {"val/f1_accident": 0.631578947368421, "val_loss": 2.099},
            "monitor_metric": "val/f1_accident",
            "mode": "max",
        }
        torch.save(state, ckpt_path)

        config = MagicMock()
        config.monitor_metric = "val/f1_accident"
        config.monitor_mode = "max"

        desc = CheckpointManager.describe(ckpt_path, state=state, config=config)

        self.assertEqual(desc["path"], str(ckpt_path.resolve()))
        self.assertEqual(desc["epoch"], 0)
        self.assertEqual(desc["training_epoch"], 1)
        self.assertEqual(desc["monitored_metric"], "val/f1_accident")
        self.assertEqual(desc["monitor_mode"], "max")
        self.assertEqual(desc["best_metric"], 0.631578947368421)
        self.assertEqual(desc["source_run_name"], "T01 Temporal GRU on TUDAT v2")
        self.assertEqual(desc["source_wandb_run_id"], "q8o10v54")
        self.assertEqual(desc["source_best_epoch"], 0)
        self.assertEqual(desc["source_best_metric"], 0.631578947368421)
        self.assertEqual(desc["source_git_commit"], "fe1a7f22bc27cac2c430d7f07ae3facd9c09778c")

    def test_describe_without_source_manifest(self):
        ckpt_path = self.root / "standalone.pt"
        state = {
            "epoch": 2,
            "metrics": {"val_loss": 0.42},
            "monitor_metric": "val_loss",
            "mode": "min",
        }
        torch.save(state, ckpt_path)

        desc = CheckpointManager.describe(ckpt_path, state=state)
        self.assertEqual(desc["path"], str(ckpt_path.resolve()))
        self.assertEqual(desc["epoch"], 2)
        self.assertEqual(desc["training_epoch"], 3)
        self.assertEqual(desc["monitored_metric"], "val_loss")
        self.assertEqual(desc["best_metric"], 0.42)


class TestTrainingOutputManagerManifest(unittest.TestCase):
    """Tests for TrainingOutputManager manifest recording and file tracking."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        cfg_file = self.root / "config.yaml"
        cfg_file.write_text("project:\n  name: RADS\noutput:\n  maintain_latest: false\n  track_run_history: false\n", encoding="utf-8")
        self.config = load_training_config(cfg_file)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_manifest_records_config_path_and_checkpoints(self):
        om = TrainingOutputManager(self.config, allow_checkpoint_writes=True)
        # Create a checkpoint directly in checkpoints directory
        best_pt = om.checkpoints_dir / "best.pt"
        best_pt.write_text("mock", encoding="utf-8")

        manifest = om.finalize(status="success")
        self.assertEqual(manifest["config_path"], str((self.root / "config.yaml").resolve()))
        self.assertIn("best.pt", manifest["files"]["checkpoints"])

    def test_manifest_preserves_evaluation_checkpoint_lineage(self):
        om = TrainingOutputManager(self.config, allow_checkpoint_writes=False)
        extra_checkpoint = {
            "path": "E:/Rads/training/outputs/2026-08-24_20-12-10/checkpoints/best.pt",
            "source_run_name": "T01 Temporal GRU on TUDAT v2",
            "source_wandb_run_id": "q8o10v54",
            "training_epoch": 1,
            "best_metric": 0.6315,
        }
        manifest = om.finalize(
            status="test",
            extra_manifest={"checkpoint": extra_checkpoint},
        )
        self.assertEqual(manifest["execution"]["status"], "test")
        self.assertEqual(manifest["checkpoint"]["source_wandb_run_id"], "q8o10v54")
        self.assertEqual(manifest["checkpoint"]["training_epoch"], 1)


class TestEvaluatorArtifactSaving(unittest.TestCase):
    """Tests that Evaluator outputs test metrics, predictions, and confusion matrices."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        cfg_file = self.root / "config.yaml"
        cfg_file.write_text("data:\n  label_mode: binary\noutput:\n  maintain_latest: false\n  track_run_history: false\n", encoding="utf-8")
        self.config = load_training_config(cfg_file)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_evaluator_saves_all_artifacts(self):
        model = DummyClassificationModel(num_classes=2)
        loss_fn = nn.CrossEntropyLoss()
        device = torch.device("cpu")
        class_names = ["accident", "non-accident"]

        evaluator = Evaluator(model, loss_fn, device, class_names)
        x = torch.randn(10, 4)
        y = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=torch.long)
        loader = DataLoader(TensorDataset(x, y), batch_size=5)

        om = TrainingOutputManager(self.config, allow_checkpoint_writes=False)
        metrics = evaluator.evaluate(loader, split_name="test", output_manager=om)

        self.assertIn("test/loss", metrics)
        self.assertIn("test/top1_accuracy", metrics)
        self.assertIn("test_metrics.json", om._exported_metrics)
        self.assertIn("test_predictions.json", om._exported_predictions)
        self.assertIn("test_confusion_matrix.json", om._exported_predictions)

        # Verify files exist on disk
        self.assertTrue((om.metrics_dir / "test_metrics.json").is_file())
        self.assertTrue((om.predictions_dir / "test_predictions.json").is_file())
        self.assertTrue((om.predictions_dir / "test_confusion_matrix.json").is_file())


class TestExperimentReportGeneration(unittest.TestCase):
    """Tests experiment report generation on training and evaluation runs."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_report_for_evaluation_run(self):
        # 1. Create mock source training run
        source_dir = self.root / "outputs" / "2026-08-24_20-12-10"
        (source_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        (source_dir / "metrics").mkdir(parents=True, exist_ok=True)

        source_manifest = {
            "timestamp": "2026-08-24T20:12:10.119156+00:00",
            "run_name": "T01 Temporal GRU on TUDAT v2",
            "git_commit": "fe1a7f22bc27cac2c430d7f07ae3facd9c09778c",
            "wandb_run_id": "q8o10v54",
            "execution": {
                "duration_seconds": 3442.91,
                "status": "success",
                "total_epochs": 2,
                "best_metric": 0.6315,
                "best_epoch": 0,
            },
        }
        (source_dir / "manifest.json").write_text(json.dumps(source_manifest), encoding="utf-8")
        training_history = {
            "train_loss": [0.54, 0.42],
            "val_loss": [2.09, 1.95],
            "val/f1_accident": [0.6315, 0.60],
        }
        (source_dir / "metrics" / "training_history.json").write_text(json.dumps(training_history), encoding="utf-8")
        (source_dir / "checkpoints" / "best.pt").write_text("mock", encoding="utf-8")

        # 2. Create mock test evaluation run
        test_dir = self.root / "outputs" / "2026-08-24_21-00-00"
        (test_dir / "metrics").mkdir(parents=True, exist_ok=True)
        (test_dir / "predictions").mkdir(parents=True, exist_ok=True)

        test_manifest = {
            "timestamp": "2026-08-24T21:00:00.000000+00:00",
            "run_name": "T01 Temporal GRU on TUDAT v2",
            "config_path": str(self.root / "config.yaml"),
            "model": "resnet18",
            "dataset": "tudat",
            "git_commit": "fe1a7f22bc27cac2c430d7f07ae3facd9c09778c",
            "seed": 42,
            "files": {
                "checkpoints": [],
                "metrics": ["test_metrics.json"],
                "predictions": ["test_predictions.json", "test_confusion_matrix.json"],
                "logs": [],
            },
            "execution": {
                "duration_seconds": 12.5,
                "status": "test",
            },
            "checkpoint": {
                "path": str(source_dir / "checkpoints" / "best.pt"),
                "training_epoch": 1,
                "epoch": 0,
                "monitored_metric": "val/f1_accident",
                "best_metric": 0.6315,
                "source_run_name": "T01 Temporal GRU on TUDAT v2",
                "source_run_dir": str(source_dir),
                "source_wandb_run_id": "q8o10v54",
                "source_git_commit": "fe1a7f22bc27cac2c430d7f07ae3facd9c09778c",
                "source_best_epoch": 0,
                "source_best_metric": 0.6315,
            },
        }
        (test_dir / "manifest.json").write_text(json.dumps(test_manifest), encoding="utf-8")

        test_metrics = {
            "test/loss": 0.85,
            "test/top1_accuracy": 0.70,
            "test/macro_f1": 0.68,
            "test/precision_accident": 0.75,
            "test/recall_accident": 0.66,
            "test/f1_accident": 0.70,
        }
        (test_dir / "metrics" / "test_metrics.json").write_text(json.dumps(test_metrics), encoding="utf-8")

        test_preds = {
            "predictions": [0, 1, 0, 1],
            "targets": [0, 1, 1, 1],
            "confidences": [[0.8, 0.2], [0.1, 0.9], [0.6, 0.4], [0.3, 0.7]],
            "class_names": ["accident", "non-accident"],
        }
        (test_dir / "predictions" / "test_predictions.json").write_text(json.dumps(test_preds), encoding="utf-8")

        test_cm = {
            "class_names": ["accident", "non-accident"],
            "matrix": [[1, 0], [1, 2]],
        }
        (test_dir / "predictions" / "test_confusion_matrix.json").write_text(json.dumps(test_cm), encoding="utf-8")

        report_path = generate_experiment_report(test_dir)
        self.assertTrue(report_path.is_file())

        content = report_path.read_text(encoding="utf-8")
        self.assertIn("q8o10v54", content)
        self.assertIn("T01 Temporal GRU on TUDAT v2", content)
        self.assertIn("0.85", content)  # test loss
        self.assertIn("0.7", content)  # test accuracy
        self.assertIn("VALID", content)


if __name__ == "__main__":
    unittest.main()
