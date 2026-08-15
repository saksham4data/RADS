# ─────────────────────────────────────────────────────────────
# Training Output Manager — Timestamped Versioning & Manifest
# ─────────────────────────────────────────────────────────────
"""
Manages timestamped output directories for training runs:
checkpoints, metrics, predictions, and logs.

Pattern: adapted from ``eda.utils.output_manager.OutputManager``.
"""

from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from training.configs.config import TrainingConfig


class TrainingOutputManager:
    """Creates and manages versioned output directories for training.

    Each training run gets a timestamped subdirectory under
    ``outputs/`` with subdirectories for checkpoints, metrics,
    predictions, and logs.

    Usage::

        om = TrainingOutputManager(config, run_name="sanity_check")
        om.save_metrics(metrics_dict, "epoch_0")
        om.save_predictions(preds, labels, confs, "val_epoch_0")
        om.save_confusion_matrix(cm, class_names, "val_epoch_0")
        om.finalize(wandb_run_id="abc")
    """

    def __init__(
        self,
        config: TrainingConfig,
        run_name: Optional[str] = None,
        *,
        allow_checkpoint_writes: bool = True,
        timestamp: Optional[datetime] = None,
    ) -> None:
        self.config = config
        self.run_name = run_name or config.experiment_name
        self.allow_checkpoint_writes = allow_checkpoint_writes
        self._ts = timestamp or datetime.now(timezone.utc)
        self._ts_str = self._ts.strftime("%Y-%m-%d_%H-%M-%S")

        # Timestamped output root
        self.run_dir = config.outputs_dir / self._ts_str

        # Sub-directories
        self.checkpoints_dir = self.run_dir / "checkpoints"
        self.metrics_dir = self.run_dir / "metrics"
        self.predictions_dir = self.run_dir / "predictions"
        self.logs_dir = self.run_dir / "logs"

        # Track exported files
        self._exported_checkpoints: List[str] = []
        self._exported_metrics: List[str] = []
        self._exported_predictions: List[str] = []
        self._exported_logs: List[str] = []

        # Ensure directories exist
        dirs = [self.metrics_dir, self.predictions_dir, self.logs_dir]
        if self.allow_checkpoint_writes:
            dirs.insert(0, self.checkpoints_dir)
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)

    # ── Save methods ────────────────────────────────────────

    def save_metrics(
        self,
        metrics: Dict[str, Any],
        name: str = "metrics",
    ) -> Path:
        """Save metrics dict as JSON."""
        filename = f"{name}.json"
        path = self.metrics_dir / filename
        path.write_text(
            json.dumps(metrics, indent=2, default=str, ensure_ascii=False),
            encoding="utf-8",
        )
        self._exported_metrics.append(filename)
        return path

    def save_training_history(
        self,
        history: Dict[str, List[Any]],
        name: str = "training_history",
    ) -> Path:
        """Save training history (loss/metric curves) as JSON."""
        filename = f"{name}.json"
        path = self.metrics_dir / filename
        # Convert numpy arrays to lists for JSON serialization
        serializable = {}
        for key, values in history.items():
            serializable[key] = [
                float(v) if isinstance(v, (np.floating, float)) else v
                for v in values
            ]
        path.write_text(
            json.dumps(serializable, indent=2, default=str, ensure_ascii=False),
            encoding="utf-8",
        )
        self._exported_metrics.append(filename)
        return path

    def save_predictions(
        self,
        predictions: List[int],
        targets: List[int],
        confidences: Optional[List[List[float]]] = None,
        class_names: Optional[List[str]] = None,
        label_mode: Optional[str] = None,
        name: str = "predictions",
    ) -> Path:
        """Save prediction results as JSON.

        Parameters
        ----------
        predictions : list[int]
            Predicted class indices.
        targets : list[int]
            Ground-truth class indices.
        confidences : list[list[float]], optional
            Per-class confidence scores for each sample.
        name : str
            Base filename.
        """
        filename = f"{name}.json"
        path = self.predictions_dir / filename
        data: Dict[str, Any] = {
            "predictions": predictions,
            "targets": targets,
        }
        if confidences is not None:
            data["confidences"] = confidences
        if class_names is not None:
            data["class_names"] = class_names
        if label_mode is not None:
            data["label_mode"] = label_mode
        path.write_text(
            json.dumps(data, indent=2, default=str, ensure_ascii=False),
            encoding="utf-8",
        )
        self._exported_predictions.append(filename)
        return path

    def save_confusion_matrix(
        self,
        cm: np.ndarray,
        class_names: List[str],
        name: str = "confusion_matrix",
    ) -> Path:
        """Save a confusion matrix as JSON.

        Parameters
        ----------
        cm : np.ndarray
            Confusion matrix of shape ``[N, N]``.
        class_names : list[str]
            Class names corresponding to matrix indices.
        name : str
            Base filename.
        """
        filename = f"{name}.json"
        path = self.predictions_dir / filename
        data = {
            "class_names": class_names,
            "matrix": cm.tolist(),
        }
        path.write_text(
            json.dumps(data, indent=2, default=str, ensure_ascii=False),
            encoding="utf-8",
        )
        self._exported_predictions.append(filename)
        return path

    def get_checkpoint_path(self, name: str = "best") -> Path:
        """Return the path where a checkpoint should be saved.

        Parameters
        ----------
        name : str
            Checkpoint name (e.g. ``"best"``, ``"last"``).
        """
        if not self.allow_checkpoint_writes:
            raise RuntimeError(
                "Checkpoint writes are disabled for this output manager."
            )
        return self.checkpoints_dir / f"{name}.pt"

    def register_checkpoint(self, filename: str) -> None:
        """Register a saved checkpoint file."""
        if not self.allow_checkpoint_writes:
            raise RuntimeError(
                "Checkpoint registration is disabled for this output manager."
            )
        self._exported_checkpoints.append(filename)

    # ── Manifest & History ──────────────────────────────────

    def _generate_manifest(
        self,
        *,
        wandb_run_id: Optional[str] = None,
        wandb_run_url: Optional[str] = None,
        git_commit: Optional[str] = None,
        total_epochs: Optional[int] = None,
        best_metric: Optional[float] = None,
        best_epoch: Optional[int] = None,
        duration_seconds: float = 0.0,
        status: str = "success",
        extra_manifest: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Build the manifest dict for this run."""
        execution: Dict[str, Any] = {
            "duration_seconds": round(duration_seconds, 2),
            "status": status,
        }
        if total_epochs is not None:
            execution["total_epochs"] = total_epochs
        if best_metric is not None:
            execution["best_metric"] = best_metric
        if best_epoch is not None:
            execution["best_epoch"] = best_epoch

        manifest = {
            "timestamp": self._ts.isoformat(),
            "run_name": self.run_name,
            "training_version": self.config.training_version,
            "model": self.config.model_name,
            "dataset": self.config.dataset_name,
            "git_commit": git_commit,
            "seed": self.config.seed,
            "wandb_run_id": wandb_run_id,
            "wandb_run_url": wandb_run_url,
            "files": {
                "checkpoints": self._exported_checkpoints,
                "metrics": self._exported_metrics,
                "predictions": self._exported_predictions,
                "logs": self._exported_logs,
            },
            "execution": execution,
        }
        if extra_manifest:
            manifest.update(extra_manifest)
        return manifest

    def _write_manifest(self, manifest: Dict[str, Any]) -> None:
        """Write manifest.json into the run directory."""
        content = json.dumps(
            manifest, indent=2, default=str, ensure_ascii=False,
        )
        (self.run_dir / "manifest.json").write_text(content, encoding="utf-8")

    def _update_latest(self) -> None:
        """Update the ``latest/`` reference directory."""
        latest = self.config.outputs_dir / "latest"
        try:
            if latest.is_symlink() or latest.is_dir():
                if latest.is_symlink():
                    latest.unlink()
                else:
                    shutil.rmtree(latest)
            os.symlink(self.run_dir, latest, target_is_directory=True)
        except (OSError, NotImplementedError):
            # Fallback: copy directory contents
            if latest.exists():
                shutil.rmtree(latest)
            shutil.copytree(self.run_dir, latest)

    def _update_run_history(self, manifest: Dict[str, Any]) -> None:
        """Append this run's manifest to ``run_history.json``."""
        history_path = self.config.run_history_path
        history: List[Dict[str, Any]] = []
        if history_path.is_file():
            try:
                history = json.loads(
                    history_path.read_text(encoding="utf-8"),
                )
            except (json.JSONDecodeError, ValueError):
                history = []
        history.append(manifest)
        history_path.write_text(
            json.dumps(history, indent=2, default=str, ensure_ascii=False),
            encoding="utf-8",
        )

    def finalize(
        self,
        *,
        wandb_run_id: Optional[str] = None,
        wandb_run_url: Optional[str] = None,
        git_commit: Optional[str] = None,
        total_epochs: Optional[int] = None,
        best_metric: Optional[float] = None,
        best_epoch: Optional[int] = None,
        duration_seconds: float = 0.0,
        status: str = "success",
        extra_manifest: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Write manifest, update latest, update run_history.

        Returns the manifest dict for further use (e.g. W&B).
        """
        manifest = self._generate_manifest(
            wandb_run_id=wandb_run_id,
            wandb_run_url=wandb_run_url,
            git_commit=git_commit,
            total_epochs=total_epochs,
            best_metric=best_metric,
            best_epoch=best_epoch,
            duration_seconds=duration_seconds,
            status=status,
            extra_manifest=extra_manifest,
        )
        if self.config.output.get("generate_manifest", True):
            self._write_manifest(manifest)
        if self.allow_checkpoint_writes and self.config.output.get("maintain_latest", True):
            self._update_latest()
        if self.config.output.get("track_run_history", True):
            self._update_run_history(manifest)
        return manifest

    # ── Convenience ─────────────────────────────────────────

    @property
    def all_exported_files(self) -> List[str]:
        """All files exported during this run."""
        return (
            self._exported_checkpoints
            + self._exported_metrics
            + self._exported_predictions
            + self._exported_logs
        )
