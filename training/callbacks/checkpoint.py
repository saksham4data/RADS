# ─────────────────────────────────────────────────────────────
# Checkpoint Manager
# ─────────────────────────────────────────────────────────────
"""
Saves and loads model checkpoints with support for best-metric
tracking and full training state resume.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manages saving and loading of model checkpoints.

    Tracks the best metric value to save ``best.pt`` and always
    saves ``last.pt`` at the end of each epoch (if enabled).

    Parameters
    ----------
    checkpoint_dir : Path
        Directory to save checkpoint files.
    monitor_metric : str
        Metric key to monitor (e.g. ``"val_loss"``).
    mode : str
        ``"min"`` if lower is better, ``"max"`` if higher is better.
    save_best : bool
        If ``True``, save ``best.pt`` when the metric improves.
    save_last : bool
        If ``True``, save ``last.pt`` after every epoch.

    Usage::

        ckpt = CheckpointManager(Path("outputs/checkpoints"))
        ckpt.save(model, optimizer, scheduler, epoch=0, metrics={"val_loss": 0.5})
        state = ckpt.load(Path("outputs/checkpoints/best.pt"))
        ckpt.resume_from(Path("best.pt"), model, optimizer, scheduler)
    """

    def __init__(
        self,
        checkpoint_dir: Path,
        monitor_metric: str = "val_loss",
        mode: str = "min",
        save_best: bool = True,
        save_last: bool = True,
    ) -> None:
        self.checkpoint_dir = checkpoint_dir
        self.monitor_metric = monitor_metric
        self.mode = mode
        self.save_best = save_best
        self.save_last = save_last

        self._best_value: Optional[float] = None
        self._best_epoch: int = -1

        # Ensure directory exists
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _is_improvement(self, value: float) -> bool:
        """Check if the new value is an improvement."""
        if self._best_value is None:
            return True
        if self.mode == "min":
            return value < self._best_value
        return value > self._best_value

    def save(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any],
        epoch: int,
        metrics: Dict[str, float],
    ) -> Dict[str, Any]:
        """Save checkpoint(s) for the current epoch.

        Parameters
        ----------
        model : nn.Module
            The model to save.
        optimizer : Optimizer
            Current optimizer state.
        scheduler : LRScheduler or None
            Current scheduler state.
        epoch : int
            Current epoch number (0-indexed).
        metrics : dict
            Metrics dict containing the monitored metric.

        Returns
        -------
        dict
            Summary with ``saved_best`` and ``saved_last`` flags.
        """
        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": (
                scheduler.state_dict() if scheduler is not None else None
            ),
            "metrics": metrics,
            "monitor_metric": self.monitor_metric,
            "mode": self.mode,
        }

        result = {"saved_best": False, "saved_last": False}

        # ── Save best ──
        if self.save_best:
            metric_value = metrics.get(self.monitor_metric)
            if metric_value is not None and self._is_improvement(metric_value):
                self._best_value = metric_value
                self._best_epoch = epoch
                best_path = self.checkpoint_dir / "best.pt"
                torch.save(state, best_path)
                result["saved_best"] = True
                logger.info(
                    "Checkpoint saved: best.pt (epoch %d, %s=%.4f)",
                    epoch + 1, self.monitor_metric, metric_value,
                )

        # ── Save last ──
        if self.save_last:
            last_path = self.checkpoint_dir / "last.pt"
            torch.save(state, last_path)
            result["saved_last"] = True
            logger.debug("Checkpoint saved: last.pt (epoch %d)", epoch + 1)

        return result

    @staticmethod
    def load(path: Path) -> Dict[str, Any]:
        """Load a checkpoint from disk.

        Parameters
        ----------
        path : Path
            Path to the ``.pt`` checkpoint file.

        Returns
        -------
        dict
            Checkpoint state dict with keys: ``epoch``,
            ``model_state_dict``, ``optimizer_state_dict``,
            ``scheduler_state_dict``, ``metrics``.
        """
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        state = torch.load(path, map_location="cpu", weights_only=False)
        logger.info("Checkpoint loaded: %s (epoch %d)", path.name, state["epoch"] + 1)
        return state

    @staticmethod
    def describe(
        path: Path,
        state: Optional[Dict[str, Any]] = None,
        config: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Return metadata describing a loaded checkpoint and its source lineage."""
        import json

        checkpoint_path = path.resolve()
        checkpoint_state = state if state is not None else CheckpointManager.load(checkpoint_path)

        metadata: Dict[str, Any] = {
            "path": str(checkpoint_path),
        }
        epoch = checkpoint_state.get("epoch")
        if isinstance(epoch, int):
            metadata["epoch"] = epoch
            metadata["training_epoch"] = epoch + 1
        metrics = checkpoint_state.get("metrics")
        if isinstance(metrics, dict) and metrics:
            metadata["training_metrics"] = metrics

        monitored_metric = getattr(config, "monitor_metric", None) or checkpoint_state.get("monitor_metric")
        monitor_mode = getattr(config, "monitor_mode", None) or checkpoint_state.get("mode")
        if monitored_metric:
            metadata["monitored_metric"] = monitored_metric
        if monitor_mode:
            metadata["monitor_mode"] = monitor_mode

        # Attempt to resolve source training run manifest from checkpoint directory
        try:
            source_run_dir = checkpoint_path.parent.parent
            source_manifest_path = source_run_dir / "manifest.json"
            if source_manifest_path.is_file():
                source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
                metadata["source_run_name"] = source_manifest.get("run_name")
                metadata["source_run_dir"] = str(source_run_dir.resolve())
                metadata["source_timestamp"] = source_manifest.get("timestamp")
                metadata["source_wandb_run_id"] = source_manifest.get("wandb_run_id")
                metadata["source_wandb_run_url"] = source_manifest.get("wandb_run_url")
                metadata["source_git_commit"] = source_manifest.get("git_commit")
                exec_info = source_manifest.get("execution", {})
                if isinstance(exec_info, dict):
                    metadata["source_best_epoch"] = exec_info.get("best_epoch")
                    metadata["source_best_metric"] = exec_info.get("best_metric")
                    metadata["source_total_epochs"] = exec_info.get("total_epochs")
                    if "best_metric" not in metadata and exec_info.get("best_metric") is not None:
                        metadata["best_metric"] = exec_info.get("best_metric")
        except Exception as err:
            logger.debug("Failed to resolve source manifest for checkpoint %s: %s", checkpoint_path, err)

        if "best_metric" not in metadata and metrics and monitored_metric:
            metadata["best_metric"] = metrics.get(monitored_metric)

        return metadata

    def resume_from(
        self,
        path: Path,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None,
    ) -> int:
        """Resume training from a checkpoint.

        Restores model, optimizer, and scheduler state.

        Parameters
        ----------
        path : Path
            Path to the checkpoint file.
        model : nn.Module
            Model to restore.
        optimizer : Optimizer
            Optimizer to restore.
        scheduler : LRScheduler or None
            Scheduler to restore.

        Returns
        -------
        int
            The epoch number to resume from (next epoch).
        """
        state = self.load(path)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])

        if scheduler is not None and state.get("scheduler_state_dict"):
            scheduler.load_state_dict(state["scheduler_state_dict"])

        resume_epoch = state["epoch"] + 1

        # Restore best tracking
        metrics = state.get("metrics", {})
        metric_value = metrics.get(self.monitor_metric)
        if metric_value is not None:
            self._best_value = metric_value
            self._best_epoch = state["epoch"]

        logger.info(
            "Resumed from %s — continuing at epoch %d",
            path.name, resume_epoch + 1,
        )
        return resume_epoch

    # ── Properties ──────────────────────────────────────────

    @property
    def best_value(self) -> Optional[float]:
        """Best monitored metric value seen so far."""
        return self._best_value

    @property
    def best_epoch(self) -> int:
        """Epoch that achieved the best metric (-1 if none)."""
        return self._best_epoch
