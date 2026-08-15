# ─────────────────────────────────────────────────────────────
# Core Trainer — Training Loop Engine
# ─────────────────────────────────────────────────────────────
"""
Orchestrates the full training loop:

    epoch → batch → forward → loss → backward → step → validate
    → callbacks → log → repeat

Integrates every component: model, loss, optimizer, scheduler,
metrics, checkpoint, early stopping, LR monitor, W&B, logger.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from training.callbacks.checkpoint import CheckpointManager
from training.callbacks.early_stopping import EarlyStopping
from training.callbacks.lr_monitor import LearningRateMonitor
from training.configs.config import TrainingConfig
from training.metrics.classification_metrics import MetricsTracker
from training.utils.logger import TrainingLogger
from training.utils.wandb_manager import TrainingWandbManager

logger = logging.getLogger(__name__)


class Trainer:
    """Core training engine for classification models.

    Parameters
    ----------
    model : nn.Module
        The classification model.
    loss_fn : nn.Module
        Loss function (e.g. CrossEntropyLoss).
    optimizer : Optimizer
        Optimizer instance.
    scheduler : LRScheduler or None
        Learning rate scheduler.
    train_loader : DataLoader
        Training data loader.
    val_loader : DataLoader
        Validation data loader.
    config : TrainingConfig
        Full training configuration.
    device : torch.device
        Compute device.
    training_logger : TrainingLogger, optional
        Logger instance.
    wandb_manager : TrainingWandbManager, optional
        W&B manager instance.
    checkpoint_manager : CheckpointManager, optional
        Checkpoint callback.
    early_stopping : EarlyStopping, optional
        Early stopping callback.
    lr_monitor : LearningRateMonitor, optional
        LR logging callback.
    class_names : list[str], optional
        Ordered class names for metrics.

    Usage::

        trainer = Trainer(model, loss_fn, optimizer, scheduler,
                          train_loader, val_loader, config, device)
        history = trainer.fit()
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any],
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: TrainingConfig,
        device: torch.device,
        *,
        training_logger: Optional[TrainingLogger] = None,
        wandb_manager: Optional[TrainingWandbManager] = None,
        checkpoint_manager: Optional[CheckpointManager] = None,
        early_stopping: Optional[EarlyStopping] = None,
        lr_monitor: Optional[LearningRateMonitor] = None,
        class_names: Optional[List[str]] = None,
    ) -> None:
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device

        self.training_logger = training_logger
        self.wandb_manager = wandb_manager
        self.checkpoint_manager = checkpoint_manager
        self.early_stopping = early_stopping
        self.lr_monitor = lr_monitor or LearningRateMonitor()

        self.class_names = class_names or []

        # Mixed precision
        self._use_amp = config.mixed_precision and device.type == "cuda"
        self._scaler = (
            torch.amp.GradScaler("cuda") if self._use_amp else None
        )

    def fit(
        self,
        start_epoch: int = 0,
    ) -> Dict[str, List[Any]]:
        """Run the full training loop.

        Parameters
        ----------
        start_epoch : int
            Epoch to start from (for resume support).

        Returns
        -------
        dict[str, list]
            Training history with keys like ``train_loss``,
            ``val_loss``, ``val_top1_accuracy``, ``lr``, etc.
        """
        total_epochs = self.config.epochs
        history: Dict[str, List[Any]] = {
            "train_loss": [],
            "val_loss": [],
            "lr": [],
        }

        if self.training_logger:
            self.training_logger.log_start()

        for epoch in range(start_epoch, total_epochs):
            epoch_start = time.time()

            if self.training_logger:
                self.training_logger.log_epoch_start(epoch, total_epochs)

            # ── Train ──
            train_loss, train_metrics = self._train_one_epoch(epoch)

            # ── Validate ──
            val_loss, val_metrics = self._validate_one_epoch(epoch)

            epoch_duration = time.time() - epoch_start
            current_lr = self.optimizer.param_groups[0]["lr"]

            # ── Record history ──
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["lr"].append(current_lr)

            # Copy additional val metrics into history
            for key, value in val_metrics.items():
                if isinstance(value, (int, float)):
                    if key not in history:
                        history[key] = []
                    history[key].append(value)

            # ── LR Monitor ──
            self.lr_monitor.step(
                self.optimizer, epoch, self.wandb_manager,
            )

            # ── Scheduler step ──
            if self.scheduler is not None:
                self.scheduler.step()

            # ── Logger ──
            if self.training_logger:
                self.training_logger.log_epoch_end(
                    epoch,
                    train_loss=train_loss,
                    val_loss=val_loss,
                    val_metrics={
                        k: v for k, v in val_metrics.items()
                        if isinstance(v, (int, float))
                    },
                    lr=current_lr,
                    duration_s=epoch_duration,
                )

            # ── W&B logging ──
            if self.wandb_manager:
                log_dict = {
                    "epoch": epoch,
                    "train/loss": train_loss,
                    "val/loss": val_loss,
                }
                # Add scalar val metrics
                for key, value in val_metrics.items():
                    if isinstance(value, (int, float)):
                        log_dict[key] = value
                # Add scalar train metrics
                for key, value in train_metrics.items():
                    if isinstance(value, (int, float)):
                        log_dict[key] = value
                self.wandb_manager.log_epoch_metrics(epoch, log_dict)

            # ── Checkpoint ──
            if self.checkpoint_manager:
                ckpt_metrics = {"val_loss": val_loss}
                ckpt_metrics.update({
                    k: v for k, v in val_metrics.items()
                    if isinstance(v, (int, float))
                })
                self.checkpoint_manager.save(
                    self.model, self.optimizer, self.scheduler,
                    epoch, ckpt_metrics,
                )

            # ── Early stopping ──
            if self.early_stopping:
                # Use the same metric as checkpoint selection when available;
                # fall back to val_loss so behaviour is identical to before
                # if the metric has not yet been computed.
                es_metric_key = (
                    self.checkpoint_manager.monitor_metric
                    if self.checkpoint_manager is not None
                    else "val_loss"
                )
                es_value = ckpt_metrics.get(es_metric_key, val_loss)
                if self.early_stopping.step(es_value):
                    logger.info(
                        "Early stopping triggered at epoch %d (monitor=%s, value=%.4f)",
                        epoch + 1, es_metric_key, es_value,
                    )
                    break

        return history

    # ── Training epoch ──────────────────────────────────────

    def _train_one_epoch(
        self,
        epoch: int,
    ) -> Tuple[float, Dict[str, Any]]:
        """Run one training epoch.

        Returns
        -------
        tuple[float, dict]
            (average_loss, metrics_dict)
        """
        self.model.train()
        running_loss = 0.0
        num_batches = 0

        metrics_tracker = MetricsTracker(
            self.class_names, prefix="train",
        )

        pbar = tqdm(
            self.train_loader,
            desc=f"Train Epoch {epoch + 1}",
            leave=False,
            dynamic_ncols=True,
        )

        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            # Skip empty batches
            if images.numel() == 0:
                continue

            self.optimizer.zero_grad()

            if self._use_amp:
                with torch.amp.autocast("cuda"):
                    logits = self.model(images)
                    loss = self.loss_fn(logits, labels)
                self._scaler.scale(loss).backward()
                self._scaler.step(self.optimizer)
                self._scaler.update()
            else:
                logits = self.model(images)
                loss = self.loss_fn(logits, labels)
                loss.backward()
                self.optimizer.step()

            running_loss += loss.item()
            num_batches += 1

            # Track metrics
            with torch.no_grad():
                preds = logits.argmax(dim=1)
                metrics_tracker.update(preds, labels)

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = running_loss / max(num_batches, 1)
        metrics = metrics_tracker.compute()
        return avg_loss, metrics

    # ── Validation epoch ────────────────────────────────────

    @torch.no_grad()
    def _validate_one_epoch(
        self,
        epoch: int,
    ) -> Tuple[float, Dict[str, Any]]:
        """Run one validation epoch.

        Returns
        -------
        tuple[float, dict]
            (average_loss, metrics_dict)
        """
        self.model.eval()
        running_loss = 0.0
        num_batches = 0

        metrics_tracker = MetricsTracker(
            self.class_names, prefix="val",
        )

        pbar = tqdm(
            self.val_loader,
            desc=f"Val   Epoch {epoch + 1}",
            leave=False,
            dynamic_ncols=True,
        )

        for images, labels in pbar:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            if images.numel() == 0:
                continue

            if self._use_amp:
                with torch.amp.autocast("cuda"):
                    logits = self.model(images)
                    loss = self.loss_fn(logits, labels)
            else:
                logits = self.model(images)
                loss = self.loss_fn(logits, labels)

            running_loss += loss.item()
            num_batches += 1

            # Softmax probabilities for AUROC
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            metrics_tracker.update(preds, labels, confidences=probs)

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = running_loss / max(num_batches, 1)
        metrics = metrics_tracker.compute()
        return avg_loss, metrics
