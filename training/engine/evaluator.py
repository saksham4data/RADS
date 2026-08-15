# ─────────────────────────────────────────────────────────────
# Standalone Evaluator
# ─────────────────────────────────────────────────────────────
"""
Evaluates a model on a given split (val/test), computes all
metrics, and saves prediction results and confusion matrices
to ``outputs/predictions/``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from training.metrics.classification_metrics import MetricsTracker
from training.utils.output_manager import TrainingOutputManager
from training.utils.wandb_manager import TrainingWandbManager

logger = logging.getLogger(__name__)


class Evaluator:
    """Standalone evaluation engine for classification models.

    Runs forward passes on a data loader, computes all metrics,
    and optionally saves prediction results.

    Parameters
    ----------
    model : nn.Module
        The classification model.
    loss_fn : nn.Module
        Loss function.
    device : torch.device
        Compute device.
    class_names : list[str]
        Ordered class names.

    Usage::

        evaluator = Evaluator(model, loss_fn, device, class_names)
        metrics = evaluator.evaluate(test_loader, split_name="test")
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        device: torch.device,
        class_names: List[str],
    ) -> None:
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.device = device
        self.class_names = class_names

    @torch.no_grad()
    def evaluate(
        self,
        dataloader: DataLoader,
        split_name: str = "test",
        *,
        output_manager: Optional[TrainingOutputManager] = None,
        wandb_manager: Optional[TrainingWandbManager] = None,
    ) -> Dict[str, Any]:
        """Evaluate the model on a data loader.

        Parameters
        ----------
        dataloader : DataLoader
            Data loader to evaluate.
        split_name : str
            Split name for metric prefix (e.g. ``"test"``).
        output_manager : TrainingOutputManager, optional
            If provided, saves predictions and confusion matrix.
        wandb_manager : TrainingWandbManager, optional
            If provided, logs metrics and confusion matrix to W&B.

        Returns
        -------
        dict
            Full metrics dict including loss, accuracy, F1, etc.
        """
        self.model.eval()
        running_loss = 0.0
        num_batches = 0

        tracker = MetricsTracker(self.class_names, prefix=split_name)

        all_confidences: List[List[float]] = []

        pbar = tqdm(
            dataloader,
            desc=f"Evaluating ({split_name})",
            leave=False,
            dynamic_ncols=True,
        )

        for images, labels in pbar:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            if images.numel() == 0:
                continue

            logits = self.model(images)
            loss = self.loss_fn(logits, labels)

            running_loss += loss.item()
            num_batches += 1

            # Softmax for confidences
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)

            tracker.update(preds, labels, confidences=probs)
            all_confidences.extend(probs.cpu().tolist())

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # ── Compute metrics ──
        avg_loss = running_loss / max(num_batches, 1)
        metrics = tracker.compute()
        metrics[f"{split_name}/loss"] = avg_loss

        # ── Log summary ──
        top1 = metrics.get(f"{split_name}/top1_accuracy", 0)
        macro_f1 = metrics.get(f"{split_name}/macro_f1", 0)
        logger.info(
            "Evaluation [%s] — loss=%.4f top1_acc=%.4f macro_f1=%.4f "
            "(%d samples)",
            split_name, avg_loss, top1, macro_f1, tracker.num_samples,
        )

        # ── Save predictions ──
        if output_manager:
            output_manager.save_predictions(
                predictions=tracker.predictions,
                targets=tracker.targets,
                confidences=all_confidences,
                class_names=self.class_names,
                label_mode=getattr(output_manager.config, "label_mode", None),
                name=f"{split_name}_predictions",
            )

            cm = metrics.get(f"{split_name}/confusion_matrix")
            if cm is not None:
                output_manager.save_confusion_matrix(
                    cm, self.class_names,
                    name=f"{split_name}_confusion_matrix",
                )

        # ── W&B logging ──
        if wandb_manager:
            scalar_metrics = {
                k: v for k, v in metrics.items()
                if isinstance(v, (int, float))
            }
            wandb_manager.log_stats(scalar_metrics)

            cm = metrics.get(f"{split_name}/confusion_matrix")
            if cm is not None:
                wandb_manager.log_confusion_matrix(
                    cm, self.class_names,
                    title=f"{split_name}_confusion_matrix",
                )

        return metrics
