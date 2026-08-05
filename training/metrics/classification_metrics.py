# ─────────────────────────────────────────────────────────────
# Classification Metrics Tracker
# ─────────────────────────────────────────────────────────────
"""
Accumulates predictions and targets per epoch and computes:

- **Top-1 Accuracy** — overall classification accuracy
- **Per-class Precision** — per-class precision
- **Per-class Recall** — per-class recall
- **Per-class F1** — per-class harmonic mean
- **Macro F1** — unweighted mean across classes
- **Confusion Matrix** — full N x N matrix

Returns a flat dict suitable for ``wandb.log()``.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

logger = logging.getLogger(__name__)


class MetricsTracker:
    """Accumulates per-batch predictions and computes epoch metrics.

    Parameters
    ----------
    class_names : list[str]
        Ordered list of class names (index 0, 1, ...).
    prefix : str
        Prefix for metric keys (e.g. ``"train"`` or ``"val"``).

    Usage::

        tracker = MetricsTracker(["accident", "non-accident"], prefix="val")
        tracker.update(preds_batch, targets_batch)
        metrics = tracker.compute()
        # → {"val/top1_accuracy": 0.85, "val/macro_f1": 0.72, ...}
        tracker.reset()
    """

    def __init__(
        self,
        class_names: List[str],
        prefix: str = "metrics",
    ) -> None:
        self.class_names = class_names
        self.prefix = prefix
        self._all_preds: List[int] = []
        self._all_targets: List[int] = []
        self._all_confidences: List[List[float]] = []

    def reset(self) -> None:
        """Clear all accumulated predictions and targets."""
        self._all_preds.clear()
        self._all_targets.clear()
        self._all_confidences.clear()

    def update(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        confidences: Optional[torch.Tensor] = None,
    ) -> None:
        """Add a batch of predictions and targets.

        Parameters
        ----------
        predictions : torch.Tensor
            Predicted class indices of shape ``[B]`` or logits of
            shape ``[B, C]`` (argmax is applied automatically).
        targets : torch.Tensor
            Ground-truth class indices of shape ``[B]``.
        confidences : torch.Tensor, optional
            Softmax probabilities of shape ``[B, C]``.
        """
        # Handle logits → predicted classes
        if predictions.dim() > 1:
            predictions = predictions.argmax(dim=1)

        self._all_preds.extend(predictions.cpu().tolist())
        self._all_targets.extend(targets.cpu().tolist())

        if confidences is not None:
            self._all_confidences.extend(confidences.cpu().tolist())

    def compute(self) -> Dict[str, Any]:
        """Compute all metrics from accumulated predictions.

        Returns
        -------
        dict[str, Any]
            Flat dict of metric name → value.  Suitable for
            ``wandb.log()`` or ``TrainingLogger``.
        """
        if len(self._all_preds) == 0:
            logger.warning("MetricsTracker.compute() called with no data.")
            return {}

        preds = np.array(self._all_preds)
        targets = np.array(self._all_targets)
        num_classes = len(self.class_names)
        labels = list(range(num_classes))

        # ── Top-1 Accuracy ──
        top1_acc = accuracy_score(targets, preds)

        # ── Per-class Precision / Recall / F1 ──
        per_precision = precision_score(
            targets, preds, labels=labels, average=None, zero_division=0,
        )
        per_recall = recall_score(
            targets, preds, labels=labels, average=None, zero_division=0,
        )
        per_f1 = f1_score(
            targets, preds, labels=labels, average=None, zero_division=0,
        )

        # ── Macro F1 ──
        macro_f1 = f1_score(
            targets, preds, labels=labels, average="macro", zero_division=0,
        )

        # ── Confusion Matrix ──
        cm = confusion_matrix(targets, preds, labels=labels)

        # ── Build flat metrics dict ──
        p = self.prefix
        metrics: Dict[str, Any] = {
            f"{p}/top1_accuracy": float(top1_acc),
            f"{p}/macro_f1": float(macro_f1),
        }

        for i, name in enumerate(self.class_names):
            safe_name = name.replace("-", "_").replace(" ", "_")
            metrics[f"{p}/precision_{safe_name}"] = float(per_precision[i])
            metrics[f"{p}/recall_{safe_name}"] = float(per_recall[i])
            metrics[f"{p}/f1_{safe_name}"] = float(per_f1[i])

        # Confusion matrix as numpy (not logged to W&B directly — use log_confusion_matrix)
        metrics[f"{p}/confusion_matrix"] = cm

        logger.debug(
            "%s metrics: top1_acc=%.4f macro_f1=%.4f",
            p, top1_acc, macro_f1,
        )

        return metrics

    # ── Properties ──────────────────────────────────────────

    @property
    def predictions(self) -> List[int]:
        """All accumulated predicted class indices."""
        return self._all_preds.copy()

    @property
    def targets(self) -> List[int]:
        """All accumulated ground-truth class indices."""
        return self._all_targets.copy()

    @property
    def confidences(self) -> List[List[float]]:
        """All accumulated per-class confidence scores."""
        return self._all_confidences.copy()

    @property
    def num_samples(self) -> int:
        """Number of accumulated samples."""
        return len(self._all_preds)
