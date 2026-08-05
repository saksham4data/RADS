# ─────────────────────────────────────────────────────────────
# Classification Loss Wrapper
# ─────────────────────────────────────────────────────────────
"""
Wraps ``nn.CrossEntropyLoss`` with optional class weights
computed from the training set distribution.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import torch
import torch.nn as nn

from training.configs.config import TrainingConfig

logger = logging.getLogger(__name__)


def create_loss(
    config: TrainingConfig,
    class_weights: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None,
) -> nn.Module:
    """Create a classification loss function.

    Parameters
    ----------
    config : TrainingConfig
        Training configuration (currently unused but available
        for future label-smoothing / focal-loss extensions).
    class_weights : torch.Tensor, optional
        Per-class weights of shape ``[num_classes]`` for handling
        class imbalance.  Typically obtained via
        ``VideoFrameDataset.class_weights``.
    device : torch.device, optional
        Device to place the weight tensor on.

    Returns
    -------
    nn.CrossEntropyLoss
        Configured loss function.
    """
    weight_tensor = None

    if class_weights is not None:
        weight_tensor = class_weights.clone()
        if device is not None:
            weight_tensor = weight_tensor.to(device)
        logger.info(
            "CrossEntropyLoss with class weights: %s",
            [f"{w:.3f}" for w in weight_tensor.tolist()],
        )
    else:
        logger.info("CrossEntropyLoss with uniform weights")

    loss_fn = nn.CrossEntropyLoss(weight=weight_tensor)
    return loss_fn
