# ─────────────────────────────────────────────────────────────
# Learning Rate Monitor Callback
# ─────────────────────────────────────────────────────────────
"""
Lightweight callback that reads the current learning rate from
the optimizer and logs it to the console and W&B at each epoch.

No state, no side effects — purely observational.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch.optim

logger = logging.getLogger(__name__)


class LearningRateMonitor:
    """Logs the current learning rate at the end of each epoch.

    Reads ``optimizer.param_groups[0]['lr']`` and logs it to:
    - Console via standard logging
    - W&B via ``TrainingWandbManager.log_learning_rate()``

    Parameters
    ----------
    log_to_console : bool
        Whether to log LR to the console.

    Usage::

        lr_monitor = LearningRateMonitor()
        for epoch in range(epochs):
            train(...)
            scheduler.step()
            lr_monitor.step(optimizer, epoch, wandb_manager=wb)
    """

    def __init__(self, log_to_console: bool = True) -> None:
        self.log_to_console = log_to_console

    def step(
        self,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        wandb_manager: Optional[object] = None,
    ) -> float:
        """Read and log the current learning rate.

        Parameters
        ----------
        optimizer : Optimizer
            The optimizer whose LR to read.
        epoch : int
            Current epoch number (0-indexed).
        wandb_manager : TrainingWandbManager, optional
            If provided, logs LR to W&B.

        Returns
        -------
        float
            The current learning rate.
        """
        lr = optimizer.param_groups[0]["lr"]

        if self.log_to_console:
            logger.info("LR at epoch %d: %.6f", epoch + 1, lr)

        if wandb_manager is not None and hasattr(wandb_manager, "log_learning_rate"):
            wandb_manager.log_learning_rate(lr, epoch)

        return lr
