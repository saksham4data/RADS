# ─────────────────────────────────────────────────────────────
# Early Stopping Callback
# ─────────────────────────────────────────────────────────────
"""
Patience-based early stopping to prevent overfitting.
Stops training when a monitored metric plateaus.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Stops training when a monitored metric stops improving.

    Parameters
    ----------
    patience : int
        Number of epochs to wait after the last improvement.
    min_delta : float
        Minimum change to qualify as an improvement.
    mode : str
        ``"min"`` if lower is better, ``"max"`` if higher is better.

    Usage::

        es = EarlyStopping(patience=5)
        for epoch in range(100):
            val_loss = train_one_epoch()
            if es.step(val_loss):
                print(f"Early stopping at epoch {epoch}")
                break
    """

    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.001,
        mode: str = "min",
    ) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode

        self._best_value: Optional[float] = None
        self._counter: int = 0
        self._stopped: bool = False

    def step(self, metric_value: float) -> bool:
        """Check whether training should stop.

        Parameters
        ----------
        metric_value : float
            Current value of the monitored metric.

        Returns
        -------
        bool
            ``True`` if training should stop (patience exhausted).
        """
        if self._best_value is None:
            self._best_value = metric_value
            return False

        improved = self._is_improvement(metric_value)

        if improved:
            self._best_value = metric_value
            self._counter = 0
        else:
            self._counter += 1
            logger.debug(
                "EarlyStopping: no improvement for %d/%d epochs "
                "(best=%.4f, current=%.4f)",
                self._counter, self.patience,
                self._best_value, metric_value,
            )

        if self._counter >= self.patience:
            self._stopped = True
            logger.info(
                "EarlyStopping triggered after %d epochs without "
                "improvement (best=%.4f)",
                self.patience, self._best_value,
            )
            return True

        return False

    def _is_improvement(self, value: float) -> bool:
        """Check if the new value is a sufficient improvement."""
        if self._best_value is None:
            return True
        if self.mode == "min":
            return value < (self._best_value - self.min_delta)
        return value > (self._best_value + self.min_delta)

    def reset(self) -> None:
        """Reset the early stopping state."""
        self._best_value = None
        self._counter = 0
        self._stopped = False

    @property
    def stopped(self) -> bool:
        """Whether early stopping has been triggered."""
        return self._stopped

    @property
    def counter(self) -> int:
        """Current patience counter."""
        return self._counter
