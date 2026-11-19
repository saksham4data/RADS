# ─────────────────────────────────────────────────────────────
# Training Losses Sub-package
# ─────────────────────────────────────────────────────────────
"""
Loss function wrappers for classification training.
"""

from baseline.losses.classification_loss import create_loss

__all__ = ["create_loss"]
