# ─────────────────────────────────────────────────────────────
# Training Engine Sub-package
# ─────────────────────────────────────────────────────────────
"""
Core training and evaluation loops.
"""

from training.engine.trainer import Trainer
from training.engine.evaluator import Evaluator

__all__ = ["Trainer", "Evaluator"]
