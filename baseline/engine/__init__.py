# ─────────────────────────────────────────────────────────────
# Training Engine Sub-package
# ─────────────────────────────────────────────────────────────
"""
Core training and evaluation loops.
"""

from baseline.engine.trainer import Trainer
from baseline.engine.evaluator import Evaluator

__all__ = ["Trainer", "Evaluator"]
