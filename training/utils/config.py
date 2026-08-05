# ─────────────────────────────────────────────────────────────
# Training Configuration — Re-export
# ─────────────────────────────────────────────────────────────
"""
Re-exports ``TrainingConfig`` and ``load_training_config`` from
``training.configs.config`` so that downstream modules can use
the canonical ``from training.utils.config import ...`` path.
"""

from training.configs.config import TrainingConfig, load_training_config

__all__ = ["TrainingConfig", "load_training_config"]
