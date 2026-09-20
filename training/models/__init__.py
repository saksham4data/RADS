# ─────────────────────────────────────────────────────────────
# Training Models Sub-package
# ─────────────────────────────────────────────────────────────
"""
Classification model wrappers and model factory.
"""

from training.models.classification_model import ClassificationModel
from training.models.model_factory import create_model

__all__ = ["ClassificationModel", "create_model"]
