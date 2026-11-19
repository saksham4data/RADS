# ─────────────────────────────────────────────────────────────
# Training Models Sub-package
# ─────────────────────────────────────────────────────────────
"""
Classification model wrappers and model factory.
"""

from baseline.models.classification_model import ClassificationModel
from baseline.models.model_factory import create_model

__all__ = ["ClassificationModel", "create_model"]
