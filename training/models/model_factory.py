# ─────────────────────────────────────────────────────────────
# Model Factory — Registry Pattern
# ─────────────────────────────────────────────────────────────
"""
Maps configuration names to ``ClassificationModel`` constructors.

Both ``resnet18`` and ``efficientnet_b0`` are registered from
day one.  To add a new backbone, add one entry to ``_REGISTRY``
and one ``elif`` branch in ``ClassificationModel._build_backbone``.
"""

from __future__ import annotations

import logging
from typing import Callable, Dict

import torch.nn as nn

from training.configs.config import TrainingConfig
from training.models.classification_model import ClassificationModel
from training.models.temporal_model import TemporalClassificationModel

logger = logging.getLogger(__name__)


# ── Registry ────────────────────────────────────────────────

_REGISTRY: Dict[str, Callable[[TrainingConfig], nn.Module]] = {
    "resnet18": lambda cfg: ClassificationModel(
        backbone_name="resnet18",
        num_classes=cfg.num_classes,
        pretrained=cfg.pretrained,
        freeze_backbone=cfg.freeze_backbone,
    ),
    "efficientnet_b0": lambda cfg: ClassificationModel(
        backbone_name="efficientnet_b0",
        num_classes=cfg.num_classes,
        pretrained=cfg.pretrained,
        freeze_backbone=cfg.freeze_backbone,
    ),
}


def _create_temporal_model(cfg: TrainingConfig) -> nn.Module:
    """Build a TemporalClassificationModel from config."""
    return TemporalClassificationModel(
        backbone_name=cfg.model_name,
        num_classes=cfg.num_classes,
        temporal_arch=cfg.temporal_architecture,
        temporal_hidden_dim=cfg.temporal_hidden_dim,
        temporal_num_layers=cfg.temporal_num_layers,
        temporal_dropout=cfg.temporal_dropout,
        temporal_bidirectional=cfg.temporal_bidirectional,
        pretrained=cfg.pretrained,
        freeze_backbone=cfg.freeze_backbone,
        classifier_dropout=cfg.temporal_classifier_dropout,
    )


def create_model(config: TrainingConfig) -> nn.Module:
    """Instantiate a classification model from configuration.

    Parameters
    ----------
    config : TrainingConfig
        Training configuration.  The model is determined by
        ``config.model_name`` (e.g. ``"resnet18"``, ``"efficientnet_b0"``).

    Returns
    -------
    nn.Module
        Instantiated model ready for ``.to(device)``.

    Raises
    ------
    ValueError
        If the model name is not in the registry.
    """
    name = config.model_name

    # ── Temporal mode: compose spatial + temporal encoders ──
    if config.temporal_enabled:
        model = _create_temporal_model(config)
        logger.info(
            "Temporal model created via factory: '%s' + '%s'",
            name, config.temporal_architecture,
        )
        return model

    # ── Standard frame-level model ──
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown model '{name}'. "
            f"Available models: {list(_REGISTRY.keys())}"
        )

    model = _REGISTRY[name](config)
    logger.info("Model created via factory: '%s'", name)
    return model


def list_available_models() -> list[str]:
    """Return the names of all registered models."""
    return list(_REGISTRY.keys())
