# ─────────────────────────────────────────────────────────────
# Generic Classification Model
# ─────────────────────────────────────────────────────────────
"""
Backbone-agnostic classification wrapper supporting multiple
pretrained architectures from ``torchvision.models``.

Currently registered backbones:
    resnet18        — 512-dim features, lightweight baseline
    efficientnet_b0 — 1280-dim features, higher accuracy

Adding a new backbone requires only one additional ``elif`` branch
in ``_build_backbone()`` — no file rename or structural changes.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
from torchvision import models

logger = logging.getLogger(__name__)


class ClassificationModel(nn.Module):
    """Multi-backbone classification model.

    Wraps a pretrained ``torchvision`` backbone and replaces its
    classification head with a new linear layer matching
    ``num_classes``.

    Parameters
    ----------
    backbone_name : str
        Name of the backbone architecture (e.g. ``"resnet18"``).
    num_classes : int
        Number of output classes.
    pretrained : bool
        If ``True``, load ImageNet-pretrained weights.
    freeze_backbone : bool
        If ``True``, freeze all backbone parameters (transfer learning).

    Usage::

        model = ClassificationModel("resnet18", num_classes=3)
        logits = model(torch.randn(4, 3, 224, 224))
        assert logits.shape == (4, 3)
    """

    # Supported backbones and their feature dimensions
    _BACKBONE_FEATURES = {
        "resnet18": 512,
        "efficientnet_b0": 1280,
    }

    def __init__(
        self,
        backbone_name: str,
        num_classes: int,
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone_name
        self.num_classes = num_classes
        self._pretrained = pretrained

        if backbone_name not in self._BACKBONE_FEATURES:
            raise ValueError(
                f"Unknown backbone '{backbone_name}'. "
                f"Available: {list(self._BACKBONE_FEATURES.keys())}"
            )

        self._feature_dim = self._BACKBONE_FEATURES[backbone_name]

        # Build backbone + head
        self.backbone, self.classifier = self._build_backbone(
            backbone_name, num_classes, pretrained,
        )

        # Optional freeze
        if freeze_backbone:
            self.freeze_backbone()

        # Log architecture summary
        total_params = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            "ClassificationModel: %s | classes=%d | params=%s | "
            "trainable=%s | pretrained=%s | frozen=%s",
            backbone_name,
            num_classes,
            f"{total_params:,}",
            f"{trainable:,}",
            pretrained,
            freeze_backbone,
        )

    # ── Backbone construction ───────────────────────────────

    @staticmethod
    def _build_backbone(
        name: str,
        num_classes: int,
        pretrained: bool,
    ) -> tuple[nn.Module, nn.Linear]:
        """Build the backbone and classification head.

        Returns
        -------
        tuple[nn.Module, nn.Linear]
            (backbone_with_head_replaced, reference_to_new_head)
        """
        if name == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            backbone = models.resnet18(weights=weights)
            # Replace the FC layer
            in_features = backbone.fc.in_features
            head = nn.Linear(in_features, num_classes)
            backbone.fc = head
            return backbone, head

        elif name == "efficientnet_b0":
            weights = (
                models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            )
            backbone = models.efficientnet_b0(weights=weights)
            # Replace the last layer in the classifier sequential
            in_features = backbone.classifier[-1].in_features
            head = nn.Linear(in_features, num_classes)
            backbone.classifier[-1] = head
            return backbone, head

        else:
            raise ValueError(f"Unsupported backbone: {name}")

    # ── Forward ─────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``[B, 3, H, W]``.

        Returns
        -------
        torch.Tensor
            Logits of shape ``[B, num_classes]``.
        """
        return self.backbone(x)

    # ── Freeze / Unfreeze ───────────────────────────────────

    def freeze_backbone(self) -> None:
        """Freeze all backbone parameters (only head trains)."""
        for name, param in self.backbone.named_parameters():
            # Keep the classifier head trainable
            if self.backbone_name == "resnet18" and name.startswith("fc."):
                continue
            if self.backbone_name == "efficientnet_b0" and "classifier" in name:
                continue
            param.requires_grad = False

        trainable = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )
        logger.info("Backbone frozen — %s trainable params remain", f"{trainable:,}")

    def unfreeze_backbone(self) -> None:
        """Unfreeze all backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        trainable = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )
        logger.info("Backbone unfrozen — %s trainable params", f"{trainable:,}")

    # ── Properties ──────────────────────────────────────────

    def get_feature_dim(self) -> int:
        """Return the backbone feature dimension before the head."""
        return self._feature_dim
