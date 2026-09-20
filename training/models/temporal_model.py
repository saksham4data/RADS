# ─────────────────────────────────────────────────────────────
# Temporal Classification Model
# ─────────────────────────────────────────────────────────────
"""
Composes a spatial encoder (ResNet18 backbone without its fc head)
with a temporal encoder (GRU / LSTM / Conv1D / Pooling) and a
linear classifier.

Pipeline::

    [B, T, 3, H, W]
      -> SpatialEncoder  -> [B, T, F]
      -> TemporalEncoder -> [B, D]
      -> Classifier       -> [B, C]

The spatial encoder and temporal encoder have explicit interfaces
so that future experiments can swap components independently.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
from torchvision import models

logger = logging.getLogger(__name__)


# ── Spatial Encoder ─────────────────────────────────────────

class SpatialEncoder(nn.Module):
    """Extracts per-frame spatial features using a pretrained backbone.

    Strips the classification head from the backbone and returns
    the pooled feature vector for each frame.

    Parameters
    ----------
    backbone_name : str
        Name of the backbone (``"resnet18"`` or ``"efficientnet_b0"``).
    pretrained : bool
        If ``True``, load ImageNet-pretrained weights.
    freeze : bool
        If ``True``, freeze all backbone parameters.
    """

    _FEATURE_DIMS = {
        "resnet18": 512,
        "efficientnet_b0": 1280,
    }

    def __init__(
        self,
        backbone_name: str = "resnet18",
        pretrained: bool = True,
        freeze: bool = False,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone_name

        if backbone_name not in self._FEATURE_DIMS:
            raise ValueError(
                f"Unknown backbone '{backbone_name}'. "
                f"Available: {list(self._FEATURE_DIMS.keys())}"
            )

        self.feature_dim = self._FEATURE_DIMS[backbone_name]
        self.backbone = self._build_backbone(backbone_name, pretrained)

        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False
            logger.info("SpatialEncoder: backbone frozen")

    @staticmethod
    def _build_backbone(name: str, pretrained: bool) -> nn.Module:
        """Build the backbone with its classification head removed."""
        if name == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            backbone = models.resnet18(weights=weights)
            # Remove fc layer, keep avgpool
            backbone.fc = nn.Identity()
            return backbone

        elif name == "efficientnet_b0":
            weights = (
                models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            )
            backbone = models.efficientnet_b0(weights=weights)
            # Remove classifier, keep avgpool
            backbone.classifier = nn.Identity()
            return backbone

        raise ValueError(f"Unsupported backbone: {name}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from a batch of frame sequences.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape ``[B, T, C, H, W]``.

        Returns
        -------
        torch.Tensor
            Features of shape ``[B, T, F]`` where F is the
            backbone feature dimension.
        """
        B, T, C, H, W = x.shape
        # Merge batch and time: [B*T, C, H, W]
        x_flat = x.reshape(B * T, C, H, W)
        # Extract features: [B*T, F]
        features = self.backbone(x_flat)
        # Reshape back: [B, T, F]
        features = features.reshape(B, T, self.feature_dim)
        return features


# ── Temporal Encoders ───────────────────────────────────────

class TemporalGRU(nn.Module):
    """GRU-based temporal encoder.

    Parameters
    ----------
    input_dim : int
        Spatial feature dimension (F).
    hidden_dim : int
        GRU hidden state dimension.
    num_layers : int
        Number of stacked GRU layers.
    dropout : float
        Dropout between GRU layers (only when num_layers > 1).
    bidirectional : bool
        If ``True``, use bidirectional GRU.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 1,
        dropout: float = 0.0,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.output_dim = hidden_dim * (2 if bidirectional else 1)

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a sequence of frame features.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape ``[B, T, F]``.

        Returns
        -------
        torch.Tensor
            Video-level representation of shape ``[B, D]``
            where D = hidden_dim (or 2*hidden_dim if bidirectional).
            Uses the last hidden state.
        """
        # output: [B, T, D], h_n: [num_layers*num_dirs, B, hidden_dim]
        output, h_n = self.gru(x)
        if self.bidirectional:
            # Concatenate final hidden states from both directions
            # h_n shape: [num_layers*2, B, hidden_dim]
            # Take last layer: forward = h_n[-2], backward = h_n[-1]
            h_forward = h_n[-2]
            h_backward = h_n[-1]
            return torch.cat([h_forward, h_backward], dim=1)
        else:
            # h_n[-1]: last layer, shape [B, hidden_dim]
            return h_n[-1]


class TemporalLSTM(nn.Module):
    """LSTM-based temporal encoder.

    Same interface as :class:`TemporalGRU`.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 1,
        dropout: float = 0.0,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.output_dim = hidden_dim * (2 if bidirectional else 1)

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a sequence.  Returns last hidden state [B, D]."""
        output, (h_n, c_n) = self.lstm(x)
        if self.bidirectional:
            return torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            return h_n[-1]


class TemporalConv1D(nn.Module):
    """1D temporal convolution encoder.

    Applies Conv1D over the temporal dimension followed by
    global average pooling.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        kernel_size: int = 3,
        num_layers: int = 1,
        dropout: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__()
        self.output_dim = hidden_dim

        layers = []
        in_channels = input_dim
        for i in range(num_layers):
            layers.append(nn.Conv1d(
                in_channels, hidden_dim,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            ))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_channels = hidden_dim

        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a sequence.  Returns pooled representation [B, D].

        Parameters
        ----------
        x : torch.Tensor
            Input of shape ``[B, T, F]``.
        """
        # Conv1d expects [B, C, T]
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        # Global average pool over time
        x = x.mean(dim=2)
        return x


class TemporalPool(nn.Module):
    """Simple temporal pooling (mean or max).

    Useful as a minimal baseline / control for temporal modeling.
    """

    def __init__(
        self,
        input_dim: int,
        pool_type: str = "mean",
        **kwargs,
    ) -> None:
        super().__init__()
        self.output_dim = input_dim
        self.pool_type = pool_type

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pool over the temporal dimension.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape ``[B, T, F]``.

        Returns
        -------
        torch.Tensor
            Pooled representation of shape ``[B, F]``.
        """
        if self.pool_type == "max":
            return x.max(dim=1).values
        return x.mean(dim=1)


# ── Temporal Encoder Factory ────────────────────────────────

_TEMPORAL_ENCODERS = {
    "gru": TemporalGRU,
    "lstm": TemporalLSTM,
    "conv1d": TemporalConv1D,
    "pool": TemporalPool,
}


def create_temporal_encoder(
    architecture: str,
    input_dim: int,
    hidden_dim: int = 256,
    num_layers: int = 1,
    dropout: float = 0.0,
    bidirectional: bool = False,
    **kwargs,
) -> nn.Module:
    """Factory function for temporal encoders.

    Parameters
    ----------
    architecture : str
        One of ``"gru"``, ``"lstm"``, ``"conv1d"``, ``"pool"``.
    input_dim : int
        Spatial feature dimension.
    hidden_dim : int
        Hidden dimension for the temporal encoder.
    num_layers : int
        Number of layers.
    dropout : float
        Dropout rate.
    bidirectional : bool
        Bidirectional flag (GRU/LSTM only).

    Returns
    -------
    nn.Module
        Temporal encoder instance with an ``output_dim`` attribute.
    """
    if architecture not in _TEMPORAL_ENCODERS:
        raise ValueError(
            f"Unknown temporal architecture '{architecture}'. "
            f"Available: {list(_TEMPORAL_ENCODERS.keys())}"
        )

    cls = _TEMPORAL_ENCODERS[architecture]
    return cls(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        bidirectional=bidirectional,
        **kwargs,
    )


# ── Composed Model ──────────────────────────────────────────

class TemporalClassificationModel(nn.Module):
    """Full temporal classification model.

    Composes::

        SpatialEncoder -> TemporalEncoder -> Linear Classifier

    Parameters
    ----------
    backbone_name : str
        Spatial backbone name.
    num_classes : int
        Number of output classes.
    temporal_arch : str
        Temporal encoder architecture name.
    temporal_hidden_dim : int
        Temporal encoder hidden dimension.
    temporal_num_layers : int
        Number of temporal encoder layers.
    temporal_dropout : float
        Temporal encoder dropout.
    temporal_bidirectional : bool
        Bidirectional flag.
    pretrained : bool
        Use pretrained backbone weights.
    freeze_backbone : bool
        Freeze spatial backbone parameters.
    classifier_dropout : float
        Dropout before the classifier head.
    """

    def __init__(
        self,
        backbone_name: str = "resnet18",
        num_classes: int = 2,
        temporal_arch: str = "gru",
        temporal_hidden_dim: int = 256,
        temporal_num_layers: int = 1,
        temporal_dropout: float = 0.0,
        temporal_bidirectional: bool = False,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        classifier_dropout: float = 0.3,
    ) -> None:
        super().__init__()

        # Spatial encoder
        self.spatial_encoder = SpatialEncoder(
            backbone_name=backbone_name,
            pretrained=pretrained,
            freeze=freeze_backbone,
        )

        spatial_dim = self.spatial_encoder.feature_dim

        # Temporal encoder
        self.temporal_encoder = create_temporal_encoder(
            architecture=temporal_arch,
            input_dim=spatial_dim,
            hidden_dim=temporal_hidden_dim,
            num_layers=temporal_num_layers,
            dropout=temporal_dropout,
            bidirectional=temporal_bidirectional,
        )

        temporal_out_dim = self.temporal_encoder.output_dim

        # Classifier head
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(temporal_out_dim, num_classes)

        # Log summary
        total_params = sum(p.numel() for p in self.parameters())
        trainable = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )
        logger.info(
            "TemporalClassificationModel: %s + %s | "
            "spatial_dim=%d temporal_dim=%d | "
            "classes=%d | params=%s | trainable=%s",
            backbone_name,
            temporal_arch,
            spatial_dim,
            temporal_out_dim,
            num_classes,
            f"{total_params:,}",
            f"{trainable:,}",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape ``[B, T, 3, H, W]``.

        Returns
        -------
        torch.Tensor
            Logits of shape ``[B, num_classes]``.
        """
        # [B, T, 3, H, W] -> [B, T, F]
        features = self.spatial_encoder(x)
        # [B, T, F] -> [B, D]
        temporal_repr = self.temporal_encoder(features)
        # [B, D] -> [B, C]
        temporal_repr = self.dropout(temporal_repr)
        logits = self.classifier(temporal_repr)
        return logits
