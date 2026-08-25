"""
Tests for the temporal modeling infrastructure (Phase 1).

Validates:
- SpatialEncoder output shapes
- Temporal encoder output shapes (GRU, LSTM, Conv1D, Pool)
- TemporalClassificationModel end-to-end shapes
- Config loading with and without temporal section
- Config backward compatibility (temporal disabled by default)
"""

import unittest

import torch

from training.models.temporal_model import (
    SpatialEncoder,
    TemporalClassificationModel,
    TemporalConv1D,
    TemporalGRU,
    TemporalLSTM,
    TemporalPool,
    create_temporal_encoder,
)


class TestSpatialEncoder(unittest.TestCase):
    """Tests for SpatialEncoder shape correctness."""

    def test_resnet18_output_shape(self):
        encoder = SpatialEncoder("resnet18", pretrained=False)
        # [B=2, T=4, C=3, H=224, W=224]
        x = torch.randn(2, 4, 3, 224, 224)
        out = encoder(x)
        self.assertEqual(out.shape, (2, 4, 512))

    def test_feature_dim_attribute(self):
        encoder = SpatialEncoder("resnet18", pretrained=False)
        self.assertEqual(encoder.feature_dim, 512)

    def test_frozen_backbone_no_grad(self):
        encoder = SpatialEncoder("resnet18", pretrained=False, freeze=True)
        for param in encoder.backbone.parameters():
            self.assertFalse(param.requires_grad)

    def test_single_frame_sequence(self):
        encoder = SpatialEncoder("resnet18", pretrained=False)
        x = torch.randn(1, 1, 3, 224, 224)
        out = encoder(x)
        self.assertEqual(out.shape, (1, 1, 512))


class TestTemporalGRU(unittest.TestCase):
    """Tests for TemporalGRU shape correctness."""

    def test_output_shape(self):
        gru = TemporalGRU(input_dim=512, hidden_dim=256)
        x = torch.randn(2, 8, 512)
        out = gru(x)
        self.assertEqual(out.shape, (2, 256))

    def test_output_dim_attribute(self):
        gru = TemporalGRU(input_dim=512, hidden_dim=128)
        self.assertEqual(gru.output_dim, 128)

    def test_bidirectional_output_shape(self):
        gru = TemporalGRU(input_dim=512, hidden_dim=128, bidirectional=True)
        x = torch.randn(2, 8, 512)
        out = gru(x)
        self.assertEqual(out.shape, (2, 256))
        self.assertEqual(gru.output_dim, 256)

    def test_multi_layer(self):
        gru = TemporalGRU(input_dim=512, hidden_dim=256, num_layers=2, dropout=0.1)
        x = torch.randn(2, 8, 512)
        out = gru(x)
        self.assertEqual(out.shape, (2, 256))


class TestTemporalLSTM(unittest.TestCase):
    """Tests for TemporalLSTM shape correctness."""

    def test_output_shape(self):
        lstm = TemporalLSTM(input_dim=512, hidden_dim=256)
        x = torch.randn(2, 8, 512)
        out = lstm(x)
        self.assertEqual(out.shape, (2, 256))

    def test_bidirectional(self):
        lstm = TemporalLSTM(input_dim=512, hidden_dim=128, bidirectional=True)
        x = torch.randn(2, 8, 512)
        out = lstm(x)
        self.assertEqual(out.shape, (2, 256))


class TestTemporalConv1D(unittest.TestCase):
    """Tests for TemporalConv1D shape correctness."""

    def test_output_shape(self):
        conv = TemporalConv1D(input_dim=512, hidden_dim=256)
        x = torch.randn(2, 8, 512)
        out = conv(x)
        self.assertEqual(out.shape, (2, 256))

    def test_multi_layer(self):
        conv = TemporalConv1D(input_dim=512, hidden_dim=256, num_layers=3)
        x = torch.randn(2, 8, 512)
        out = conv(x)
        self.assertEqual(out.shape, (2, 256))


class TestTemporalPool(unittest.TestCase):
    """Tests for TemporalPool shape correctness."""

    def test_mean_pool(self):
        pool = TemporalPool(input_dim=512, pool_type="mean")
        x = torch.randn(2, 8, 512)
        out = pool(x)
        self.assertEqual(out.shape, (2, 512))

    def test_max_pool(self):
        pool = TemporalPool(input_dim=512, pool_type="max")
        x = torch.randn(2, 8, 512)
        out = pool(x)
        self.assertEqual(out.shape, (2, 512))

    def test_output_dim_matches_input(self):
        pool = TemporalPool(input_dim=512)
        self.assertEqual(pool.output_dim, 512)


class TestCreateTemporalEncoder(unittest.TestCase):
    """Tests for the temporal encoder factory function."""

    def test_gru(self):
        enc = create_temporal_encoder("gru", input_dim=512, hidden_dim=128)
        self.assertEqual(enc.output_dim, 128)

    def test_lstm(self):
        enc = create_temporal_encoder("lstm", input_dim=512, hidden_dim=128)
        self.assertEqual(enc.output_dim, 128)

    def test_conv1d(self):
        enc = create_temporal_encoder("conv1d", input_dim=512, hidden_dim=128)
        self.assertEqual(enc.output_dim, 128)

    def test_pool(self):
        enc = create_temporal_encoder("pool", input_dim=512)
        self.assertEqual(enc.output_dim, 512)

    def test_unknown_raises(self):
        with self.assertRaises(ValueError):
            create_temporal_encoder("transformer", input_dim=512)


class TestTemporalClassificationModel(unittest.TestCase):
    """End-to-end tests for the composed temporal model."""

    def test_gru_end_to_end_shape(self):
        model = TemporalClassificationModel(
            backbone_name="resnet18",
            num_classes=2,
            temporal_arch="gru",
            temporal_hidden_dim=256,
            pretrained=False,
        )
        x = torch.randn(2, 8, 3, 224, 224)
        logits = model(x)
        self.assertEqual(logits.shape, (2, 2))

    def test_lstm_end_to_end_shape(self):
        model = TemporalClassificationModel(
            backbone_name="resnet18",
            num_classes=2,
            temporal_arch="lstm",
            temporal_hidden_dim=128,
            pretrained=False,
        )
        x = torch.randn(1, 4, 3, 224, 224)
        logits = model(x)
        self.assertEqual(logits.shape, (1, 2))

    def test_pool_end_to_end_shape(self):
        model = TemporalClassificationModel(
            backbone_name="resnet18",
            num_classes=2,
            temporal_arch="pool",
            pretrained=False,
        )
        x = torch.randn(2, 8, 3, 224, 224)
        logits = model(x)
        self.assertEqual(logits.shape, (2, 2))

    def test_conv1d_end_to_end_shape(self):
        model = TemporalClassificationModel(
            backbone_name="resnet18",
            num_classes=2,
            temporal_arch="conv1d",
            temporal_hidden_dim=128,
            pretrained=False,
        )
        x = torch.randn(1, 8, 3, 224, 224)
        logits = model(x)
        self.assertEqual(logits.shape, (1, 2))

    def test_output_is_differentiable(self):
        model = TemporalClassificationModel(
            backbone_name="resnet18",
            num_classes=2,
            temporal_arch="gru",
            temporal_hidden_dim=64,
            pretrained=False,
        )
        x = torch.randn(1, 4, 3, 224, 224)
        logits = model(x)
        loss = logits.sum()
        loss.backward()
        # Verify gradients exist on temporal encoder parameters
        for name, param in model.temporal_encoder.named_parameters():
            self.assertIsNotNone(
                param.grad,
                f"No gradient for temporal_encoder.{name}",
            )


class TestConfigTemporalDefaults(unittest.TestCase):
    """Tests that config backward compatibility is preserved."""

    def test_temporal_disabled_by_default(self):
        from training.configs.config import TrainingConfig
        config = TrainingConfig()
        self.assertFalse(config.temporal_enabled)

    def test_temporal_properties_have_defaults(self):
        from training.configs.config import TrainingConfig
        config = TrainingConfig()
        self.assertEqual(config.temporal_architecture, "gru")
        self.assertEqual(config.temporal_hidden_dim, 256)
        self.assertEqual(config.temporal_num_layers, 1)
        self.assertEqual(config.temporal_dropout, 0.0)
        self.assertFalse(config.temporal_bidirectional)
        self.assertEqual(config.temporal_classifier_dropout, 0.3)

    def test_temporal_config_loading(self):
        from training.configs.config import load_training_config
        config = load_training_config(
            "training/config/training_config_temporal.yaml"
        )
        self.assertTrue(config.temporal_enabled)
        self.assertEqual(config.temporal_architecture, "gru")
        self.assertEqual(config.temporal_hidden_dim, 256)
        self.assertEqual(config.experiment_id, "T01")

    def test_e07_config_temporal_disabled(self):
        from training.configs.config import load_training_config
        config = load_training_config(
            "training/config/training_config_v2.yaml"
        )
        self.assertFalse(config.temporal_enabled)


class TestModelFactoryTemporalRouting(unittest.TestCase):
    """Tests that model factory correctly routes temporal vs frame-level."""

    def test_temporal_enabled_creates_temporal_model(self):
        from training.configs.config import load_training_config
        from training.models.model_factory import create_model

        config = load_training_config(
            "training/config/training_config_temporal.yaml"
        )
        model = create_model(config)
        self.assertIsInstance(model, TemporalClassificationModel)

    def test_temporal_disabled_creates_standard_model(self):
        from training.configs.config import load_training_config
        from training.models.classification_model import ClassificationModel
        from training.models.model_factory import create_model

        config = load_training_config(
            "training/config/training_config_v2.yaml"
        )
        model = create_model(config)
        self.assertIsInstance(model, ClassificationModel)


if __name__ == "__main__":
    unittest.main()
