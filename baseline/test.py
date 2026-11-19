#!/usr/bin/env python
# ─────────────────────────────────────────────────────────────
# Production Test Entry Point
# ─────────────────────────────────────────────────────────────
"""
Final evaluation on the held-out test split.

Usage::

    python baseline/test.py --config baseline/p02.yaml --checkpoint baseline/outputs/latest/checkpoints/best.pt
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Add project root to sys.path so 'baseline' module can be imported
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RADS — Test Script (Pipeline 4)",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to training config YAML.",
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to model checkpoint (.pt).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from baseline.utils.config import load_training_config
    from baseline.utils.seed import set_global_seed
    from baseline.utils.device import get_device
    from baseline.utils.logger import TrainingLogger
    from baseline.utils.output_manager import TrainingOutputManager
    from baseline.models.model_factory import create_model
    from baseline.losses.classification_loss import create_loss
    from baseline.callbacks.checkpoint import CheckpointManager
    from baseline.datasets.dataloader import create_test_dataloader
    from baseline.engine.evaluator import Evaluator
    from baseline.utils.experiment_report import generate_experiment_report

    checkpoint_path = Path(args.checkpoint).resolve()
    config = load_training_config(args.config, checkpoint_path=checkpoint_path)
    set_global_seed(config.seed)
    device = get_device()
    test_start = time.time()

    tlogger = TrainingLogger.setup(config)
    output_mgr = TrainingOutputManager(config, allow_checkpoint_writes=False)

    # Build model and load checkpoint
    model = create_model(config)
    state = CheckpointManager.load(checkpoint_path)
    model.load_state_dict(state["model_state_dict"])
    tlogger.info(
        "Loaded checkpoint: %s (epoch %d)",
        checkpoint_path, state["epoch"] + 1,
    )

    # Build test data loader
    if config.temporal_enabled:
        from baseline.datasets.dataloader import create_temporal_test_dataloader
        test_loader = create_temporal_test_dataloader(config)
    else:
        test_loader = create_test_dataloader(config)
    class_names = test_loader.dataset.class_names  # type: ignore[attr-defined]

    # Build loss
    loss_fn = create_loss(config, device=device)

    # Evaluate
    evaluator = Evaluator(model, loss_fn, device, class_names)
    metrics = evaluator.evaluate(
        test_loader,
        split_name="test",
        output_manager=output_mgr,
    )

    output_mgr.finalize(
        duration_seconds=time.time() - test_start,
        status="test",
        extra_manifest={
            "checkpoint": CheckpointManager.describe(checkpoint_path, state, config=config),
        },
    )
    generate_experiment_report(output_mgr.run_dir, config=config)

    tlogger.info("Test complete. Results at: %s", output_mgr.run_dir)


if __name__ == "__main__":
    main()
