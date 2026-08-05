#!/usr/bin/env python
# ─────────────────────────────────────────────────────────────
# Production Validation Entry Point
# ─────────────────────────────────────────────────────────────
"""
Evaluate a trained model on the validation split.

Usage::

    python training/validate.py --checkpoint training/outputs/latest/checkpoints/best.pt
    python training/validate.py --config custom.yaml --checkpoint best.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RADS — Validation Script (Pipeline 4)",
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

    from training.utils.config import load_training_config
    from training.utils.seed import set_global_seed
    from training.utils.device import get_device
    from training.utils.logger import TrainingLogger
    from training.utils.output_manager import TrainingOutputManager
    from training.models.model_factory import create_model
    from training.losses.classification_loss import create_loss
    from training.callbacks.checkpoint import CheckpointManager
    from training.datasets.dataloader import create_dataloaders
    from training.engine.evaluator import Evaluator

    config = load_training_config(args.config)
    set_global_seed(config.seed)
    device = get_device()

    tlogger = TrainingLogger.setup(config, run_name="validate")
    output_mgr = TrainingOutputManager(config, run_name="validate")

    # Build model and load checkpoint
    model = create_model(config)
    state = CheckpointManager.load(Path(args.checkpoint))
    model.load_state_dict(state["model_state_dict"])
    tlogger.info(
        "Loaded checkpoint: %s (epoch %d)",
        args.checkpoint, state["epoch"] + 1,
    )

    # Build data loader (val split only)
    _, val_loader = create_dataloaders(config)
    class_names = val_loader.dataset.class_names  # type: ignore[attr-defined]

    # Build loss
    loss_fn = create_loss(config, device=device)

    # Evaluate
    evaluator = Evaluator(model, loss_fn, device, class_names)
    metrics = evaluator.evaluate(
        val_loader,
        split_name="val",
        output_manager=output_mgr,
    )

    output_mgr.finalize(
        total_epochs=state["epoch"] + 1,
        status="validation",
    )

    tlogger.info("Validation complete. Results at: %s", output_mgr.run_dir)


if __name__ == "__main__":
    main()
