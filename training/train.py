#!/usr/bin/env python
# ─────────────────────────────────────────────────────────────
# Production Training Entry Point
# ─────────────────────────────────────────────────────────────
"""
End-to-end training script for RADS classification models.

Usage::

    python training/train.py --config training/config/training_config_v1.yaml
    python training/train.py --config training/config/training_config_v2.yaml
    python training/train.py --resume training/outputs/latest/checkpoints/last.pt
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="RADS — Training Script (Pipeline 4)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to training config YAML (default: auto-detect).",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint .pt file to resume from.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the complete training pipeline."""
    args = parse_args()

    # ── 1. Load Configuration ──
    from training.utils.config import load_training_config
    config = load_training_config(args.config)

    # ── 2. Seed ──
    from training.utils.seed import set_global_seed
    set_global_seed(config.seed)

    # ── 3. Device ──
    from training.utils.device import get_device
    device = get_device()

    # ── 4. Logger ──
    from training.utils.logger import TrainingLogger
    tlogger = TrainingLogger.setup(config)

    # ── 5. Output Manager ──
    from training.utils.output_manager import TrainingOutputManager
    output_mgr = TrainingOutputManager(config, allow_checkpoint_writes=True)

    # ── 6. W&B ──
    from training.utils.wandb_manager import TrainingWandbManager
    wb = TrainingWandbManager(config)
    wb.init(git_commit=TrainingLogger.get_git_commit())

    # ── 7. System Monitor ──
    from training.utils.system_monitor import SystemMonitor
    from training.utils.experiment_report import generate_experiment_report
    monitor = SystemMonitor()
    start_snap = monitor.snapshot("training_start")
    if wb.is_active:
        wb.log_system_info(start_snap)

    # ── 8. Data ──
    from training.datasets.dataloader import create_dataloaders
    train_loader, val_loader = create_dataloaders(config)
    class_names = train_loader.dataset.class_names  # type: ignore[attr-defined]

    # ── 9. Model ──
    from training.models.model_factory import create_model
    model = create_model(config)

    # ── 10. Loss ──
    from training.losses.classification_loss import create_loss
    class_weights = train_loader.dataset.class_weights  # type: ignore[attr-defined]
    loss_fn = create_loss(config, class_weights=class_weights, device=device)

    # ── 11. Optimizer ──
    optimizer = _build_optimizer(config, model)

    # ── 12. Scheduler ──
    scheduler = _build_scheduler(config, optimizer)

    # ── 13. Callbacks ──
    from training.callbacks.checkpoint import CheckpointManager
    from training.callbacks.early_stopping import EarlyStopping
    from training.callbacks.lr_monitor import LearningRateMonitor

    ckpt_mgr = CheckpointManager(
        checkpoint_dir=output_mgr.checkpoints_dir,
        monitor_metric=config.monitor_metric,
        mode=config.monitor_mode,
        save_best=config.save_best,
        save_last=config.save_last,
    )

    early_stop = None
    if config.early_stopping_enabled:
        es_cfg = config.early_stopping_config
        early_stop = EarlyStopping(
            patience=es_cfg["patience"],
            min_delta=es_cfg.get("min_delta", 0.001),
            mode=es_cfg.get("mode", "min"),
        )

    lr_monitor = LearningRateMonitor()

    # ── 14. Resume ──
    start_epoch = 0
    if args.resume:
        resume_path = Path(args.resume)
        start_epoch = ckpt_mgr.resume_from(
            resume_path, model, optimizer, scheduler,
        )
        tlogger.info("Resumed from %s at epoch %d", resume_path, start_epoch + 1)

    # ── 15. Train ──
    from training.engine.trainer import Trainer

    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        training_logger=tlogger,
        wandb_manager=wb,
        checkpoint_manager=ckpt_mgr,
        early_stopping=early_stop,
        lr_monitor=lr_monitor,
        class_names=class_names,
    )

    train_start = time.time()
    history = trainer.fit(start_epoch=start_epoch)
    duration = time.time() - train_start

    # ── 16. Save training history ──
    output_mgr.save_training_history(history, "training_history")

    # ── 17. End system snapshot ──
    end_snap = monitor.snapshot("training_end")
    if wb.is_active:
        wb.log_system_info(end_snap)

    # ── 18. Log best checkpoint to W&B ──
    if wb.is_active and ckpt_mgr.best_value is not None:
        best_path = output_mgr.checkpoints_dir / "best.pt"
        if best_path.is_file():
            wb.log_model_checkpoint(
                best_path,
                metadata={
                    "best_metric": ckpt_mgr.best_value,
                    "best_epoch": ckpt_mgr.best_epoch,
                },
            )

    # ── 19. Training Summary ──
    tlogger.log_training_summary(
        total_epochs=len(history["train_loss"]),
        best_metric=ckpt_mgr.best_value or 0.0,
        best_epoch=ckpt_mgr.best_epoch,
        duration_seconds=duration,
        wandb_run_id=wb.run_id,
        wandb_run_url=wb.run_url,
    )

    # ── 20. Finalize ──
    manifest = output_mgr.finalize(
        wandb_run_id=wb.run_id,
        wandb_run_url=wb.run_url,
        git_commit=TrainingLogger.get_git_commit(),
        total_epochs=len(history["train_loss"]),
        best_metric=ckpt_mgr.best_value,
        best_epoch=ckpt_mgr.best_epoch,
        duration_seconds=duration,
        extra_manifest={
            "augmentation": config.augmentation_config,
            "augmentation_notes": "Validation and test splits use deterministic transforms.",
        },
    )
    generate_experiment_report(output_mgr.run_dir, config=config)

    wb.finish()
    tlogger.info("All outputs saved to: %s", output_mgr.run_dir)


# ── Helper: Optimizer builder ───────────────────────────────

def _build_optimizer(config, model) -> torch.optim.Optimizer:
    """Build the optimizer from config."""
    name = config.optimizer_name.lower()
    params = model.parameters()

    if name == "adam":
        return torch.optim.Adam(
            params, lr=config.learning_rate, weight_decay=config.weight_decay,
        )
    elif name == "adamw":
        return torch.optim.AdamW(
            params, lr=config.learning_rate, weight_decay=config.weight_decay,
        )
    elif name == "sgd":
        return torch.optim.SGD(
            params, lr=config.learning_rate, weight_decay=config.weight_decay,
            momentum=0.9,
        )
    else:
        raise ValueError(f"Unknown optimizer: '{name}'")


# ── Helper: Scheduler builder ──────────────────────────────

def _build_scheduler(config, optimizer):
    """Build the LR scheduler from config."""
    name = config.scheduler_name.lower()
    sched_cfg = config.scheduler_config

    if name == "step_lr":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=sched_cfg.get("step_size", 5),
            gamma=sched_cfg.get("gamma", 0.1),
        )
    elif name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.epochs,
        )
    elif name == "none":
        return None
    else:
        raise ValueError(f"Unknown scheduler: '{name}'")


if __name__ == "__main__":
    main()
