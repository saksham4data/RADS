#!/usr/bin/env python
"""
E06 W&B Replay Script
=====================
Uploads the completed E06 training history to Weights & Biases as a proper
training run, exactly mirroring what train.py would have logged in real-time.

Reads:
  training/outputs/latest/metrics/training_history.json
  training/outputs/latest/manifest.json
  training/outputs/latest/checkpoints/best.pt  (logged as artifact)

Run from project root:
  python scripts/replay_e06_to_wandb.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    # ── Paths ──────────────────────────────────────────────
    root = Path(__file__).resolve().parent.parent
    latest = root / "training" / "outputs" / "latest"
    history_path = latest / "metrics" / "training_history.json"
    manifest_path = latest / "manifest.json"
    best_ckpt = latest / "checkpoints" / "best.pt"

    for p in [history_path, manifest_path]:
        if not p.is_file():
            logger.error("Required file not found: %s", p)
            sys.exit(1)

    with open(history_path, "r") as f:
        history: dict = json.load(f)

    with open(manifest_path, "r") as f:
        manifest: dict = json.load(f)

    # ── Load config to reuse TrainingWandbManager ──────────
    sys.path.insert(0, str(root))
    from training.utils.config import load_training_config
    from training.utils.wandb_manager import TrainingWandbManager

    config = load_training_config()

    wb = TrainingWandbManager(config)
    run = wb.init(
        git_commit=manifest.get("git_commit"),
        extra_config={
            "replay": True,
            "replay_source": "training_history.json",
            "original_run_timestamp": manifest.get("timestamp"),
        },
    )

    if not wb.is_active:
        logger.error("W&B run failed to initialise. Aborting.")
        sys.exit(1)

    logger.info("W&B run: %s  |  URL: %s", run.id, wb.run_url)

    # ── Replay epoch-by-epoch ──────────────────────────────
    num_epochs = len(history.get("train_loss", []))
    logger.info("Replaying %d epochs ...", num_epochs)

    # Collect all per-epoch scalar series (everything that is a list of numbers)
    series: dict[str, list] = {
        k: v for k, v in history.items()
        if isinstance(v, list) and all(isinstance(x, (int, float)) for x in v)
    }

    for epoch_idx in range(num_epochs):
        log_dict: dict = {"epoch": epoch_idx}

        for key, values in series.items():
            if epoch_idx < len(values):
                val = values[epoch_idx]
                # Map flat history keys to the W&B namespaced keys used in train.py
                if key == "train_loss":
                    log_dict["train/loss"] = val
                elif key == "val_loss":
                    log_dict["val/loss"] = val
                elif key == "lr":
                    log_dict["lr/learning_rate"] = val
                else:
                    # val/*, train/* keys are already namespaced correctly
                    log_dict[key] = val

        run.log(log_dict)
        logger.info(
            "  epoch %d/%d  train/loss=%.4f  val/loss=%.4f  val/f1_accident=%s",
            epoch_idx + 1, num_epochs,
            log_dict.get("train/loss", float("nan")),
            log_dict.get("val/loss", float("nan")),
            f"{log_dict.get('val/f1_accident', 0):.4f}" if "val/f1_accident" in log_dict else "N/A",
        )

    # ── Log best-checkpoint artifact ───────────────────────
    execution = manifest.get("execution", {})
    best_metric = execution.get("best_metric")
    best_epoch  = execution.get("best_epoch")

    if best_ckpt.is_file():
        wb.log_model_checkpoint(
            best_ckpt,
            metadata={
                "best_metric": best_metric,
                "best_metric_name": config.monitor_metric,
                "best_epoch": best_epoch,
                "total_epochs": num_epochs,
                "replay": True,
            },
        )
        logger.info("Model artifact logged: best.pt (best_epoch=%s, metric=%.4f)",
                    best_epoch, best_metric or 0)
    else:
        logger.warning("best.pt not found — skipping artifact upload")

    # ── Summary ────────────────────────────────────────────
    run.summary.update({
        "best_metric": best_metric,
        "best_metric_name": config.monitor_metric,
        "best_epoch": best_epoch,
        "total_epochs": num_epochs,
        "duration_seconds": execution.get("duration_seconds"),
    })

    wb.finish()
    logger.info(
        "Done. View run at: %s",
        wb.run_url or "(URL unavailable after finish)",
    )


if __name__ == "__main__":
    main()
