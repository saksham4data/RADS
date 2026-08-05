# ─────────────────────────────────────────────────────────────
# Training Dual Logging System
# ─────────────────────────────────────────────────────────────
"""
Provides two parallel log sinks for training:

1. **Training execution log** → ``training/outputs/logs/training_<ts>.log``
   Detailed, per-run, JSON-structured entries.

2. **Root module log** → ``logs/training/training.log``
   High-level, append-mode, project-wide events.

Pattern: mirrors ``eda.utils.logger.EDALogger`` exactly.
"""

from __future__ import annotations

import json
import logging
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from training.configs.config import TrainingConfig


# ── JSON Formatter ──────────────────────────────────────────

class _JSONFormatter(logging.Formatter):
    """Emit each log record as a single JSON line."""

    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        if record.exc_info and record.exc_info[1]:
            entry["exception"] = self.formatException(record.exc_info)
        # Attach any extra fields
        for key in ("phase", "epoch", "batch", "wandb_run_id",
                     "wandb_run_url", "duration_s", "loss", "lr"):
            val = getattr(record, key, None)
            if val is not None:
                entry[key] = val
        return json.dumps(entry, default=str)


# ── Readable Console Formatter ──────────────────────────────

class _ConsoleFormatter(logging.Formatter):
    """Coloured, human-friendly console output."""

    GREY = "\033[38;5;245m"
    CYAN = "\033[36m"
    YELLOW = "\033[33m"
    RED = "\033[31m"
    BOLD_RED = "\033[1;31m"
    GREEN = "\033[32m"
    RESET = "\033[0m"

    COLOURS = {
        logging.DEBUG: GREY,
        logging.INFO: CYAN,
        logging.WARNING: YELLOW,
        logging.ERROR: RED,
        logging.CRITICAL: BOLD_RED,
    }

    def format(self, record: logging.LogRecord) -> str:
        colour = self.COLOURS.get(record.levelno, self.RESET)
        ts = datetime.fromtimestamp(record.created).strftime("%H:%M:%S")
        return (
            f"{colour}[{ts}] {record.levelname:<8}{self.RESET} "
            f"{record.getMessage()}"
        )


# ── Training Logger ─────────────────────────────────────────

class TrainingLogger:
    """Dual logging system for training scripts and notebooks.

    Usage::

        logger = TrainingLogger.setup(config, run_name="sanity_check")
        logger.info("Loaded %d samples", 555)
        logger.log_epoch_end(epoch=0, train_loss=1.23, val_loss=0.98)
    """

    _instance: Optional[TrainingLogger] = None

    def __init__(
        self,
        config: TrainingConfig,
        run_name: str,
        *,
        console_level: int = logging.INFO,
        file_level: int = logging.DEBUG,
    ) -> None:
        self.config = config
        self.run_name = run_name
        self.start_time = datetime.now(timezone.utc)
        self._timestamp_str = self.start_time.strftime("%Y-%m-%d_%H-%M-%S")

        # ── Create directories ──
        self.config.training_logs_dir.mkdir(parents=True, exist_ok=True)
        self.config.root_training_log_dir.mkdir(parents=True, exist_ok=True)

        # ── Training execution log (per-run) ──
        self._training_log_path = (
            self.config.training_logs_dir
            / f"training_{self._timestamp_str}.log"
        )
        self._training_handler = logging.FileHandler(
            self._training_log_path, encoding="utf-8"
        )
        self._training_handler.setLevel(file_level)
        self._training_handler.setFormatter(_JSONFormatter())

        # ── Root module log (append) ──
        self._root_log_path = (
            self.config.root_training_log_dir / "training.log"
        )
        self._root_handler = logging.FileHandler(
            self._root_log_path, mode="a", encoding="utf-8"
        )
        self._root_handler.setLevel(logging.INFO)
        self._root_handler.setFormatter(_JSONFormatter())

        # ── Console handler ──
        self._console_handler = logging.StreamHandler(sys.stdout)
        self._console_handler.setLevel(console_level)
        self._console_handler.setFormatter(_ConsoleFormatter())

        # ── Logger instance ──
        self._logger = logging.getLogger(f"training.{run_name}")
        self._logger.setLevel(logging.DEBUG)
        self._logger.handlers.clear()
        self._logger.addHandler(self._training_handler)
        self._logger.addHandler(self._root_handler)
        self._logger.addHandler(self._console_handler)
        self._logger.propagate = False

    # ── Proxy standard logging methods ──────────────────────

    def debug(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self._logger.debug(msg, *args, **kwargs)

    def info(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self._logger.info(msg, *args, **kwargs)

    def warning(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self._logger.warning(msg, *args, **kwargs)

    def error(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self._logger.error(msg, *args, **kwargs)

    def critical(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self._logger.critical(msg, *args, **kwargs)

    def exception(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self._logger.exception(msg, *args, **kwargs)

    # ── Structured events ───────────────────────────────────

    def log_start(self) -> None:
        """Log training run start with environment info."""
        info = self._get_environment_info()
        self.info(
            "=== TRAINING START: %s === | v%s | Python %s | %s",
            self.run_name,
            self.config.training_version,
            platform.python_version(),
            platform.system(),
        )
        self.debug("Environment: %s", json.dumps(info, default=str))

    def log_epoch_start(self, epoch: int, total_epochs: int) -> None:
        """Log the beginning of an epoch."""
        self.info(
            "-- Epoch %d/%d ------------------------------------------------",
            epoch + 1, total_epochs,
        )

    def log_epoch_end(
        self,
        epoch: int,
        *,
        train_loss: float,
        val_loss: Optional[float] = None,
        train_metrics: Optional[Dict[str, Any]] = None,
        val_metrics: Optional[Dict[str, Any]] = None,
        lr: Optional[float] = None,
        duration_s: Optional[float] = None,
    ) -> None:
        """Log epoch completion with metrics."""
        parts = [f"Epoch {epoch + 1} complete"]
        parts.append(f"train_loss={train_loss:.4f}")
        if val_loss is not None:
            parts.append(f"val_loss={val_loss:.4f}")
        if lr is not None:
            parts.append(f"lr={lr:.6f}")
        if duration_s is not None:
            parts.append(f"time={duration_s:.1f}s")
        self.info(" | ".join(parts))

        if val_metrics:
            self.debug(
                "Val metrics: %s",
                json.dumps(val_metrics, default=str),
            )

    def log_training_summary(
        self,
        *,
        total_epochs: int,
        best_metric: float,
        best_epoch: int,
        duration_seconds: float,
        wandb_run_id: Optional[str] = None,
        wandb_run_url: Optional[str] = None,
    ) -> None:
        """Log a final summary when training completes."""
        summary = {
            "run_name": self.run_name,
            "start_time": self.start_time.isoformat(),
            "end_time": datetime.now(timezone.utc).isoformat(),
            "total_epochs": total_epochs,
            "best_metric": best_metric,
            "best_epoch": best_epoch,
            "duration_seconds": round(duration_seconds, 2),
            "wandb_run_id": wandb_run_id,
            "wandb_run_url": wandb_run_url,
        }
        self.info(
            "=== TRAINING COMPLETE: %s === | %d epochs | %.1fs | best@epoch %d",
            self.run_name, total_epochs, duration_seconds, best_epoch + 1,
        )
        self.debug("Training summary: %s", json.dumps(summary, default=str))

    # ── Helpers ─────────────────────────────────────────────

    @staticmethod
    def get_git_commit() -> Optional[str]:
        """Return the current git commit hash, or None."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        return None

    def _get_environment_info(self) -> Dict[str, Any]:
        """Gather environment metadata."""
        info: Dict[str, Any] = {
            "python_version": platform.python_version(),
            "os": platform.system(),
            "os_version": platform.version(),
            "platform": platform.platform(),
            "training_version": self.config.training_version,
            "git_commit": self.get_git_commit(),
            "run_name": self.run_name,
            "config_path": str(self.config.config_path),
        }
        try:
            import torch
            info["pytorch_version"] = torch.__version__
            info["cuda_available"] = torch.cuda.is_available()
            if torch.cuda.is_available():
                info["gpu_name"] = torch.cuda.get_device_name(0)
        except ImportError:
            info["pytorch_version"] = "not installed"
        return info

    @property
    def log_path(self) -> Path:
        """Path to the current execution log file."""
        return self._training_log_path

    # ── Class-level factory ─────────────────────────────────

    @classmethod
    def setup(
        cls,
        config: TrainingConfig,
        run_name: str,
        **kwargs: Any,
    ) -> TrainingLogger:
        """Create (or replace) the singleton TrainingLogger."""
        cls._instance = cls(config, run_name, **kwargs)
        return cls._instance

    @classmethod
    def get(cls) -> TrainingLogger:
        """Return the active logger (must call ``setup`` first)."""
        if cls._instance is None:
            raise RuntimeError("TrainingLogger.setup() has not been called yet.")
        return cls._instance
