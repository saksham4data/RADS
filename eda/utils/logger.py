# ─────────────────────────────────────────────────────────────
# EDA Dual Logging System
# ─────────────────────────────────────────────────────────────
"""
Provides two parallel log sinks:

1. **EDA execution log** → ``eda/logs/eda_<timestamp>.log``
   Detailed, per-run, JSON-structured entries.

2. **Root module log** → ``logs/eda/eda.log``
   High-level, append-mode, project-wide events.

Both loggers are initialised via `EDALogger.setup()` and
share the same Python ``logging`` infrastructure.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from eda.utils.config import EDAConfig


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
        # Attach any extra fields set via `logger.info("msg", extra={...})`
        for key in ("notebook", "wandb_run_id", "wandb_run_url",
                     "rows", "columns", "duration_s", "files_exported"):
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


# ── EDA Logger ──────────────────────────────────────────────

class EDALogger:
    """Dual logging system for EDA notebooks.

    Usage::

        logger = EDALogger.setup(config, notebook_name="01_dataset_overview")
        logger.info("Loaded %d rows", 5338)
        logger.log_execution_summary(...)
    """

    _instance: Optional[EDALogger] = None

    def __init__(
        self,
        config: EDAConfig,
        notebook_name: str,
        *,
        console_level: int = logging.INFO,
        file_level: int = logging.DEBUG,
    ) -> None:
        self.config = config
        self.notebook_name = notebook_name
        self.start_time = datetime.now(timezone.utc)
        self._timestamp_str = self.start_time.strftime("%Y-%m-%d_%H-%M-%S")

        # ── Create directories ──
        self.config.eda_logs_dir.mkdir(parents=True, exist_ok=True)
        self.config.root_eda_log_dir.mkdir(parents=True, exist_ok=True)

        # ── EDA execution log (per-run) ──
        self._eda_log_path = (
            self.config.eda_logs_dir / f"eda_{self._timestamp_str}.log"
        )
        self._eda_handler = logging.FileHandler(
            self._eda_log_path, encoding="utf-8"
        )
        self._eda_handler.setLevel(file_level)
        self._eda_handler.setFormatter(_JSONFormatter())

        # ── Root module log (append) ──
        self._root_log_path = self.config.root_eda_log_dir / "eda.log"
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
        self._logger = logging.getLogger(f"eda.{notebook_name}")
        self._logger.setLevel(logging.DEBUG)
        self._logger.handlers.clear()
        self._logger.addHandler(self._eda_handler)
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
        """Log notebook execution start with environment info."""
        info = self._get_environment_info()
        self.info(
            "═══ EDA START: %s ═══ | EDA v%s | Python %s | %s",
            self.notebook_name,
            self.config.eda_version,
            platform.python_version(),
            platform.system(),
        )
        self.debug("Environment: %s", json.dumps(info, default=str))

    def log_data_loaded(self, rows: int, columns: int, metadata_hash: str) -> None:
        """Log metadata loading event."""
        self.info(
            "Data loaded: %d rows × %d columns | hash=%s",
            rows, columns, metadata_hash[:16] + "...",
        )

    def log_execution_summary(
        self,
        *,
        duration_seconds: float,
        files_exported: List[str],
        wandb_run_id: Optional[str] = None,
        wandb_run_url: Optional[str] = None,
        stats_generated: Optional[List[str]] = None,
        warnings_count: int = 0,
        errors_count: int = 0,
    ) -> None:
        """Log a final summary when a notebook completes."""
        summary = {
            "notebook": self.notebook_name,
            "start_time": self.start_time.isoformat(),
            "end_time": datetime.now(timezone.utc).isoformat(),
            "duration_seconds": round(duration_seconds, 2),
            "files_exported": files_exported,
            "wandb_run_id": wandb_run_id,
            "wandb_run_url": wandb_run_url,
            "stats_generated": stats_generated or [],
            "warnings": warnings_count,
            "errors": errors_count,
        }
        self.info(
            "═══ EDA COMPLETE: %s ═══ | %.1fs | %d files exported",
            self.notebook_name, duration_seconds, len(files_exported),
        )
        self.debug("Execution summary: %s", json.dumps(summary, default=str))

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
        return {
            "python_version": platform.python_version(),
            "os": platform.system(),
            "os_version": platform.version(),
            "platform": platform.platform(),
            "eda_version": self.config.eda_version,
            "git_commit": self.get_git_commit(),
            "notebook": self.notebook_name,
            "config_path": str(self.config.config_path),
        }

    @property
    def log_path(self) -> Path:
        """Path to the current execution log file."""
        return self._eda_log_path

    # ── Class-level factory ─────────────────────────────────

    @classmethod
    def setup(
        cls,
        config: EDAConfig,
        notebook_name: str,
        **kwargs: Any,
    ) -> EDALogger:
        """Create (or replace) the singleton EDALogger."""
        cls._instance = cls(config, notebook_name, **kwargs)
        return cls._instance

    @classmethod
    def get(cls) -> EDALogger:
        """Return the active logger (must call ``setup`` first)."""
        if cls._instance is None:
            raise RuntimeError("EDALogger.setup() has not been called yet.")
        return cls._instance
