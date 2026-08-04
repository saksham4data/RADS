# ─────────────────────────────────────────────────────────────
# W&B Lifecycle & Artifact Manager
# ─────────────────────────────────────────────────────────────
"""
Wraps Weights & Biases initialisation, logging, artifact
uploads, and teardown into a reusable context manager.

Gracefully degrades if W&B is not installed or authenticated.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from eda.utils.config import EDAConfig

logger = logging.getLogger(__name__)

# ── Safe W&B import ─────────────────────────────────────────
_WANDB_AVAILABLE = False
try:
    import wandb  # type: ignore[import-untyped]
    _WANDB_AVAILABLE = True
except ImportError:
    wandb = None  # type: ignore[assignment]


class WandbManager:
    """Manages W&B run lifecycle with artifact support.

    Usage (context manager)::

        with WandbManager(config, "01_dataset_overview") as wb:
            wb.log_stats({"rows": 5338})
            wb.log_figure(fig, "missing_values")
            wb.log_table(df, "column_completeness")
            wb.log_notebook_outputs(figures_dir, reports_dir, exports_dir)

    Usage (explicit)::

        wb = WandbManager(config, "01_dataset_overview")
        wb.init(metadata_hash="sha256:abc...", git_commit="a1b2c3d")
        wb.log_stats({"rows": 5338})
        wb.finish()
    """

    def __init__(
        self,
        config: EDAConfig,
        notebook_name: str,
    ) -> None:
        self.config = config
        self.notebook_name = notebook_name
        self._run: Any = None
        self._enabled = _WANDB_AVAILABLE

    # ── Lifecycle ───────────────────────────────────────────

    def init(
        self,
        *,
        metadata_hash: Optional[str] = None,
        git_commit: Optional[str] = None,
        extra_config: Optional[Dict[str, Any]] = None,
    ) -> Optional[Any]:
        """Initialise a W&B run."""
        if not self._enabled:
            logger.warning("W&B is not available — running without tracking.")
            return None

        # Auth
        api_key = os.environ.get("WANDB_API_KEY")
        if api_key:
            wandb.login(key=api_key, relogin=False)

        # Build config
        run_config: Dict[str, Any] = {
            "eda_version": self.config.eda_version,
            "notebook": self.notebook_name,
            "seed": self.config.seed,
            "metadata_hash": metadata_hash,
            "git_commit": git_commit,
        }
        if extra_config:
            run_config.update(extra_config)

        wb_cfg = self.config.wandb
        tags = list(wb_cfg.get("tags", [])) + [self.notebook_name]

        try:
            self._run = wandb.init(
                project=wb_cfg["project"],
                entity=wb_cfg.get("entity"),
                job_type=wb_cfg.get("job_type", "eda"),
                group=wb_cfg.get("group", "exploratory-analysis"),
                tags=tags,
                name=f"eda-{self.notebook_name}",
                config=run_config,
                reinit=True,
            )
            logger.info(
                "W&B run initialised: %s (ID: %s)",
                self._run.name, self._run.id,
            )
        except Exception as exc:
            logger.warning("Failed to init W&B run: %s", exc)
            self._enabled = False
            self._run = None

        return self._run

    def finish(self) -> None:
        """Finish the active W&B run."""
        if self._run is not None:
            try:
                self._run.finish()
                logger.info("W&B run finished: %s", self._run.id)
            except Exception as exc:
                logger.warning("Error finishing W&B run: %s", exc)
            self._run = None

    # ── Context manager ─────────────────────────────────────

    def __enter__(self) -> WandbManager:
        self.init()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.finish()

    # ── Logging methods ─────────────────────────────────────

    def log_stats(self, stats: Dict[str, Any]) -> None:
        """Log key-value metrics."""
        if self._run is None:
            return
        try:
            self._run.log(stats)
        except Exception as exc:
            logger.warning("W&B log_stats failed: %s", exc)

    def log_figure(
        self,
        fig: Any,
        name: str,
        *,
        caption: Optional[str] = None,
    ) -> None:
        """Log a matplotlib figure as a W&B Image."""
        if self._run is None:
            return
        try:
            self._run.log({
                name: wandb.Image(fig, caption=caption),
            })
        except Exception as exc:
            logger.warning("W&B log_figure failed for '%s': %s", name, exc)

    def log_table(
        self,
        df: pd.DataFrame,
        name: str,
    ) -> None:
        """Log a DataFrame as a W&B Table."""
        if self._run is None:
            return
        try:
            table = wandb.Table(dataframe=df.reset_index(drop=True))
            self._run.log({name: table})
        except Exception as exc:
            logger.warning("W&B log_table failed for '%s': %s", name, exc)

    def log_system_info(self, system_snapshot: Dict[str, Any]) -> None:
        """Log system resource metrics using standardized hierarchical keys.

        This is the **single source of truth** for converting raw
        :class:`SystemMonitor` snapshots into W&B metric names.  Only
        numeric/string metrics relevant to dashboards are emitted;
        metadata fields (``label``, ``timestamp``, ``epoch``,
        ``software``) are intentionally excluded.
        """
        if self._run is None:
            return

        flat: Dict[str, Any] = {
            "system/cpu/percent": system_snapshot["cpu"]["percent"],
            "system/cpu/cores": system_snapshot["cpu"]["count_logical"],
            "system/memory/total_gb": system_snapshot["memory"]["total_gb"],
            "system/memory/used_gb": system_snapshot["memory"]["used_gb"],
            "system/memory/percent": system_snapshot["memory"]["percent"],
            "system/disk/used_gb": system_snapshot["disk"]["used_gb"],
            "system/disk/percent": system_snapshot["disk"]["percent"],
        }

        # GPU metrics (optional — snapshot may have gpu=None)
        gpus = system_snapshot.get("gpu")
        if gpus:
            for gpu in gpus:
                prefix = f"system/gpu_{gpu['id']}"
                flat[f"{prefix}/name"] = gpu["name"]
                flat[f"{prefix}/memory_used_mb"] = gpu["memory_used_mb"]
                flat[f"{prefix}/load_percent"] = gpu.get("load_percent")

        self.log_stats(flat)

    # ── Artifact methods ────────────────────────────────────

    def log_artifact(
        self,
        name: str,
        artifact_type: str,
        file_paths: List[Union[str, Path]],
        *,
        metadata: Optional[Dict[str, Any]] = None,
        description: Optional[str] = None,
    ) -> None:
        """Log files as a versioned W&B Artifact."""
        if self._run is None:
            return
        try:
            artifact = wandb.Artifact(
                name=name,
                type=artifact_type,
                description=description,
                metadata=metadata or {},
            )
            for fp in file_paths:
                fp = Path(fp)
                if fp.is_file():
                    artifact.add_file(str(fp))
                elif fp.is_dir():
                    artifact.add_dir(str(fp))
            self._run.log_artifact(artifact)
            logger.info("W&B artifact logged: %s (%s)", name, artifact_type)
        except Exception as exc:
            logger.warning("W&B log_artifact failed for '%s': %s", name, exc)

    def log_notebook_outputs(
        self,
        figures_dir: Path,
        reports_dir: Path,
        exports_dir: Path,
        *,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Upload all notebook outputs as a single W&B Artifact."""
        if self._run is None:
            return

        artifact_name = f"eda-{self.notebook_name}-outputs"
        try:
            artifact = wandb.Artifact(
                name=artifact_name,
                type="eda-output",
                description=f"Complete outputs from {self.notebook_name}",
                metadata=metadata or {},
            )
            for d, label in [
                (figures_dir, "figures"),
                (reports_dir, "reports"),
                (exports_dir, "exports"),
            ]:
                if d.is_dir() and any(d.iterdir()):
                    artifact.add_dir(str(d), name=label)

            self._run.log_artifact(artifact)
            logger.info("W&B notebook outputs artifact logged: %s", artifact_name)
        except Exception as exc:
            logger.warning(
                "W&B log_notebook_outputs failed for '%s': %s",
                artifact_name, exc,
            )

    # ── Properties ──────────────────────────────────────────

    @property
    def run_id(self) -> Optional[str]:
        """The W&B run ID, or None."""
        return self._run.id if self._run else None

    @property
    def run_url(self) -> Optional[str]:
        """The W&B run URL, or None."""
        return self._run.get_url() if self._run else None

    @property
    def is_active(self) -> bool:
        """True if a W&B run is active."""
        return self._run is not None
