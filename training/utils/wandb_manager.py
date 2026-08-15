# ─────────────────────────────────────────────────────────────
# Training W&B Lifecycle & Artifact Manager
# ─────────────────────────────────────────────────────────────
"""
Wraps Weights & Biases initialisation, logging, artifact
uploads, and teardown for training runs.

Extends the EDA W&B pattern with training-specific methods:
``log_epoch_metrics``, ``log_model_checkpoint``,
``log_confusion_matrix``, ``log_learning_rate``.

Gracefully degrades if W&B is not installed or authenticated.

Pattern: mirrors ``eda.utils.wandb_manager.WandbManager``.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from training.configs.config import TrainingConfig

logger = logging.getLogger(__name__)
# Ensure W&B warnings always reach stderr even when no training logger
# is attached to this module's logger (which caused silent failures in E06).
if not logger.handlers:
    _fallback = logging.StreamHandler()
    _fallback.setLevel(logging.WARNING)
    _fallback.setFormatter(logging.Formatter("%(levelname)s [%(name)s] %(message)s"))
    logger.addHandler(_fallback)
    logger.propagate = True   # also reach root logger if configured

# ── Safe W&B import ─────────────────────────────────────────
_WANDB_AVAILABLE = False
try:
    import wandb  # type: ignore[import-untyped]
    _WANDB_AVAILABLE = True
except ImportError:
    wandb = None  # type: ignore[assignment]


class TrainingWandbManager:
    """Manages W&B run lifecycle for training.

    Usage (context manager)::

        with TrainingWandbManager(config, "sanity_check") as wb:
            wb.log_epoch_metrics(0, {"train_loss": 1.23, "val_loss": 0.98})
            wb.log_learning_rate(0.001, 0)
            wb.log_model_checkpoint(Path("best.pt"))

    Usage (explicit)::

        wb = TrainingWandbManager(config, "baseline_run")
        wb.init(git_commit="a1b2c3d")
        wb.log_stats({"loss": 0.5})
        wb.finish()
    """

    def __init__(
        self,
        config: TrainingConfig,
        run_name: Optional[str] = None,
    ) -> None:
        self.config = config
        self.run_name = run_name or config.experiment_name
        self._run: Any = None
        self._enabled = _WANDB_AVAILABLE

    # ── Lifecycle ───────────────────────────────────────────

    def init(
        self,
        *,
        git_commit: Optional[str] = None,
        extra_config: Optional[Dict[str, Any]] = None,
    ) -> Optional[Any]:
        """Initialise a W&B run."""
        if not self._enabled:
            logger.warning("W&B is not available — running without tracking.")
            return None

        # Build config
        run_config: Dict[str, Any] = {
            "training_version": self.config.training_version,
            "run_name": self.run_name,
            "model": self.config.model_name,
            "dataset": self.config.dataset_name,
            "epochs": self.config.epochs,
            "batch_size": self.config.batch_size,
            "learning_rate": self.config.learning_rate,
            "optimizer": self.config.optimizer_name,
            "scheduler": self.config.scheduler_name,
            "image_size": self.config.image_size,
            "frames_per_video": self.config.frames_per_video,
            "seed": self.config.seed,
            "git_commit": git_commit,
            "augmentation": self.config.augmentation_config,
        }
        if extra_config:
            run_config.update(extra_config)

        wb_cfg = self.config.wandb
        # W&B enforces a 64-character maximum on tag strings (wandb ≥ 0.18 / pydantic).
        # Truncate run_name to avoid silent init failure when experiment names are long.
        run_name_tag = self.run_name[:64] if self.run_name else ""
        tags = list(wb_cfg.get("tags", [])) + [run_name_tag]

        try:
            # Auth — inside try/except so auth errors are caught and logged
            api_key = os.environ.get("WANDB_API_KEY")
            if api_key:
                wandb.login(key=api_key, relogin=False)

            self._run = wandb.init(
                project=wb_cfg["project"],
                entity=wb_cfg.get("entity"),
                job_type=wb_cfg.get("job_type", "training"),
                group=wb_cfg.get("group", "baseline-classification"),
                tags=tags,
                name=f"train-{self.run_name}",
                config=run_config,
                reinit=True,
            )

            # wandb 0.18+ may return a NoopRun if init silently fails
            if self._run is None or getattr(self._run, "id", None) is None:
                logger.warning(
                    "W&B init returned an inactive run (NoopRun or None) — "
                    "W&B logging disabled for this run."
                )
                self._enabled = False
                self._run = None
                return None

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

    def __enter__(self) -> TrainingWandbManager:
        self.init()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.finish()

    # ── Core logging methods ────────────────────────────────

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

    def log_system_info(self, system_snapshot: Dict[str, Any]) -> None:
        """Log system resource metrics using hierarchical keys.

        Reuses the same key structure as
        ``eda.utils.wandb_manager.WandbManager.log_system_info``.
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

        # GPU metrics (optional)
        gpus = system_snapshot.get("gpu")
        if gpus:
            for gpu in gpus:
                prefix = f"system/gpu_{gpu['id']}"
                flat[f"{prefix}/name"] = gpu["name"]
                flat[f"{prefix}/memory_used_mb"] = gpu["memory_used_mb"]
                flat[f"{prefix}/load_percent"] = gpu.get("load_percent")

        self.log_stats(flat)

    # ── Training-specific logging ───────────────────────────

    def log_epoch_metrics(
        self,
        epoch: int,
        metrics: Dict[str, Any],
    ) -> None:
        """Log all metrics for a given epoch.

        Parameters
        ----------
        epoch : int
            Current epoch number (0-indexed).
        metrics : dict
            Flat dict of metric name → value.
        """
        if self._run is None:
            return
        try:
            log_dict = {"epoch": epoch}
            log_dict.update(metrics)
            self._run.log(log_dict)
        except Exception as exc:
            logger.warning("W&B log_epoch_metrics failed: %s", exc)

    def log_learning_rate(self, lr: float, epoch: int) -> None:
        """Log the current learning rate.

        Parameters
        ----------
        lr : float
            Current learning rate.
        epoch : int
            Current epoch number (0-indexed).
        """
        if self._run is None:
            return
        try:
            self._run.log({
                "lr/learning_rate": lr,
                "epoch": epoch,
            })
        except Exception as exc:
            logger.warning("W&B log_learning_rate failed: %s", exc)

    def log_model_checkpoint(
        self,
        checkpoint_path: Path,
        *,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log a model checkpoint as a W&B Artifact.

        Parameters
        ----------
        checkpoint_path : Path
            Path to the ``.pt`` checkpoint file.
        metadata : dict, optional
            Extra metadata (epoch, metrics, etc.).
        """
        if self._run is None:
            return
        try:
            # Artifact names may only contain alphanumeric chars, dashes,
            # underscores, and dots — sanitize run_name accordingly.
            import re
            safe_name = re.sub(r"[^\w.\-]", "-", self.run_name)[:128]
            artifact = wandb.Artifact(
                name=f"model-{safe_name}",
                type="model",
                description=f"Checkpoint from {self.run_name}",
                metadata=metadata or {},
            )
            artifact.add_file(str(checkpoint_path))
            self._run.log_artifact(artifact)
            logger.info("W&B model artifact logged: %s", checkpoint_path.name)
        except Exception as exc:
            logger.warning(
                "W&B log_model_checkpoint failed: %s", exc,
            )

    def log_confusion_matrix(
        self,
        cm: np.ndarray,
        class_names: List[str],
        *,
        title: str = "confusion_matrix",
    ) -> None:
        """Log a confusion matrix as a W&B plot.

        Parameters
        ----------
        cm : np.ndarray
            Confusion matrix of shape ``[N, N]``.
        class_names : list[str]
            Names corresponding to each class index.
        title : str
            Key name for the logged plot.
        """
        if self._run is None:
            return
        try:
            # Build the wandb confusion matrix data
            data = []
            for i, actual in enumerate(class_names):
                for j, predicted in enumerate(class_names):
                    data.append([actual, predicted, int(cm[i][j])])

            table = wandb.Table(
                data=data,
                columns=["Actual", "Predicted", "Count"],
            )
            self._run.log({
                title: wandb.plot.confusion_matrix(
                    probs=None,
                    y_true=[class_names[i] for i in range(len(class_names))
                            for _ in range(len(class_names))],
                    preds=[class_names[j] for _ in range(len(class_names))
                           for j in range(len(class_names))],
                    class_names=class_names,
                    title=title,
                ) if hasattr(wandb.plot, "confusion_matrix") else table,
            })
        except Exception as exc:
            logger.warning("W&B log_confusion_matrix failed: %s", exc)

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

    # ── Properties ──────────────────────────────────────────

    @property
    def run_id(self) -> Optional[str]:
        """The W&B run ID, or None."""
        return self._run.id if self._run else None

    @property
    def run_url(self) -> Optional[str]:
        """The W&B run URL, or None."""
        if self._run is None:
            return None
        # run.url is the current API; get_url() is deprecated in 0.18+
        url = getattr(self._run, "url", None)
        if url is None and hasattr(self._run, "get_url"):
            url = self._run.get_url()  # fallback for older SDK versions
        return url

    @property
    def is_active(self) -> bool:
        """True if a real W&B run is active (not a NoopRun or None)."""
        return self._run is not None and getattr(self._run, "id", None) is not None
