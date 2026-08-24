# ─────────────────────────────────────────────────────────────
# Training Configuration Loader
# ─────────────────────────────────────────────────────────────
"""
Loads ``training_config_v1.yaml`` / ``training_config_v2.yaml`` and provides typed access to all
configuration sections.  Includes sensible defaults so scripts
and notebooks work even without a config file present.

Pattern: mirrors ``eda.utils.config.EDAConfig`` exactly.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml


# ── Defaults ────────────────────────────────────────────────

_DEFAULTS: Dict[str, Any] = {
    "project": {
        "name": "RADS",
        "training_version": "1.0.0",
    },
    "experiment": {
        "id": None,
        "name": None,
    },
    "data": {
        "metadata_path": "Datasets/processed/global_master_metadata.csv",
        "dataset_name": "tudat",
        "raw_base_dir": "Datasets/raw",
        "label_column": "type",
        "label_mode": "multiclass",
        "split_column": "split_in_distribution",
        "class_mapping": None,
        "split_ratios": {
            "train": 0.70,
            "val": 0.15,
            "test": 0.15,
        },
        "frame_sampling": {
            "strategy": "uniform",
            "frames_per_video": 5,
        },
        "image_size": [224, 224],
        "augmentation": {
            "enabled": False,
            "preset": "none",
            "horizontal_flip": {"enabled": True, "p": 0.5},
            "rotation": {"enabled": True, "degrees": 5},
            "brightness_contrast": {"enabled": True, "brightness": 0.15, "contrast": 0.15},
            "saturation": {"enabled": False, "value": 0.0},
            "hue": {"enabled": False, "value": 0.0},
            "gaussian_blur": {"enabled": False, "kernel_size": 3, "sigma": [0.1, 0.5]},
        },
    },
    "model": {
        "name": "resnet18",
        "pretrained": True,
        "num_classes": 3,
        "freeze_backbone": False,
    },
    "training": {
        "epochs": 1,
        "batch_size": 16,
        "num_workers": 2,
        "pin_memory": True,
        "mixed_precision": False,
    },
    "optimizer": {
        "name": "adam",
        "lr": 0.001,
        "weight_decay": 0.0001,
    },
    "scheduler": {
        "name": "step_lr",
        "step_size": 5,
        "gamma": 0.1,
    },
    "checkpoint": {
        "save_best": True,
        "save_last": True,
        "monitor_metric": "val_loss",
        "mode": "min",
    },
    "early_stopping": {
        "enabled": False,
        "patience": 5,
        "min_delta": 0.001,
    },
    "wandb": {
        "project": "RADS",
        "entity": None,
        "job_type": "training",
        "group": "baseline-classification",
        "tags": ["pipeline-4", "baseline", "resnet18", "tudat"],
    },
    "reproducibility": {
        "seed": 42,
    },
    "output": {
        "timestamped": True,
        "maintain_latest": True,
        "track_run_history": True,
        "generate_manifest": True,
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base* (non-destructive)."""
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


# ── Path helpers ────────────────────────────────────────────

def _find_project_root() -> Path:
    """Walk upward from this file to find the project root.

    Uses multiple markers (``Datasets/raw`` AND ``eda/``) to
    avoid false positives on case-insensitive filesystems where
    ``training/datasets/`` could match ``Datasets/``.
    """
    current = Path(__file__).resolve().parent
    for _ in range(10):
        if (current / "Datasets" / "raw").is_dir() and (current / "eda").is_dir():
            return current
        parent = current.parent
        if parent == current:
            break
        current = parent
    # Fallback: three levels up from training/configs/config.py
    return Path(__file__).resolve().parent.parent.parent


# ── Config dataclass ────────────────────────────────────────

@dataclass
class TrainingConfig:
    """Typed, validated access to the training configuration.

    Mirrors ``eda.utils.config.EDAConfig`` for consistency.
    """

    # Raw merged dictionary
    _raw: Dict[str, Any] = field(default_factory=dict, repr=False)

    # Resolved paths
    project_root: Path = field(default_factory=_find_project_root)
    training_root: Path = field(init=False)
    config_path: Optional[Path] = None

    # ── Lifecycle ───────────────────────────────────────────

    def __post_init__(self) -> None:
        self.training_root = self.project_root / "training"
        if not self._raw:
            self._raw = _DEFAULTS.copy()

    # ── Project ─────────────────────────────────────────────

    @property
    def project_name(self) -> str:
        return self._raw["project"]["name"]

    @property
    def training_version(self) -> str:
        return self._raw["project"]["training_version"]

    # ── Experiment ──────────────────────────────────────────

    @property
    def experiment_id(self) -> str:
        exp_id = self._raw.get("experiment", {}).get("id")
        if exp_id:
            return exp_id
        return f"{self.model_name}_{self.dataset_name}"

    @property
    def experiment_name(self) -> str:
        exp_name = self._raw.get("experiment", {}).get("name")
        if exp_name:
            return exp_name
        return self.experiment_id

    # ── Data ────────────────────────────────────────────────

    @property
    def metadata_path(self) -> Path:
        raw = self._raw["data"]["metadata_path"]
        p = Path(raw)
        if not p.is_absolute():
            p = self.project_root / p
        return p

    @property
    def dataset_name(self) -> str:
        return self._raw["data"]["dataset_name"]

    @property
    def raw_base_dir(self) -> Path:
        raw = self._raw["data"]["raw_base_dir"]
        p = Path(raw)
        if not p.is_absolute():
            p = self.project_root / p
        return p

    @property
    def label_column(self) -> str:
        return self._raw["data"]["label_column"]

    @property
    def label_mode(self) -> str:
        mode = str(self._raw["data"].get("label_mode", "multiclass")).lower()
        if mode not in {"multiclass", "binary"}:
            raise ValueError(
                f"Unsupported label_mode '{mode}'. "
                "Expected one of: ['multiclass', 'binary']"
            )
        return mode

    @property
    def split_column(self) -> str:
        return self._raw["data"]["split_column"]

    @property
    def class_mapping(self) -> Optional[Dict[str, int]]:
        return self._raw["data"]["class_mapping"]

    @property
    def default_class_mapping(self) -> Dict[str, int]:
        if self.label_mode == "binary":
            return {"accident": 0, "non-accident": 1}
        return {
            "accident": 0,
            "challenging": 1,
            "non-accident": 2,
        }

    @property
    def resolved_class_names(self) -> List[str]:
        mapping = self.class_mapping or self.default_class_mapping
        return [
            label for label, _ in sorted(mapping.items(), key=lambda item: item[1])
        ]

    @property
    def split_ratios(self) -> Dict[str, float]:
        return self._raw["data"]["split_ratios"]

    @property
    def frame_sampling(self) -> Dict[str, Any]:
        return self._raw["data"]["frame_sampling"]

    @property
    def frames_per_video(self) -> int:
        return self._raw["data"]["frame_sampling"]["frames_per_video"]

    @property
    def sampling_strategy(self) -> str:
        return self._raw["data"]["frame_sampling"]["strategy"]

    @property
    def image_size(self) -> Tuple[int, int]:
        size = self._raw["data"]["image_size"]
        return (size[0], size[1])

    @property
    def augmentation_config(self) -> Dict[str, Any]:
        return self._raw["data"].get("augmentation", {})

    @property
    def augmentation_enabled(self) -> bool:
        return self.augmentation_config.get("enabled", False)

    # ── Model ───────────────────────────────────────────────

    @property
    def model_name(self) -> str:
        return self._raw["model"]["name"]

    @property
    def pretrained(self) -> bool:
        return self._raw["model"]["pretrained"]

    @property
    def num_classes(self) -> int:
        mapping = self.class_mapping
        if mapping:
            return len(set(mapping.values()))
        return len(self.default_class_mapping)

    @property
    def freeze_backbone(self) -> bool:
        return self._raw["model"]["freeze_backbone"]

    # ── Training ────────────────────────────────────────────

    @property
    def epochs(self) -> int:
        return self._raw["training"]["epochs"]

    @property
    def batch_size(self) -> int:
        return self._raw["training"]["batch_size"]

    @property
    def num_workers(self) -> int:
        return self._raw["training"]["num_workers"]

    @property
    def pin_memory(self) -> bool:
        return self._raw["training"]["pin_memory"]

    @property
    def mixed_precision(self) -> bool:
        return self._raw["training"]["mixed_precision"]

    # ── Optimizer ───────────────────────────────────────────

    @property
    def optimizer_name(self) -> str:
        return self._raw["optimizer"]["name"]

    @property
    def learning_rate(self) -> float:
        return self._raw["optimizer"]["lr"]

    @property
    def weight_decay(self) -> float:
        return self._raw["optimizer"]["weight_decay"]

    @property
    def optimizer_config(self) -> Dict[str, Any]:
        return self._raw["optimizer"]

    # ── Scheduler ───────────────────────────────────────────

    @property
    def scheduler_name(self) -> str:
        return self._raw["scheduler"]["name"]

    @property
    def scheduler_config(self) -> Dict[str, Any]:
        return self._raw["scheduler"]

    # ── Checkpoint ──────────────────────────────────────────

    @property
    def checkpoint_config(self) -> Dict[str, Any]:
        return self._raw["checkpoint"]

    @property
    def save_best(self) -> bool:
        return self._raw["checkpoint"]["save_best"]

    @property
    def save_last(self) -> bool:
        return self._raw["checkpoint"]["save_last"]

    @property
    def monitor_metric(self) -> str:
        return self._raw["checkpoint"]["monitor_metric"]

    @property
    def monitor_mode(self) -> str:
        return self._raw["checkpoint"]["mode"]

    # ── Early Stopping ──────────────────────────────────────

    @property
    def early_stopping_config(self) -> Dict[str, Any]:
        return self._raw["early_stopping"]

    @property
    def early_stopping_enabled(self) -> bool:
        return self._raw["early_stopping"]["enabled"]

    # ── W&B ─────────────────────────────────────────────────

    @property
    def wandb(self) -> Dict[str, Any]:
        return self._raw["wandb"]

    # ── Reproducibility ─────────────────────────────────────

    @property
    def seed(self) -> int:
        return self._raw["reproducibility"]["seed"]

    # ── Output ──────────────────────────────────────────────

    @property
    def output(self) -> Dict[str, Any]:
        return self._raw["output"]

    # ── Directory helpers ───────────────────────────────────

    @property
    def notebooks_dir(self) -> Path:
        return self.training_root / "notebooks"

    @property
    def outputs_dir(self) -> Path:
        return self.training_root / "outputs"

    @property
    def checkpoints_dir(self) -> Path:
        return self.outputs_dir / "checkpoints"

    @property
    def metrics_dir(self) -> Path:
        return self.outputs_dir / "metrics"

    @property
    def predictions_dir(self) -> Path:
        return self.outputs_dir / "predictions"

    @property
    def training_logs_dir(self) -> Path:
        return self.outputs_dir / "logs"

    @property
    def root_training_log_dir(self) -> Path:
        return self.project_root / "logs" / "training"

    @property
    def run_history_path(self) -> Path:
        return self.training_root / "run_history.json"

    # ── Serialisation ───────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        """Return the full merged configuration as a plain dict."""
        return self._raw.copy()


# ── Factory ─────────────────────────────────────────────────

def load_training_config(
    config_path: Optional[str | Path] = None,
) -> TrainingConfig:
    """Load and validate the training configuration.

    Parameters
    ----------
    config_path : str or Path, optional
        Explicit path to a YAML config file.  When *None* the
        loader looks for ``training/config/training_config_v1.yaml``
        (or ``training_config_v2.yaml``) relative to the detected
        project root.

    Returns
    -------
    TrainingConfig
        A fully-resolved configuration object.
    """
    project_root = _find_project_root()

    # Resolve config file
    if config_path is not None:
        cfg_file = Path(config_path)
    else:
        # Default priority: v2 → v1 (backward compatibility) → legacy
        default_candidates = [
            project_root / "training" / "config" / "training_config_v2.yaml",
            project_root / "training" / "config" / "training_config_v1.yaml",
            project_root / "training" / "config" / "training_config.yaml",
        ]
        cfg_file = next((p for p in default_candidates if p.is_file()), default_candidates[0])

    # Load YAML (if it exists)
    user_cfg: Dict[str, Any] = {}
    resolved_path: Optional[Path] = None
    if cfg_file.is_file():
        with open(cfg_file, "r", encoding="utf-8") as fh:
            user_cfg = yaml.safe_load(fh) or {}
        resolved_path = cfg_file.resolve()

    # Merge defaults ← user overrides
    merged = _deep_merge(_DEFAULTS, user_cfg)

    return TrainingConfig(
        _raw=merged,
        project_root=project_root,
        config_path=resolved_path,
    )
