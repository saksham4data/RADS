# ─────────────────────────────────────────────────────────────
# EDA Configuration Loader
# ─────────────────────────────────────────────────────────────
"""
Loads eda_config.yaml and provides typed access to all
configuration sections. Includes sensible defaults so
notebooks work even without a config file present.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


# ── Defaults ────────────────────────────────────────────────

_DEFAULTS: Dict[str, Any] = {
    "project": {
        "name": "RADS",
        "eda_version": "1.0.0",
    },
    "wandb": {
        "project": "RADS",
        "entity": None,
        "job_type": "eda",
        "group": "exploratory-analysis",
        "tags": ["eda", "metadata-analysis"],
    },
    "data": {
        "metadata_path": "Datasets/processed/global_master_metadata.csv",
        "primary_label": "type",
        "secondary_labels": ["weather", "day_time", "scene_layout", "quality"],
        "split_columns": ["split", "split_in_distribution", "split_geo_aware"],
        "bbox_columns": ["center_x", "center_y", "x1", "y1", "x2", "y2"],
        "video_columns": ["fps", "duration", "no_frames", "height", "width"],
        "split_normalization_enabled": True,
        "unknown_split_policy": "warn",  # "warn", "drop", or "keep"
    },
    "reproducibility": {
        "seed": 42,
        "track_metadata_hash": True,
        "track_git_commit": True,
    },
    "visualization": {
        "video_sampling": {
            "enabled": True,
            "max_samples": 20,
            "seed": 42,
        },
        "figure_dpi": 300,
        "figure_formats": ["png", "svg"],
        "theme": "dark_professional",
    },
    "quality_score": {
        "weights": {
            "missing_values": 0.20,
            "duplicates": 0.15,
            "class_imbalance": 0.25,
            "invalid_bboxes": 0.20,
            "corruption": 0.20,
        },
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
    """Walk upward from this file to find the project root
    (identified by the presence of the ``Datasets`` directory)."""
    current = Path(__file__).resolve().parent
    for _ in range(10):
        if (current / "Datasets").is_dir():
            return current
        parent = current.parent
        if parent == current:
            break
        current = parent
    # Fallback: two levels up from eda/utils/config.py
    return Path(__file__).resolve().parent.parent.parent


# ── Config dataclass ────────────────────────────────────────

@dataclass
class EDAConfig:
    """Typed, validated access to the EDA configuration."""

    # Raw merged dictionary
    _raw: Dict[str, Any] = field(default_factory=dict, repr=False)

    # Resolved paths
    project_root: Path = field(default_factory=_find_project_root)
    eda_root: Path = field(init=False)
    config_path: Optional[Path] = None

    # ── Lifecycle ───────────────────────────────────────────

    def __post_init__(self) -> None:
        self.eda_root = self.project_root / "eda"
        if not self._raw:
            self._raw = _DEFAULTS.copy()

    # ── Convenience properties ──────────────────────────────

    # Project
    @property
    def project_name(self) -> str:
        return self._raw["project"]["name"]

    @property
    def eda_version(self) -> str:
        return self._raw["project"]["eda_version"]

    # W&B
    @property
    def wandb(self) -> Dict[str, Any]:
        return self._raw["wandb"]

    # Data
    @property
    def metadata_path(self) -> Path:
        raw = self._raw["data"]["metadata_path"]
        p = Path(raw)
        if not p.is_absolute():
            p = self.project_root / p
        return p

    @property
    def primary_label(self) -> str:
        return self._raw["data"]["primary_label"]

    @property
    def secondary_labels(self) -> List[str]:
        return self._raw["data"]["secondary_labels"]

    @property
    def split_columns(self) -> List[str]:
        return self._raw["data"]["split_columns"]

    @property
    def bbox_columns(self) -> List[str]:
        return self._raw["data"]["bbox_columns"]

    @property
    def video_columns(self) -> List[str]:
        return self._raw["data"]["video_columns"]

    @property
    def split_normalization_enabled(self) -> bool:
        return self._raw["data"].get("split_normalization_enabled", True)

    @property
    def unknown_split_policy(self) -> str:
        return self._raw["data"].get("unknown_split_policy", "warn")

    # Reproducibility
    @property
    def seed(self) -> int:
        return self._raw["reproducibility"]["seed"]

    @property
    def track_metadata_hash(self) -> bool:
        return self._raw["reproducibility"]["track_metadata_hash"]

    @property
    def track_git_commit(self) -> bool:
        return self._raw["reproducibility"]["track_git_commit"]

    # Visualization
    @property
    def visualization(self) -> Dict[str, Any]:
        return self._raw["visualization"]

    @property
    def figure_dpi(self) -> int:
        return self._raw["visualization"]["figure_dpi"]

    @property
    def figure_formats(self) -> List[str]:
        return self._raw["visualization"]["figure_formats"]

    @property
    def video_sampling(self) -> Dict[str, Any]:
        return self._raw["visualization"]["video_sampling"]

    # Quality score
    @property
    def quality_weights(self) -> Dict[str, float]:
        return self._raw["quality_score"]["weights"]

    # Output
    @property
    def output(self) -> Dict[str, Any]:
        return self._raw["output"]

    # ── Directory helpers ───────────────────────────────────

    @property
    def notebooks_dir(self) -> Path:
        return self.eda_root / "notebooks"

    @property
    def figures_dir(self) -> Path:
        return self.eda_root / "figures"

    @property
    def reports_dir(self) -> Path:
        return self.eda_root / "reports"

    @property
    def exports_dir(self) -> Path:
        return self.eda_root / "exports"

    @property
    def eda_logs_dir(self) -> Path:
        return self.eda_root / "logs"

    @property
    def root_eda_log_dir(self) -> Path:
        return self.project_root / "logs" / "eda"

    @property
    def run_history_path(self) -> Path:
        return self.eda_root / "run_history.json"

    # ── Serialisation ───────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        """Return the full merged configuration as a plain dict."""
        return self._raw.copy()


# ── Factory ─────────────────────────────────────────────────

def load_config(config_path: Optional[str | Path] = None) -> EDAConfig:
    """Load and validate the EDA configuration.

    Parameters
    ----------
    config_path : str or Path, optional
        Explicit path to a YAML config file.  When *None* the
        loader looks for ``eda/config/eda_config.yaml`` relative
        to the detected project root.

    Returns
    -------
    EDAConfig
        A fully-resolved configuration object.
    """
    project_root = _find_project_root()

    # Resolve config file
    if config_path is not None:
        cfg_file = Path(config_path)
    else:
        cfg_file = project_root / "eda" / "config" / "eda_config.yaml"

    # Load YAML (if it exists)
    user_cfg: Dict[str, Any] = {}
    resolved_path: Optional[Path] = None
    if cfg_file.is_file():
        with open(cfg_file, "r", encoding="utf-8") as fh:
            user_cfg = yaml.safe_load(fh) or {}
        resolved_path = cfg_file.resolve()

    # Merge defaults ← user overrides
    merged = _deep_merge(_DEFAULTS, user_cfg)

    return EDAConfig(
        _raw=merged,
        project_root=project_root,
        config_path=resolved_path,
    )
