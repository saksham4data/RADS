# ─────────────────────────────────────────────────────────────
# Read-Only Metadata Loader
# ─────────────────────────────────────────────────────────────
"""
Loads ``global_master_metadata.csv`` with proper dtype inference,
computes a SHA-256 hash for reproducibility, and provides
filtered views by dataset / media type.

This module NEVER mutates the source file.

Preprocessing pipeline (when normalization is enabled)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
1. Load raw CSV.
2. Run ``MetadataPreprocessor.preprocess()`` to normalize split
   names, validate the schema, and generate a quality report.
3. Cast cleaned columns to ``category`` dtype for memory
   efficiency.

The categorical cast happens *after* normalization so that all
split columns share a consistent set of category values,
preventing the ``TypeError: Categoricals can only be compared``
error that occurs when comparing columns with different
category sets.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from eda.utils.config import EDAConfig
from eda.utils.metadata_preprocessor import MetadataPreprocessor

logger = logging.getLogger(__name__)


class MetadataLoader:
    """Read-only accessor for the master metadata CSV.

    Usage::

        loader = MetadataLoader(config)
        df = loader.load()
        print(loader.metadata_hash)
        print(loader.quality_report)
        videos = loader.get_videos()
    """

    # Columns that should be treated as categorical
    # (applied *after* normalization to ensure clean categories)
    _CATEGORICAL_COLS = [
        "dataset_name", "source_type", "video_path_mode",
        "validation_status", "processing_status", "preprocessing_status",
        "type", "weather", "split", "split_in_distribution",
        "split_geo_aware", "region", "scene_layout", "day_time",
        "quality", "media_type", "metadata_source", "codec",
    ]

    def __init__(self, config: EDAConfig) -> None:
        self.config = config
        self._df: Optional[pd.DataFrame] = None
        self._metadata_hash: Optional[str] = None
        self._quality_report: Optional[Dict[str, Any]] = None
        self._file_path: Path = config.metadata_path
        self._preprocessor = MetadataPreprocessor()

    # ── Loading ─────────────────────────────────────────────

    def load(self) -> pd.DataFrame:
        """Load the metadata CSV, preprocess, and compute its hash.

        Returns a copy to prevent accidental mutation.
        """
        if self._df is not None:
            return self._df.copy()

        path = self._file_path
        if not path.is_file():
            raise FileNotFoundError(
                f"Metadata file not found: {path}\n"
                f"Expected at: {path.resolve()}"
            )

        # Compute SHA-256
        if self.config.track_metadata_hash:
            self._metadata_hash = self._compute_hash(path)

        # Load CSV
        self._df = pd.read_csv(path, low_memory=False)

        # ── Preprocessing pipeline ──
        # Run normalization BEFORE categorical casting so that
        # all split columns have consistent string values.
        if self.config.split_normalization_enabled:
            self._df, self._quality_report = self._preprocessor.preprocess(
                self._df, self.config
            )
            # Log quality warnings
            for warning in self._quality_report.get(
                "preprocessing_warnings", []
            ):
                logger.warning("Preprocessing: %s", warning)
        else:
            # Still generate a quality report without normalization
            _, self._quality_report = self._preprocessor.preprocess(
                self._df.copy(), self.config
            )

        # Optimise dtypes (after normalization)
        for col in self._CATEGORICAL_COLS:
            if col in self._df.columns:
                self._df[col] = self._df[col].astype("category")

        # Validate expected columns
        self._validate_columns()

        logger.info(
            "Loaded metadata: %d rows × %d columns from %s",
            len(self._df), len(self._df.columns), path.name,
        )

        return self._df.copy()

    # ── Hash ────────────────────────────────────────────────

    @staticmethod
    def _compute_hash(path: Path, algorithm: str = "sha256") -> str:
        """Compute the SHA-256 hash of a file."""
        h = hashlib.new(algorithm)
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return f"{algorithm}:{h.hexdigest()}"

    @property
    def metadata_hash(self) -> str:
        """SHA-256 hash of the metadata file. Call ``load()`` first."""
        if self._metadata_hash is None:
            if self._file_path.is_file():
                self._metadata_hash = self._compute_hash(self._file_path)
            else:
                return "unknown"
        return self._metadata_hash

    @property
    def dataset_version(self) -> str:
        """Extract dataset version from metadata or config."""
        if self._df is not None and "dataset_version" in self._df.columns:
            versions = self._df["dataset_version"].dropna().unique()
            if len(versions) == 1:
                return str(versions[0])
            return ",".join(str(v) for v in sorted(versions))
        return "unknown"

    @property
    def quality_report(self) -> Optional[Dict[str, Any]]:
        """Metadata quality report from preprocessing.

        Call ``load()`` first to populate this property.
        """
        return self._quality_report

    # ── Validation ──────────────────────────────────────────

    def _validate_columns(self) -> None:
        """Warn about expected but missing columns."""
        expected = set(
            self.config.bbox_columns
            + self.config.video_columns
            + self.config.split_columns
            + [self.config.primary_label]
            + self.config.secondary_labels
        )
        present = set(self._df.columns)
        missing = expected - present
        if missing:
            logger.warning(
                "Missing expected columns (will be skipped): %s",
                sorted(missing),
            )

    # ── Column presence checks ──────────────────────────────

    def has_bbox(self) -> bool:
        """True if all bounding box columns are present."""
        if self._df is None:
            return False
        return all(c in self._df.columns for c in self.config.bbox_columns)

    def has_video_stats(self) -> bool:
        """True if all video statistics columns are present."""
        if self._df is None:
            return False
        return all(c in self._df.columns for c in self.config.video_columns)

    def has_splits(self) -> bool:
        """True if at least one split column is present."""
        if self._df is None:
            return False
        return any(c in self._df.columns for c in self.config.split_columns)

    def has_column(self, col: str) -> bool:
        """True if a specific column exists."""
        if self._df is None:
            return False
        return col in self._df.columns

    # ── Filtered views ──────────────────────────────────────

    def get_videos(self) -> pd.DataFrame:
        """Return rows identified as video media."""
        df = self.load()
        if "media_type" in df.columns:
            mask = df["media_type"].astype(str).str.lower().isin(
                ["video", ""]
            ) | df["media_type"].isna()
        else:
            # Heuristic: rows with non-null fps or duration are videos
            mask = df["fps"].notna() | df["duration"].notna()
        return df[mask].copy()

    def get_images(self) -> pd.DataFrame:
        """Return rows identified as image media."""
        df = self.load()
        if "media_type" in df.columns:
            mask = df["media_type"].astype(str).str.lower() == "image"
        else:
            # Heuristic: rows without fps and duration
            mask = df["fps"].isna() & df["duration"].isna()
        return df[mask].copy()

    def get_dataset(self, name: str) -> pd.DataFrame:
        """Return rows for a specific dataset."""
        df = self.load()
        return df[df["dataset_name"] == name].copy()

    def get_dataset_names(self) -> List[str]:
        """Return a sorted list of unique dataset names."""
        df = self.load()
        return sorted(df["dataset_name"].dropna().unique().tolist())

    # ── Info ────────────────────────────────────────────────

    @property
    def shape(self) -> tuple:
        """(rows, columns) of the loaded DataFrame."""
        if self._df is None:
            return (0, 0)
        return self._df.shape

    @property
    def columns(self) -> List[str]:
        """Column names of the loaded DataFrame."""
        if self._df is None:
            return []
        return self._df.columns.tolist()

    @property
    def memory_usage_mb(self) -> float:
        """Approximate memory usage in MB."""
        if self._df is None:
            return 0.0
        return self._df.memory_usage(deep=True).sum() / (1024 ** 2)
