# ─────────────────────────────────────────────────────────────
# Metadata Preprocessor — Normalization & Validation
# ─────────────────────────────────────────────────────────────
"""
Centralised metadata preprocessing: split-name normalization,
categorical cleaning, schema validation, and quality reporting.

This module runs **before** any analyzer and guarantees that
downstream code receives consistently-typed, consistently-named
data.  It never mutates the source CSV — it returns a cleaned
*copy* of the DataFrame plus a structured quality report.

Design decisions
~~~~~~~~~~~~~~~~
* ``val`` is the canonical validation split name because it is
  the dominant convention in ML frameworks (PyTorch, HuggingFace)
  and is shorter, reducing visual noise in reports.
* Unknown split values are **warned about** but never silently
  dropped.  This ensures future datasets with novel naming
  produce visible diagnostics instead of silent data loss.
* All normalization is idempotent — running it twice produces
  the same result.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Split alias map ─────────────────────────────────────────
# Applied *after* strip + lowercase, so keys are all lowercase.
# The map is intentionally exhaustive to handle mixed-case
# inputs that somehow bypass the lowercase step.

SPLIT_ALIASES: Dict[str, str] = {
    "validation": "val",
    "valid": "val",
    "dev": "val",
    "development": "val",
    "training": "train",
    "testing": "test",
}

# Canonical split names — anything that isn't in this set (after
# alias resolution) triggers a warning.
CANONICAL_SPLITS: Set[str] = {"train", "val", "test"}


# ── Preprocessor ────────────────────────────────────────────

class MetadataPreprocessor:
    """Centralised metadata cleaning and validation.

    Usage::

        from eda.utils.config import EDAConfig
        preprocessor = MetadataPreprocessor()
        df_clean, report = preprocessor.preprocess(df, config)
    """

    def __init__(
        self,
        *,
        split_aliases: Optional[Dict[str, str]] = None,
        canonical_splits: Optional[Set[str]] = None,
    ) -> None:
        self.split_aliases = split_aliases or SPLIT_ALIASES
        self.canonical_splits = canonical_splits or CANONICAL_SPLITS

    # ── Public API ──────────────────────────────────────────

    def preprocess(
        self,
        df: pd.DataFrame,
        config: Any,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Run all preprocessing steps and return (cleaned_df, report).

        Parameters
        ----------
        df : pd.DataFrame
            Raw metadata DataFrame (will NOT be mutated).
        config : EDAConfig
            Configuration object providing column lists.

        Returns
        -------
        tuple[pd.DataFrame, dict]
            A cleaned copy and a structured quality report.
        """
        df = df.copy()
        warnings_list: List[str] = []

        # 1. Normalize split columns
        split_cols = [c for c in getattr(config, "split_columns", [])
                      if c in df.columns]
        if split_cols:
            df, split_warnings = self.normalize_split_columns(df, split_cols)
            warnings_list.extend(split_warnings)

        # 2. Validate schema
        schema_report = self.validate_schema(df, config)
        warnings_list.extend(schema_report.get("warnings", []))

        # 3. Generate quality report
        quality_report = self.generate_quality_report(df, config)
        quality_report["preprocessing_warnings"] = warnings_list

        return df, quality_report

    # ── Split normalization ─────────────────────────────────

    def normalize_split_columns(
        self,
        df: pd.DataFrame,
        split_cols: List[str],
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Normalize split column values in-place on a copy.

        Steps:
        1. Convert to plain string dtype (strip categorical metadata).
        2. Strip whitespace.
        3. Lowercase.
        4. Apply alias map (e.g. ``validation`` → ``val``).
        5. Warn about unrecognised values.

        Returns
        -------
        tuple[pd.DataFrame, list[str]]
            The modified DataFrame and a list of warning messages.
        """
        warnings_list: List[str] = []

        for col in split_cols:
            if col not in df.columns:
                continue

            # Step 1: Convert to plain string, preserving NaN
            original_dtype = df[col].dtype
            series = df[col].copy()

            # Remove categorical wrapper if present
            if hasattr(series, "cat"):
                series = series.astype(object)

            # Step 2–3: Strip + lowercase (NaN-safe)
            mask_notna = series.notna()
            series.loc[mask_notna] = (
                series.loc[mask_notna]
                .astype(str)
                .str.strip()
                .str.lower()
            )

            # Step 4: Apply alias map
            series = series.map(
                lambda v: self.split_aliases.get(v, v) if pd.notna(v) else v
            )

            # Step 5: Detect unrecognised values
            unique_vals = set(series.dropna().unique())
            unknown = unique_vals - self.canonical_splits
            if unknown:
                msg = (
                    f"Column '{col}': unrecognised split values "
                    f"after normalization: {sorted(unknown)}. "
                    f"Expected one of {sorted(self.canonical_splits)}."
                )
                warnings_list.append(msg)
                logger.warning(msg)

            # Log normalization summary
            new_unique = sorted(series.dropna().unique())
            logger.info(
                "Normalized split column '%s': %s → %s",
                col, sorted(df[col].dropna().unique()), new_unique,
            )

            df[col] = series

        return df, warnings_list

    # ── Schema validation ───────────────────────────────────

    def validate_schema(
        self,
        df: pd.DataFrame,
        config: Any,
    ) -> Dict[str, Any]:
        """Validate the DataFrame schema against expected columns.

        Checks:
        - Required columns missing
        - Duplicate column names
        - Completely empty columns
        - Split columns present and non-empty
        - Label columns present
        - Dataset column present

        Returns a report dict with ``valid``, ``warnings``, and
        ``details`` keys.
        """
        warnings_list: List[str] = []
        details: Dict[str, Any] = {}

        # Duplicate column names
        col_counts = pd.Series(df.columns).value_counts()
        duplicates = col_counts[col_counts > 1]
        if not duplicates.empty:
            dup_names = duplicates.index.tolist()
            warnings_list.append(
                f"Duplicate column names detected: {dup_names}"
            )
            details["duplicate_columns"] = dup_names

        # Expected columns
        expected_groups: Dict[str, List[str]] = {
            "split_columns": getattr(config, "split_columns", []),
            "bbox_columns": getattr(config, "bbox_columns", []),
            "video_columns": getattr(config, "video_columns", []),
            "label_columns": (
                [getattr(config, "primary_label", "type")]
                + getattr(config, "secondary_labels", [])
            ),
        }

        missing_by_group: Dict[str, List[str]] = {}
        for group, cols in expected_groups.items():
            missing = [c for c in cols if c not in df.columns]
            if missing:
                missing_by_group[group] = missing
                warnings_list.append(
                    f"Missing {group}: {missing}"
                )
        details["missing_columns"] = missing_by_group

        # Completely empty columns
        empty_cols = [
            c for c in df.columns if df[c].isna().all()
        ]
        if empty_cols:
            warnings_list.append(
                f"Completely empty columns: {empty_cols}"
            )
            details["empty_columns"] = empty_cols

        # Dataset column
        if "dataset_name" not in df.columns:
            warnings_list.append("Missing 'dataset_name' column")
        elif df["dataset_name"].isna().any():
            n = int(df["dataset_name"].isna().sum())
            warnings_list.append(
                f"{n} rows missing 'dataset_name'"
            )

        return {
            "valid": len(warnings_list) == 0,
            "warnings": warnings_list,
            "details": details,
        }

    # ── Quality report ──────────────────────────────────────

    def generate_quality_report(
        self,
        df: pd.DataFrame,
        config: Any,
    ) -> Dict[str, Any]:
        """Generate a comprehensive metadata quality report.

        Covers:
        - Missing values per column
        - Unexpected split names
        - Invalid / missing labels
        - Duplicate rows, hashes, paths
        - Rows missing dataset / label / split
        - Inconsistent split naming across columns
        - Unused categories
        - Potential leakage indicators
        """
        report: Dict[str, Any] = {}

        # ── Missing values ──
        total = len(df)
        missing_summary = []
        for col in df.columns:
            n_missing = int(df[col].isna().sum())
            if n_missing > 0:
                missing_summary.append({
                    "column": col,
                    "missing_count": n_missing,
                    "missing_percent": round(n_missing / total * 100, 2)
                    if total > 0 else 0,
                })
        report["missing_values"] = sorted(
            missing_summary, key=lambda x: x["missing_count"], reverse=True
        )

        # ── Split analysis ──
        split_cols = [
            c for c in getattr(config, "split_columns", [])
            if c in df.columns
        ]
        split_report: Dict[str, Any] = {}
        for col in split_cols:
            vals = df[col].dropna().unique().tolist()
            canonical = [v for v in vals if v in self.canonical_splits]
            unexpected = [v for v in vals if v not in self.canonical_splits]
            n_missing = int(df[col].isna().sum())
            split_report[col] = {
                "unique_values": vals,
                "canonical_values": canonical,
                "unexpected_values": unexpected,
                "missing_count": n_missing,
                "missing_percent": round(n_missing / total * 100, 2)
                if total > 0 else 0,
            }
        report["split_columns"] = split_report

        # ── Label columns ──
        primary_label = getattr(config, "primary_label", None)
        if primary_label and primary_label in df.columns:
            n_missing = int(df[primary_label].isna().sum())
            report["primary_label"] = {
                "column": primary_label,
                "missing_count": n_missing,
                "unique_values": int(df[primary_label].nunique(dropna=True)),
            }
        else:
            report["primary_label"] = {
                "column": primary_label,
                "available": False,
            }

        # ── Duplicate detection ──
        dup_report: Dict[str, Any] = {
            "exact_duplicates": int(df.duplicated().sum()),
        }

        if "file_hash" in df.columns:
            hash_counts = df["file_hash"].dropna().value_counts()
            dup_hashes = hash_counts[hash_counts > 1]
            dup_report["duplicate_hashes"] = {
                "groups": len(dup_hashes),
                "total_rows": int(dup_hashes.sum()) if len(dup_hashes) > 0 else 0,
            }

        if "original_path" in df.columns:
            path_counts = df["original_path"].dropna().value_counts()
            dup_paths = path_counts[path_counts > 1]
            dup_report["duplicate_paths"] = {
                "groups": len(dup_paths),
                "total_rows": int(dup_paths.sum()) if len(dup_paths) > 0 else 0,
            }

        report["duplicates"] = dup_report

        # ── Rows missing key fields ──
        key_fields: Dict[str, int] = {}
        for col_name in ["dataset_name", primary_label] + split_cols:
            if col_name and col_name in df.columns:
                n = int(df[col_name].isna().sum())
                if n > 0:
                    key_fields[col_name] = n
        report["rows_missing_key_fields"] = key_fields

        # ── Inconsistent split naming ──
        # Check if the same row has different normalized split values
        # across split columns (which is expected but worth flagging)
        if len(split_cols) >= 2:
            inconsistencies: Dict[str, Any] = {}
            for i, c1 in enumerate(split_cols):
                for c2 in split_cols[i + 1:]:
                    valid_mask = df[c1].notna() & df[c2].notna()
                    if valid_mask.any():
                        s1 = df.loc[valid_mask, c1].astype(str)
                        s2 = df.loc[valid_mask, c2].astype(str)
                        disagree = (s1 != s2).sum()
                        inconsistencies[f"{c1}_vs_{c2}"] = {
                            "comparable_rows": int(valid_mask.sum()),
                            "disagreements": int(disagree),
                            "agreement_pct": round(
                                (1 - disagree / valid_mask.sum()) * 100, 2
                            ) if valid_mask.sum() > 0 else 0,
                        }
            report["split_consistency"] = inconsistencies

        # ── Unused categories ──
        unused: Dict[str, List[str]] = {}
        for col in df.columns:
            if hasattr(df[col], "cat"):
                used = set(df[col].dropna().unique())
                all_cats = set(df[col].cat.categories)
                diff = all_cats - used
                if diff:
                    unused[col] = sorted(diff)
        if unused:
            report["unused_categories"] = unused

        # ── Potential leakage ──
        leakage: Dict[str, Any] = {}
        if "file_hash" in df.columns and split_cols:
            for col in split_cols:
                if col not in df.columns:
                    continue
                valid = df[[col, "file_hash"]].dropna()
                if valid.empty:
                    continue
                hash_splits = valid.groupby("file_hash")[col].nunique()
                multi = hash_splits[hash_splits > 1]
                if len(multi) > 0:
                    leakage[f"{col}_hash_leakage"] = {
                        "hashes_in_multiple_splits": len(multi),
                        "affected_rows": int(
                            valid[valid["file_hash"].isin(multi.index)].shape[0]
                        ),
                    }
        report["potential_leakage"] = leakage

        return report
