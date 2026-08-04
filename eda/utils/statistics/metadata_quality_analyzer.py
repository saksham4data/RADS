# ─────────────────────────────────────────────────────────────
# Metadata Quality Analyzer
# ─────────────────────────────────────────────────────────────
"""
Generates a comprehensive metadata quality report covering
missing values, unexpected split names, invalid labels,
duplicates, leakage indicators, and consistency checks.

This analyzer consolidates quality diagnostics that were
previously scattered across multiple analyzers and the
preprocessor, providing a single entry point for metadata
quality assessment.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Set

import numpy as np
import pandas as pd

from eda.utils.statistics.base_analyzer import BaseAnalyzer
from eda.utils.metadata_preprocessor import CANONICAL_SPLITS

logger = logging.getLogger(__name__)


class MetadataQualityAnalyzer(BaseAnalyzer):
    """Comprehensive metadata quality analysis.

    Produces a structured report covering:
    - Missing values per column
    - Unexpected split names
    - Invalid / missing labels
    - Duplicate rows, hashes, file paths
    - Rows missing dataset name, label, or split
    - Rows belonging to multiple datasets (via hash)
    - Inconsistent split naming across columns
    - Unused categories
    - Potential leakage indicators
    """

    def _analyze(self) -> Dict[str, Any]:
        results: Dict[str, Any] = {"available": True}

        try:
            results["missing_values"] = self._analyze_missing_values()
        except Exception as exc:
            self.add_warning(f"Missing values analysis failed: {exc}")
            results["missing_values"] = []

        try:
            results["split_quality"] = self._analyze_split_quality()
        except Exception as exc:
            self.add_warning(f"Split quality analysis failed: {exc}")
            results["split_quality"] = {}

        try:
            results["label_quality"] = self._analyze_label_quality()
        except Exception as exc:
            self.add_warning(f"Label quality analysis failed: {exc}")
            results["label_quality"] = {}

        try:
            results["duplicates"] = self._analyze_duplicates()
        except Exception as exc:
            self.add_warning(f"Duplicate analysis failed: {exc}")
            results["duplicates"] = {}

        try:
            results["key_field_coverage"] = self._analyze_key_field_coverage()
        except Exception as exc:
            self.add_warning(f"Key field coverage failed: {exc}")
            results["key_field_coverage"] = {}

        try:
            results["multi_dataset_rows"] = self._analyze_multi_dataset_rows()
        except Exception as exc:
            self.add_warning(f"Multi-dataset analysis failed: {exc}")
            results["multi_dataset_rows"] = {}

        try:
            results["unused_categories"] = self._analyze_unused_categories()
        except Exception as exc:
            self.add_warning(f"Unused categories analysis failed: {exc}")
            results["unused_categories"] = {}

        try:
            results["leakage_indicators"] = self._analyze_leakage()
        except Exception as exc:
            self.add_warning(f"Leakage analysis failed: {exc}")
            results["leakage_indicators"] = {}

        # Overall quality summary
        results["summary"] = self._build_summary(results)

        self._results = results
        return results

    # ── Missing values ──────────────────────────────────────

    def _analyze_missing_values(self) -> List[Dict[str, Any]]:
        """Per-column missing value counts and percentages."""
        total = len(self.df)
        missing_info = []
        for col in self.df.columns:
            n_missing = int(self.df[col].isna().sum())
            if n_missing > 0:
                missing_info.append({
                    "column": col,
                    "missing_count": n_missing,
                    "missing_percent": round(
                        n_missing / total * 100, 2
                    ) if total > 0 else 0,
                })
        return sorted(
            missing_info, key=lambda x: x["missing_count"], reverse=True
        )

    # ── Split quality ───────────────────────────────────────

    def _analyze_split_quality(self) -> Dict[str, Any]:
        """Analyze split column quality and consistency."""
        split_cols = self.available_columns(self.config.split_columns)
        if not split_cols:
            return {"available": False, "reason": "No split columns"}

        report: Dict[str, Any] = {"columns": {}}

        for col in split_cols:
            # Convert to string for safe analysis
            values = self.df[col].astype(str).where(
                self.df[col].notna(), other=np.nan
            )
            unique_vals = set(values.dropna().unique())
            expected = set(CANONICAL_SPLITS)
            unexpected = unique_vals - expected

            col_report: Dict[str, Any] = {
                "unique_values": sorted(unique_vals),
                "missing_count": int(self.df[col].isna().sum()),
                "missing_percent": round(
                    self.df[col].isna().mean() * 100, 2
                ),
            }

            if unexpected:
                col_report["unexpected_values"] = sorted(unexpected)
                self.add_warning(
                    f"Split column '{col}' has unexpected values: "
                    f"{sorted(unexpected)}"
                )

            # Value distribution
            counts = self.df[col].value_counts(dropna=False)
            col_report["distribution"] = {
                str(k): int(v) for k, v in counts.items()
            }

            report["columns"][col] = col_report

        # Cross-column consistency
        if len(split_cols) >= 2:
            consistency: Dict[str, Any] = {}
            for i, c1 in enumerate(split_cols):
                for c2 in split_cols[i + 1:]:
                    both_valid = self.df[c1].notna() & self.df[c2].notna()
                    n_both = int(both_valid.sum())
                    if n_both > 0:
                        s1 = self.df.loc[both_valid, c1].astype(str)
                        s2 = self.df.loc[both_valid, c2].astype(str)
                        n_agree = int((s1.values == s2.values).sum())
                        consistency[f"{c1}_vs_{c2}"] = {
                            "comparable_rows": n_both,
                            "agreement_count": n_agree,
                            "agreement_pct": round(
                                n_agree / n_both * 100, 2
                            ),
                        }
                    else:
                        consistency[f"{c1}_vs_{c2}"] = {
                            "comparable_rows": 0,
                            "note": "No overlapping non-null rows",
                        }
            report["cross_column_consistency"] = consistency

        return report

    # ── Label quality ───────────────────────────────────────

    def _analyze_label_quality(self) -> Dict[str, Any]:
        """Analyze primary and secondary label quality."""
        report: Dict[str, Any] = {}

        # Primary label
        primary = self.config.primary_label
        if primary in self.df.columns:
            n_missing = int(self.df[primary].isna().sum())
            n_unique = int(self.df[primary].nunique(dropna=True))
            report["primary"] = {
                "column": primary,
                "missing_count": n_missing,
                "missing_percent": round(
                    n_missing / len(self.df) * 100, 2
                ) if len(self.df) > 0 else 0,
                "unique_values": n_unique,
                "values": sorted(
                    self.df[primary].dropna().unique().astype(str).tolist()
                ),
            }
        else:
            report["primary"] = {
                "column": primary,
                "available": False,
            }

        # Secondary labels
        secondary: Dict[str, Any] = {}
        for col in self.config.secondary_labels:
            if col in self.df.columns:
                secondary[col] = {
                    "missing_count": int(self.df[col].isna().sum()),
                    "unique_values": int(
                        self.df[col].nunique(dropna=True)
                    ),
                }
            else:
                secondary[col] = {"available": False}
        report["secondary"] = secondary

        return report

    # ── Duplicates ──────────────────────────────────────────

    def _analyze_duplicates(self) -> Dict[str, Any]:
        """Comprehensive duplicate detection."""
        report: Dict[str, Any] = {
            "total_rows": len(self.df),
            "exact_row_duplicates": int(self.df.duplicated().sum()),
        }

        # Hash duplicates
        if "file_hash" in self.df.columns:
            hash_counts = self.df["file_hash"].dropna().value_counts()
            dup_hashes = hash_counts[hash_counts > 1]
            report["duplicate_hashes"] = {
                "groups": len(dup_hashes),
                "affected_rows": int(dup_hashes.sum()) - len(dup_hashes)
                if len(dup_hashes) > 0 else 0,
            }
        else:
            report["duplicate_hashes"] = {"available": False}

        # Path duplicates
        if "original_path" in self.df.columns:
            path_counts = self.df["original_path"].dropna().value_counts()
            dup_paths = path_counts[path_counts > 1]
            report["duplicate_paths"] = {
                "groups": len(dup_paths),
                "affected_rows": int(dup_paths.sum()) - len(dup_paths)
                if len(dup_paths) > 0 else 0,
            }
        else:
            report["duplicate_paths"] = {"available": False}

        # Flagged duplicates
        if "is_duplicate" in self.df.columns:
            dup_col = self.df["is_duplicate"]
            if dup_col.dtype == bool:
                report["flagged_duplicates"] = int(dup_col.sum())
            else:
                report["flagged_duplicates"] = int(
                    (dup_col == True).sum()  # noqa: E712
                )
        else:
            report["flagged_duplicates"] = {"available": False}

        return report

    # ── Key field coverage ──────────────────────────────────

    def _analyze_key_field_coverage(self) -> Dict[str, Any]:
        """Check coverage of critical columns."""
        total = len(self.df)
        report: Dict[str, Any] = {"total_rows": total}

        key_columns = {
            "dataset_name": "dataset_name",
            "primary_label": self.config.primary_label,
        }
        # Add split columns
        for col in self.config.split_columns:
            key_columns[f"split_{col}"] = col

        for label, col in key_columns.items():
            if col in self.df.columns:
                n_missing = int(self.df[col].isna().sum())
                report[label] = {
                    "column": col,
                    "present": total - n_missing,
                    "missing": n_missing,
                    "coverage_pct": round(
                        (total - n_missing) / total * 100, 2
                    ) if total > 0 else 0,
                }
            else:
                report[label] = {
                    "column": col,
                    "available": False,
                }

        return report

    # ── Multi-dataset rows ──────────────────────────────────

    def _analyze_multi_dataset_rows(self) -> Dict[str, Any]:
        """Find rows whose file_hash appears in multiple datasets."""
        if "file_hash" not in self.df.columns or "dataset_name" not in self.df.columns:
            return {
                "available": False,
                "reason": "Need both file_hash and dataset_name columns",
            }

        valid = self.df[["file_hash", "dataset_name"]].dropna()
        if valid.empty:
            return {"multi_dataset_hashes": 0, "affected_rows": 0}

        hash_ds = valid.groupby("file_hash")["dataset_name"].nunique()
        multi = hash_ds[hash_ds > 1]
        affected = 0
        if len(multi) > 0:
            affected = int(
                valid[valid["file_hash"].isin(multi.index)].shape[0]
            )

        return {
            "multi_dataset_hashes": len(multi),
            "affected_rows": affected,
        }

    # ── Unused categories ───────────────────────────────────

    def _analyze_unused_categories(self) -> Dict[str, List[str]]:
        """Find categorical columns with unused category levels."""
        unused: Dict[str, List[str]] = {}
        for col in self.df.columns:
            if hasattr(self.df[col], "cat"):
                used = set(self.df[col].dropna().unique())
                all_cats = set(self.df[col].cat.categories)
                diff = all_cats - used
                if diff:
                    unused[col] = sorted(str(v) for v in diff)
        return unused

    # ── Leakage indicators ──────────────────────────────────

    def _analyze_leakage(self) -> Dict[str, Any]:
        """Check for potential data leakage across splits."""
        split_cols = self.available_columns(self.config.split_columns)
        if not split_cols:
            return {"available": False, "reason": "No split columns"}

        leakage: Dict[str, Any] = {}

        for col in split_cols:
            # Hash leakage
            if "file_hash" in self.df.columns:
                valid = self.df[[col, "file_hash"]].dropna()
                if not valid.empty:
                    hash_splits = valid.groupby("file_hash")[col].nunique()
                    multi = hash_splits[hash_splits > 1]
                    if len(multi) > 0:
                        affected = int(
                            valid[
                                valid["file_hash"].isin(multi.index)
                            ].shape[0]
                        )
                        leakage[f"{col}_hash"] = {
                            "hashes_in_multiple_splits": len(multi),
                            "affected_rows": affected,
                        }
                        self.add_warning(
                            f"Potential leakage: {len(multi)} hashes appear "
                            f"in multiple '{col}' splits ({affected} rows)"
                        )

            # Path leakage
            if "original_path" in self.df.columns:
                valid = self.df[[col, "original_path"]].dropna()
                if not valid.empty:
                    path_splits = valid.groupby("original_path")[col].nunique()
                    multi = path_splits[path_splits > 1]
                    if len(multi) > 0:
                        affected = int(
                            valid[
                                valid["original_path"].isin(multi.index)
                            ].shape[0]
                        )
                        leakage[f"{col}_path"] = {
                            "paths_in_multiple_splits": len(multi),
                            "affected_rows": affected,
                        }

        if not leakage:
            leakage["status"] = "No leakage detected"

        return leakage

    # ── Summary ─────────────────────────────────────────────

    def _build_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Build a high-level quality summary."""
        issues: List[str] = []

        # Missing values
        missing = results.get("missing_values", [])
        critical_missing = [
            m for m in missing if m["missing_percent"] > 50
        ]
        if critical_missing:
            cols = [m["column"] for m in critical_missing]
            issues.append(f"Columns >50% missing: {cols}")

        # Duplicates
        dups = results.get("duplicates", {})
        exact_dups = dups.get("exact_row_duplicates", 0)
        if isinstance(exact_dups, int) and exact_dups > 0:
            issues.append(f"{exact_dups} exact row duplicates")

        # Unexpected splits
        split_q = results.get("split_quality", {})
        for col, info in split_q.get("columns", {}).items():
            unexpected = info.get("unexpected_values", [])
            if unexpected:
                issues.append(
                    f"Unexpected split values in '{col}': {unexpected}"
                )

        # Leakage
        leakage = results.get("leakage_indicators", {})
        for key, info in leakage.items():
            if isinstance(info, dict):
                n = info.get(
                    "hashes_in_multiple_splits",
                    info.get("paths_in_multiple_splits", 0)
                )
                if isinstance(n, int) and n > 0:
                    issues.append(f"Potential leakage in {key}: {n} items")

        return {
            "total_issues": len(issues),
            "issues": issues,
            "warnings": list(self._warnings),
        }
