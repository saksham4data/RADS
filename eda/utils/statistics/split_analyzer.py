# ─────────────────────────────────────────────────────────────
# Split Strategy Analyzer
# ─────────────────────────────────────────────────────────────
"""
Compares train/val/test split strategies, analyzes class
balance per split, detects agreement/disagreement between
split columns, and identifies potential data leakage.

Design notes
~~~~~~~~~~~~
* All comparisons convert to normalized string values before
  comparison.  This avoids the ``TypeError: Categoricals can
  only be compared if 'categories' are the same`` that occurs
  when pandas categorical columns have different category sets.
* ``compare_split_strategies()`` adds semantic warnings when
  two split columns appear to represent fundamentally different
  splitting concepts (e.g. one has only train/test while the
  other has train/val/test).
* Every sub-analysis is wrapped in try/except so that a failure
  in one does not prevent the others from running.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Set

import numpy as np
import pandas as pd

from eda.utils.statistics.base_analyzer import BaseAnalyzer

logger = logging.getLogger(__name__)


class SplitAnalyzer(BaseAnalyzer):
    """Analyzes and compares split strategies."""

    def _analyze(self) -> Dict[str, Any]:
        split_cols = self.available_columns(self.config.split_columns)
        if not split_cols:
            return self._safe_result(False, reason="No split columns found")

        results: Dict[str, Any] = {"available": True, "split_columns": split_cols}

        # Per-split-column analysis
        for col in split_cols:
            try:
                results[col] = {
                    "distribution": self.analyze_split_distribution(col),
                    "class_balance": self.analyze_split_class_balance(
                        col, self.config.primary_label
                    ),
                }
            except Exception as exc:
                self.add_warning(
                    f"Failed to analyze split column '{col}': {exc}"
                )
                results[col] = {
                    "distribution": {"available": False, "reason": str(exc)},
                    "class_balance": {"available": False, "reason": str(exc)},
                }

        try:
            results["strategy_comparison"] = self.compare_split_strategies()
        except Exception as exc:
            self.add_warning(f"Strategy comparison failed: {exc}")
            results["strategy_comparison"] = {
                "available": False, "reason": str(exc),
            }

        try:
            results["leakage_indicators"] = self.detect_leakage_indicators()
        except Exception as exc:
            self.add_warning(f"Leakage detection failed: {exc}")
            results["leakage_indicators"] = {}

        try:
            results["per_dataset"] = self.analyze_per_dataset()
        except Exception as exc:
            self.add_warning(f"Per-dataset analysis failed: {exc}")
            results["per_dataset"] = {}

        self._results = results
        return results

    # ── Split distribution ──────────────────────────────────

    def analyze_split_distribution(self, split_col: str) -> Dict[str, Any]:
        """Sample counts per split value."""
        if split_col not in self.df.columns:
            return {"available": False, "reason": f"Column '{split_col}' not found"}

        counts = self.df[split_col].value_counts(dropna=False)
        total = len(self.df)
        if total == 0:
            return {"available": True, "counts": {}, "percentages": {}, "total": 0}
        return {
            "available": True,
            "counts": {str(k): int(v) for k, v in counts.items()},
            "percentages": {
                str(k): round(v / total * 100, 2)
                for k, v in counts.items()
            },
            "total": total,
        }

    # ── Class balance per split ─────────────────────────────

    def analyze_split_class_balance(
        self,
        split_col: str,
        label_col: str,
    ) -> Dict[str, Any]:
        """Per-split class distribution."""
        if split_col not in self.df.columns or label_col not in self.df.columns:
            return {"available": False, "reason": "Required columns missing"}

        result: Dict[str, Any] = {}
        for split_val in self.df[split_col].dropna().unique():
            subset = self.df[self.df[split_col] == split_val]
            if subset.empty:
                continue
            counts = subset[label_col].value_counts(dropna=False)
            total = len(subset)
            result[str(split_val)] = {
                "total": total,
                "distribution": {str(k): int(v) for k, v in counts.items()},
                "percentages": {
                    str(k): round(v / total * 100, 2) if total > 0 else 0
                    for k, v in counts.items()
                },
            }
        return result

    # ── Strategy comparison ─────────────────────────────────

    def compare_split_strategies(self) -> Dict[str, Any]:
        """Agreement/disagreement between split columns.

        Converts both columns to normalized strings before
        comparison to avoid categorical dtype conflicts.
        Also performs semantic analysis: if two columns have
        very different value sets, a warning is emitted.
        """
        split_cols = self.available_columns(self.config.split_columns)
        if len(split_cols) < 2:
            return {"available": False, "reason": "Need ≥2 split columns"}

        comparisons: Dict[str, Any] = {}
        comparison_warnings: List[str] = []

        for i, col1 in enumerate(split_cols):
            for col2 in split_cols[i + 1:]:
                key = f"{col1}_vs_{col2}"
                valid = self.df[[col1, col2]].dropna()
                total = len(valid)

                if total == 0:
                    comparisons[key] = {
                        "agreement_count": 0,
                        "total": 0,
                        "note": "No rows with both columns non-null",
                    }
                    continue

                # ── Semantic check ──
                # Detect when two split columns represent different
                # splitting concepts (different unique value sets).
                vals1: Set[str] = set(
                    valid[col1].astype(str).str.strip().str.lower().unique()
                )
                vals2: Set[str] = set(
                    valid[col2].astype(str).str.strip().str.lower().unique()
                )
                if vals1 != vals2:
                    missing_in_2 = vals1 - vals2
                    missing_in_1 = vals2 - vals1
                    note_parts = []
                    if missing_in_2:
                        note_parts.append(
                            f"values in {col1} but not {col2}: {sorted(missing_in_2)}"
                        )
                    if missing_in_1:
                        note_parts.append(
                            f"values in {col2} but not {col1}: {sorted(missing_in_1)}"
                        )
                    warn_msg = (
                        f"{col1} and {col2} have different value sets "
                        f"({', '.join(note_parts)}). "
                        f"Agreement % may not be meaningful — "
                        f"these may represent different splitting concepts."
                    )
                    comparison_warnings.append(warn_msg)
                    self.add_warning(warn_msg)

                # ── Compare as strings (never as categoricals) ──
                left = valid[col1].astype(str).str.strip().str.lower()
                right = valid[col2].astype(str).str.strip().str.lower()
                agree = int((left == right).sum())

                comparisons[key] = {
                    "total": total,
                    "agreement_count": agree,
                    "agreement_percent": round(agree / total * 100, 2),
                    "disagreement_count": total - agree,
                    "disagreement_percent": round(
                        (total - agree) / total * 100, 2
                    ),
                    "value_sets_match": vals1 == vals2,
                }

                # Confusion-style breakdown (also cast to string)
                try:
                    ct = pd.crosstab(
                        left.rename(col1),
                        right.rename(col2),
                    )
                    comparisons[key]["crosstab"] = ct.to_dict()
                except Exception as exc:
                    self.add_warning(
                        f"Crosstab for {key} failed: {exc}"
                    )

        result: Dict[str, Any] = {
            "available": True,
            "comparisons": comparisons,
        }
        if comparison_warnings:
            result["warnings"] = comparison_warnings
        return result

    # ── Leakage detection ───────────────────────────────────

    def detect_leakage_indicators(self) -> Dict[str, Any]:
        """Identify potential data leakage signals.

        Resilient to missing columns, missing hashes, missing
        paths, and duplicate metadata.
        """
        indicators: Dict[str, Any] = {}
        split_cols = self.available_columns(self.config.split_columns)

        if not split_cols:
            self.add_warning("No split columns available for leakage detection")
            return indicators

        for col in split_cols:
            if col not in self.df.columns:
                continue

            # ── Hash-based leakage ──
            if "file_hash" in self.df.columns:
                valid = self.df[[col, "file_hash"]].dropna()
                if valid.empty:
                    self.add_warning(
                        f"No valid rows for hash leakage check on '{col}'"
                    )
                    indicators[f"{col}_hash_leakage"] = {
                        "hashes_in_multiple_splits": 0,
                        "affected_rows": 0,
                        "note": "No rows with both split and hash",
                    }
                else:
                    hash_splits = valid.groupby("file_hash")[col].nunique()
                    multi_split = hash_splits[hash_splits > 1]
                    affected = 0
                    if len(multi_split) > 0:
                        affected = int(
                            valid[
                                valid["file_hash"].isin(multi_split.index)
                            ].shape[0]
                        )
                    indicators[f"{col}_hash_leakage"] = {
                        "hashes_in_multiple_splits": len(multi_split),
                        "affected_rows": affected,
                    }
            else:
                self.add_warning(
                    f"Column 'file_hash' missing — "
                    f"cannot check hash leakage for '{col}'"
                )

            # ── Path-based leakage ──
            if "original_path" in self.df.columns:
                valid = self.df[[col, "original_path"]].dropna()
                if valid.empty:
                    indicators[f"{col}_path_leakage"] = {
                        "paths_in_multiple_splits": 0,
                        "affected_rows": 0,
                        "note": "No rows with both split and path",
                    }
                else:
                    path_splits = valid.groupby("original_path")[col].nunique()
                    multi_path = path_splits[path_splits > 1]
                    affected = 0
                    if len(multi_path) > 0:
                        affected = int(
                            valid[
                                valid["original_path"].isin(multi_path.index)
                            ].shape[0]
                        )
                    indicators[f"{col}_path_leakage"] = {
                        "paths_in_multiple_splits": len(multi_path),
                        "affected_rows": affected,
                    }
            else:
                self.add_warning(
                    f"Column 'original_path' missing — "
                    f"cannot check path leakage for '{col}'"
                )

        return indicators

    # ── Per-dataset ─────────────────────────────────────────

    def analyze_per_dataset(self) -> Dict[str, Dict[str, Any]]:
        """Repeat split analysis per dataset."""
        results = {}
        split_cols = self.available_columns(self.config.split_columns)

        for ds in self.iter_datasets():
            ds_df = self.get_dataset_df(ds)
            ds_result: Dict[str, Any] = {"total": len(ds_df)}
            for col in split_cols:
                if col in ds_df.columns:
                    counts = ds_df[col].value_counts(dropna=False)
                    ds_result[col] = {
                        str(k): int(v) for k, v in counts.items()
                    }
            results[ds] = ds_result

        return results
