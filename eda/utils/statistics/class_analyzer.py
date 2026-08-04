# ─────────────────────────────────────────────────────────────
# Class Distribution Analyzer
# ─────────────────────────────────────────────────────────────
"""
Analyzes class balance across primary (accident type) and
secondary label dimensions (weather, day_time, etc.).

All cross-tabulations cast to string before ``pd.crosstab``
to avoid categorical comparison errors when columns have
different category sets.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from eda.utils.statistics.base_analyzer import BaseAnalyzer

logger = logging.getLogger(__name__)


class ClassAnalyzer(BaseAnalyzer):
    """Analyzes label distributions and class balance."""

    def _analyze(self) -> Dict[str, Any]:
        results: Dict[str, Any] = {"available": True}

        analyses = [
            ("primary_label", self.analyze_primary_label),
            ("secondary_labels", self.analyze_secondary_labels),
            ("cross_tabulations", self.analyze_cross_tabulation),
            ("label_cooccurrence", self.analyze_label_cooccurrence),
            ("per_dataset", self.analyze_per_dataset),
        ]

        for key, fn in analyses:
            try:
                results[key] = fn()
            except Exception as exc:
                self.add_warning(f"{key} analysis failed: {exc}")
                results[key] = {"available": False, "reason": str(exc)}

        # Extra analyses that don't follow the same pattern
        try:
            results["region"] = self._analyze_column("region")
        except Exception as exc:
            self.add_warning(f"Region analysis failed: {exc}")
            results["region"] = {"available": False}

        try:
            results["rollover"] = self._analyze_column("rollover")
        except Exception as exc:
            self.add_warning(f"Rollover analysis failed: {exc}")
            results["rollover"] = {"available": False}

        try:
            results["quality_components"] = {
                "class_imbalance_score": self.compute_class_imbalance_score(),
            }
        except Exception as exc:
            self.add_warning(f"Class imbalance scoring failed: {exc}")
            results["quality_components"] = {
                "class_imbalance_score": 1.0,
            }

        self._results = results
        return results

    # ── Primary label ───────────────────────────────────────

    def analyze_primary_label(self) -> Dict[str, Any]:
        """Analyze the primary classification label."""
        col = self.config.primary_label
        if col not in self.df.columns:
            logger.warning("Primary label '%s' not found.", col)
            return {"available": False}

        counts = self.df[col].value_counts(dropna=False)
        total = len(self.df)
        if total == 0:
            return {"available": True, "column": col, "unique_classes": 0}

        return {
            "available": True,
            "column": col,
            "unique_classes": int(counts.nunique()),
            "distribution": {str(k): int(v) for k, v in counts.items()},
            "percentages": {
                str(k): round(v / total * 100, 2)
                for k, v in counts.items()
            },
            "imbalance_ratio": round(
                float(counts.max() / counts.min()) if counts.min() > 0 else float("inf"),
                2,
            ),
            "entropy": round(self._entropy(counts.values), 4),
        }

    # ── Secondary labels ────────────────────────────────────

    def analyze_secondary_labels(self) -> Dict[str, Dict[str, Any]]:
        """Analyze each secondary label column."""
        results = {}
        for col in self.config.secondary_labels:
            try:
                results[col] = self._analyze_column(col)
            except Exception as exc:
                self.add_warning(f"Secondary label '{col}' analysis failed: {exc}")
                results[col] = {"available": False, "column": col}
        return results

    def _analyze_column(self, col: str) -> Dict[str, Any]:
        """Generic categorical column analysis."""
        if col not in self.df.columns:
            return {"available": False, "column": col}

        counts = self.df[col].value_counts(dropna=False)
        total = len(self.df)
        if total == 0:
            return {"available": True, "column": col, "unique_values": 0}

        return {
            "available": True,
            "column": col,
            "unique_values": int(self.df[col].nunique(dropna=False)),
            "distribution": {str(k): int(v) for k, v in counts.items()},
            "percentages": {
                str(k): round(v / total * 100, 2)
                for k, v in counts.items()
            },
            "missing_count": int(self.df[col].isna().sum()),
        }

    # ── Cross-tabulations ───────────────────────────────────

    def analyze_cross_tabulation(self) -> Dict[str, Any]:
        """Cross-tabulate primary label with each secondary label.

        Casts columns to string before ``pd.crosstab`` to avoid
        categorical comparison errors.
        """
        primary = self.config.primary_label
        if primary not in self.df.columns:
            return {}

        cross_tabs = {}
        for sec in self.config.secondary_labels:
            if sec not in self.df.columns:
                continue
            try:
                # Cast to string to avoid categorical dtype conflicts
                primary_str = self.df[primary].astype(str)
                sec_str = self.df[sec].astype(str)
                ct = pd.crosstab(primary_str, sec_str)
                cross_tabs[f"{primary}_x_{sec}"] = ct.to_dict()
            except Exception as exc:
                self.add_warning(
                    f"Crosstab {primary} × {sec} failed: {exc}"
                )

        return cross_tabs

    # ── Label co-occurrence ─────────────────────────────────

    def analyze_label_cooccurrence(self) -> Dict[str, Any]:
        """Analyze how labels co-occur across label columns."""
        label_cols = [self.config.primary_label] + self.config.secondary_labels
        present = self.available_columns(label_cols)
        if len(present) < 2:
            return {"available": False}

        # Compute non-null overlap counts
        cooccur: Dict[str, Dict[str, int]] = {}
        for c1 in present:
            cooccur[c1] = {}
            for c2 in present:
                both_present = int(
                    (self.df[c1].notna() & self.df[c2].notna()).sum()
                )
                cooccur[c1][c2] = both_present

        return {"available": True, "cooccurrence_matrix": cooccur}

    # ── Per-dataset ─────────────────────────────────────────

    def analyze_per_dataset(self) -> Dict[str, Dict[str, Any]]:
        """Repeat primary label analysis per dataset."""
        results = {}
        primary = self.config.primary_label
        if primary not in self.df.columns:
            return results

        for ds in self.iter_datasets():
            ds_df = self.get_dataset_df(ds)
            counts = ds_df[primary].value_counts(dropna=False)
            total = len(ds_df)
            results[ds] = {
                "total": total,
                "distribution": {str(k): int(v) for k, v in counts.items()},
                "percentages": {
                    str(k): round(v / total * 100, 2) if total > 0 else 0
                    for k, v in counts.items()
                },
            }
        return results

    # ── Quality component ───────────────────────────────────

    def compute_class_imbalance_score(self) -> float:
        """Normalized 0–1 score (1 = perfectly balanced).

        Uses normalised entropy: H(X) / log(k), where k = # classes.
        """
        col = self.config.primary_label
        if col not in self.df.columns:
            return 1.0

        counts = self.df[col].value_counts(dropna=True)
        if len(counts) <= 1:
            return 1.0

        h = self._entropy(counts.values)
        max_h = np.log2(len(counts))
        if max_h == 0:
            return 1.0
        return float(h / max_h)

    # ── Helpers ─────────────────────────────────────────────

    @staticmethod
    def _entropy(counts: np.ndarray) -> float:
        """Shannon entropy in bits.

        Handles empty arrays and single-element arrays safely.
        """
        if len(counts) == 0:
            return 0.0
        total = counts.sum()
        if total == 0:
            return 0.0
        probs = counts / total
        probs = probs[probs > 0]
        if len(probs) == 0:
            return 0.0
        return float(-np.sum(probs * np.log2(probs)))
