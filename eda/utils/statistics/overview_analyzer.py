# ─────────────────────────────────────────────────────────────
# Overview Analyzer
# ─────────────────────────────────────────────────────────────
"""
First-look dataset overview: dimensions, per-dataset counts,
media types, source types, missing values, duplicates,
processing status, version summary, and memory footprint.

All sub-analyses are independently try/excepted to ensure
partial results are returned even when individual analyses fail.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from eda.utils.statistics.base_analyzer import BaseAnalyzer

logger = logging.getLogger(__name__)


class OverviewAnalyzer(BaseAnalyzer):
    """Comprehensive dataset overview analyzer."""

    def _analyze(self) -> Dict[str, Any]:
        results: Dict[str, Any] = {"available": True}

        analyses = [
            ("dimensions", self.analyze_dimensions),
            ("per_dataset_counts", self.analyze_per_dataset_counts),
            ("media_types", self.analyze_media_types),
            ("source_types", self.analyze_source_types),
            ("missing_values", self.analyze_missing_values),
            ("duplicates", self.analyze_duplicates),
            ("processing_status", self.analyze_processing_status),
            ("versions", self.analyze_versions),
            ("memory_footprint", self.analyze_memory_footprint),
        ]

        for key, fn in analyses:
            try:
                results[key] = fn()
            except Exception as exc:
                self.add_warning(f"{key} analysis failed: {exc}")
                results[key] = {"available": False, "reason": str(exc)}

        try:
            results["quality_components"] = {
                "missing_values_score": self.compute_missing_values_score(),
                "duplicate_score": self.compute_duplicate_score(),
            }
        except Exception as exc:
            self.add_warning(f"Quality component scoring failed: {exc}")
            results["quality_components"] = {
                "missing_values_score": 1.0,
                "duplicate_score": 1.0,
            }

        self._results = results
        return results

    # ── Individual analyses ─────────────────────────────────

    def analyze_dimensions(self) -> Dict[str, Any]:
        """Basic shape, dtype breakdown."""
        dtypes = self.df.dtypes.value_counts()
        return {
            "rows": len(self.df),
            "columns": len(self.df.columns),
            "dtypes": {str(k): int(v) for k, v in dtypes.items()},
            "column_names": self.df.columns.tolist(),
        }

    def analyze_per_dataset_counts(self) -> List[Dict[str, Any]]:
        """Record counts per dataset_name."""
        if "dataset_name" not in self.df.columns:
            return []
        counts = self.df["dataset_name"].value_counts()
        total = len(self.df)
        if total == 0:
            return []
        return [
            {
                "dataset": name,
                "count": int(count),
                "percentage": round(count / total * 100, 2),
            }
            for name, count in counts.items()
        ]

    def analyze_media_types(self) -> Dict[str, Any]:
        """Video vs image distribution."""
        if "media_type" in self.df.columns:
            counts = self.df["media_type"].value_counts(dropna=False)
        else:
            # Heuristic based on fps/duration
            has_video = self.df["fps"].notna() if "fps" in self.df.columns else pd.Series(False, index=self.df.index)
            counts = pd.Series({
                "video": int(has_video.sum()),
                "image_or_unknown": int((~has_video).sum()),
            })
        return {
            "distribution": {str(k): int(v) for k, v in counts.items()},
            "total": len(self.df),
        }

    def analyze_source_types(self) -> Dict[str, Any]:
        """Real vs synthetic distribution."""
        if "source_type" not in self.df.columns:
            return {"available": False}
        counts = self.df["source_type"].value_counts()
        return {
            "available": True,
            "distribution": {str(k): int(v) for k, v in counts.items()},
        }

    def analyze_missing_values(self) -> List[Dict[str, Any]]:
        """Per-column missing value counts and percentages."""
        total = len(self.df)
        missing = self.df.isnull().sum()
        result = []
        for col in self.df.columns:
            count = int(missing[col])
            result.append({
                "column": col,
                "missing_count": count,
                "missing_percent": round(count / total * 100, 2) if total > 0 else 0,
            })
        return sorted(result, key=lambda x: x["missing_count"], reverse=True)

    def analyze_duplicates(self) -> Dict[str, Any]:
        """Duplicate record detection."""
        result: Dict[str, Any] = {"total_rows": len(self.df)}

        # Via is_duplicate column
        if "is_duplicate" in self.df.columns:
            dup_col = self.df["is_duplicate"]
            flagged = int(dup_col.sum()) if dup_col.dtype == bool else int((dup_col == True).sum())  # noqa: E712
            result["flagged_duplicates"] = flagged
        else:
            result["flagged_duplicates"] = None

        # Via file_hash
        if "file_hash" in self.df.columns:
            hash_counts = self.df["file_hash"].dropna().value_counts()
            dup_hashes = hash_counts[hash_counts > 1]
            result["duplicate_hash_groups"] = len(dup_hashes)
            result["rows_with_duplicate_hash"] = int(
                dup_hashes.sum() - len(dup_hashes)
            ) if len(dup_hashes) > 0 else 0
        else:
            result["duplicate_hash_groups"] = None
            result["rows_with_duplicate_hash"] = None

        # Exact row duplicates
        result["exact_row_duplicates"] = int(self.df.duplicated().sum())

        return result

    def analyze_processing_status(self) -> Dict[str, Any]:
        """Distribution of validation / processing / preprocessing status."""
        statuses: Dict[str, Any] = {}
        for col in ["validation_status", "processing_status", "preprocessing_status"]:
            if col in self.df.columns:
                counts = self.df[col].value_counts(dropna=False)
                statuses[col] = {str(k): int(v) for k, v in counts.items()}
        return statuses

    def analyze_versions(self) -> Dict[str, Any]:
        """Dataset and pipeline version summary."""
        result: Dict[str, Any] = {}
        for col in ["dataset_version", "pipeline_version"]:
            if col in self.df.columns:
                versions = self.df[col].dropna().unique().tolist()
                result[col] = [str(v) for v in versions]
        return result

    def analyze_memory_footprint(self) -> Dict[str, float]:
        """Memory usage of the DataFrame."""
        mem = self.df.memory_usage(deep=True)
        return {
            "total_mb": round(mem.sum() / (1024 ** 2), 2),
            "per_column_kb": {
                col: round(mem[col] / 1024, 2)
                for col in self.df.columns
            },
        }

    # ── Quality component scores ────────────────────────────

    def compute_missing_values_score(self) -> float:
        """Normalized 0–1 score (1 = no missing values)."""
        total_cells = self.df.shape[0] * self.df.shape[1]
        if total_cells == 0:
            return 1.0
        missing_cells = int(self.df.isnull().sum().sum())
        return max(0.0, 1.0 - (missing_cells / total_cells))

    def compute_duplicate_score(self) -> float:
        """Normalized 0–1 score (1 = no duplicates)."""
        total = len(self.df)
        if total == 0:
            return 1.0
        exact_dups = int(self.df.duplicated().sum())
        return max(0.0, 1.0 - (exact_dups / total))
