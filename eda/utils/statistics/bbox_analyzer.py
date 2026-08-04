# ─────────────────────────────────────────────────────────────
# Bounding Box Analyzer
# ─────────────────────────────────────────────────────────────
"""
Comprehensive bounding box analysis: area, aspect ratio,
centre distribution, edge proximity, and validation of
invalid boxes (negative, zero-area, inverted, out-of-range).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from eda.utils.statistics.base_analyzer import BaseAnalyzer

logger = logging.getLogger(__name__)


class BBoxAnalyzer(BaseAnalyzer):
    """Analyzes bounding box geometry and annotation quality."""

    def _analyze(self) -> Dict[str, Any]:
        if not self.validate_columns(self.config.bbox_columns):
            return self._safe_result(False, reason="Missing bbox columns")

        results: Dict[str, Any] = {"available": True}

        try:
            bbox_df = self._prepare_bbox_df()
            results["count"] = len(bbox_df)
        except Exception as exc:
            self.add_warning(f"BBox preparation failed: {exc}")
            return self._safe_result(False, reason=str(exc))

        analyses = [
            ("area", lambda: self.analyze_area(bbox_df)),
            ("aspect_ratio", lambda: self.analyze_aspect_ratio(bbox_df)),
            ("center_distribution", lambda: self.analyze_center_distribution(bbox_df)),
            ("edge_proximity", lambda: self.analyze_edge_proximity(bbox_df)),
            ("validation", lambda: self.validate_bboxes(bbox_df)),
            ("per_dataset", self.analyze_per_dataset),
        ]

        for key, fn in analyses:
            try:
                results[key] = fn()
            except Exception as exc:
                self.add_warning(f"{key} analysis failed: {exc}")
                results[key] = {"available": False, "reason": str(exc)}

        try:
            results["quality_components"] = {
                "invalid_bbox_score": self.compute_invalid_bbox_score(),
            }
        except Exception as exc:
            self.add_warning(f"Quality scoring failed: {exc}")
            results["quality_components"] = {"invalid_bbox_score": 1.0}

        self._results = results
        return results

    # ── Preparation ─────────────────────────────────────────

    def _prepare_bbox_df(self) -> pd.DataFrame:
        """Extract and clean bbox columns."""
        cols = self.config.bbox_columns
        bbox = self.df[cols].copy()
        # Drop rows where all bbox cols are NaN
        bbox = bbox.dropna(how="all")
        return bbox

    # ── Area ────────────────────────────────────────────────

    def analyze_area(self, bbox_df: pd.DataFrame) -> Dict[str, Any]:
        """Normalized bounding box area distribution."""
        df = bbox_df.copy()
        df["width"] = (df["x2"] - df["x1"]).clip(lower=0)
        df["height"] = (df["y2"] - df["y1"]).clip(lower=0)
        df["area"] = df["width"] * df["height"]
        s = df["area"].dropna()
        if s.empty:
            return {"count": 0}
        return {
            "count": len(s),
            "mean": round(float(s.mean()), 6),
            "median": round(float(s.median()), 6),
            "std": round(float(s.std()), 6),
            "min": round(float(s.min()), 6),
            "max": round(float(s.max()), 6),
            "q1": round(float(s.quantile(0.25)), 6),
            "q3": round(float(s.quantile(0.75)), 6),
            "zero_area_count": int((s == 0).sum()),
        }

    # ── Aspect ratio ────────────────────────────────────────

    def analyze_aspect_ratio(self, bbox_df: pd.DataFrame) -> Dict[str, Any]:
        """Width / height ratio of bounding boxes."""
        df = bbox_df.copy()
        df["bw"] = (df["x2"] - df["x1"]).clip(lower=0)
        df["bh"] = (df["y2"] - df["y1"]).clip(lower=0)
        # Avoid division by zero
        valid = df[df["bh"] > 0].copy()
        if valid.empty:
            return {"count": 0}
        valid["aspect_ratio"] = valid["bw"] / valid["bh"]
        s = valid["aspect_ratio"]
        return {
            "count": len(s),
            "mean": round(float(s.mean()), 4),
            "median": round(float(s.median()), 4),
            "std": round(float(s.std()), 4),
            "min": round(float(s.min()), 4),
            "max": round(float(s.max()), 4),
        }

    # ── Centre distribution ─────────────────────────────────

    def analyze_center_distribution(self, bbox_df: pd.DataFrame) -> Dict[str, Any]:
        """Statistics on centre_x, centre_y positions."""
        result: Dict[str, Any] = {}
        for col in ["center_x", "center_y"]:
            if col in bbox_df.columns:
                s = bbox_df[col].dropna()
                if not s.empty:
                    result[col] = {
                        "mean": round(float(s.mean()), 4),
                        "median": round(float(s.median()), 4),
                        "std": round(float(s.std()), 4),
                        "min": round(float(s.min()), 4),
                        "max": round(float(s.max()), 4),
                    }
        return result

    # ── Edge proximity ──────────────────────────────────────

    def analyze_edge_proximity(
        self, bbox_df: pd.DataFrame, threshold: float = 0.05
    ) -> Dict[str, Any]:
        """Count bboxes touching or near image boundaries."""
        df = bbox_df.copy()
        near_left = (df["x1"] < threshold).sum()
        near_top = (df["y1"] < threshold).sum()
        near_right = (df["x2"] > 1 - threshold).sum()
        near_bottom = (df["y2"] > 1 - threshold).sum()
        total = len(df)
        return {
            "threshold": threshold,
            "near_left": int(near_left),
            "near_top": int(near_top),
            "near_right": int(near_right),
            "near_bottom": int(near_bottom),
            "near_any_edge": int(
                ((df["x1"] < threshold) | (df["y1"] < threshold) |
                 (df["x2"] > 1 - threshold) | (df["y2"] > 1 - threshold)).sum()
            ),
            "near_any_edge_percent": round(
                ((df["x1"] < threshold) | (df["y1"] < threshold) |
                 (df["x2"] > 1 - threshold) | (df["y2"] > 1 - threshold)).mean() * 100, 2
            ) if total > 0 else 0,
        }

    # ── Validation ──────────────────────────────────────────

    def validate_bboxes(self, bbox_df: pd.DataFrame) -> Dict[str, Any]:
        """Detect invalid bounding boxes."""
        df = bbox_df.copy()
        total = len(df)
        if total == 0:
            return {"total": 0}

        issues: Dict[str, int] = {}

        # Negative coordinates
        for col in ["x1", "y1", "x2", "y2"]:
            neg = int((df[col] < 0).sum())
            if neg > 0:
                issues[f"negative_{col}"] = neg

        # Inverted (x1 > x2 or y1 > y2)
        inverted_x = int((df["x1"] > df["x2"]).sum())
        inverted_y = int((df["y1"] > df["y2"]).sum())
        if inverted_x:
            issues["inverted_x"] = inverted_x
        if inverted_y:
            issues["inverted_y"] = inverted_y

        # Zero-area boxes
        zero_w = int(((df["x2"] - df["x1"]).abs() < 1e-8).sum())
        zero_h = int(((df["y2"] - df["y1"]).abs() < 1e-8).sum())
        if zero_w:
            issues["zero_width"] = zero_w
        if zero_h:
            issues["zero_height"] = zero_h

        # Out of [0, 1] range (normalised coords)
        oob = int(
            ((df["x1"] > 1.0) | (df["y1"] > 1.0) |
             (df["x2"] > 1.0) | (df["y2"] > 1.0)).sum()
        )
        if oob:
            issues["out_of_range"] = oob

        total_invalid = len(set().union(*(
            set(df.index[(df[col] < 0)])
            for col in ["x1", "y1", "x2", "y2"]
            if col in df.columns
        )) | set(df.index[df["x1"] > df["x2"]])
           | set(df.index[df["y1"] > df["y2"]]))

        return {
            "total": total,
            "issues": issues,
            "total_invalid": total_invalid,
            "invalid_percent": round(total_invalid / total * 100, 2) if total > 0 else 0,
            "valid_count": total - total_invalid,
        }

    # ── Per-dataset ─────────────────────────────────────────

    def analyze_per_dataset(self) -> Dict[str, Dict[str, Any]]:
        """Repeat bbox analysis per dataset."""
        results = {}
        bbox_cols = self.config.bbox_columns
        for ds in self.iter_datasets():
            ds_df = self.get_dataset_df(ds)
            bbox = ds_df[bbox_cols].dropna(how="all")
            results[ds] = {
                "count": len(bbox),
                "area": self.analyze_area(bbox) if not bbox.empty else {},
            }
        return results

    # ── Quality component ───────────────────────────────────

    def compute_invalid_bbox_score(self) -> float:
        """Normalized 0–1 score (1 = no invalid bboxes)."""
        if not self.has_columns(self.config.bbox_columns):
            return 1.0
        bbox_df = self._prepare_bbox_df()
        validation = self.validate_bboxes(bbox_df)
        total = validation["total"]
        if total == 0:
            return 1.0
        invalid = validation["total_invalid"]
        return max(0.0, 1.0 - (invalid / total))
