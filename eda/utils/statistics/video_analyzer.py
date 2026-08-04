# ─────────────────────────────────────────────────────────────
# Video / Media Statistics Analyzer
# ─────────────────────────────────────────────────────────────
"""
Deep analysis of video and media technical properties:
duration, FPS, resolution, frame count, file size, codec,
outlier detection, and image-subset analysis.

All sub-analyses are independently try/excepted. The
``_describe`` helper gracefully handles edge cases like
single-value series where skewness/kurtosis are undefined.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from eda.utils.statistics.base_analyzer import BaseAnalyzer

logger = logging.getLogger(__name__)


class VideoAnalyzer(BaseAnalyzer):
    """Analyzes video/media technical properties."""

    def _analyze(self) -> Dict[str, Any]:
        results: Dict[str, Any] = {"available": True}

        analyses = [
            ("duration", self.analyze_duration),
            ("fps", self.analyze_fps),
            ("resolution", self.analyze_resolution),
            ("frame_count", self.analyze_frame_count),
            ("file_size", self.analyze_file_size),
            ("codec_channels", self.analyze_codec_channels),
            ("image_subset", self.analyze_image_subset),
            ("outliers", self.detect_outliers),
            ("per_dataset", self.analyze_per_dataset),
        ]

        for key, fn in analyses:
            try:
                results[key] = fn()
            except Exception as exc:
                self.add_warning(f"{key} analysis failed: {exc}")
                results[key] = {"available": False, "reason": str(exc)}

        self._results = results
        return results

    # ── Duration ────────────────────────────────────────────

    def analyze_duration(self) -> Dict[str, Any]:
        """Duration distribution and statistics."""
        if "duration" not in self.df.columns:
            return {"available": False}
        s = self.df["duration"].dropna()
        if s.empty:
            return {"available": True, "count": 0}
        return {
            "available": True,
            "count": len(s),
            "stats": self._describe(s),
        }

    # ── FPS ─────────────────────────────────────────────────

    def analyze_fps(self) -> Dict[str, Any]:
        """FPS distribution and common values."""
        if "fps" not in self.df.columns:
            return {"available": False}
        s = self.df["fps"].dropna()
        if s.empty:
            return {"available": True, "count": 0}
        common = s.round(0).value_counts().head(10)
        return {
            "available": True,
            "count": len(s),
            "stats": self._describe(s),
            "common_fps": {str(k): int(v) for k, v in common.items()},
        }

    # ── Resolution ──────────────────────────────────────────

    def analyze_resolution(self) -> Dict[str, Any]:
        """Height × width distribution and common resolutions."""
        if not self.has_columns(["height", "width"]):
            return {"available": False}
        df = self.df[["height", "width"]].dropna()
        if df.empty:
            return {"available": True, "count": 0}

        # Common resolutions
        df = df.copy()
        df["resolution"] = df["height"].astype(int).astype(str) + "×" + df["width"].astype(int).astype(str)
        common = df["resolution"].value_counts().head(10)

        # Aspect ratios (guard against zero height)
        valid_h = df[df["height"] > 0].copy()
        ar_stats = {}
        if not valid_h.empty:
            valid_h["aspect_ratio"] = valid_h["width"] / valid_h["height"]
            ar_stats = self._describe(valid_h["aspect_ratio"])

        return {
            "available": True,
            "count": len(df),
            "height_stats": self._describe(df["height"]),
            "width_stats": self._describe(df["width"]),
            "aspect_ratio_stats": ar_stats,
            "common_resolutions": {str(k): int(v) for k, v in common.items()},
        }

    # ── Frame count ─────────────────────────────────────────

    def analyze_frame_count(self) -> Dict[str, Any]:
        """Frame count distribution."""
        if "no_frames" not in self.df.columns:
            return {"available": False}
        s = self.df["no_frames"].dropna()
        if s.empty:
            return {"available": True, "count": 0}
        return {
            "available": True,
            "count": len(s),
            "stats": self._describe(s),
        }

    # ── File size ───────────────────────────────────────────

    def analyze_file_size(self) -> Dict[str, Any]:
        """File size distribution and total storage."""
        if "file_size_bytes" not in self.df.columns:
            return {"available": False}
        s = self.df["file_size_bytes"].dropna()
        if s.empty:
            return {"available": True, "count": 0}
        return {
            "available": True,
            "count": len(s),
            "stats": self._describe(s),
            "total_gb": round(s.sum() / (1024 ** 3), 2),
            "mean_mb": round(s.mean() / (1024 ** 2), 2),
        }

    # ── Codec / channels ───────────────────────────────────

    def analyze_codec_channels(self) -> Dict[str, Any]:
        """Codec and channel distributions."""
        result: Dict[str, Any] = {}
        for col in ["codec", "channels"]:
            if col in self.df.columns:
                counts = self.df[col].value_counts(dropna=False)
                result[col] = {str(k): int(v) for k, v in counts.items()}
            else:
                result[col] = {"available": False}
        return result

    # ── Image subset ────────────────────────────────────────

    def analyze_image_subset(self) -> Dict[str, Any]:
        """Separate analysis for image-only records."""
        # Identify images
        if "media_type" in self.df.columns:
            mask = self.df["media_type"].astype(str).str.lower() == "image"
        elif "fps" in self.df.columns and "duration" in self.df.columns:
            mask = self.df["fps"].isna() & self.df["duration"].isna()
        else:
            return {"available": False}

        images = self.df[mask]
        if images.empty:
            return {"available": True, "count": 0}

        result: Dict[str, Any] = {
            "available": True,
            "count": len(images),
        }
        # Resolution stats for images
        if self.has_columns(["height", "width"]):
            h = images["height"].dropna()
            w = images["width"].dropna()
            if not h.empty:
                result["height_stats"] = self._describe(h)
                result["width_stats"] = self._describe(w)
        # File size for images
        if "file_size_bytes" in images.columns:
            s = images["file_size_bytes"].dropna()
            if not s.empty:
                result["file_size_stats"] = self._describe(s)

        return result

    # ── Outlier detection ───────────────────────────────────

    def detect_outliers(self) -> Dict[str, Any]:
        """Detect extreme values using IQR method."""
        outliers: Dict[str, Any] = {}
        for col in ["duration", "fps", "no_frames", "file_size_bytes"]:
            if col not in self.df.columns:
                continue
            s = self.df[col].dropna()
            if s.empty or len(s) < 4:
                # Need at least a few values for meaningful IQR
                continue
            q1 = s.quantile(0.25)
            q3 = s.quantile(0.75)
            iqr = q3 - q1
            if iqr == 0:
                # All values are the same — no outliers
                outliers[col] = {
                    "count": 0,
                    "percentage": 0.0,
                    "note": "Zero IQR (constant values)",
                }
                continue
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outlier_mask = (s < lower) | (s > upper)
            outliers[col] = {
                "count": int(outlier_mask.sum()),
                "percentage": round(outlier_mask.mean() * 100, 2),
                "lower_bound": round(float(lower), 2),
                "upper_bound": round(float(upper), 2),
                "min_outlier": round(float(s[outlier_mask].min()), 2) if outlier_mask.any() else None,
                "max_outlier": round(float(s[outlier_mask].max()), 2) if outlier_mask.any() else None,
            }
        return outliers

    # ── Per-dataset ─────────────────────────────────────────

    def analyze_per_dataset(self) -> Dict[str, Dict[str, Any]]:
        """Repeat key analyses per dataset."""
        results = {}
        for ds in self.iter_datasets():
            ds_df = self.get_dataset_df(ds)
            ds_result: Dict[str, Any] = {"total": len(ds_df)}
            for col in ["duration", "fps", "no_frames", "file_size_bytes"]:
                if col in ds_df.columns:
                    s = ds_df[col].dropna()
                    if not s.empty:
                        ds_result[col] = self._describe(s)
            results[ds] = ds_result
        return results

    # ── Helpers ─────────────────────────────────────────────

    @staticmethod
    def _describe(s: pd.Series) -> Dict[str, float]:
        """Return descriptive statistics as a dict.

        Handles edge cases:
        - Single-value series: skewness/kurtosis return NaN in
          pandas, so we catch that and return 0.0.
        - Empty series: returns empty dict (caller should check).
        """
        if s.empty:
            return {}

        desc = s.describe()
        result = {k: round(float(v), 4) for k, v in desc.items()}

        # Skewness and kurtosis may be NaN for single-value series
        try:
            skew_val = float(s.skew())
            result["skewness"] = round(skew_val, 4) if np.isfinite(skew_val) else 0.0
        except (ValueError, TypeError):
            result["skewness"] = 0.0

        try:
            kurt_val = float(s.kurtosis())
            result["kurtosis"] = round(kurt_val, 4) if np.isfinite(kurt_val) else 0.0
        except (ValueError, TypeError):
            result["kurtosis"] = 0.0

        # Quartiles explicitly
        result["q1"] = round(float(s.quantile(0.25)), 4)
        result["median"] = round(float(s.median()), 4)
        result["q3"] = round(float(s.quantile(0.75)), 4)
        return result
