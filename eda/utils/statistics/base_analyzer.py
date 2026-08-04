# ─────────────────────────────────────────────────────────────
# Base Analyzer — Abstract Interface
# ─────────────────────────────────────────────────────────────
"""
Abstract base class for all EDA analyzers. Provides a
consistent interface, column validation, serialisation,
and crash-proof execution via ``safe_analyze()``.

Defensive design
~~~~~~~~~~~~~~~~
Every analyzer *must* implement ``analyze()`` which returns a
results dictionary.  The base class provides ``safe_analyze()``
which wraps ``analyze()`` in a try/except and guarantees a
well-formed result dict even on failure.  This ensures no
single analyzer can crash the entire EDA pipeline.
"""

from __future__ import annotations

import json
import logging
import traceback
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from eda.utils.config import EDAConfig

logger = logging.getLogger(__name__)


class BaseAnalyzer(ABC):
    """Abstract base for EDA analyzers.

    Subclasses must implement ``analyze()`` which returns a
    results dictionary.  The base class provides column
    validation, per-dataset iteration, serialisation, and
    crash-proof execution.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        config: EDAConfig,
        *,
        name: Optional[str] = None,
    ) -> None:
        self.df = df
        self.config = config
        self.name = name or self.__class__.__name__
        self._results: Optional[Dict[str, Any]] = None
        self._warnings: List[str] = []

    # ── Abstract ────────────────────────────────────────────

    @abstractmethod
    def _analyze(self) -> Dict[str, Any]:
        """Subclasses implement this method for actual computation."""
        ...

    # ── Safe execution ──────────────────────────────────────

    def analyze(self) -> Dict[str, Any]:
        """Run ``_analyze()`` with crash protection.

        If ``_analyze()`` raises an exception, this method catches
        it, logs the error, and returns a well-formed result dict
        with ``available=False`` and the error details.  This
        guarantees the overall EDA pipeline continues even if one
        analyzer fails.

        Returns
        -------
        dict
            The analysis results, always containing at least
            ``available`` and ``warnings`` keys.
        """
        try:
            results = self._analyze()
            # Inject accumulated warnings
            if self._warnings:
                results.setdefault("warnings", [])
                results["warnings"].extend(self._warnings)
            self._results = results
            return results
        except Exception as exc:
            tb = traceback.format_exc()
            logger.error(
                "%s crashed during _analyze(): %s\n%s",
                self.name, exc, tb,
            )
            err_res = {
                "available": False,
                "reason": f"{self.name} failed: {exc}",
                "error_type": type(exc).__name__,
                "warnings": list(self._warnings),
            }
            self._results = err_res
            return err_res

    def safe_analyze(self) -> Dict[str, Any]:
        """Legacy alias for ``analyze()``."""
        return self.analyze()

    # ── Warning helpers ─────────────────────────────────────

    def add_warning(self, msg: str) -> None:
        """Accumulate a warning message for the results."""
        self._warnings.append(msg)
        logger.warning("%s: %s", self.name, msg)

    def _safe_result(
        self,
        available: bool,
        reason: str = "",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Build a consistent result dict.

        Parameters
        ----------
        available : bool
            Whether the analysis is available / succeeded.
        reason : str
            Explanation when ``available`` is False.
        **kwargs
            Additional keys to include in the result.
        """
        result: Dict[str, Any] = {"available": available}
        if not available and reason:
            result["reason"] = reason
        if self._warnings:
            result["warnings"] = list(self._warnings)
        result.update(kwargs)
        return result

    # ── Column helpers ──────────────────────────────────────

    def validate_columns(self, required: List[str]) -> bool:
        """Return True if all required columns are present."""
        missing = [c for c in required if c not in self.df.columns]
        if missing:
            logger.warning(
                "%s: missing required columns %s — skipping related analyses.",
                self.name, missing,
            )
            return False
        return True

    def has_columns(self, columns: List[str]) -> bool:
        """Return True if ALL columns are present (no warning)."""
        return all(c in self.df.columns for c in columns)

    def available_columns(self, candidates: List[str]) -> List[str]:
        """Return the subset of *candidates* that exist in the DataFrame."""
        return [c for c in candidates if c in self.df.columns]

    # ── Per-dataset iteration ───────────────────────────────

    def iter_datasets(self) -> List[str]:
        """Return sorted list of unique dataset names."""
        if "dataset_name" in self.df.columns:
            return sorted(
                self.df["dataset_name"].dropna().unique().tolist()
            )
        return []

    def get_dataset_df(self, name: str) -> pd.DataFrame:
        """Return rows for a specific dataset."""
        return self.df[self.df["dataset_name"] == name]

    # ── Serialisation ───────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        """Return cached results (calls ``analyze()`` if needed)."""
        if self._results is None:
            self._results = self.analyze()
        return self._results

    def to_dataframe(self, key: Optional[str] = None) -> pd.DataFrame:
        """Convert results (or a sub-key) to a DataFrame.

        Works best when the value is a list of dicts or a flat dict.
        """
        results = self.to_dict()
        data = results.get(key) if key else results
        if isinstance(data, pd.DataFrame):
            return data
        if isinstance(data, list):
            return pd.DataFrame(data)
        if isinstance(data, dict):
            return pd.DataFrame([data])
        raise ValueError(f"Cannot convert {type(data)} to DataFrame")

    def export_csv(self, path: Path, key: Optional[str] = None) -> Path:
        """Export results as CSV."""
        df = self.to_dataframe(key)
        df.to_csv(path, index=False, encoding="utf-8")
        return path

    def export_json(self, path: Path) -> Path:
        """Export results as JSON."""
        path.write_text(
            json.dumps(self.to_dict(), indent=2, default=str),
            encoding="utf-8",
        )
        return path
