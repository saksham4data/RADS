# ─────────────────────────────────────────────────────────────
# Plotting Engine — Dark Professional Theme
# ─────────────────────────────────────────────────────────────
"""
Provides a consistent dark professional visual theme, a
curated colour palette, and reusable plot helper functions.

Compatibility notes
~~~~~~~~~~~~~~~~~~~
* ``safe_boxplot()`` auto-detects the matplotlib version and
  uses ``tick_labels`` (≥3.9) or ``labels`` (<3.9) accordingly.
* All plot helpers guard against empty data and return the
  axes object unchanged when there is nothing to plot.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from packaging.version import Version

logger = logging.getLogger(__name__)

# ── Matplotlib version ──────────────────────────────────────

_MPL_VERSION = Version(mpl.__version__)
_MPL_HAS_TICK_LABELS = _MPL_VERSION >= Version("3.9")

# ── Colour Palette ──────────────────────────────────────────

PALETTE = {
    "primary": "#6C63FF",      # Vivid indigo
    "secondary": "#FF6584",    # Warm coral
    "accent1": "#43E97B",      # Mint green
    "accent2": "#F9D423",      # Amber
    "accent3": "#38F9D7",      # Teal
    "accent4": "#FA709A",      # Rose
    "accent5": "#A18CD1",      # Lavender
    "accent6": "#FBC2EB",      # Soft pink
    "bg_dark": "#1A1A2E",      # Dark navy
    "bg_card": "#16213E",      # Card background
    "bg_plot": "#0F3460",      # Plot area
    "text": "#E8E8E8",         # Light text
    "text_muted": "#A0A0B0",   # Muted text
    "grid": "#2A2A4A",         # Grid lines
    "success": "#43E97B",
    "warning": "#F9D423",
    "error": "#FF6584",
}

# Ordered list for bar charts / categorical plots
CATEGORICAL_COLOURS = [
    PALETTE["primary"],
    PALETTE["secondary"],
    PALETTE["accent1"],
    PALETTE["accent2"],
    PALETTE["accent3"],
    PALETTE["accent4"],
    PALETTE["accent5"],
    PALETTE["accent6"],
    "#667EEA",  # Extra: blue
    "#764BA2",  # Extra: purple
    "#F093FB",  # Extra: pink
    "#4FACFE",  # Extra: sky blue
]

# Per-dataset colour mapping
DATASET_COLOURS = {
    "picek": PALETTE["primary"],
    "tudat": PALETTE["accent1"],
    "kaggle_images": PALETTE["accent2"],
}


# ── Theme Setup ─────────────────────────────────────────────

class PlotEngine:
    """Manages plot styling and provides helper functions."""

    def __init__(self, dpi: int = 300, formats: Optional[List[str]] = None) -> None:
        self.dpi = dpi
        self.formats = formats or ["png", "svg"]
        self._apply_theme()

    def _apply_theme(self) -> None:
        """Apply the dark professional theme globally."""
        # Use seaborn darkgrid as base
        sns.set_theme(style="darkgrid")

        # Override with custom colours
        mpl.rcParams.update({
            # Figure
            "figure.facecolor": PALETTE["bg_dark"],
            "figure.edgecolor": PALETTE["bg_dark"],
            "figure.dpi": self.dpi,
            "figure.figsize": (12, 7),
            # Axes
            "axes.facecolor": PALETTE["bg_card"],
            "axes.edgecolor": PALETTE["grid"],
            "axes.labelcolor": PALETTE["text"],
            "axes.titlecolor": PALETTE["text"],
            "axes.grid": True,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "axes.titleweight": "bold",
            # Grid
            "grid.color": PALETTE["grid"],
            "grid.alpha": 0.3,
            "grid.linewidth": 0.5,
            # Ticks
            "xtick.color": PALETTE["text_muted"],
            "ytick.color": PALETTE["text_muted"],
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            # Text
            "text.color": PALETTE["text"],
            "font.size": 11,
            # Legend
            "legend.facecolor": PALETTE["bg_card"],
            "legend.edgecolor": PALETTE["grid"],
            "legend.fontsize": 10,
            # Savefig
            "savefig.facecolor": PALETTE["bg_dark"],
            "savefig.edgecolor": "none",
            "savefig.bbox": "tight",
        })

    # ── Helper: new figure ──────────────────────────────────

    @staticmethod
    def create_figure(
        nrows: int = 1,
        ncols: int = 1,
        figsize: Optional[Tuple[float, float]] = None,
        **kwargs: Any,
    ) -> Tuple[plt.Figure, Any]:
        """Create a figure with the dark theme applied."""
        figsize = figsize or (12, 7)
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, **kwargs)
        fig.set_facecolor(PALETTE["bg_dark"])
        return fig, axes

    # ── Safe boxplot ────────────────────────────────────────

    @staticmethod
    def safe_boxplot(
        ax: plt.Axes,
        data: Sequence,
        labels: Sequence[str],
        **kwargs: Any,
    ) -> Optional[Dict[str, Any]]:
        """Version-compatible boxplot wrapper.

        Uses ``tick_labels`` on matplotlib ≥3.9, ``labels`` on
        older versions.  Gracefully handles empty data.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Target axes.
        data : sequence of arrays
            Data to plot (list of arrays, one per box).
        labels : sequence of str
            Tick labels for each box.
        **kwargs
            Additional keyword arguments passed to ``ax.boxplot``.

        Returns
        -------
        dict or None
            The boxplot result dict, or None if data is empty.
        """
        # Guard: empty data
        if not data or len(data) == 0:
            logger.warning("safe_boxplot: no data to plot")
            return None

        # Filter out empty arrays
        filtered_data = []
        filtered_labels = []
        for d, lbl in zip(data, labels):
            arr = np.asarray(d)
            if len(arr) > 0:
                filtered_data.append(arr)
                filtered_labels.append(lbl)

        if not filtered_data:
            logger.warning("safe_boxplot: all data arrays are empty")
            return None

        # Version-compatible label argument
        if _MPL_HAS_TICK_LABELS:
            kwargs["tick_labels"] = filtered_labels
        else:
            kwargs["labels"] = filtered_labels

        return ax.boxplot(filtered_data, **kwargs)

    # ── Plot helpers ────────────────────────────────────────

    @staticmethod
    def plot_distribution(
        data: pd.Series,
        *,
        title: str,
        xlabel: str,
        ylabel: str = "Count",
        bins: int = 30,
        color: Optional[str] = None,
        ax: Optional[plt.Axes] = None,
    ) -> Optional[plt.Axes]:
        """Histogram with KDE overlay.

        Returns None if data is empty or all-NaN.
        """
        clean = data.dropna()
        if clean.empty:
            logger.warning(
                "plot_distribution: no data for '%s'", title
            )
            return ax

        # Single-value series: disable KDE (would fail)
        use_kde = clean.nunique() > 1

        if ax is None:
            _, ax = plt.subplots(figsize=(12, 7))
        sns.histplot(
            clean,
            bins=bins,
            color=color or PALETTE["primary"],
            edgecolor=PALETTE["bg_dark"],
            alpha=0.8,
            kde=use_kde,
            ax=ax,
        )
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        return ax

    @staticmethod
    def plot_bar_chart(
        data: pd.Series,
        *,
        title: str,
        xlabel: str,
        ylabel: str = "Count",
        horizontal: bool = False,
        palette: Optional[List[str]] = None,
        ax: Optional[plt.Axes] = None,
    ) -> Optional[plt.Axes]:
        """Bar chart for categorical data.

        Returns None if data is empty.
        """
        clean = data.dropna()
        if clean.empty:
            logger.warning("plot_bar_chart: no data for '%s'", title)
            return ax

        if ax is None:
            _, ax = plt.subplots(figsize=(12, 7))

        counts = clean.value_counts()
        colours = (palette or CATEGORICAL_COLOURS)[: len(counts)]

        if horizontal:
            ax.barh(counts.index.astype(str), counts.values, color=colours)
            ax.set_xlabel(ylabel)
            ax.set_ylabel(xlabel)
        else:
            ax.bar(counts.index.astype(str), counts.values, color=colours)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            # Use axis-level tick rotation (not global plt.xticks)
            ax.tick_params(axis="x", rotation=45)
            for label in ax.get_xticklabels():
                label.set_ha("right")

        ax.set_title(title, fontweight="bold")
        return ax

    @staticmethod
    def plot_heatmap(
        data: pd.DataFrame,
        *,
        title: str,
        annot: bool = True,
        fmt: str = ".1f",
        cmap: str = "magma",
        ax: Optional[plt.Axes] = None,
    ) -> Optional[plt.Axes]:
        """Heatmap for correlation or cross-tabulation.

        Returns the axes unchanged if data is empty.
        """
        if data.empty:
            logger.warning("plot_heatmap: empty DataFrame for '%s'", title)
            return ax

        if ax is None:
            _, ax = plt.subplots(figsize=(14, 10))
        sns.heatmap(
            data,
            annot=annot,
            fmt=fmt,
            cmap=cmap,
            linewidths=0.5,
            linecolor=PALETTE["grid"],
            ax=ax,
            cbar_kws={"shrink": 0.8},
        )
        ax.set_title(title, fontweight="bold")
        return ax

    @staticmethod
    def plot_correlation(
        df: pd.DataFrame,
        *,
        title: str = "Feature Correlation Matrix",
        ax: Optional[plt.Axes] = None,
    ) -> Optional[plt.Axes]:
        """Correlation heatmap for numeric columns.

        Guards against <2 numeric columns.
        """
        numeric = df.select_dtypes(include="number")
        if numeric.shape[1] < 2:
            logger.warning(
                "plot_correlation: need ≥2 numeric columns, got %d",
                numeric.shape[1],
            )
            return ax

        corr = numeric.corr()
        if corr.empty:
            return ax

        if ax is None:
            _, ax = plt.subplots(figsize=(14, 10))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(
            corr,
            mask=mask,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            center=0,
            linewidths=0.5,
            linecolor=PALETTE["grid"],
            ax=ax,
            vmin=-1, vmax=1,
            cbar_kws={"shrink": 0.8},
        )
        ax.set_title(title, fontweight="bold")
        return ax

    @staticmethod
    def plot_bbox_heatmap(
        center_x: pd.Series,
        center_y: pd.Series,
        *,
        title: str = "Bounding Box Centre Distribution",
        bins: int = 50,
        ax: Optional[plt.Axes] = None,
    ) -> Optional[plt.Axes]:
        """2D histogram heatmap of bounding box centres.

        Guards against empty data.
        """
        valid = pd.DataFrame({"cx": center_x, "cy": center_y}).dropna()
        if valid.empty:
            logger.warning("plot_bbox_heatmap: no valid data")
            return ax

        if ax is None:
            _, ax = plt.subplots(figsize=(10, 8))
        h = ax.hist2d(
            valid["cx"], valid["cy"],
            bins=bins,
            cmap="inferno",
            range=[[0, 1], [0, 1]],
        )
        plt.colorbar(h[3], ax=ax, label="Count")
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Centre X (normalised)")
        ax.set_ylabel("Centre Y (normalised)")
        ax.invert_yaxis()
        return ax

    @staticmethod
    def plot_quality_score_breakdown(
        components: Dict[str, float],
        overall: float,
        grade: str,
        *,
        ax: Optional[plt.Axes] = None,
    ) -> Optional[plt.Axes]:
        """Horizontal bar chart showing quality score components.

        Guards against empty components.
        """
        if not components:
            logger.warning("plot_quality_score_breakdown: no components")
            return ax

        if ax is None:
            _, ax = plt.subplots(figsize=(10, 6))
        names = list(components.keys())
        scores = [components[n] * 100 for n in names]
        colours = [
            PALETTE["success"] if s >= 80
            else PALETTE["warning"] if s >= 60
            else PALETTE["error"]
            for s in scores
        ]
        ax.barh(names, scores, color=colours, edgecolor=PALETTE["bg_dark"])
        ax.set_xlim(0, 100)
        ax.set_xlabel("Score (0–100)")
        ax.set_title(
            f"Dataset Quality Score: {overall:.1f} ({grade})",
            fontweight="bold",
        )
        # Annotate bars
        for i, (s, n) in enumerate(zip(scores, names)):
            ax.text(s + 1, i, f"{s:.1f}", va="center", color=PALETTE["text"])
        return ax

    # ── Per-dataset helpers ─────────────────────────────────

    @staticmethod
    def get_dataset_color(name: str) -> str:
        """Return the assigned colour for a dataset."""
        return DATASET_COLOURS.get(name, PALETTE["accent5"])
