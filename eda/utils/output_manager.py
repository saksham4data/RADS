# ─────────────────────────────────────────────────────────────
# Output Manager — Timestamped Versioning & Manifest
# ─────────────────────────────────────────────────────────────
"""
Manages timestamped output directories, ``latest/`` references,
``manifest.json`` generation, and ``run_history.json`` updates.
"""

from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from eda.utils.config import EDAConfig


class OutputManager:
    """Creates and manages versioned output directories.

    Each EDA run gets a timestamped subdirectory under
    ``figures/``, ``reports/``, and ``exports/``.

    Usage::

        om = OutputManager(config, notebook_name="01_dataset_overview")
        om.save_figure(fig, "missing_values_heatmap")
        om.save_csv(df, "column_completeness")
        om.save_report("# Report\\n...", "dataset_overview_report")
        om.save_summary_json(stats_dict, "summary")
        om.finalize(wandb_run_id="abc", wandb_run_url="https://...")
    """

    def __init__(
        self,
        config: EDAConfig,
        notebook_name: str,
        *,
        timestamp: Optional[datetime] = None,
    ) -> None:
        self.config = config
        self.notebook_name = notebook_name
        self._ts = timestamp or datetime.now(timezone.utc)
        self._ts_str = self._ts.strftime("%Y-%m-%d_%H-%M")

        # Timestamped output dirs
        self.figures_dir = config.figures_dir / self._ts_str
        self.reports_dir = config.reports_dir / self._ts_str
        self.exports_dir = config.exports_dir / self._ts_str

        # Track exported files
        self._exported_figures: List[str] = []
        self._exported_reports: List[str] = []
        self._exported_exports: List[str] = []

        # Ensure directories exist
        for d in (self.figures_dir, self.reports_dir, self.exports_dir):
            d.mkdir(parents=True, exist_ok=True)

    # ── Save methods ────────────────────────────────────────

    def save_figure(
        self,
        fig: Any,
        name: str,
        *,
        formats: Optional[List[str]] = None,
        dpi: Optional[int] = None,
    ) -> List[Path]:
        """Save a matplotlib figure in configured formats."""
        formats = formats or self.config.figure_formats
        dpi = dpi or self.config.figure_dpi
        paths: List[Path] = []
        for fmt in formats:
            filename = f"{name}.{fmt}"
            path = self.figures_dir / filename
            fig.savefig(
                path,
                format=fmt,
                dpi=dpi if fmt != "svg" else None,
                bbox_inches="tight",
                facecolor=fig.get_facecolor(),
                edgecolor="none",
            )
            self._exported_figures.append(filename)
            paths.append(path)
        return paths

    def save_csv(self, df: pd.DataFrame, name: str) -> Path:
        """Save a DataFrame as CSV."""
        filename = f"{name}.csv"
        path = self.exports_dir / filename
        df.to_csv(path, index=False, encoding="utf-8")
        self._exported_exports.append(filename)
        return path

    def save_report(self, content: str, name: str) -> Path:
        """Save a Markdown report."""
        filename = f"{name}.md"
        path = self.reports_dir / filename
        path.write_text(content, encoding="utf-8")
        self._exported_reports.append(filename)
        return path

    def save_summary_json(self, data: Dict[str, Any], name: str = "summary") -> Path:
        """Save a machine-readable summary JSON."""
        filename = f"{name}.json"
        path = self.reports_dir / filename
        path.write_text(
            json.dumps(data, indent=2, default=str, ensure_ascii=False),
            encoding="utf-8",
        )
        self._exported_reports.append(filename)
        return path

    # ── Manifest & History ──────────────────────────────────

    def _generate_manifest(
        self,
        *,
        wandb_run_id: Optional[str] = None,
        wandb_run_url: Optional[str] = None,
        git_commit: Optional[str] = None,
        metadata_hash: Optional[str] = None,
        rows_analyzed: int = 0,
        duration_seconds: float = 0.0,
        status: str = "success",
    ) -> Dict[str, Any]:
        """Build the manifest dict for this run."""
        return {
            "timestamp": self._ts.isoformat(),
            "notebook": self.notebook_name,
            "eda_version": self.config.eda_version,
            "dataset_version": self.config._raw.get("data", {}).get(
                "dataset_version", "unknown"
            ),
            "metadata_hash": metadata_hash,
            "git_commit": git_commit,
            "seed": self.config.seed,
            "wandb_run_id": wandb_run_id,
            "wandb_run_url": wandb_run_url,
            "files": {
                "reports": self._exported_reports,
                "figures": self._exported_figures,
                "exports": self._exported_exports,
            },
            "execution": {
                "duration_seconds": round(duration_seconds, 2),
                "status": status,
                "rows_analyzed": rows_analyzed,
            },
        }

    def _write_manifest(self, manifest: Dict[str, Any]) -> None:
        """Write manifest.json into each timestamped output dir."""
        content = json.dumps(manifest, indent=2, default=str, ensure_ascii=False)
        for d in (self.figures_dir, self.reports_dir, self.exports_dir):
            (d / "manifest.json").write_text(content, encoding="utf-8")

    def _update_latest(self) -> None:
        """Update the ``latest/`` reference directories."""
        for src, parent in [
            (self.figures_dir, self.config.figures_dir),
            (self.reports_dir, self.config.reports_dir),
            (self.exports_dir, self.config.exports_dir),
        ]:
            latest = parent / "latest"
            # On Windows, symlinks may not work — use directory copy
            try:
                if latest.is_symlink() or latest.is_dir():
                    if latest.is_symlink():
                        latest.unlink()
                    else:
                        shutil.rmtree(latest)
                os.symlink(src, latest, target_is_directory=True)
            except (OSError, NotImplementedError):
                # Fallback: copy directory contents
                if latest.exists():
                    shutil.rmtree(latest)
                shutil.copytree(src, latest)

    def _update_run_history(self, manifest: Dict[str, Any]) -> None:
        """Append this run's manifest to ``run_history.json``."""
        history_path = self.config.run_history_path
        history: List[Dict[str, Any]] = []
        if history_path.is_file():
            try:
                history = json.loads(history_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, ValueError):
                history = []
        history.append(manifest)
        history_path.write_text(
            json.dumps(history, indent=2, default=str, ensure_ascii=False),
            encoding="utf-8",
        )

    def finalize(
        self,
        *,
        wandb_run_id: Optional[str] = None,
        wandb_run_url: Optional[str] = None,
        git_commit: Optional[str] = None,
        metadata_hash: Optional[str] = None,
        rows_analyzed: int = 0,
        duration_seconds: float = 0.0,
        status: str = "success",
    ) -> Dict[str, Any]:
        """Write manifest, update latest, update run_history.

        Returns the manifest dict for further use (e.g. W&B).
        """
        manifest = self._generate_manifest(
            wandb_run_id=wandb_run_id,
            wandb_run_url=wandb_run_url,
            git_commit=git_commit,
            metadata_hash=metadata_hash,
            rows_analyzed=rows_analyzed,
            duration_seconds=duration_seconds,
            status=status,
        )
        if self.config.output.get("generate_manifest", True):
            self._write_manifest(manifest)
        if self.config.output.get("maintain_latest", True):
            self._update_latest()
        if self.config.output.get("track_run_history", True):
            self._update_run_history(manifest)
        return manifest

    # ── Convenience ─────────────────────────────────────────

    @property
    def all_exported_files(self) -> List[str]:
        """All files exported during this run."""
        return (
            self._exported_figures
            + self._exported_reports
            + self._exported_exports
        )

    def get_all_output_paths(self) -> List[Path]:
        """Return absolute paths to all exported files."""
        paths: List[Path] = []
        for f in self._exported_figures:
            paths.append(self.figures_dir / f)
        for f in self._exported_reports:
            paths.append(self.reports_dir / f)
        for f in self._exported_exports:
            paths.append(self.exports_dir / f)
        return paths
