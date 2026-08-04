# ─────────────────────────────────────────────────────────────
# Report Generator — Markdown + JSON
# ─────────────────────────────────────────────────────────────
"""
Generates both human-readable Markdown reports and
machine-readable summary.json files for each notebook.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from tabulate import tabulate


class ReportGenerator:
    """Builds Markdown reports and summary JSON files.

    Usage::

        rg = ReportGenerator(
            notebook_name="01_dataset_overview",
            eda_version="1.0.0",
            metadata_hash="sha256:abc...",
            git_commit="a1b2c3d",
            seed=42,
        )
        rg.add_heading("Dataset Overview")
        rg.add_paragraph("This report covers ...")
        rg.add_table(df, "Key Statistics")
        rg.add_figure("missing_values_heatmap.png", "Missing Values")
        rg.add_stat("total_rows", 5338)

        md_content = rg.build_markdown()
        json_content = rg.build_summary_json()
    """

    def __init__(
        self,
        notebook_name: str,
        *,
        eda_version: str = "1.0.0",
        dataset_version: str = "unknown",
        metadata_hash: Optional[str] = None,
        git_commit: Optional[str] = None,
        seed: int = 42,
    ) -> None:
        self.notebook_name = notebook_name
        self.eda_version = eda_version
        self.dataset_version = dataset_version
        self.metadata_hash = metadata_hash
        self.git_commit = git_commit
        self.seed = seed
        self.timestamp = datetime.now(timezone.utc)

        # Accumulate sections
        self._sections: List[str] = []
        self._stats: Dict[str, Any] = {}
        self._tables: Dict[str, List[Dict[str, Any]]] = {}
        self._figures: List[Dict[str, str]] = []

    # ── Markdown building blocks ────────────────────────────

    def add_heading(self, text: str, level: int = 2) -> None:
        """Add a Markdown heading."""
        prefix = "#" * level
        self._sections.append(f"\n{prefix} {text}\n")

    def add_paragraph(self, text: str) -> None:
        """Add a paragraph of text."""
        self._sections.append(f"\n{text}\n")

    def add_table(
        self,
        df: pd.DataFrame,
        title: Optional[str] = None,
        *,
        key: Optional[str] = None,
    ) -> None:
        """Add a formatted table from a DataFrame."""
        if title:
            self._sections.append(f"\n### {title}\n")
        md_table = tabulate(
            df.head(100),  # Cap at 100 rows for readability
            headers="keys",
            tablefmt="pipe",
            showindex=False,
            floatfmt=".4f",
        )
        self._sections.append(f"\n{md_table}\n")

        # Also store for JSON
        store_key = key or (title or "table").lower().replace(" ", "_")
        self._tables[store_key] = df.to_dict(orient="records")

    def add_figure(self, filename: str, caption: str) -> None:
        """Add a figure reference."""
        self._sections.append(f"\n![{caption}]({filename})\n")
        self._figures.append({"filename": filename, "caption": caption})

    def add_stat(self, key: str, value: Any) -> None:
        """Record a statistic for the summary JSON."""
        self._stats[key] = value

    def add_stats(self, stats: Dict[str, Any]) -> None:
        """Record multiple statistics at once."""
        self._stats.update(stats)

    def add_key_value_list(
        self,
        items: Dict[str, Any],
        title: Optional[str] = None,
    ) -> None:
        """Add a formatted key-value list."""
        if title:
            self._sections.append(f"\n### {title}\n")
        lines = [f"- **{k}:** {v}" for k, v in items.items()]
        self._sections.append("\n".join(lines) + "\n")

    def add_warning(self, text: str) -> None:
        """Add a warning callout."""
        self._sections.append(f"\n> ⚠️ **Warning:** {text}\n")

    def add_finding(self, text: str) -> None:
        """Add a key finding callout."""
        self._sections.append(f"\n> 📊 **Finding:** {text}\n")

    # ── Build outputs ───────────────────────────────────────

    def build_markdown(self) -> str:
        """Generate the complete Markdown report."""
        header = (
            f"# {self.notebook_name.replace('_', ' ').title()}\n\n"
            f"*Generated: {self.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}*\n\n"
            f"| Metadata | Value |\n"
            f"|---|---|\n"
            f"| EDA Version | {self.eda_version} |\n"
            f"| Dataset Version | {self.dataset_version} |\n"
            f"| Metadata Hash | `{(self.metadata_hash or 'N/A')[:24]}...` |\n"
            f"| Git Commit | `{self.git_commit or 'N/A'}` |\n"
            f"| Seed | {self.seed} |\n"
            f"\n---\n"
        )
        return header + "\n".join(self._sections)

    def build_summary_json(self) -> Dict[str, Any]:
        """Generate the machine-readable summary dictionary."""
        return {
            "notebook": self.notebook_name,
            "timestamp": self.timestamp.isoformat(),
            "reproducibility": {
                "eda_version": self.eda_version,
                "dataset_version": self.dataset_version,
                "metadata_hash": self.metadata_hash,
                "git_commit": self.git_commit,
                "seed": self.seed,
            },
            "statistics": self._stats,
            "tables": {
                k: v[:20] if len(v) > 20 else v  # Cap for JSON size
                for k, v in self._tables.items()
            },
            "figures": self._figures,
        }
