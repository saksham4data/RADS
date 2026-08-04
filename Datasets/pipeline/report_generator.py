"""
Report, statistics, and processing manifest generation.

Generates:
- dataset_statistics.json — machine-readable aggregate metrics
- dataset_report.md — human-readable summary with tables
- processing_manifest.json — reproducibility snapshot (config + hashes)
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from pipeline.config import PipelineConfig
from pipeline.utils import compute_file_hash, ensure_dir


def generate_statistics(
    df: pd.DataFrame,
    config: PipelineConfig,
    duplicates_found: int = 0,
) -> Dict[str, Any]:
    """
    Compute aggregate dataset statistics from the unified metadata DataFrame.

    Args:
        df: The master metadata DataFrame (all rows, including invalid).
        config: Pipeline configuration.
        duplicates_found: Number of duplicate video groups detected.

    Returns:
        JSON-serializable dictionary of dataset statistics.
    """
    stats: Dict[str, Any] = {
        "pipeline_version": config.pipeline_version,
        "dataset_version": config.dataset_version,
        "dataset_name": config.dataset_name,
        "processed_at": datetime.now(timezone.utc).isoformat(),
        "video_mode": config.video_mode,
        "total_samples": len(df),
        "real_samples": int((df["source_type"] == "real").sum()),
        "synthetic_samples": int((df["source_type"] == "synthetic").sum()),
    }

    # ── Validation summary ───────────────────────────────────────────────
    if "validation_status" in df.columns:
        stats["validation_summary"] = _safe_value_counts(df, "validation_status")

    # ── Processing summary ───────────────────────────────────────────────
    if "processing_status" in df.columns:
        stats["processing_summary"] = _safe_value_counts(df, "processing_status")

    # ── Preprocessing status summary ─────────────────────────────────────
    if "preprocessing_status" in df.columns:
        stats["preprocessing_status_summary"] = _safe_value_counts(df, "preprocessing_status")

    # ── Accident type distribution ───────────────────────────────────────
    if "type" in df.columns:
        stats["by_accident_type"] = _safe_value_counts(df, "type")

    # ── Weather distribution ─────────────────────────────────────────────
    if "weather" in df.columns:
        stats["by_weather"] = _safe_value_counts(df, "weather")

    # ── Source × Type cross-tab ──────────────────────────────────────────
    if "type" in df.columns and "source_type" in df.columns:
        cross = pd.crosstab(df["source_type"], df["type"]).to_dict(orient="index")
        stats["source_type_by_accident"] = {
            src: {k: int(v) for k, v in types.items()}
            for src, types in cross.items()
        }

    # ── Split distribution ───────────────────────────────────────────────
    if "split" in df.columns:
        stats["split_distribution"] = _safe_value_counts(df, "split")

    # ── Resolution distribution ──────────────────────────────────────────
    if "height" in df.columns and "width" in df.columns:
        valid_res = df.dropna(subset=["height", "width"])
        if not valid_res.empty:
            res_series = (
                valid_res["height"].astype(int).astype(str)
                + "x"
                + valid_res["width"].astype(int).astype(str)
            )
            stats["video_resolution_distribution"] = res_series.value_counts().to_dict()

    # ── Duration statistics ──────────────────────────────────────────────
    if "duration" in df.columns:
        dur = pd.to_numeric(df["duration"], errors="coerce").dropna()
        if len(dur) > 0:
            stats["duration_stats"] = {
                "min": round(float(dur.min()), 3),
                "max": round(float(dur.max()), 3),
                "mean": round(float(dur.mean()), 3),
                "median": round(float(dur.median()), 3),
                "std": round(float(dur.std()), 3),
            }

    # ── FPS statistics ───────────────────────────────────────────────────
    if "fps" in df.columns:
        fps = pd.to_numeric(df["fps"], errors="coerce").dropna()
        fps = fps[fps > 0]
        if len(fps) > 0:
            stats["fps_stats"] = {
                "min": round(float(fps.min()), 3),
                "max": round(float(fps.max()), 3),
                "mean": round(float(fps.mean()), 3),
                "unique_values": sorted(fps.unique().tolist()),
            }

    # ── Total file size ──────────────────────────────────────────────────
    if "file_size_bytes" in df.columns:
        total_bytes = int(pd.to_numeric(df["file_size_bytes"], errors="coerce").sum())
        stats["total_size_bytes"] = total_bytes
        stats["total_size_gb"] = round(total_bytes / (1024 ** 3), 2)

    # ── Duplicates ───────────────────────────────────────────────────────
    stats["duplicates_found"] = duplicates_found
    if "is_duplicate" in df.columns:
        stats["duplicate_samples"] = int(df["is_duplicate"].sum())

    return stats


def generate_report(
    df: pd.DataFrame,
    stats: Dict[str, Any],
    config: PipelineConfig,
    duplicate_groups: Optional[pd.DataFrame] = None,
) -> str:
    """
    Generate a human-readable Markdown report.

    Args:
        df: Master metadata DataFrame.
        stats: Statistics dictionary from generate_statistics().
        config: Pipeline configuration.
        duplicate_groups: DataFrame of duplicate groups (from find_duplicates).

    Returns:
        Markdown string.
    """
    lines: List[str] = []

    lines.append(f"# {config.dataset_name.title()} Dataset — Processing Report")
    lines.append("")
    lines.append(f"**Generated**: {stats.get('processed_at', 'N/A')}")
    lines.append(f"**Pipeline Version**: {config.pipeline_version}")
    lines.append(f"**Dataset Version**: {config.dataset_version}")
    lines.append(f"**Video Mode**: {config.video_mode}")
    lines.append("")

    # ── Overview ─────────────────────────────────────────────────────────
    lines.append("## Overview")
    lines.append("")
    lines.append("| Metric | Count |")
    lines.append("|--------|-------|")
    lines.append(f"| Total Samples | {stats.get('total_samples', 0)} |")
    lines.append(f"| Real Videos | {stats.get('real_samples', 0)} |")
    lines.append(f"| Synthetic Videos | {stats.get('synthetic_samples', 0)} |")
    if "total_size_gb" in stats:
        lines.append(f"| Total Size | {stats['total_size_gb']} GB |")
    lines.append(f"| Duplicates Found | {stats.get('duplicates_found', 0)} groups |")
    lines.append("")

    # ── Validation Summary ───────────────────────────────────────────────
    _add_dict_table(lines, "Validation Summary", stats.get("validation_summary"))

    # ── Processing Summary ───────────────────────────────────────────────
    _add_dict_table(lines, "Processing Summary", stats.get("processing_summary"))

    # ── Preprocessing Status ─────────────────────────────────────────────
    _add_dict_table(lines, "Preprocessing Status", stats.get("preprocessing_status_summary"))

    # ── Accident Type Distribution ───────────────────────────────────────
    _add_dict_table(lines, "Accident Type Distribution", stats.get("by_accident_type"), sort_desc=True)

    # ── Weather Distribution ─────────────────────────────────────────────
    _add_dict_table(lines, "Weather Conditions", stats.get("by_weather"), sort_desc=True)

    # ── Split Distribution ───────────────────────────────────────────────
    _add_dict_table(lines, "Split Distribution", stats.get("split_distribution"))

    # ── Duration Stats ───────────────────────────────────────────────────
    dur_stats = stats.get("duration_stats")
    if dur_stats:
        lines.append("## Video Duration Statistics (seconds)")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        for k, v in dur_stats.items():
            lines.append(f"| {k} | {v} |")
        lines.append("")

    # ── FPS Stats ────────────────────────────────────────────────────────
    fps_stats = stats.get("fps_stats")
    if fps_stats:
        lines.append("## FPS Statistics")
        lines.append("")
        lines.append(f"- Min: {fps_stats['min']}")
        lines.append(f"- Max: {fps_stats['max']}")
        lines.append(f"- Mean: {fps_stats['mean']}")
        lines.append(f"- Unique values: {fps_stats['unique_values']}")
        lines.append("")

    # ── Resolution Distribution ──────────────────────────────────────────
    _add_dict_table(lines, "Video Resolution Distribution",
                    stats.get("video_resolution_distribution"), sort_desc=True,
                    col1="Resolution (HxW)", col2="Count")

    # ── Duplicates ───────────────────────────────────────────────────────
    if duplicate_groups is not None and len(duplicate_groups) > 0:
        lines.append("## Duplicate Videos")
        lines.append("")
        lines.append(f"Found **{len(duplicate_groups)}** duplicate group(s):")
        lines.append("")
        lines.append("| Group # | Count | Paths |")
        lines.append("|---------|-------|-------|")
        for i, row in duplicate_groups.head(30).iterrows():
            paths = ", ".join(str(p) for p in row["original_paths"][:3])
            if len(row["original_paths"]) > 3:
                paths += f" (+{len(row['original_paths']) - 3} more)"
            lines.append(f"| {i + 1} | {row['count']} | {paths} |")
        if len(duplicate_groups) > 30:
            lines.append(f"| ... | ... | ({len(duplicate_groups) - 30} more groups) |")
        lines.append("")

    # ── Invalid Entries Detail ───────────────────────────────────────────
    invalid = df[df["validation_status"] != "valid"]
    if len(invalid) > 0:
        lines.append("## Invalid Entries")
        lines.append("")
        lines.append(f"Total invalid: **{len(invalid)}**")
        lines.append("")
        lines.append("| video_id | source | status | original_path |")
        lines.append("|----------|--------|--------|---------------|")
        for _, row in invalid.head(50).iterrows():
            lines.append(
                f"| {row.get('video_id', 'N/A')} "
                f"| {row.get('source_type', '')} "
                f"| {row.get('validation_status', '')} "
                f"| {row.get('original_path', '')} |"
            )
        if len(invalid) > 50:
            lines.append(f"| ... | ... | ... | ({len(invalid) - 50} more) |")
        lines.append("")

    # ── Configuration Snapshot ────────────────────────────────────────────
    lines.append("## Pipeline Configuration")
    lines.append("")
    lines.append("```yaml")
    for key, value in config.to_dict().items():
        lines.append(f"{key}: {value}")
    lines.append("```")
    lines.append("")

    lines.append("---")
    lines.append(f"*Report generated by the {config.dataset_name} preprocessing pipeline v{config.pipeline_version}.*")
    return "\n".join(lines)


def generate_manifest(
    config: PipelineConfig,
    df: pd.DataFrame,
    stats: Dict[str, Any],
    duplicates_found: int = 0,
    output_files: Optional[Dict[str, Path]] = None,
) -> Dict[str, Any]:
    """
    Generate a processing manifest for reproducibility.

    The manifest captures everything needed to verify or reproduce a run:
    config snapshot, input file hashes, output file hashes, summary counts.

    Args:
        config: Pipeline configuration used for this run.
        df: Master metadata DataFrame.
        stats: Generated statistics dictionary.
        duplicates_found: Number of duplicate groups.
        output_files: Dict of name → path for generated output files.

    Returns:
        JSON-serializable manifest dictionary.
    """
    manifest: Dict[str, Any] = {
        "pipeline_version": config.pipeline_version,
        "dataset_version": config.dataset_version,
        "dataset_name": config.dataset_name,
        "processed_at": datetime.now(timezone.utc).isoformat(),
        "config": config.to_dict(),
    }

    # Input file checksums
    input_files = {}
    raw_dir = config.raw_dir
    for csv_name in ["metadata-real.csv", "metadata-synthetic.csv"]:
        csv_path = raw_dir / csv_name
        if csv_path.exists():
            input_files[csv_name] = {
                "sha256": compute_file_hash(csv_path, "sha256"),
                "size_bytes": csv_path.stat().st_size,
            }
    manifest["input_files"] = input_files

    # Output file checksums
    out_checksums = {}
    if output_files:
        for name, path in output_files.items():
            if path and Path(path).exists():
                out_checksums[name] = {
                    "sha256": compute_file_hash(Path(path), "sha256"),
                    "size_bytes": Path(path).stat().st_size,
                }
    manifest["output_files"] = out_checksums

    # Summary counts
    manifest["summary"] = {
        "total": len(df),
        "valid": int((df["validation_status"] == "valid").sum()) if "validation_status" in df.columns else 0,
        "invalid": int((df["validation_status"] != "valid").sum()) if "validation_status" in df.columns else 0,
        "duplicates_found": duplicates_found,
    }

    return manifest


# ── File I/O ─────────────────────────────────────────────────────────────────

def write_statistics(stats: Dict[str, Any], output_path: Path) -> None:
    """Write statistics dictionary to a JSON file."""
    ensure_dir(output_path.parent)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False, default=str)


def write_report(report_md: str, output_path: Path) -> None:
    """Write the Markdown report to a file."""
    ensure_dir(output_path.parent)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_md)


def write_manifest(manifest: Dict[str, Any], output_path: Path) -> None:
    """Write the processing manifest to a JSON file."""
    ensure_dir(output_path.parent)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False, default=str)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _safe_value_counts(df: pd.DataFrame, column: str) -> Dict[str, int]:
    """Get value counts as a plain dict with int values."""
    return {str(k): int(v) for k, v in df[column].value_counts().items()}


def _add_dict_table(
    lines: List[str],
    title: str,
    data: Optional[Dict],
    sort_desc: bool = False,
    col1: str = "Status",
    col2: str = "Count",
) -> None:
    """Append a markdown table for a simple key→value dict."""
    if not data:
        return
    lines.append(f"## {title}")
    lines.append("")
    lines.append(f"| {col1} | {col2} |")
    lines.append("|--------|-------|")
    items = data.items()
    if sort_desc:
        items = sorted(items, key=lambda x: -x[1])
    else:
        items = sorted(items)
    for k, v in items:
        lines.append(f"| {k} | {v} |")
    lines.append("")
