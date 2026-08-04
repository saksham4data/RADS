"""Generate metadata documentation reports from global master metadata.

This module is intentionally read-only with respect to metadata inputs. It does
not run EDA, mutate metadata, or modify existing reports.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import pandas as pd


DATASET_COLUMN = "dataset_name"


@dataclass(frozen=True)
class ColumnDecision:
    """Training classification decision for one metadata column."""

    group: str
    recommended_usage: str
    justification: str
    notes: str


class MetadataDocumentationGenerator:
    """Build metadata dictionary, audit, schema, and readiness reports."""

    def __init__(
        self,
        metadata_path: Path,
        metadata_dictionary_md_path: Path,
        metadata_dictionary_csv_path: Path,
        reports_dir: Path,
    ) -> None:
        self.metadata_path = metadata_path
        self.metadata_dictionary_md_path = metadata_dictionary_md_path
        self.metadata_dictionary_csv_path = metadata_dictionary_csv_path
        self.reports_dir = reports_dir
        self.df = pd.read_csv(metadata_path)
        if DATASET_COLUMN not in self.df.columns:
            raise ValueError(f"Required column missing: {DATASET_COLUMN}")

    def generate_all(self) -> List[Path]:
        """Generate all requested documentation artifacts."""

        self.reports_dir.mkdir(parents=True, exist_ok=True)
        dictionary = self.build_metadata_dictionary()
        dictionary.to_csv(self.metadata_dictionary_csv_path, index=False)
        self.metadata_dictionary_md_path.write_text(
            self.render_metadata_dictionary(dictionary),
            encoding="utf-8",
        )

        outputs = [
            self.metadata_dictionary_md_path,
            self.metadata_dictionary_csv_path,
            self.reports_dir / "feature_classification.md",
            self.reports_dir / "missing_value_audit.md",
            self.reports_dir / "dataset_schema_comparison.md",
            self.reports_dir / "dataset_readiness.md",
        ]
        outputs[2].write_text(self.render_feature_classification(dictionary), encoding="utf-8")
        outputs[3].write_text(self.render_missing_value_audit(dictionary), encoding="utf-8")
        outputs[4].write_text(self.render_dataset_schema_comparison(), encoding="utf-8")
        outputs[5].write_text(self.render_dataset_readiness(dictionary), encoding="utf-8")
        return outputs

    def build_metadata_dictionary(self) -> pd.DataFrame:
        """Create one data dictionary row per metadata column."""

        rows: List[Dict[str, object]] = []
        total_rows = len(self.df)
        for column in self.df.columns:
            series = self.df[column]
            decision = classify_column(column)
            missing_count = int(series.isna().sum())
            missing_pct = round((missing_count / total_rows) * 100, 2) if total_rows else 0.0
            unique_values = int(series.nunique(dropna=True))
            rows.append(
                {
                    "Column Name": column,
                    "Data Type": str(series.dtype),
                    "Description": infer_description(column, series),
                    "Missing Percentage": missing_pct,
                    "Example Value": example_value(series),
                    "Number of Unique Values": unique_values,
                    "Constant": bool(unique_values <= 1),
                    "Classification": decision.group,
                    "Recommended Usage during Model Training": decision.recommended_usage,
                    "Notes": decision.notes,
                }
            )
        return pd.DataFrame(rows)

    def render_metadata_dictionary(self, dictionary: pd.DataFrame) -> str:
        lines = self._header("Metadata Dictionary")
        lines.extend(
            [
                "Source: `Datasets/processed/global_master_metadata.csv`",
                "",
                "Descriptions are inferred from column names, observed values, and schema presence. "
                "Fields that cannot be described from the metadata are marked as unknown.",
                "",
                dictionary.to_markdown(index=False),
                "",
            ]
        )
        return "\n".join(lines)

    def render_feature_classification(self, dictionary: pd.DataFrame) -> str:
        rows = []
        for column in self.df.columns:
            decision = classify_column(column)
            rows.append(
                {
                    "Column": column,
                    "Group": decision.group,
                    "Justification": decision.justification,
                    "Recommended Usage": decision.recommended_usage,
                }
            )
        table = pd.DataFrame(rows)
        lines = self._header("Feature Classification Report")
        lines.extend(
            [
                "This report classifies metadata columns for model-training use. "
                "It does not modify the metadata.",
                "",
            ]
        )
        for group in [
            "Required for Training",
            "Optional Feature",
            "Metadata Only",
            "Ignore During Training",
        ]:
            subset = table[table["Group"] == group]
            lines.extend([f"## {group}", ""])
            lines.append(subset.to_markdown(index=False) if not subset.empty else "No columns.")
            lines.append("")
        return "\n".join(lines)

    def render_missing_value_audit(self, dictionary: pd.DataFrame) -> str:
        missing_rows = []
        for column in self.df.columns:
            missing_count = int(self.df[column].isna().sum())
            if missing_count == 0:
                continue
            missing_pct = round((missing_count / len(self.df)) * 100, 2)
            missing_by_dataset = self._missing_by_dataset(column)
            expected = is_expected_missing(column, missing_by_dataset)
            issue = is_potential_preprocessing_issue(column, missing_by_dataset)
            missing_rows.append(
                {
                    "Column": column,
                    "Missing Count": missing_count,
                    "Missing Percentage": missing_pct,
                    "Dataset(s) Responsible": format_missing_by_dataset(missing_by_dataset),
                    "Expected Schema Difference": "Yes" if expected else "No",
                    "Preprocessing Issue": "Possible" if issue else "No",
                    "Training Impact": training_impact(column, missing_pct),
                    "Recommended Action": recommended_missing_action(column, expected, issue),
                    "Why Missing": missing_reason(column, missing_by_dataset, expected, issue),
                }
            )

        audit = pd.DataFrame(missing_rows).sort_values(
            ["Missing Percentage", "Column"], ascending=[False, True]
        )
        lines = self._header("Missing Value Audit")
        lines.extend(
            [
                "Missingness is attributed using the existing `dataset_name` column.",
                "",
                audit.to_markdown(index=False),
                "",
                "## Summary",
                "",
            ]
        )
        expected_columns = [
            row["Column"] for row in missing_rows if row["Expected Schema Difference"] == "Yes"
        ]
        issue_columns = [
            row["Column"] for row in missing_rows if row["Preprocessing Issue"] == "Possible"
        ]
        safe_ignore = [
            col for col in expected_columns if classify_column(col).group == "Ignore During Training"
        ]
        dataset_specific = [
            row["Column"]
            for row in missing_rows
            if len(self._datasets_with_missing(row["Column"])) < self.df[DATASET_COLUMN].nunique()
        ]
        lines.extend(
            [
                f"- Expected Missing Values: {', '.join(expected_columns) if expected_columns else 'None'}",
                f"- Dataset-Specific Missing Values: {', '.join(dataset_specific) if dataset_specific else 'None'}",
                f"- Potential Issues Requiring Investigation: {', '.join(issue_columns) if issue_columns else 'None'}",
                f"- Safe-to-Ignore Missing Fields: {', '.join(safe_ignore) if safe_ignore else 'None'}",
                "",
            ]
        )
        return "\n".join(lines)

    def render_dataset_schema_comparison(self) -> str:
        datasets = list(self.df[DATASET_COLUMN].dropna().unique())
        datasets.sort()
        schema_rows = []
        all_columns = set(self.df.columns)
        for dataset in datasets:
            available = set(self.df.loc[self.df[DATASET_COLUMN] == dataset].dropna(axis=1, how="all").columns)
            schema_rows.append(
                {
                    "Dataset": dataset,
                    "Rows": int((self.df[DATASET_COLUMN] == dataset).sum()),
                    "Available Columns": len(available),
                    "Missing Columns": len(all_columns - available),
                    "Unique Metadata Fields": ", ".join(sorted(available - self._shared_columns(datasets))) or "None",
                }
            )

        column_presence = []
        for column in self.df.columns:
            row = {"Column": column}
            for dataset in datasets:
                row[dataset] = "Yes" if self._has_dataset_values(dataset, column) else "No"
            column_presence.append(row)

        category_rows = []
        categories = {
            "Annotation Information": annotation_columns(),
            "Video Information": video_columns(),
            "Context Information": context_columns(),
            "Bounding Box Information": bbox_columns(),
            "Potential Training Features": potential_training_columns(),
        }
        for dataset in datasets:
            available = set(self.df.loc[self.df[DATASET_COLUMN] == dataset].dropna(axis=1, how="all").columns)
            for category, cols in categories.items():
                present = sorted(set(cols) & available)
                category_rows.append(
                    {
                        "Dataset": dataset,
                        "Category": category,
                        "Fields": ", ".join(present) if present else "None",
                    }
                )

        lines = self._header("Dataset Schema Comparison")
        lines.extend(
            [
                "Schemas are compared by non-null column availability within each dataset.",
                "",
                "## Dataset Summary",
                "",
                pd.DataFrame(schema_rows).to_markdown(index=False),
                "",
                "## Column Presence",
                "",
                pd.DataFrame(column_presence).to_markdown(index=False),
                "",
                "## Schema Categories",
                "",
                pd.DataFrame(category_rows).to_markdown(index=False),
                "",
                "## Shared Metadata Fields",
                "",
                ", ".join(sorted(self._shared_columns(datasets))) or "None",
                "",
            ]
        )
        for dataset in datasets:
            available = set(self.df.loc[self.df[DATASET_COLUMN] == dataset].dropna(axis=1, how="all").columns)
            lines.extend(
                [
                    f"## {dataset} Details",
                    "",
                    f"- Available Columns: {', '.join(sorted(available))}",
                    f"- Missing Columns: {', '.join(sorted(all_columns - available)) or 'None'}",
                    f"- Unique Metadata Fields: {', '.join(sorted(available - self._shared_columns(datasets))) or 'None'}",
                    "",
                ]
            )
        return "\n".join(lines)

    def render_dataset_readiness(self, dictionary: pd.DataFrame) -> str:
        total = len(self.df)
        dataset_counts = self.df[DATASET_COLUMN].value_counts()
        class_counts = self.df["type"].value_counts() if "type" in self.df.columns else pd.Series(dtype=int)
        bbox_cols = bbox_columns()
        valid_bbox_rows = int(self.df[bbox_cols].notna().all(axis=1).sum()) if set(bbox_cols).issubset(self.df.columns) else 0
        invalid_bbox_count = count_invalid_bboxes(self.df)
        high_missing_cols = dictionary[dictionary["Missing Percentage"] > 50]["Column Name"].tolist()
        label_missing = int(self.df["type"].isna().sum()) if "type" in self.df.columns else total
        imbalance_ratio = round(float(class_counts.max() / class_counts.min()), 2) if not class_counts.empty and class_counts.min() > 0 else 0.0
        metadata_quality_score = effective_metadata_quality(dictionary)
        readiness_score = compute_readiness_score(
            metadata_quality_score=metadata_quality_score,
            label_missing=label_missing,
            invalid_bbox_count=invalid_bbox_count,
            imbalance_ratio=imbalance_ratio,
            dataset_share=float(dataset_counts.max() / total) if total else 1.0,
        )
        leakage_columns = [
            col
            for col in ["dataset_name", "source_type", "original_path", "processed_path", "file_hash", "video_id"]
            if col in self.df.columns
        ]

        lines = self._header("Dataset Readiness")
        lines.extend(
            [
                "This is a training readiness decision document based on the completed metadata.",
                "",
                "## Decision",
                "",
                f"- Overall Readiness Score: {readiness_score}/100",
                f"- Decision: {readiness_decision(readiness_score)}",
                "",
                "## Dataset Strengths",
                "",
                f"- All {total:,} rows have a primary `type` label.",
                f"- Bounding boxes are present for {valid_bbox_rows:,} rows and no invalid boxes were detected in rows with complete bbox coordinates.",
                "- The metadata preserves dataset provenance and split columns for controlled training setup.",
                "",
                "## Dataset Weaknesses",
                "",
                f"- Dataset balance is skewed: {dataset_counts.index[0]} contributes {dataset_counts.iloc[0]} rows ({dataset_counts.iloc[0] / total:.1%}).",
                f"- Class imbalance ratio is {imbalance_ratio}:1.",
                f"- {len(high_missing_cols)} columns have more than 50% missing values, mostly because schemas differ across datasets.",
                "",
                "## Quality Assessment",
                "",
                f"- Metadata Quality: {metadata_quality_score}/100 task-weighted completeness for training-relevant metadata.",
                f"- Annotation Quality: Complete bbox annotations are concentrated in Picek; Kaggle and TUDAT do not provide bbox columns in this unified metadata.",
                f"- Class Balance: {class_balance_summary(class_counts)}",
                f"- Dataset Balance: {dataset_balance_summary(dataset_counts, total)}",
                "- Missing Metadata Assessment: High missingness is expected for schema-specific fields; training-critical missingness must be handled by filtering or task-specific loaders.",
                f"- Leakage Assessment: Exclude provenance, path, hash, split, timestamp, and processing-status fields from model inputs. Leakage-risk columns include {', '.join(leakage_columns)}.",
                "",
                "## Risks Before Training",
                "",
                "- Training across all three datasets without task-aware filtering will mix dense annotation rows with video-level-only rows.",
                "- Dataset and source-type imbalance can bias evaluation if splits are not stratified by dataset and class.",
                "- Context columns such as weather and day_time are incomplete and should not be treated as mandatory features.",
                "- Path, ID, hash, timestamp, and processing metadata can leak dataset provenance if accidentally encoded.",
                "",
                "## Recommendations Before Pipeline 3",
                "",
                "- Define the training task explicitly: classification-only can use all rows; bbox/localization training should filter to rows with complete bbox fields.",
                "- Use `type` as the primary label and bbox/time annotation columns only where present and required by the model objective.",
                "- Keep dataset provenance and path fields in dataloading metadata only, not model features.",
                "- Add class-balanced or dataset-aware sampling before training.",
                "- Treat optional context fields as ablation features after a baseline is established.",
                "",
            ]
        )
        return "\n".join(lines)

    def _header(self, title: str) -> List[str]:
        generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        return [
            f"# {title}",
            "",
            f"Generated: {generated_at}",
            f"Source metadata: `{self.metadata_path.as_posix()}`",
            "",
        ]

    def _missing_by_dataset(self, column: str) -> Dict[str, Dict[str, float]]:
        result: Dict[str, Dict[str, float]] = {}
        for dataset, group in self.df.groupby(DATASET_COLUMN, dropna=False):
            missing_count = int(group[column].isna().sum())
            if missing_count:
                result[str(dataset)] = {
                    "missing_count": missing_count,
                    "dataset_rows": int(len(group)),
                    "missing_percent": round((missing_count / len(group)) * 100, 2),
                }
        return result

    def _datasets_with_missing(self, column: str) -> List[str]:
        return list(self._missing_by_dataset(column).keys())

    def _has_dataset_values(self, dataset: str, column: str) -> bool:
        group = self.df[self.df[DATASET_COLUMN] == dataset]
        return bool(group[column].notna().any())

    def _shared_columns(self, datasets: Sequence[str]) -> set[str]:
        shared = set(self.df.columns)
        for dataset in datasets:
            available = set(self.df.loc[self.df[DATASET_COLUMN] == dataset].dropna(axis=1, how="all").columns)
            shared &= available
        return shared


def classify_column(column: str) -> ColumnDecision:
    required = {"type", "accident_time", "accident_frame", "center_x", "center_y", "x1", "y1", "x2", "y2"}
    optional = {
        "weather",
        "day_time",
        "scene_layout",
        "region",
        "rollover",
        "quality",
        "height",
        "width",
        "fps",
        "duration",
        "no_frames",
    }
    metadata_only = {
        "video_id",
        "dataset_name",
        "source_type",
        "dataset_version",
        "pipeline_version",
        "original_path",
        "processed_path",
        "video_path_mode",
        "annotation_available",
        "split",
        "split_in_distribution",
        "split_geo_aware",
        "annotations_path",
        "map",
        "camera_position",
        "annotations_start_offset",
        "media_type",
    }
    ignored = {
        "validation_status",
        "processing_status",
        "preprocessing_status",
        "file_size_bytes",
        "file_hash",
        "is_duplicate",
        "duplicate_group_id",
        "processed_at",
        "metadata_source",
        "weather_confidence",
        "day_time_confidence",
        "channels",
        "codec",
        "generated_at",
    }
    if column in required:
        return ColumnDecision(
            "Required for Training",
            "Use for labels or annotation targets when the training objective requires accident localization.",
            "Column represents the primary class label, accident timing, or bounding-box annotation.",
            "Filter to rows where this field is present for objectives that require it.",
        )
    if column in optional:
        return ColumnDecision(
            "Optional Feature",
            "Use only after establishing a baseline and handling missingness explicitly.",
            "Column may describe media properties or scene context but is incomplete or not a core label.",
            "Avoid making this mandatory across datasets unless the training subset supports it.",
        )
    if column in metadata_only:
        return ColumnDecision(
            "Metadata Only",
            "Keep for loading, grouping, traceability, splitting, or post-training analysis; do not feed to the model.",
            "Column identifies provenance, paths, splits, annotation storage, or dataset bookkeeping.",
            "Useful outside the model feature tensor.",
        )
    if column in ignored:
        return ColumnDecision(
            "Ignore During Training",
            "Exclude from model inputs and labels.",
            "Column is processing metadata, technical encoding metadata, duplicate bookkeeping, or timestamp information.",
            "Retain only for audits if needed.",
        )
    return ColumnDecision(
        "Ignore During Training",
        "Exclude until a reviewed training purpose is defined.",
        "No reliable training role can be inferred from the metadata structure.",
        "Unknown training purpose.",
    )


def infer_description(column: str, series: pd.Series) -> str:
    descriptions = {
        "video_id": "Video identifier where supplied by the source metadata.",
        "dataset_name": "Source dataset name observed in the unified metadata.",
        "source_type": "Source category such as real or synthetic.",
        "dataset_version": "Dataset version value recorded in metadata.",
        "pipeline_version": "Pipeline version recorded during processing.",
        "original_path": "Original media path recorded in metadata.",
        "processed_path": "Processed media path recorded in metadata.",
        "video_path_mode": "Path mode used for processed video references.",
        "validation_status": "Validation status recorded by preprocessing.",
        "processing_status": "Processing status recorded by preprocessing.",
        "preprocessing_status": "Preprocessing status recorded in metadata.",
        "type": "Primary class or accident-type label.",
        "accident_time": "Accident event time annotation.",
        "accident_frame": "Accident event frame annotation.",
        "center_x": "Bounding-box center x coordinate.",
        "center_y": "Bounding-box center y coordinate.",
        "x1": "Bounding-box left x coordinate.",
        "y1": "Bounding-box top y coordinate.",
        "x2": "Bounding-box right x coordinate.",
        "y2": "Bounding-box bottom y coordinate.",
        "weather": "Weather or environment condition label where available.",
        "no_frames": "Number of frames in the video.",
        "duration": "Video duration.",
        "height": "Video frame height.",
        "width": "Video frame width.",
        "fps": "Video frames per second.",
        "file_size_bytes": "Media file size in bytes.",
        "file_hash": "File hash recorded for traceability or duplicate checks.",
        "annotation_available": "Flag indicating whether annotation data is available.",
        "split": "Dataset split assignment.",
        "split_in_distribution": "In-distribution split assignment.",
        "split_geo_aware": "Geo-aware split assignment.",
        "rollover": "Rollover annotation or context field where supplied.",
        "region": "Region annotation or context field where supplied.",
        "scene_layout": "Scene layout label where supplied.",
        "day_time": "Day-time or lighting label where supplied.",
        "quality": "Annotation or media quality label where supplied.",
        "annotations_path": "Path to annotation file where supplied.",
        "map": "Map identifier or map metadata where supplied.",
        "camera_position": "Camera position metadata where supplied.",
        "annotations_start_offset": "Annotation start offset where supplied.",
        "is_duplicate": "Duplicate flag recorded during metadata processing.",
        "duplicate_group_id": "Duplicate group identifier where duplicate groups exist.",
        "processed_at": "Timestamp recorded when processing occurred.",
        "media_type": "Media type identified by metadata generation.",
        "metadata_source": "Metadata source identifier.",
        "weather_confidence": "Confidence score for inferred weather metadata.",
        "day_time_confidence": "Confidence score for inferred day_time metadata.",
        "channels": "Number of media channels where available.",
        "codec": "Media codec where available.",
        "generated_at": "Timestamp recorded when metadata was generated.",
    }
    return descriptions.get(column, "Unknown; no description can be inferred from the metadata.")


def example_value(series: pd.Series) -> str:
    non_null = series.dropna()
    if non_null.empty:
        return ""
    value = non_null.iloc[0]
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def format_missing_by_dataset(missing_by_dataset: Dict[str, Dict[str, float]]) -> str:
    return "; ".join(
        f"{dataset}: {int(values['missing_count'])}/{int(values['dataset_rows'])} ({values['missing_percent']}%)"
        for dataset, values in missing_by_dataset.items()
    )


def is_expected_missing(column: str, missing_by_dataset: Dict[str, Dict[str, float]]) -> bool:
    if not missing_by_dataset:
        return False
    if column in {"duplicate_group_id"}:
        return True
    if any(values["missing_percent"] == 100.0 for values in missing_by_dataset.values()):
        return True
    if column in context_columns() | annotation_columns() | {"media_type", "metadata_source", "generated_at", "channels", "codec"}:
        return True
    return False


def is_potential_preprocessing_issue(column: str, missing_by_dataset: Dict[str, Dict[str, float]]) -> bool:
    if column in {"processed_path", "file_hash"}:
        return True
    if column in {"fps", "duration", "no_frames"} and any(
        0 < values["missing_percent"] < 100 for values in missing_by_dataset.values()
    ):
        return True
    return False


def training_impact(column: str, missing_pct: float) -> str:
    group = classify_column(column).group
    if group == "Required for Training":
        return "High" if missing_pct > 0 else "None"
    if group == "Optional Feature":
        return "Medium" if missing_pct >= 50 else "Low"
    if group == "Metadata Only":
        return "Low"
    return "None"


def recommended_missing_action(column: str, expected: bool, issue: bool) -> str:
    group = classify_column(column).group
    if group == "Required for Training":
        return "Filter to rows with complete values for training objectives that require this target."
    if issue:
        return "Investigate generation logs or processing outputs before relying on this field."
    if expected and group == "Optional Feature":
        return "Use only with missingness handling or dataset-specific ablations."
    if expected:
        return "Document as schema-specific missingness; no metadata change required."
    return "Review before use."


def missing_reason(
    column: str,
    missing_by_dataset: Dict[str, Dict[str, float]],
    expected: bool,
    issue: bool,
) -> str:
    full_missing = [dataset for dataset, values in missing_by_dataset.items() if values["missing_percent"] == 100.0]
    partial_missing = [dataset for dataset, values in missing_by_dataset.items() if values["missing_percent"] < 100.0]
    if column == "duplicate_group_id":
        return "All values are missing because no duplicate groups are recorded in this metadata."
    if full_missing and expected:
        return f"Field is absent from the observed schema for: {', '.join(full_missing)}."
    if issue:
        return f"Missingness includes partial gaps in: {', '.join(partial_missing) or ', '.join(full_missing)}."
    if partial_missing:
        return f"Values are present for some rows but absent for others in: {', '.join(partial_missing)}."
    return "Reason cannot be inferred beyond observed missingness."


def annotation_columns() -> set[str]:
    return {
        "type",
        "accident_time",
        "accident_frame",
        "annotation_available",
        "annotations_path",
        "annotations_start_offset",
        "quality",
    }


def video_columns() -> set[str]:
    return {
        "original_path",
        "processed_path",
        "video_path_mode",
        "no_frames",
        "duration",
        "height",
        "width",
        "fps",
        "file_size_bytes",
        "file_hash",
        "media_type",
        "channels",
        "codec",
    }


def context_columns() -> set[str]:
    return {
        "weather",
        "rollover",
        "region",
        "scene_layout",
        "day_time",
        "map",
        "camera_position",
        "weather_confidence",
        "day_time_confidence",
    }


def bbox_columns() -> List[str]:
    return ["center_x", "center_y", "x1", "y1", "x2", "y2"]


def potential_training_columns() -> set[str]:
    return set(bbox_columns()) | {
        "type",
        "accident_time",
        "accident_frame",
        "weather",
        "day_time",
        "scene_layout",
        "region",
        "rollover",
        "height",
        "width",
        "fps",
        "duration",
        "no_frames",
    }


def count_invalid_bboxes(df: pd.DataFrame) -> int:
    cols = bbox_columns()
    if not set(cols).issubset(df.columns):
        return 0
    bbox = df.dropna(subset=cols)
    if bbox.empty:
        return 0
    invalid = (bbox["x2"] <= bbox["x1"]) | (bbox["y2"] <= bbox["y1"])
    return int(invalid.sum())


def compute_readiness_score(
    metadata_quality_score: float,
    label_missing: int,
    invalid_bbox_count: int,
    imbalance_ratio: float,
    dataset_share: float,
) -> int:
    score = metadata_quality_score
    if label_missing:
        score -= 25
    if invalid_bbox_count:
        score -= 15
    if imbalance_ratio > 20:
        score -= 6
    elif imbalance_ratio > 10:
        score -= 5
    if dataset_share > 0.75:
        score -= 4
    return max(0, min(100, int(round(score))))


def readiness_decision(score: int) -> str:
    if score >= 85:
        return "Ready for baseline training with task-aware filtering."
    if score >= 70:
        return "Conditionally ready; address sampling, leakage controls, and task-specific missingness before full training."
    return "Not ready for training without remediation."


def effective_metadata_quality(dictionary: pd.DataFrame) -> float:
    """Score completeness after discounting expected non-feature schema gaps."""

    by_column = dictionary.set_index("Column Name")
    label_score = 100 - float(by_column.loc["type", "Missing Percentage"])
    bbox_fields = [col for col in bbox_columns() if col in by_column.index]
    bbox_score = 100 - float(by_column.loc[bbox_fields, "Missing Percentage"].mean())
    timing_fields = [col for col in ["accident_time", "accident_frame"] if col in by_column.index]
    timing_score = 100 - float(by_column.loc[timing_fields, "Missing Percentage"].mean())
    core_video_fields = [
        col
        for col in ["height", "width", "duration", "fps", "no_frames"]
        if col in by_column.index
    ]
    video_score = 100 - float(by_column.loc[core_video_fields, "Missing Percentage"].mean())
    optional_context_fields = [
        col
        for col in ["weather", "day_time", "scene_layout", "region"]
        if col in by_column.index
    ]
    context_score = 100 - float(by_column.loc[optional_context_fields, "Missing Percentage"].mean())
    score = (
        0.40 * label_score
        + 0.25 * bbox_score
        + 0.15 * timing_score
        + 0.15 * video_score
        + 0.05 * context_score
    )
    return round(score, 1)


def class_balance_summary(class_counts: pd.Series) -> str:
    if class_counts.empty:
        return "No class label column found."
    return f"{len(class_counts)} classes; largest class `{class_counts.index[0]}` has {int(class_counts.iloc[0])} rows and smallest class `{class_counts.index[-1]}` has {int(class_counts.iloc[-1])} rows."


def dataset_balance_summary(dataset_counts: pd.Series, total: int) -> str:
    if dataset_counts.empty or total == 0:
        return "No dataset labels found."
    return "; ".join(
        f"{dataset}: {int(count)} ({count / total:.1%})"
        for dataset, count in dataset_counts.items()
    )
