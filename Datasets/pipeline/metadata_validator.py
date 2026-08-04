"""
Metadata row-level validation for dataset preprocessing.

Each row from a raw CSV is checked for:
- Required fields being present and non-empty (configurable)
- Numeric fields being valid numbers in expected ranges
- Categorical fields matching known valid values
- Bounding box coordinates within configurable range

All thresholds and required fields are read from PipelineConfig,
so nothing is hardcoded.
"""

from typing import List, Tuple

from pipeline.config import PipelineConfig
from pipeline.utils import VALID_ACCIDENT_TYPES, VALID_QUALITY_LABELS


def _is_numeric(value, allow_float: bool = True) -> bool:
    """Check if a value can be interpreted as a number."""
    if value is None:
        return False
    try:
        float(value) if allow_float else int(value)
        return True
    except (ValueError, TypeError):
        return False


def _in_range(value, low: float, high: float) -> bool:
    """Check if a numeric value falls within [low, high]."""
    try:
        v = float(value)
        return low <= v <= high
    except (ValueError, TypeError):
        return False


def validate_row(
    row: dict,
    source_type: str,
    config: PipelineConfig,
) -> Tuple[bool, List[str]]:
    """
    Validate a single metadata row.

    Args:
        row: Dictionary of column → value for one CSV row.
        source_type: "real" or "synthetic".
        config: Pipeline configuration (provides required_fields, bbox_max_value).

    Returns:
        (is_valid, issues): Tuple of overall validity flag and list of
        human-readable issue descriptions. If is_valid is True, issues is empty.
    """
    issues: List[str] = []

    # ── 1. Path field must exist ─────────────────────────────────────────
    path_val = row.get("original_path", "")
    if not path_val or str(path_val).strip() == "":
        issues.append("Missing or empty video path")

    # ── 2. Accident type must be recognized ──────────────────────────────
    acc_type = str(row.get("type", "")).strip().lower()
    if acc_type not in VALID_ACCIDENT_TYPES:
        issues.append(f"Unknown accident type: '{row.get('type', '')}'")

    # ── 3. Required numeric fields ───────────────────────────────────────
    numeric_checks = {
        "accident_time": {"min": 0, "required_positive": False},
        "accident_frame": {"min": 0, "required_positive": False},
        "no_frames": {"min": 0, "required_positive": True},
        "duration": {"min": 0, "required_positive": True},
        "height": {"min": 0, "required_positive": True},
        "width": {"min": 0, "required_positive": True},
    }

    for field_name, checks in numeric_checks.items():
        if field_name not in config.required_fields:
            continue
        val = row.get(field_name)
        if not _is_numeric(val):
            issues.append(f"{field_name} is not numeric: '{val}'")
        elif checks["required_positive"] and float(val) <= 0:
            issues.append(f"{field_name} must be > 0: {val}")
        elif float(val) < checks["min"]:
            issues.append(f"{field_name} is negative: {val}")

    # ── 4. Bounding box coordinates ──────────────────────────────────────
    bbox_fields = ["center_x", "center_y", "x1", "y1", "x2", "y2"]
    bbox_max = config.bbox_max_value
    for field_name in bbox_fields:
        val = row.get(field_name)
        if not _is_numeric(val):
            issues.append(f"{field_name} is not numeric: '{val}'")
        elif not _in_range(val, 0.0, bbox_max):
            issues.append(f"{field_name} out of range [0, {bbox_max}]: {val}")

    # ── 5. Real-specific checks ──────────────────────────────────────────
    if source_type == "real":
        quality = str(row.get("quality", "")).strip()
        if quality and quality not in VALID_QUALITY_LABELS:
            issues.append(f"Unknown quality label: '{quality}'")

    # ── 6. Synthetic-specific checks ─────────────────────────────────────
    if source_type == "synthetic":
        ann_path = row.get("annotations_path", "")
        if not ann_path or str(ann_path).strip() == "":
            issues.append("Synthetic entry missing annotations_path")

    is_valid = len(issues) == 0
    return is_valid, issues
