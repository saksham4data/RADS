"""
Shared utilities for the preprocessing pipeline.

Contains:
- Typed enums for status fields and video modes
- Canonical column order (42-column schema)
- Path helpers
- File hashing (SHA-256)
- Deterministic split assignment
- Dual-output logger
"""

import enum
import hashlib
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


# ── Enums ────────────────────────────────────────────────────────────────────

class VideoMode(enum.Enum):
    """How video files are handled in the output directory."""
    REFERENCE = "reference"   # Store path to raw/ — no file I/O
    COPY = "copy"             # shutil.copy2 into processed/
    SYMLINK = "symlink"       # os.symlink into processed/


class ValidationStatus(enum.Enum):
    """Result of metadata + file validation."""
    VALID = "valid"
    INVALID_METADATA = "invalid_metadata"
    MISSING_VIDEO = "missing_video"
    CORRUPTED_VIDEO = "corrupted_video"


class PreprocessingStatus(enum.Enum):
    """
    Lifecycle stage of a sample.

    Pipeline progression:
        raw → validated → processed → frame_extracted → annotation_ready → training_ready
    Invalid samples stay at 'invalid'.
    """
    RAW = "raw"
    VALIDATED = "validated"
    PROCESSED = "processed"
    INVALID = "invalid"
    # Future stages (set by downstream pipelines):
    FRAME_EXTRACTED = "frame_extracted"
    ANNOTATION_READY = "annotation_ready"
    TRAINING_READY = "training_ready"


class ProcessingStatus(enum.Enum):
    """Outcome of the file-handling step (copy/symlink/reference)."""
    SUCCESS = "success"
    SKIPPED = "skipped"
    FAILED = "failed"


# ── Constants ────────────────────────────────────────────────────────────────

VALID_ACCIDENT_TYPES = frozenset({
    "rear-end", "t-bone", "single", "head-on", "sideswipe"
})

VALID_WEATHER_CONDITIONS = frozenset({
    "normal", "rain", "snow", "fog", "clear", "night", "sunset", "wet"
})

VALID_QUALITY_LABELS = frozenset({
    "Very_Poor", "Poor", "Fine", "Good"
})

VALID_SOURCE_TYPES = frozenset({"real", "synthetic"})

VIDEO_EXTENSIONS = frozenset({".mp4", ".avi", ".mkv", ".mov", ".wmv"})

# ── Column schema (42 columns) ──────────────────────────────────────────────

# Identity
IDENTITY_COLUMNS = [
    "video_id", "dataset_name", "source_type",
    "dataset_version", "pipeline_version",
]

# Paths
PATH_COLUMNS = [
    "original_path", "processed_path", "video_path_mode",
]

# Status
STATUS_COLUMNS = [
    "validation_status", "processing_status", "preprocessing_status",
]

# Shared video properties (from CSV)
SHARED_COLUMNS = [
    "type", "accident_time", "accident_frame",
    "center_x", "center_y", "x1", "y1", "x2", "y2",
    "weather", "no_frames", "duration", "height", "width",
]

# Extracted video properties
EXTRACTED_COLUMNS = [
    "fps", "file_size_bytes", "file_hash", "annotation_available",
]

# Splits
SPLIT_COLUMNS = [
    "split", "split_in_distribution", "split_geo_aware",
]

# Real-only columns
REAL_ONLY_COLUMNS = [
    "rollover", "region", "scene_layout", "day_time", "quality",
]

# Synthetic-only columns
SYNTHETIC_ONLY_COLUMNS = [
    "annotations_path", "map", "camera_position",
    "annotations_start_offset",
]

# Duplicate detection
DUPLICATE_COLUMNS = [
    "is_duplicate", "duplicate_group_id",
]

# Audit
AUDIT_COLUMNS = [
    "processed_at",
]

# Final master column order (42 columns)
MASTER_COLUMN_ORDER = (
    IDENTITY_COLUMNS
    + PATH_COLUMNS
    + STATUS_COLUMNS
    + SHARED_COLUMNS
    + EXTRACTED_COLUMNS
    + SPLIT_COLUMNS
    + REAL_ONLY_COLUMNS
    + SYNTHETIC_ONLY_COLUMNS
    + DUPLICATE_COLUMNS
    + AUDIT_COLUMNS
)


# ── Path Helpers ─────────────────────────────────────────────────────────────

def resolve_raw_path(raw_base: Path, relative_path: str) -> Path:
    """
    Resolve a relative video path from metadata against the raw dataset root.

    Args:
        raw_base: Absolute path to raw/<dataset_name>/ directory.
        relative_path: Value from the CSV path column (e.g. "real_videos/X.mp4").

    Returns:
        Absolute resolved path.
    """
    cleaned = relative_path.replace("\\", "/").strip()
    return raw_base / cleaned


def ensure_dir(directory: Path) -> Path:
    """Create a directory (and parents) if it doesn't exist. Returns the path."""
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def now_iso() -> str:
    """Return current UTC time as ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


# ── Hashing ──────────────────────────────────────────────────────────────────

def compute_file_hash(
    path: Path,
    algorithm: str = "sha256",
    chunk_size: int = 65536,
) -> str:
    """
    Compute a hex digest hash of a file using chunked reads.

    Args:
        path: Absolute path to the file.
        algorithm: Hash algorithm ("sha256" or "md5").
        chunk_size: Read chunk size in bytes (default 64KB).

    Returns:
        Hex digest string (64 chars for SHA-256, 32 for MD5).
        Returns empty string if the file cannot be read.
    """
    try:
        h = hashlib.new(algorithm)
        with open(path, "rb") as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                h.update(chunk)
        return h.hexdigest()
    except (OSError, ValueError):
        return ""


# ── Splits ───────────────────────────────────────────────────────────────────

def assign_deterministic_split(
    video_id: str,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
    seed: int = 42,
) -> str:
    """
    Assign a sample to train/validation/test based on a hash of its video_id.

    This is deterministic: the same video_id always gets the same split,
    regardless of row order, run order, or platform.

    Args:
        video_id: Unique identifier for the sample.
        train_ratio: Fraction assigned to train.
        validation_ratio: Fraction assigned to validation.
        seed: Integer seed mixed into the hash.

    Returns:
        One of "train", "validation", "test".
    """
    key = f"{seed}:{video_id}"
    hash_val = int(hashlib.sha256(key.encode()).hexdigest(), 16)
    bucket = (hash_val % 10000) / 10000.0  # [0, 1) with 4 decimal places

    if bucket < train_ratio:
        return "train"
    elif bucket < train_ratio + validation_ratio:
        return "validation"
    else:
        return "test"


# ── Logging ──────────────────────────────────────────────────────────────────

class ProcessingLogger:
    """
    Dual-output logger: writes to both console (stdout) and a log file.
    All pipeline operations go through this logger for full traceability.
    """

    def __init__(
        self,
        log_file: Optional[Path] = None,
        name: str = "pipeline",
        console_level: str = "INFO",
        file_level: str = "DEBUG",
    ):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers.clear()  # Prevent duplicate handlers on re-init

        # Console handler
        console = logging.StreamHandler(sys.stdout)
        console.setLevel(getattr(logging, console_level.upper(), logging.INFO))
        console_fmt = logging.Formatter(
            "[%(asctime)s] %(levelname)-8s %(message)s",
            datefmt="%H:%M:%S",
        )
        console.setFormatter(console_fmt)
        self.logger.addHandler(console)

        # File handler
        if log_file is not None:
            ensure_dir(log_file.parent)
            file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
            file_handler.setLevel(getattr(logging, file_level.upper(), logging.DEBUG))
            file_fmt = logging.Formatter(
                "[%(asctime)s] %(levelname)-8s %(name)s — %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            file_handler.setFormatter(file_fmt)
            self.logger.addHandler(file_handler)

    def info(self, msg: str) -> None:
        self.logger.info(msg)

    def debug(self, msg: str) -> None:
        self.logger.debug(msg)

    def warning(self, msg: str) -> None:
        self.logger.warning(msg)

    def error(self, msg: str) -> None:
        self.logger.error(msg)

    def section(self, title: str) -> None:
        """Log a visual section divider."""
        divider = "-" * 60
        self.logger.info("")
        self.logger.info(divider)
        self.logger.info(f"  {title}")
        self.logger.info(divider)
