"""
Metadata Assembler for Pipeline 0.

Orchestrates the full metadata generation sequence:
    1. Select and instantiate the correct DatasetAdapter (by dataset_name)
    2. Discover all media files via adapter.discover_files()
    3. For each MediaFile: run the appropriate Analyzer → SceneClassifier
    4. Convert List[MediaFile] into a pandas DataFrame
    5. Write the DataFrame to metadata.csv in output_dir

This module owns the output CSV column schema for Pipeline 0. The columns
produced here are designed to be consumed by a future Pipeline 1 adapter
(TudatProcessor, KaggleProcessor, etc.) in the same way PicekProcessor
consumes Picek's author-provided CSV.

Single Responsibility: orchestration only. No analysis logic lives here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Type

import pandas as pd

from metadata_gen.adapters.base_adapter import BaseAdapter, MediaFile
from metadata_gen.adapters.tudat_adapter import TudatAdapter
from metadata_gen.adapters.kaggle_adapter import KaggleAdapter
from metadata_gen.analyzers.video_analyzer import VideoAnalyzer
from metadata_gen.analyzers.image_analyzer import ImageAnalyzer
from metadata_gen.classifiers.scene_classifier import SceneClassifier
from metadata_gen.config import MetadataGenConfig


logger = logging.getLogger(__name__)

# tqdm is optional — graceful degradation to plain iteration
try:
    from tqdm import tqdm as _tqdm
    _TQDM_AVAILABLE = True
except ImportError:
    _TQDM_AVAILABLE = False


# ── Adapter registry ──────────────────────────────────────────────────────────
# Maps dataset_name (from config) to its adapter class.
# Adding support for a new dataset = one entry here + one new adapter module.
_ADAPTER_REGISTRY: Dict[str, Type[BaseAdapter]] = {
    "tudat":  TudatAdapter,
    "kaggle": KaggleAdapter,
}

# ── Output CSV column order ───────────────────────────────────────────────────
# This is the canonical column schema for Pipeline 0's metadata.csv output.
# Pipeline 1 adapters (TudatProcessor, KaggleProcessor) must read these columns.
#
# Grouped by concern to match the Picek CSV style:
OUTPUT_COLUMNS = [
    # ── File identity ─────────────────────────────────────────────────────
    "original_path",         # Relative path from raw_dir (forward slashes)
    "dataset_name",
    "dataset_version",
    "source_type",           # "real" | "synthetic"
    "media_type",            # "video" | "image"  [Pipeline 0 addition]
    "metadata_source",       # "auto_generated" | "folder_structure"  [Pipeline 0]

    # ── Accident classification ───────────────────────────────────────────
    "type",                  # "accident" | "non-accident" | "challenging" | NULL
    "accident_time",         # NULL — cannot infer without detector
    "accident_frame",        # NULL — cannot infer without detector

    # ── Bounding box ──────────────────────────────────────────────────────
    "center_x",              # NULL — requires trained object detector
    "center_y",
    "x1", "y1", "x2", "y2",

    # ── Scene conditions ──────────────────────────────────────────────────
    "weather",               # "clear" | "rain" | "fog" | NULL
    "weather_confidence",    # float in [0,1] — always present, even if weather=NULL
    "day_time",              # "day" | "night" | NULL
    "day_time_confidence",   # float in [0,1] — always present, even if day_time=NULL
    "scene_layout",          # NULL — cannot infer reliably
    "region",                # NULL — cannot infer
    "rollover",              # NULL — cannot infer

    # ── Video/image technical properties ─────────────────────────────────
    "no_frames",             # int  (1 for images)
    "duration",              # float seconds (0.0 for images)
    "fps",                   # float (0.0 for images)
    "height",                # int pixels
    "width",                 # int pixels
    "channels",              # int (image-only; NULL for video)
    "codec",                 # str e.g. "h264" (video-only; NULL for image)
    "file_size_bytes",       # int

    # ── Splits ────────────────────────────────────────────────────────────
    "split_in_distribution", # From folder structure if available, else NULL
    "split_geo_aware",       # NULL — Pipeline 0 cannot determine this

    # ── Quality / subjective fields ───────────────────────────────────────
    "quality",               # NULL — requires human annotation

    # ── Audit ─────────────────────────────────────────────────────────────
    "generated_at",          # ISO-8601 UTC timestamp of when this row was generated
]


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class AssemblyResult:
    """
    Summary of a completed metadata generation run.

    Returned by MetadataAssembler.run(). Captures counts, the output
    artifact path, and per-file error details for post-run inspection.
    """
    dataset_name: str
    total_files: int = 0
    valid_files: int = 0
    failed_files: int = 0
    skipped_files: int = 0

    output_path: Optional[Path] = None       # Path to the written metadata.csv
    log_path: Optional[Path] = None          # Path to the written log file

    # filename → error message for files that could not be processed
    errors: Dict[str, str] = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        if self.total_files == 0:
            return 0.0
        return round(self.valid_files / self.total_files * 100, 2)

    def summary(self) -> str:
        return (
            f"{'=' * 50}\n"
            f"  Metadata Generation Complete: {self.dataset_name}\n"
            f"{'=' * 50}\n"
            f"  Total files:   {self.total_files}\n"
            f"  Valid:         {self.valid_files}\n"
            f"  Failed:        {self.failed_files}\n"
            f"  Skipped:       {self.skipped_files}\n"
            f"  Success rate:  {self.success_rate}%\n"
            f"  Output:        {self.output_path}\n"
            f"{'=' * 50}"
        )


# ── Assembler ─────────────────────────────────────────────────────────────────

class MetadataAssembler:
    """
    Orchestrates the full Pipeline 0 metadata generation sequence.

    Usage:
        config = MetadataGenConfig.from_yaml('metadata_gen_config.yaml')
        assembler = MetadataAssembler(config)
        result = assembler.run()
        print(result.summary())
    """

    def __init__(self, config: MetadataGenConfig) -> None:
        """
        Args:
            config: Fully validated Pipeline 0 configuration.

        Raises:
            ValueError: If config.dataset_name is not in the adapter registry.
        """
        if config.dataset_name not in _ADAPTER_REGISTRY:
            supported = sorted(_ADAPTER_REGISTRY.keys())
            raise ValueError(
                f"No adapter registered for dataset '{config.dataset_name}'. "
                f"Supported datasets: {supported}. "
                f"To add a new dataset, create an adapter in metadata_gen/adapters/ "
                f"and register it in metadata_gen/assembler._ADAPTER_REGISTRY."
            )

        self.config = config
        self._result = AssemblyResult(dataset_name=config.dataset_name)

        # Set up file logger in output_dir
        from metadata_gen.config import MetadataGenConfig as _C  # avoid circular
        log_path = config.output_dir / "metadata_gen.log"
        self._setup_logging(log_path)
        self._result.log_path = log_path

    # ── Public interface ──────────────────────────────────────────────────

    def run(self) -> AssemblyResult:
        """
        Execute the full metadata generation pipeline.

        Steps:
            1. Instantiate adapter → discover files
            2. For each file: analyze + classify
            3. Build DataFrame → write CSV

        Returns:
            AssemblyResult with counts and artifact paths.

        Never raises — top-level exceptions are logged and re-raised so the
        CLI entry point can exit with a non-zero code.
        """
        logger.info("")
        logger.info("=" * 60)
        logger.info("  METADATA GENERATION PIPELINE v%s", self.config.pipeline_version)
        logger.info("=" * 60)
        logger.info("  Dataset:    %s v%s", self.config.dataset_name, self.config.dataset_version)
        logger.info("  Media type: %s", self.config.media_type)
        logger.info("  Raw dir:    %s", self.config.raw_dir)
        logger.info("  Output dir: %s", self.config.output_dir)
        logger.info("")

        try:
            # Step 1: File discovery
            logger.info("-" * 60)
            logger.info("  Step 1: Discovering Files")
            logger.info("-" * 60)
            adapter = self._build_adapter()
            files = adapter.discover_files()
            self._result.total_files = len(files)
            logger.info("  Discovered %d file(s)", len(files))

            if not files:
                logger.warning("  No files discovered — nothing to process.")
                return self._result

            # Step 2: Analyze + Classify each file
            logger.info("")
            logger.info("-" * 60)
            logger.info("  Step 2: Analyzing Files")
            logger.info("-" * 60)
            files = self._process_files(files)

            # Step 3: Build DataFrame
            logger.info("")
            logger.info("-" * 60)
            logger.info("  Step 3: Building Metadata DataFrame")
            logger.info("-" * 60)
            df = self._to_dataframe(files)
            logger.info("  DataFrame shape: %d rows x %d columns", *df.shape)

            # Step 4: Write CSV
            logger.info("")
            logger.info("-" * 60)
            logger.info("  Step 4: Writing metadata.csv")
            logger.info("-" * 60)
            output_path = self._save_csv(df)
            self._result.output_path = output_path
            logger.info("  Saved: %s", output_path)

            # Final summary
            logger.info("")
            logger.info(self._result.summary())

        except Exception as exc:
            logger.error("PIPELINE FAILED: %s: %s", type(exc).__name__, exc, exc_info=True)
            raise

        return self._result

    # ── Private: pipeline steps ───────────────────────────────────────────

    def _build_adapter(self) -> BaseAdapter:
        """Instantiate the adapter for config.dataset_name."""
        adapter_cls = _ADAPTER_REGISTRY[self.config.dataset_name]
        logger.info("  Using adapter: %s", adapter_cls.__name__)
        return adapter_cls(self.config)

    def _process_files(self, files: List[MediaFile]) -> List[MediaFile]:
        """
        Run Analyzer + SceneClassifier on every discovered MediaFile.

        Files that fail analysis are kept in the list with is_valid=False
        so they still appear in the output CSV (with NULL technical fields).
        This preserves full discovery counts and makes failures visible.

        Args:
            files: List of MediaFile objects from the adapter.

        Returns:
            The same list with all fields populated (or set to None on failure).
        """
        # Instantiate analyzer and classifier once (not per-file)
        if self.config.media_type == "video":
            analyzer = VideoAnalyzer(self.config)
        else:
            analyzer = ImageAnalyzer(self.config)

        classifier = SceneClassifier(self.config)

        valid = 0
        failed = 0

        iterable = files
        if _TQDM_AVAILABLE:
            iterable = _tqdm(
                files,
                total=len(files),
                desc=f"Analyzing ({self.config.media_type})",
                unit="file",
                ncols=90,
            )

        for mf in iterable:
            try:
                # Analyze: fills technical fields, returns sampled frames
                frames = analyzer.analyze(mf)

                # Scene classify: fills day_time, weather + confidences
                if frames:
                    classifier.apply(mf, frames)
                else:
                    # No frames available (file invalid or classifier disabled)
                    mf.day_time            = None
                    mf.day_time_confidence = None
                    mf.weather             = None
                    mf.weather_confidence  = None

                if mf.is_valid():
                    valid += 1
                else:
                    failed += 1
                    err = mf.video_error or mf.image_error or "unknown error"
                    self._result.errors[mf.filename] = err
                    logger.warning(
                        "  FAILED [%s]: %s",
                        mf.relative_path,
                        err,
                    )

            except Exception as exc:
                # Defensive: should not happen since analyzers catch internally,
                # but we protect the loop from any unexpected exceptions.
                err = f"{type(exc).__name__}: {exc}"
                mf.is_video_valid = False
                mf.is_image_valid = False
                mf.video_error    = err
                mf.image_error    = err
                failed += 1
                self._result.errors[mf.filename] = err
                logger.error(
                    "  UNEXPECTED ERROR [%s]: %s",
                    mf.relative_path,
                    err,
                    exc_info=True,
                )

        self._result.valid_files  = valid
        self._result.failed_files = failed
        logger.info(
            "  Analysis complete: %d valid, %d failed",
            valid,
            failed,
        )
        return files

    def _to_dataframe(self, files: List[MediaFile]) -> pd.DataFrame:
        """
        Convert a list of MediaFile objects to a pandas DataFrame.

        Columns follow OUTPUT_COLUMNS order. Fields that are not determinable
        remain as None (written as empty in CSV, consistent with Picek's style).

        Args:
            files: Fully-processed MediaFile list.

        Returns:
            DataFrame with one row per file, columns in OUTPUT_COLUMNS order.
        """
        generated_at = datetime.now(timezone.utc).isoformat()
        rows = []

        for mf in files:
            row = {
                # File identity
                "original_path":         mf.relative_path,
                "dataset_name":          mf.dataset_name,
                "dataset_version":       mf.dataset_version,
                "source_type":           mf.source_type,
                "media_type":            mf.media_type,
                "metadata_source":       mf.metadata_source,

                # Accident classification
                "type":                  mf.label,
                "accident_time":         None,    # Cannot infer
                "accident_frame":        None,    # Cannot infer

                # Bounding box — all NULL, require object detector
                "center_x":              None,
                "center_y":              None,
                "x1":                    None,
                "y1":                    None,
                "x2":                    None,
                "y2":                    None,

                # Scene conditions
                "weather":               mf.weather,
                "weather_confidence":    mf.weather_confidence,
                "day_time":              mf.day_time,
                "day_time_confidence":   mf.day_time_confidence,
                "scene_layout":          None,    # Cannot infer reliably
                "region":                None,    # Cannot infer
                "rollover":              None,    # Cannot infer

                # Technical properties
                "no_frames":             mf.no_frames,
                "duration":              mf.duration,
                "fps":                   mf.fps,
                "height":                mf.height,
                "width":                 mf.width,
                "channels":              mf.channels,   # image-only
                "codec":                 mf.codec,      # video-only
                "file_size_bytes":       mf.file_size_bytes,

                # Splits
                "split_in_distribution": mf.split,      # From folder structure or None
                "split_geo_aware":       None,           # Cannot infer

                # Quality
                "quality":               None,           # Requires human annotation

                # Audit
                "generated_at":          generated_at,
            }
            rows.append(row)

        df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
        return df

    def _save_csv(self, df: pd.DataFrame) -> Path:
        """
        Write the DataFrame to output_dir/output_filename.

        Creates output_dir if it does not exist.

        Args:
            df: DataFrame with columns matching OUTPUT_COLUMNS.

        Returns:
            Absolute path to the written CSV file.

        Raises:
            OSError: If the directory cannot be created or file cannot be written.
        """
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.config.output_dir / self.config.output_filename

        df.to_csv(output_path, index=False, encoding="utf-8")
        return output_path

    # ── Private: logging setup ────────────────────────────────────────────

    def _setup_logging(self, log_path: Path) -> None:
        """
        Configure a file handler on the root logger for this run.

        The console handler is assumed to be configured by the CLI entry
        point (run_metadata_gen.py). This method only adds the file handler.

        Args:
            log_path: Path to the log file. Parent directory is created.
        """
        log_path.parent.mkdir(parents=True, exist_ok=True)

        root = logging.getLogger("metadata_gen")
        root.setLevel(logging.DEBUG)

        # Avoid duplicate handlers on re-instantiation
        for h in root.handlers[:]:
            if isinstance(h, logging.FileHandler):
                root.removeHandler(h)

        file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
        file_handler.setLevel(
            getattr(logging, self.config.file_log_level.upper(), logging.DEBUG)
        )
        fmt = logging.Formatter(
            "[%(asctime)s] %(levelname)-8s %(name)s — %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(fmt)
        root.addHandler(file_handler)
