"""
Abstract base class for dataset processors.

Defines the contract that all dataset-specific processors must implement.
This ensures every dataset (Picek, TU-DAT, Kaggle, etc.) produces
identically structured output directories, metadata CSVs, and reports.

The run() method is implemented as a concrete template method in the base
class, orchestrating the pipeline steps in the correct order. Subclasses
only implement the dataset-specific abstract methods.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from pipeline.config import PipelineConfig


@dataclass
class ProcessingResult:
    """
    Structured result returned by a processor run.

    Captures counts, paths to output artifacts, and any issues
    encountered during processing.
    """
    dataset_name: str
    total_samples: int = 0
    valid_samples: int = 0
    invalid_samples: int = 0
    processed_samples: int = 0
    skipped_samples: int = 0
    failed_samples: int = 0
    duplicates_found: int = 0

    # Paths to generated artifacts
    master_metadata_path: Optional[Path] = None
    statistics_path: Optional[Path] = None
    report_path: Optional[Path] = None
    manifest_path: Optional[Path] = None
    log_path: Optional[Path] = None

    # Per-sample issues (video_id → list of issue strings)
    issues: Dict[str, List[str]] = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        if self.total_samples == 0:
            return 0.0
        return round(self.valid_samples / self.total_samples * 100, 2)

    def summary(self) -> str:
        return (
            f"{'=' * 50}\n"
            f"  Processing Complete: {self.dataset_name}\n"
            f"{'=' * 50}\n"
            f"  Total samples:     {self.total_samples}\n"
            f"  Valid:             {self.valid_samples}\n"
            f"  Invalid:           {self.invalid_samples}\n"
            f"  Processed:         {self.processed_samples}\n"
            f"  Skipped:           {self.skipped_samples}\n"
            f"  Failed:            {self.failed_samples}\n"
            f"  Duplicates found:  {self.duplicates_found} groups\n"
            f"  Success rate:      {self.success_rate}%\n"
            f"{'=' * 50}"
        )


class BaseDatasetProcessor(ABC):
    """
    Abstract base class for all dataset processors.

    Subclasses implement the abstract methods below. The concrete run()
    method orchestrates them in the correct order.

    Usage:
        config = PipelineConfig.from_yaml("pipeline_config.yaml")
        processor = PicekProcessor(config)
        result = processor.run()
    """

    def __init__(self, config: PipelineConfig):
        """
        Args:
            config: Pipeline configuration object.
        """
        self.config = config
        self.raw_dir = config.raw_dir
        self.output_dir = config.output_dir
        self.dataset_name = config.dataset_name

        # Standard output subdirectories
        self.output_real = self.output_dir / "real"
        self.output_synthetic = self.output_dir / "synthetic"
        self.output_metadata = self.output_dir / "metadata"
        self.output_reports = self.output_dir / "reports"
        self.output_statistics = self.output_dir / "statistics"

    @abstractmethod
    def load_metadata(self) -> pd.DataFrame:
        """
        Load and normalize raw metadata CSVs into a single DataFrame.

        Must add at minimum:
        - 'original_path' column (video file path relative to raw_dir)
        - 'source_type' column ("real" or "synthetic")
        - 'dataset_name' column

        Returns:
            Combined DataFrame with normalized column names.
        """
        ...

    @abstractmethod
    def validate_metadata(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate each metadata row and set the 'validation_status' column.

        Args:
            df: DataFrame from load_metadata().

        Returns:
            Same DataFrame with 'validation_status' column populated.
        """
        ...

    @abstractmethod
    def verify_and_probe_videos(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Check video file existence and integrity. Update 'validation_status'
        for missing or corrupted videos. Populate:
        - file_size_bytes
        - fps (if config.extract_fps)
        - file_hash (if config.compute_hashes)

        Args:
            df: DataFrame with validation_status from validate_metadata().

        Returns:
            Same DataFrame with video-level columns populated.
        """
        ...

    @abstractmethod
    def detect_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect duplicate videos by file hash. Populate:
        - is_duplicate (bool)
        - duplicate_group_id (str, empty for non-duplicates)

        Args:
            df: DataFrame with file_hash column.

        Returns:
            Same DataFrame with duplicate columns populated.
        """
        ...

    @abstractmethod
    def process_videos(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Organize videos according to config.video_mode (reference/copy/symlink).
        Set 'processing_status', 'processed_path', 'video_path_mode' columns.
        Generate 'video_id' for each sample.
        Set 'preprocessing_status' lifecycle stage.

        Args:
            df: DataFrame with validation + video info.

        Returns:
            Final DataFrame ready to be saved as master_metadata.csv.
        """
        ...

    @abstractmethod
    def assign_splits(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Assign standardized train/validation/test splits.
        Populates the 'split' column.
        Preserves any original split columns.

        Args:
            df: DataFrame with video_id column.

        Returns:
            Same DataFrame with 'split' column populated.
        """
        ...

    @abstractmethod
    def run(self) -> ProcessingResult:
        """
        Execute the full preprocessing pipeline.

        Subclasses orchestrate their specific pipeline steps,
        calling the abstract methods in the correct order.

        Returns:
            ProcessingResult with counts and artifact paths.
        """
        ...
