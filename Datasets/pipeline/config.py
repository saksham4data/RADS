"""
Pipeline configuration loader and typed dataclass.

Reads pipeline_config.yaml and provides validated, typed access to all
settings. Serves as the single source of truth — no hardcoded behavior
elsewhere in the pipeline.
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class AggregationDatasetConfig:
    """Config for a dataset to be aggregated."""
    name: str
    raw_dir: Path
    processed_dir: Optional[Path] = None


@dataclass
class PipelineConfig:
    """
    Typed, validated pipeline configuration.

    Loaded from YAML via PipelineConfig.from_yaml(path).
    All pipeline modules receive this object instead of scattered arguments.
    """

    # ── Identity ─────────────────────────────────────────────────────────
    pipeline_version: str = "2.0.0"
    dataset_name: str = "picek"
    dataset_version: str = "1.0.0"

    # ── Paths ────────────────────────────────────────────────────────────
    raw_dir: Path = field(default_factory=lambda: Path("raw/picekl"))
    output_dir: Path = field(default_factory=lambda: Path("processed/picek"))

    # ── Processing ───────────────────────────────────────────────────────
    video_mode: str = "reference"       # reference | copy | symlink
    compute_hashes: bool = True
    hash_algorithm: str = "sha256"      # sha256 | md5
    detect_duplicates: bool = True
    probe_video_integrity: bool = True
    extract_fps: bool = True

    # ── Validation ───────────────────────────────────────────────────────
    validate_metadata: bool = True
    bbox_max_value: float = 1.5
    required_fields: List[str] = field(default_factory=lambda: [
        "type", "accident_time", "accident_frame",
        "no_frames", "duration", "height", "width",
    ])

    # ── Splits ───────────────────────────────────────────────────────────
    generate_standardized_splits: bool = True
    train_ratio: float = 0.70
    validation_ratio: float = 0.15
    test_ratio: float = 0.15
    split_seed: int = 42

    # ── Logging ──────────────────────────────────────────────────────────
    console_log_level: str = "INFO"
    file_log_level: str = "DEBUG"

    # ── Aggregation (Pipeline 2) ─────────────────────────────────────────
    aggregation_datasets: List[AggregationDatasetConfig] = field(default_factory=list)
    aggregation_required_columns: List[str] = field(default_factory=lambda: ["original_path", "dataset_name"])
    aggregation_optional_columns: List[str] = field(default_factory=list)
    aggregation_dedupe_keys: List[str] = field(default_factory=lambda: ["dataset_name", "original_path"])

    # ── Validation ───────────────────────────────────────────────────────

    def __post_init__(self):
        """Validate config after construction."""
        self.raw_dir = Path(self.raw_dir)
        self.output_dir = Path(self.output_dir)

        if self.video_mode not in ("reference", "copy", "symlink"):
            raise ValueError(
                f"Invalid video_mode: '{self.video_mode}'. "
                f"Must be one of: reference, copy, symlink"
            )

        if self.hash_algorithm not in ("sha256", "md5"):
            raise ValueError(
                f"Invalid hash_algorithm: '{self.hash_algorithm}'. "
                f"Must be one of: sha256, md5"
            )

        ratios = self.train_ratio + self.validation_ratio + self.test_ratio
        if abs(ratios - 1.0) > 0.01:
            raise ValueError(
                f"Split ratios must sum to 1.0, got {ratios:.3f}"
            )

        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if self.console_log_level.upper() not in valid_levels:
            raise ValueError(f"Invalid console_log_level: {self.console_log_level}")
        if self.file_log_level.upper() not in valid_levels:
            raise ValueError(f"Invalid file_log_level: {self.file_log_level}")

    # ── Serialization ────────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a JSON-serializable dictionary."""
        d = asdict(self)
        # Convert Path objects to strings
        d["raw_dir"] = str(self.raw_dir)
        d["output_dir"] = str(self.output_dir)
        return d

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> "PipelineConfig":
        """
        Load configuration from a YAML file.

        Args:
            yaml_path: Path to the YAML configuration file.

        Returns:
            PipelineConfig with values from the YAML file.
        """
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Config file not found: {yaml_path}")

        with open(yaml_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        if raw is None:
            raise ValueError(f"Config file is empty: {yaml_path}")

        return cls._from_nested_dict(raw)

    @classmethod
    def _from_nested_dict(cls, d: Dict[str, Any]) -> "PipelineConfig":
        """Flatten the nested YAML structure into flat dataclass fields."""
        pipeline = d.get("pipeline", {})
        dataset = d.get("dataset", {})
        paths = d.get("paths", {})
        processing = d.get("processing", {})
        validation = d.get("validation", {})
        splits = d.get("splits", {})
        logging_cfg = d.get("logging", {})
        agg_cfg = d.get("aggregation", {})

        agg_datasets = []
        for ds in agg_cfg.get("datasets", []):
            agg_datasets.append(AggregationDatasetConfig(
                name=ds.get("name"),
                raw_dir=Path(ds.get("raw_dir")),
                processed_dir=Path(ds.get("processed_dir")) if ds.get("processed_dir") else None
            ))

        return cls(
            pipeline_version=pipeline.get("version", "2.0.0"),
            dataset_name=dataset.get("name", "picek"),
            dataset_version=dataset.get("version", "1.0.0"),
            raw_dir=Path(paths.get("raw_dir", "raw/picekl")),
            output_dir=Path(paths.get("output_dir", "processed/picek")),
            video_mode=processing.get("video_mode", "reference"),
            compute_hashes=processing.get("compute_hashes", True),
            hash_algorithm=processing.get("hash_algorithm", "sha256"),
            detect_duplicates=processing.get("detect_duplicates", True),
            probe_video_integrity=processing.get("probe_video_integrity", True),
            extract_fps=processing.get("extract_fps", True),
            validate_metadata=validation.get("validate_metadata", True),
            bbox_max_value=validation.get("bbox_max_value", 1.5),
            required_fields=validation.get("required_fields", [
                "type", "accident_time", "accident_frame",
                "no_frames", "duration", "height", "width",
            ]),
            generate_standardized_splits=splits.get("generate_standardized", True),
            train_ratio=splits.get("train_ratio", 0.70),
            validation_ratio=splits.get("validation_ratio", 0.15),
            test_ratio=splits.get("test_ratio", 0.15),
            split_seed=splits.get("seed", 42),
            console_log_level=logging_cfg.get("console_level", "INFO"),
            file_log_level=logging_cfg.get("file_level", "DEBUG"),
            aggregation_datasets=agg_datasets,
            aggregation_required_columns=agg_cfg.get("required_columns", ["original_path", "dataset_name"]),
            aggregation_optional_columns=agg_cfg.get("optional_columns", []),
            aggregation_dedupe_keys=agg_cfg.get("composite_dedupe_keys", ["dataset_name", "original_path"]),
        )

    @classmethod
    def from_overrides(
        cls,
        yaml_path: Optional[Path] = None,
        raw_dir: Optional[str] = None,
        output_dir: Optional[str] = None,
        video_mode: Optional[str] = None,
    ) -> "PipelineConfig":
        """
        Load config from YAML then apply CLI overrides.

        Args:
            yaml_path: Path to YAML config (uses defaults if None).
            raw_dir: Override for raw_dir path.
            output_dir: Override for output_dir path.
            video_mode: Override for video_mode.

        Returns:
            PipelineConfig with overrides applied.
        """
        if yaml_path and Path(yaml_path).exists():
            config = cls.from_yaml(yaml_path)
        else:
            config = cls()

        if raw_dir is not None:
            config.raw_dir = Path(raw_dir)
        if output_dir is not None:
            config.output_dir = Path(output_dir)
        if video_mode is not None:
            config.video_mode = video_mode
            config.__post_init__()  # Re-validate

        return config
