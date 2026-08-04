"""
Metadata Generation Pipeline configuration.

Reads metadata_gen_config.yaml and provides validated, typed access to all
settings. Single source of truth for Pipeline 0 — no hardcoded values elsewhere.

Design mirrors pipeline/config.py (Pipeline 1) but is entirely independent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class MetadataGenConfig:
    """
    Typed, validated configuration for the Metadata Generation Pipeline.

    Loaded from YAML via MetadataGenConfig.from_yaml(path).
    Passed to all Pipeline 0 modules instead of scattered arguments.
    """

    # ── Identity ──────────────────────────────────────────────────────────
    pipeline_version: str = "1.0.0"
    dataset_name: str = "tudat"
    dataset_version: str = "1.0.0"

    # ── Paths ─────────────────────────────────────────────────────────────
    raw_dir: Path = field(default_factory=lambda: Path("raw/tudat/Final_videos"))
    output_dir: Path = field(default_factory=lambda: Path("raw/tudat"))

    # ── Media type ────────────────────────────────────────────────────────
    # "video" | "image" — controls which analyzer branch is used
    media_type: str = "video"

    # Supported file extensions per media type (lower-case, with dot)
    video_extensions: List[str] = field(
        default_factory=lambda: [".mp4", ".avi", ".mkv", ".mov", ".wmv"]
    )
    image_extensions: List[str] = field(
        default_factory=lambda: [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"]
    )

    # ── Video analysis ────────────────────────────────────────────────────
    probe_video_integrity: bool = True
    # Number of frames sampled for scene classification (spread evenly)
    scene_sample_frames: int = 5

    # ── Scene classification ──────────────────────────────────────────────
    enable_scene_classification: bool = True
    # Minimum confidence below which weather/day_time is set to NULL
    confidence_threshold: float = 0.40
    # Brightness threshold (0–255 mean) below which a frame is "night"
    night_brightness_threshold: float = 60.0
    # Saturation threshold for weather heuristics (0–255 HSV mean)
    rain_saturation_threshold: float = 50.0
    fog_contrast_threshold: float = 30.0

    # ── Output ────────────────────────────────────────────────────────────
    output_filename: str = "metadata.csv"
    # metadata_source tag written to every generated row
    metadata_source: str = "auto_generated"

    # ── Logging ───────────────────────────────────────────────────────────
    console_log_level: str = "INFO"
    file_log_level: str = "DEBUG"

    # ── Validation ────────────────────────────────────────────────────────

    def __post_init__(self) -> None:
        """Validate config values after construction."""
        self.raw_dir = Path(self.raw_dir)
        self.output_dir = Path(self.output_dir)

        valid_media_types = {"video", "image"}
        if self.media_type not in valid_media_types:
            raise ValueError(
                f"Invalid media_type: '{self.media_type}'. "
                f"Must be one of: {sorted(valid_media_types)}"
            )

        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError(
                f"confidence_threshold must be in [0.0, 1.0], "
                f"got {self.confidence_threshold}"
            )

        if self.scene_sample_frames < 1:
            raise ValueError(
                f"scene_sample_frames must be >= 1, got {self.scene_sample_frames}"
            )

        if self.night_brightness_threshold < 0 or self.night_brightness_threshold > 255:
            raise ValueError(
                f"night_brightness_threshold must be in [0, 255], "
                f"got {self.night_brightness_threshold}"
            )

        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if self.console_log_level.upper() not in valid_levels:
            raise ValueError(
                f"Invalid console_log_level: '{self.console_log_level}'"
            )
        if self.file_log_level.upper() not in valid_levels:
            raise ValueError(
                f"Invalid file_log_level: '{self.file_log_level}'"
            )

    # ── Serialization ─────────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable dictionary of all settings."""
        return {
            "pipeline_version": self.pipeline_version,
            "dataset_name": self.dataset_name,
            "dataset_version": self.dataset_version,
            "raw_dir": str(self.raw_dir),
            "output_dir": str(self.output_dir),
            "media_type": self.media_type,
            "video_extensions": self.video_extensions,
            "image_extensions": self.image_extensions,
            "probe_video_integrity": self.probe_video_integrity,
            "scene_sample_frames": self.scene_sample_frames,
            "enable_scene_classification": self.enable_scene_classification,
            "confidence_threshold": self.confidence_threshold,
            "night_brightness_threshold": self.night_brightness_threshold,
            "rain_saturation_threshold": self.rain_saturation_threshold,
            "fog_contrast_threshold": self.fog_contrast_threshold,
            "output_filename": self.output_filename,
            "metadata_source": self.metadata_source,
            "console_log_level": self.console_log_level,
            "file_log_level": self.file_log_level,
        }

    # ── Loaders ───────────────────────────────────────────────────────────

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> "MetadataGenConfig":
        """
        Load configuration from a YAML file.

        Args:
            yaml_path: Path to the YAML configuration file.

        Returns:
            MetadataGenConfig with values from the YAML file.

        Raises:
            FileNotFoundError: If the config file does not exist.
            ValueError: If the config file is empty or malformed.
        """
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(
                f"Metadata gen config file not found: {yaml_path}"
            )

        with open(yaml_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        if raw is None:
            raise ValueError(
                f"Metadata gen config file is empty: {yaml_path}"
            )
        if not isinstance(raw, dict):
            raise ValueError(
                f"Metadata gen config must be a YAML mapping, got "
                f"{type(raw).__name__}: {yaml_path}"
            )

        return cls._from_nested_dict(raw)

    @classmethod
    def _from_nested_dict(cls, d: Dict[str, Any]) -> "MetadataGenConfig":
        """Flatten the nested YAML structure into flat dataclass fields."""
        pipeline  = d.get("pipeline",  {})
        dataset   = d.get("dataset",   {})
        paths     = d.get("paths",     {})
        media     = d.get("media",     {})
        analysis  = d.get("analysis",  {})
        scene     = d.get("scene_classification", {})
        output    = d.get("output",    {})
        logging_  = d.get("logging",   {})

        return cls(
            pipeline_version=pipeline.get("version", "1.0.0"),
            dataset_name=dataset.get("name", "tudat"),
            dataset_version=dataset.get("version", "1.0.0"),
            raw_dir=Path(paths.get("raw_dir", "raw/tudat/Final_videos")),
            output_dir=Path(paths.get("output_dir", "raw/tudat")),
            media_type=media.get("type", "video"),
            video_extensions=media.get(
                "video_extensions",
                [".mp4", ".avi", ".mkv", ".mov", ".wmv"],
            ),
            image_extensions=media.get(
                "image_extensions",
                [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"],
            ),
            probe_video_integrity=analysis.get("probe_video_integrity", True),
            scene_sample_frames=analysis.get("scene_sample_frames", 5),
            enable_scene_classification=scene.get("enabled", True),
            confidence_threshold=scene.get("confidence_threshold", 0.40),
            night_brightness_threshold=scene.get("night_brightness_threshold", 60.0),
            rain_saturation_threshold=scene.get("rain_saturation_threshold", 50.0),
            fog_contrast_threshold=scene.get("fog_contrast_threshold", 30.0),
            output_filename=output.get("filename", "metadata.csv"),
            metadata_source=output.get("metadata_source", "auto_generated"),
            console_log_level=logging_.get("console_level", "INFO"),
            file_log_level=logging_.get("file_level", "DEBUG"),
        )

    @classmethod
    def from_overrides(
        cls,
        yaml_path: Optional[Path] = None,
        dataset_name: Optional[str] = None,
        raw_dir: Optional[str] = None,
        output_dir: Optional[str] = None,
        media_type: Optional[str] = None,
    ) -> "MetadataGenConfig":
        """
        Load config from YAML, then apply CLI overrides on top.

        Args:
            yaml_path:    Path to YAML config (uses defaults if None).
            dataset_name: Override for dataset_name.
            raw_dir:      Override for raw_dir path.
            output_dir:   Override for output_dir path.
            media_type:   Override for media_type.

        Returns:
            MetadataGenConfig with overrides applied and re-validated.
        """
        if yaml_path and Path(yaml_path).exists():
            config = cls.from_yaml(Path(yaml_path))
        else:
            config = cls()

        if dataset_name is not None:
            config.dataset_name = dataset_name
        if raw_dir is not None:
            config.raw_dir = Path(raw_dir)
        if output_dir is not None:
            config.output_dir = Path(output_dir)
        if media_type is not None:
            config.media_type = media_type

        config.__post_init__()  # Re-validate after overrides
        return config
