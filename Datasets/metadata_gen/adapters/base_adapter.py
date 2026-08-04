"""
Abstract base adapter for dataset file discovery.

Defines the contract every dataset adapter must implement.
An adapter is responsible for exactly two things:
  1. Discovering all valid media files in the raw dataset directory.
  2. Providing the dataset-specific metadata that can only be inferred from
     folder structure or naming conventions (label, split, source_type).

Adapters do NOT perform video/image analysis — that is the Analyzer's job.
Adapters do NOT classify scenes — that is the SceneClassifier's job.
Single Responsibility: one adapter per dataset, one concern per adapter.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from metadata_gen.config import MetadataGenConfig


# ── MediaFile ─────────────────────────────────────────────────────────────────

@dataclass
class MediaFile:
    """
    A single discovered media file with all adapter-derived metadata.

    Passed from an adapter to the Analyzer and SceneClassifier.
    Fields filled in by the adapter are guaranteed to be present.
    Fields filled in by downstream stages start as None.
    """

    # ── File identity (set by adapter) ────────────────────────────────────
    absolute_path: Path
    relative_path: str          # Relative to raw_dir (used as original_path in CSV)
    filename: str               # e.g. "v10.mov"
    extension: str              # Lower-case, with dot, e.g. ".mov"
    media_type: str             # "video" | "image"

    # ── Dataset identity (set by adapter) ─────────────────────────────────
    dataset_name: str
    dataset_version: str
    source_type: str            # "real" | "synthetic"

    # ── Adapter-inferred labels (set by adapter, from folder structure) ───
    # The accident category / class label derived from directory naming.
    # e.g. "accident", "non-accident", "challenging" for TU-DAT;
    #      "Accident", "Non Accident" for Kaggle.
    # This is written as-is to the `type` column; no normalization here.
    label: str

    # Dataset split if the folder structure encodes it (e.g. train/val/test).
    # None if the adapter cannot determine this from directory layout.
    split: Optional[str] = None

    # ── Provenance tag ────────────────────────────────────────────────────
    # "folder_structure" for label/split derived from directory naming.
    # "auto_generated" for everything filled in by analyzers/classifiers.
    metadata_source: str = "folder_structure"

    # ── Video properties (filled by VideoAnalyzer) ────────────────────────
    fps: Optional[float] = None
    no_frames: Optional[int] = None
    duration: Optional[float] = None
    height: Optional[int] = None
    width: Optional[int] = None
    codec: Optional[str] = None
    file_size_bytes: Optional[int] = None
    is_video_valid: Optional[bool] = None
    video_error: Optional[str] = None

    # ── Image properties (filled by ImageAnalyzer) ────────────────────────
    # For images: height/width reuse the video fields above.
    channels: Optional[int] = None
    is_image_valid: Optional[bool] = None
    image_error: Optional[str] = None

    # ── Scene classification (filled by SceneClassifier) ──────────────────
    day_time: Optional[str] = None              # "day" | "night" | NULL
    day_time_confidence: Optional[float] = None
    weather: Optional[str] = None               # "clear" | "rain" | "fog" | NULL
    weather_confidence: Optional[float] = None

    # ── Fields that cannot be inferred (always NULL from Pipeline 0) ──────
    # accident_time, accident_frame, center_x/y, x1/y1/x2/y2,
    # rollover, region, scene_layout, quality — all left as None.

    def is_valid(self) -> bool:
        """
        Return True if the file is usable for metadata generation.

        A file is valid if its analyzer (video or image) confirmed it can
        be opened and decoded. Files that fail analysis are still written
        to the CSV with their error recorded, but flagged.
        """
        if self.media_type == "video":
            return self.is_video_valid is True
        elif self.media_type == "image":
            return self.is_image_valid is True
        return False

    def __str__(self) -> str:
        return (
            f"MediaFile({self.dataset_name}/{self.media_type} "
            f"label={self.label!r} path={self.relative_path!r})"
        )


# ── BaseAdapter ───────────────────────────────────────────────────────────────

class BaseAdapter(ABC):
    """
    Abstract base class for dataset file discovery adapters.

    Subclasses implement discover_files() to walk their dataset's directory
    structure and return a list of MediaFile objects. Each MediaFile contains
    the complete adapter-derived metadata for that file.

    Usage:
        config = MetadataGenConfig.from_yaml("metadata_gen_config.yaml")
        adapter = TudatAdapter(config)
        files = adapter.discover_files()
        # → List[MediaFile], one per video/image found
    """

    def __init__(self, config: MetadataGenConfig) -> None:
        """
        Args:
            config: Pipeline 0 configuration object.

        Raises:
            FileNotFoundError: If config.raw_dir does not exist.
        """
        if not config.raw_dir.exists():
            raise FileNotFoundError(
                f"[{self.__class__.__name__}] raw_dir does not exist: "
                f"{config.raw_dir}"
            )
        if not config.raw_dir.is_dir():
            raise NotADirectoryError(
                f"[{self.__class__.__name__}] raw_dir is not a directory: "
                f"{config.raw_dir}"
            )
        self.config = config
        self.raw_dir = config.raw_dir

    # ── Abstract interface ────────────────────────────────────────────────

    @abstractmethod
    def discover_files(self) -> List[MediaFile]:
        """
        Walk the raw dataset directory and return all valid media files.

        Each returned MediaFile must have:
        - absolute_path, relative_path, filename, extension populated
        - media_type set to "video" or "image"
        - dataset_name, dataset_version, source_type populated
        - label set (from folder structure or naming convention)
        - split set if determinable from folder structure, else None
        - metadata_source set to "folder_structure" for label/split fields

        Files that cannot be categorized (e.g. .DS_Store, README.txt)
        must be silently skipped — not raised as errors.

        Returns:
            List of MediaFile objects, one per discovered media file.
            Empty list if no files are found (not an error).
        """
        ...

    @abstractmethod
    def dataset_name(self) -> str:
        """
        Return the canonical dataset name (e.g. "tudat", "kaggle").

        Must match config.dataset_name.
        """
        ...

    @abstractmethod
    def media_type(self) -> str:
        """
        Return the media type this adapter handles: "video" or "image".

        Must match config.media_type.
        """
        ...

    # ── Concrete helpers ──────────────────────────────────────────────────

    def _collect_media_files(
        self,
        directory: Path,
        extensions: List[str],
        recursive: bool = False,
    ) -> List[Path]:
        """
        Collect all files with matching extensions from a directory.

        Args:
            directory:  Directory to scan. Must exist.
            extensions: List of lower-case extensions with dot (e.g. [".mov"]).
            recursive:  If True, scan all subdirectories recursively.

        Returns:
            Sorted list of absolute file paths matching the extensions.
            Files starting with "." (e.g. .DS_Store) are always excluded.

        Raises:
            FileNotFoundError: If directory does not exist.
            NotADirectoryError: If the path is not a directory.
        """
        if not directory.exists():
            raise FileNotFoundError(
                f"[{self.__class__.__name__}] Directory not found: {directory}"
            )
        if not directory.is_dir():
            raise NotADirectoryError(
                f"[{self.__class__.__name__}] Expected directory, got file: {directory}"
            )

        ext_set = {e.lower() for e in extensions}
        pattern = "**/*" if recursive else "*"

        found: List[Path] = []
        for path in directory.glob(pattern):
            if not path.is_file():
                continue
            if path.name.startswith("."):
                continue  # Skip .DS_Store, hidden files
            if path.suffix.lower() in ext_set:
                found.append(path)

        return sorted(found)

    def _make_relative_path(self, absolute_path: Path) -> str:
        """
        Return a path relative to raw_dir, using forward slashes.

        This becomes the `original_path` column in the metadata CSV,
        consistent with how Picek stores paths.

        Args:
            absolute_path: Absolute path to the media file.

        Returns:
            Relative path string with forward slashes.

        Raises:
            ValueError: If absolute_path is not under raw_dir.
        """
        try:
            rel = absolute_path.resolve().relative_to(self.raw_dir.resolve())
        except ValueError:
            raise ValueError(
                f"[{self.__class__.__name__}] File is not under raw_dir.\n"
                f"  raw_dir:  {self.raw_dir}\n"
                f"  file:     {absolute_path}"
            )
        return str(rel).replace("\\", "/")

    def summary(self, files: List[MediaFile]) -> str:
        """
        Return a human-readable summary of discovered files.

        Args:
            files: List of discovered MediaFile objects.

        Returns:
            Multi-line summary string for logging.
        """
        if not files:
            return f"[{self.__class__.__name__}] No files discovered in {self.raw_dir}"

        label_counts: dict = {}
        split_counts: dict = {}
        for f in files:
            label_counts[f.label] = label_counts.get(f.label, 0) + 1
            s = f.split or "unknown"
            split_counts[s] = split_counts.get(s, 0) + 1

        lines = [
            f"[{self.__class__.__name__}] Discovered {len(files)} file(s) "
            f"in {self.raw_dir}",
            f"  Labels:  {dict(sorted(label_counts.items()))}",
            f"  Splits:  {dict(sorted(split_counts.items()))}",
        ]
        return "\n".join(lines)
