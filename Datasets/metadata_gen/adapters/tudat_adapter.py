"""
TU-DAT dataset adapter.

Discovers .mov video files from the TU-DAT (Traffic Accident Detection)
dataset and maps folder structure to categorical labels.

Expected raw directory layout:
    <raw_dir>/
        Positive_Vidoes/           ← accident videos (typo is in the original dataset)
            v2.mov
            v10.mov
            ...
        Negative_Videos/           ← non-accident videos
            v1.mov
            ...
        challenging-environment/   ← ambiguous / difficult cases
            motorbike2.mov
            v29.mov
            ...

TU-DAT provides no official train/val/test split — split is set to None.
The split will be assigned deterministically by Pipeline 1.

References:
    TU-DAT: Traffic Accident Dataset (Technical University of Denmark).
    Binary classification dataset: accident vs. non-accident dashcam videos.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

from metadata_gen.adapters.base_adapter import BaseAdapter, MediaFile
from metadata_gen.config import MetadataGenConfig


logger = logging.getLogger(__name__)

# ── Folder → label mapping ────────────────────────────────────────────────────
# Keys are the exact directory names as they appear on disk.
# Values are the normalized labels written to the `type` column.
# Note: "Positive_Vidoes" is a typo in the original TU-DAT dataset — preserved here.
_FOLDER_TO_LABEL: Dict[str, str] = {
    "Positive_Vidoes":        "accident",
    "Negative_Videos":        "non-accident",
    "challenging-environment": "challenging",
}

# Folders that are known to exist but should be skipped entirely
_SKIP_FOLDERS: frozenset = frozenset()  # None currently; future-proofing


class TudatAdapter(BaseAdapter):
    """
    File discovery adapter for the TU-DAT video dataset.

    Walks the three known subdirectories, maps each to a label, and
    returns one MediaFile per .mov file discovered.

    Unknown subdirectories found during discovery are logged as warnings
    and skipped — they do not cause an error.
    """

    # ── BaseAdapter interface ─────────────────────────────────────────────

    def dataset_name(self) -> str:
        return self.config.dataset_name  # "tudat"

    def media_type(self) -> str:
        return "video"

    def discover_files(self) -> List[MediaFile]:
        """
        Walk raw_dir, discover all .mov files, and return MediaFile objects.

        Each MediaFile has:
        - label set from folder name ("accident" | "non-accident" | "challenging")
        - split = None (TU-DAT has no official split)
        - source_type = "real"
        - metadata_source = "folder_structure" for label field

        Returns:
            List of MediaFile objects. Empty list if no files are found.

        Raises:
            FileNotFoundError: If raw_dir does not exist (caught by BaseAdapter.__init__).
        """
        discovered: List[MediaFile] = []
        extensions = self.config.video_extensions

        # Iterate subdirectories in raw_dir
        subdirs = sorted(p for p in self.raw_dir.iterdir() if p.is_dir())

        if not subdirs:
            logger.warning(
                "TudatAdapter: No subdirectories found in raw_dir: %s",
                self.raw_dir,
            )
            return discovered

        for subdir in subdirs:
            folder_name = subdir.name

            # Skip known non-data folders
            if folder_name in _SKIP_FOLDERS:
                logger.debug("TudatAdapter: Skipping folder: %s", folder_name)
                continue

            # Resolve label from folder name
            label = _FOLDER_TO_LABEL.get(folder_name)
            if label is None:
                logger.warning(
                    "TudatAdapter: Unrecognised folder '%s' in %s — skipping. "
                    "If this is a new category, add it to _FOLDER_TO_LABEL.",
                    folder_name,
                    self.raw_dir,
                )
                continue

            # Collect media files in this subdirectory (non-recursive)
            try:
                media_paths = self._collect_media_files(
                    directory=subdir,
                    extensions=extensions,
                    recursive=False,
                )
            except (FileNotFoundError, NotADirectoryError) as exc:
                logger.error(
                    "TudatAdapter: Cannot scan subdirectory '%s': %s",
                    subdir,
                    exc,
                )
                continue

            if not media_paths:
                logger.warning(
                    "TudatAdapter: Folder '%s' contains no recognised media files "
                    "(extensions: %s).",
                    subdir,
                    extensions,
                )
                continue

            logger.debug(
                "TudatAdapter: '%s' → label='%s', %d file(s)",
                folder_name,
                label,
                len(media_paths),
            )

            for abs_path in media_paths:
                relative_path = self._make_relative_path(abs_path)

                media_file = MediaFile(
                    absolute_path=abs_path,
                    relative_path=relative_path,
                    filename=abs_path.name,
                    extension=abs_path.suffix.lower(),
                    media_type="video",
                    dataset_name=self.config.dataset_name,
                    dataset_version=self.config.dataset_version,
                    source_type="real",
                    label=label,
                    split=None,          # TU-DAT provides no official split
                    metadata_source="folder_structure",
                )
                discovered.append(media_file)

        logger.info(self.summary(discovered))
        return discovered

    # ── Class-level introspection ─────────────────────────────────────────

    @staticmethod
    def expected_folders() -> List[str]:
        """Return the list of folder names this adapter expects to find."""
        return list(_FOLDER_TO_LABEL.keys())

    @staticmethod
    def label_for_folder(folder_name: str) -> Optional[str]:
        """
        Return the normalized label for a given folder name, or None if unknown.

        Useful for unit tests and for inspecting the mapping without
        instantiating the adapter.
        """
        return _FOLDER_TO_LABEL.get(folder_name)
