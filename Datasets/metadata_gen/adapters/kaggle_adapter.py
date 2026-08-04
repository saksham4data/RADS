"""
Kaggle accident image dataset adapter.

Discovers .jpg image files from the Kaggle road accident image classification
dataset and maps the folder hierarchy to split and label fields.

Expected raw directory layout:
    <raw_dir>/
        train/
            Accident/
                acc1 (1).jpg
                acc1 (13).jpg
                ...
            Non Accident/
                5_10.jpg
                ...
        val/
            Accident/
                ...
            Non Accident/
                ...
        test/
            Accident/
                ...
            Non Accident/
                ...

The folder hierarchy directly encodes both the split (train/val/test)
and the binary label (Accident / Non Accident).

This is an image-only dataset. All video-related fields (fps, duration,
no_frames, codec) will remain NULL after analysis. The ImageAnalyzer
fills height, width, channels, and file_size_bytes only.

media_type is set to "image" so that Pipeline 1 can skip video probing
when it later processes this dataset.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional

from metadata_gen.adapters.base_adapter import BaseAdapter, MediaFile
from metadata_gen.config import MetadataGenConfig


logger = logging.getLogger(__name__)

# ── Folder → label mapping ────────────────────────────────────────────────────
# Keys: exact class subdirectory names as they appear on disk.
# Values: normalized labels written to the `type` column.
_LABEL_FOLDER_TO_LABEL: Dict[str, str] = {
    "Accident":    "accident",
    "Non Accident": "non-accident",
}

# ── Recognized split folder names ─────────────────────────────────────────────
# The Kaggle dataset uses "val" — normalized to "val" (kept as-is, not "validation")
# to preserve the original split semantics.
_VALID_SPLITS: FrozenSet[str] = frozenset({"train", "val", "test"})


class KaggleAdapter(BaseAdapter):
    """
    File discovery adapter for the Kaggle road accident image dataset.

    Walks the three-level directory hierarchy (split / label / files),
    extracting split and label from folder names.

    Unrecognised split folders or label folders are logged as warnings
    and skipped — they do not cause errors.
    """

    # ── BaseAdapter interface ─────────────────────────────────────────────

    def dataset_name(self) -> str:
        return self.config.dataset_name  # "kaggle"

    def media_type(self) -> str:
        return "image"

    def discover_files(self) -> List[MediaFile]:
        """
        Walk raw_dir, discover all image files, and return MediaFile objects.

        Each MediaFile has:
        - label derived from class folder ("accident" | "non-accident")
        - split derived from top-level folder ("train" | "val" | "test")
        - media_type = "image"
        - source_type = "real"
        - metadata_source = "folder_structure"

        Returns:
            List of MediaFile objects. Empty list if no files are found.

        Raises:
            FileNotFoundError: If raw_dir does not exist (caught by BaseAdapter.__init__).
        """
        discovered: List[MediaFile] = []
        extensions = self.config.image_extensions

        # Top-level = split folders (train, val, test)
        split_dirs = sorted(p for p in self.raw_dir.iterdir() if p.is_dir())

        if not split_dirs:
            logger.warning(
                "KaggleAdapter: No subdirectories found in raw_dir: %s",
                self.raw_dir,
            )
            return discovered

        for split_dir in split_dirs:
            split_name = split_dir.name

            if split_name not in _VALID_SPLITS:
                logger.warning(
                    "KaggleAdapter: Unrecognised split folder '%s' in %s — skipping. "
                    "Expected one of: %s.",
                    split_name,
                    self.raw_dir,
                    sorted(_VALID_SPLITS),
                )
                continue

            # Second-level = label folders (Accident, Non Accident)
            label_dirs = sorted(p for p in split_dir.iterdir() if p.is_dir())

            if not label_dirs:
                logger.warning(
                    "KaggleAdapter: Split folder '%s' contains no label subdirectories.",
                    split_dir,
                )
                continue

            for label_dir in label_dirs:
                label_folder = label_dir.name
                label = _LABEL_FOLDER_TO_LABEL.get(label_folder)

                if label is None:
                    logger.warning(
                        "KaggleAdapter: Unrecognised label folder '%s' in '%s' — skipping. "
                        "If this is a new class, add it to _LABEL_FOLDER_TO_LABEL.",
                        label_folder,
                        split_dir,
                    )
                    continue

                # Collect image files in this leaf directory (non-recursive)
                try:
                    image_paths = self._collect_media_files(
                        directory=label_dir,
                        extensions=extensions,
                        recursive=False,
                    )
                except (FileNotFoundError, NotADirectoryError) as exc:
                    logger.error(
                        "KaggleAdapter: Cannot scan '%s': %s",
                        label_dir,
                        exc,
                    )
                    continue

                if not image_paths:
                    logger.warning(
                        "KaggleAdapter: '%s/%s' contains no recognised image files "
                        "(extensions: %s).",
                        split_name,
                        label_folder,
                        extensions,
                    )
                    continue

                logger.debug(
                    "KaggleAdapter: '%s/%s' → split='%s', label='%s', %d file(s)",
                    split_name,
                    label_folder,
                    split_name,
                    label,
                    len(image_paths),
                )

                for abs_path in image_paths:
                    relative_path = self._make_relative_path(abs_path)

                    media_file = MediaFile(
                        absolute_path=abs_path,
                        relative_path=relative_path,
                        filename=abs_path.name,
                        extension=abs_path.suffix.lower(),
                        media_type="image",
                        dataset_name=self.config.dataset_name,
                        dataset_version=self.config.dataset_version,
                        source_type="real",
                        label=label,
                        split=split_name,    # train | val | test
                        metadata_source="folder_structure",
                    )
                    discovered.append(media_file)

        logger.info(self.summary(discovered))
        return discovered

    # ── Class-level introspection ─────────────────────────────────────────

    @staticmethod
    def valid_splits() -> FrozenSet[str]:
        """Return the set of split folder names this adapter recognises."""
        return _VALID_SPLITS

    @staticmethod
    def label_for_folder(folder_name: str) -> Optional[str]:
        """
        Return the normalized label for a given label folder name, or None.

        Useful for unit tests and mapping inspection without instantiation.
        """
        return _LABEL_FOLDER_TO_LABEL.get(folder_name)
