"""
Image file analyzer for the Metadata Generation Pipeline.

Extracts all technically-determinable properties from an image file using
OpenCV. Populates the image-specific fields of a MediaFile in-place and
returns a list containing the loaded image for the SceneClassifier.

Responsibilities (Single Responsibility: technical image metadata only):
    - File existence and size
    - Integrity check (cv2.imread decode)
    - Resolution (height, width) and channel count
    - Returns the loaded image frame for downstream scene classification

Does NOT:
    - Classify scenes (that is SceneClassifier's job)
    - Handle video files (use VideoAnalyzer for those)
    - Make decisions about labels or splits

Interface contract:
    analyze(media_file) always returns List[np.ndarray]:
        - [image_array] on success (list of length 1)
        - []            on failure

    This mirrors VideoAnalyzer.analyze() so the SceneClassifier receives
    a consistent List[np.ndarray] input from both media types.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List

import cv2
import numpy as np

from metadata_gen.adapters.base_adapter import MediaFile
from metadata_gen.config import MetadataGenConfig


logger = logging.getLogger(__name__)


class ImageAnalyzer:
    """
    OpenCV-based image metadata extractor.

    Usage:
        analyzer = ImageAnalyzer(config)
        frames = analyzer.analyze(media_file)
        # media_file.height, .width, .channels, .file_size_bytes, ... are now set
        # frames == [img_array] ready for SceneClassifier
    """

    def __init__(self, config: MetadataGenConfig) -> None:
        """
        Args:
            config: Pipeline 0 configuration. Used for:
                - enable_scene_classification: whether to return the image frame
        """
        self.config = config

    # ── Public interface ──────────────────────────────────────────────────

    def analyze(self, media_file: MediaFile) -> List[np.ndarray]:
        """
        Load the image, extract metadata, and populate the MediaFile in-place.

        Never raises — all errors are caught, recorded on media_file, and logged.

        Args:
            media_file: A MediaFile with absolute_path set. Must have
                        media_type == "image".

        Returns:
            List containing the loaded BGR image as a single np.ndarray,
            for the SceneClassifier. Returns [] on any failure.
        """
        path = media_file.absolute_path

        # ── Guard: media_type ──────────────────────────────────────────
        if media_file.media_type != "image":
            err = (
                f"ImageAnalyzer called on non-image MediaFile "
                f"(media_type={media_file.media_type!r}): {path}"
            )
            logger.error(err)
            media_file.is_image_valid = False
            media_file.image_error = err
            return []

        # ── Guard: file existence ──────────────────────────────────────
        if not path.exists():
            err = f"File does not exist: {path}"
            logger.error("ImageAnalyzer: %s", err)
            media_file.is_image_valid = False
            media_file.image_error = err
            return []

        if not path.is_file():
            err = f"Path is not a regular file: {path}"
            logger.error("ImageAnalyzer: %s", err)
            media_file.is_image_valid = False
            media_file.image_error = err
            return []

        # ── File size ──────────────────────────────────────────────────
        media_file.file_size_bytes = _safe_file_size(path)

        if media_file.file_size_bytes == 0:
            err = f"File is empty (0 bytes): {path.name}"
            logger.warning("ImageAnalyzer: %s", err)
            media_file.is_image_valid = False
            media_file.image_error = err
            return []

        # ── Load image ─────────────────────────────────────────────────
        try:
            # cv2.IMREAD_COLOR: always loads as BGR (3-channel), even for greyscale.
            # cv2.IMREAD_UNCHANGED: preserves alpha channel if present.
            # We use IMREAD_UNCHANGED so we can report the true channel count.
            img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

        except Exception as exc:
            err = (
                f"cv2.imread raised an exception for '{path.name}': "
                f"{type(exc).__name__}: {exc}"
            )
            logger.error("ImageAnalyzer: %s", err, exc_info=True)
            media_file.is_image_valid = False
            media_file.image_error = err
            return []

        # ── Validate decode result ─────────────────────────────────────
        if img is None:
            err = (
                f"cv2.imread returned None for '{path.name}'. "
                "The file may be corrupted or an unsupported format."
            )
            logger.warning("ImageAnalyzer: %s", err)
            media_file.is_image_valid = False
            media_file.image_error = err
            return []

        # ── Extract properties from the decoded array ──────────────────
        if img.ndim == 2:
            # Greyscale image: shape is (H, W)
            height, width = img.shape
            channels = 1
        elif img.ndim == 3:
            # Colour image: shape is (H, W, C)
            height, width, channels = img.shape
        else:
            err = (
                f"Unexpected image array dimensions ({img.ndim}) "
                f"for '{path.name}'."
            )
            logger.error("ImageAnalyzer: %s", err)
            media_file.is_image_valid = False
            media_file.image_error = err
            return []

        if height == 0 or width == 0:
            err = f"Zero-dimension image ({height}×{width}) for '{path.name}'."
            logger.warning("ImageAnalyzer: %s", err)
            media_file.is_image_valid = False
            media_file.image_error = err
            return []

        # ── Populate MediaFile ─────────────────────────────────────────
        media_file.height        = height
        media_file.width         = width
        media_file.channels      = channels
        media_file.is_image_valid = True
        media_file.image_error   = None

        # Video-only fields are not applicable for images — left as None:
        # fps, no_frames, duration, codec remain None.
        # The assembler will write them as NULL in the CSV.

        logger.debug(
            "ImageAnalyzer: '%s' — %dx%d, channels=%d, size=%d bytes",
            path.name,
            width,
            height,
            channels,
            media_file.file_size_bytes,
        )

        # ── Return frame for SceneClassifier ──────────────────────────
        # Convert IMREAD_UNCHANGED result to BGR (3-channel) if needed,
        # so the SceneClassifier receives a consistent input format.
        if not self.config.enable_scene_classification:
            return []

        bgr_frame = _to_bgr(img)
        return [bgr_frame]


# ── Module-level helpers ──────────────────────────────────────────────────────

def _to_bgr(img: np.ndarray) -> np.ndarray:
    """
    Convert any OpenCV-loaded image to a standard 3-channel BGR array.

    Handles:
        - Greyscale (H, W)       → BGR (H, W, 3)
        - BGRA (H, W, 4)         → BGR (H, W, 3)   [alpha dropped]
        - BGR (H, W, 3)          → unchanged

    Args:
        img: Image array as returned by cv2.imread(IMREAD_UNCHANGED).

    Returns:
        3-channel BGR np.ndarray of the same spatial dimensions.
    """
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.ndim == 3 and img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img  # Already BGR (H, W, 3)


def _safe_file_size(path: Path) -> int:
    """
    Return file size in bytes, or 0 if the OS call fails.

    Args:
        path: Path to the file.

    Returns:
        File size in bytes, or 0 on OSError.
    """
    try:
        return os.path.getsize(path)
    except OSError:
        return 0
