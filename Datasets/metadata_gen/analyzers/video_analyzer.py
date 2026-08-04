"""
Video file analyzer for the Metadata Generation Pipeline.

Extracts all technically-determinable properties from a video file using
OpenCV. Populates the video-specific fields of a MediaFile in-place and
returns a list of sampled frames for the SceneClassifier.

Responsibilities (Single Responsibility: technical video metadata only):
    - File existence and size
    - Integrity probe (first-frame decode)
    - FPS, frame count, duration, resolution, codec
    - Evenly-sampled frames for downstream scene classification

Does NOT:
    - Classify scenes (that is SceneClassifier's job)
    - Hash files (that is the Assembler/Pipeline 1's job)
    - Make decisions about labels or splits
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from metadata_gen.adapters.base_adapter import MediaFile
from metadata_gen.config import MetadataGenConfig


logger = logging.getLogger(__name__)


class VideoAnalyzer:
    """
    OpenCV-based video metadata extractor.

    Usage:
        analyzer = VideoAnalyzer(config)
        frames = analyzer.analyze(media_file)
        # media_file.fps, .no_frames, .duration, .height, .width, ... are now set
        # frames is a List[np.ndarray] ready for SceneClassifier
    """

    def __init__(self, config: MetadataGenConfig) -> None:
        """
        Args:
            config: Pipeline 0 configuration. Used for:
                - probe_video_integrity: whether to decode first frame
                - scene_sample_frames: number of frames to sample
        """
        self.config = config

    # ── Public interface ──────────────────────────────────────────────────

    def analyze(self, media_file: MediaFile) -> List[np.ndarray]:
        """
        Extract all video metadata and populate the MediaFile in-place.

        Opens the video file once, extracts all properties, samples frames
        for scene classification, then releases the capture. Never raises —
        all errors are caught, recorded on media_file, and logged.

        Args:
            media_file: A MediaFile with absolute_path set. Must have
                        media_type == "video".

        Returns:
            List of sampled BGR frames (np.ndarray, shape H×W×3) for the
            SceneClassifier. Empty list if the file is invalid or unreadable.
        """
        path = media_file.absolute_path

        # ── Guard: media_type ──────────────────────────────────────────
        if media_file.media_type != "video":
            err = (
                f"VideoAnalyzer called on non-video MediaFile "
                f"(media_type={media_file.media_type!r}): {path}"
            )
            logger.error(err)
            media_file.is_video_valid = False
            media_file.video_error = err
            return []

        # ── Guard: file existence ──────────────────────────────────────
        if not path.exists():
            err = f"File does not exist: {path}"
            logger.error("VideoAnalyzer: %s", err)
            media_file.is_video_valid = False
            media_file.video_error = err
            return []

        if not path.is_file():
            err = f"Path is not a regular file: {path}"
            logger.error("VideoAnalyzer: %s", err)
            media_file.is_video_valid = False
            media_file.video_error = err
            return []

        # ── File size ──────────────────────────────────────────────────
        media_file.file_size_bytes = _safe_file_size(path)

        if media_file.file_size_bytes == 0:
            err = f"File is empty (0 bytes): {path.name}"
            logger.warning("VideoAnalyzer: %s", err)
            media_file.is_video_valid = False
            media_file.video_error = err
            return []

        # ── Open with OpenCV ───────────────────────────────────────────
        cap: Optional[cv2.VideoCapture] = None
        sampled_frames: List[np.ndarray] = []

        try:
            cap = cv2.VideoCapture(str(path))

            if not cap.isOpened():
                err = f"cv2.VideoCapture failed to open: {path.name}"
                logger.warning("VideoAnalyzer: %s", err)
                media_file.is_video_valid = False
                media_file.video_error = err
                return []

            # ── Extract container-level properties ─────────────────────
            raw_fps         = cap.get(cv2.CAP_PROP_FPS)
            raw_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            raw_width       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            raw_height      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            raw_fourcc      = int(cap.get(cv2.CAP_PROP_FOURCC))

            # ── Integrity probe: decode first frame ────────────────────
            if self.config.probe_video_integrity:
                ret, first_frame = cap.read()
                if not ret or first_frame is None:
                    err = f"Failed to decode first frame: {path.name}"
                    logger.warning("VideoAnalyzer: %s", err)
                    media_file.is_video_valid = False
                    media_file.video_error = err
                    return []
                # Seek back to beginning for frame sampling
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            else:
                first_frame = None

            # ── Validate extracted properties ──────────────────────────
            # FPS: OpenCV occasionally reports 0 for certain codecs/containers
            fps: Optional[float] = None
            if raw_fps > 0:
                fps = round(raw_fps, 4)
            else:
                logger.warning(
                    "VideoAnalyzer: FPS reported as 0 for '%s'. "
                    "The container may not store FPS metadata.",
                    path.name,
                )

            # Resolution: 0 means OpenCV could not determine it
            height: Optional[int] = raw_height if raw_height > 0 else None
            width:  Optional[int] = raw_width  if raw_width  > 0 else None

            if height is None or width is None:
                logger.warning(
                    "VideoAnalyzer: Zero resolution reported for '%s' "
                    "(height=%d, width=%d). File may be corrupted.",
                    path.name, raw_height, raw_width,
                )

            # Frame count: OpenCV's CAP_PROP_FRAME_COUNT is unreliable for
            # some containers (e.g. .mov). We use it when > 0 but do not
            # treat 0 as a hard error.
            no_frames: Optional[int] = raw_frame_count if raw_frame_count > 0 else None

            # Duration: computed from frame count ÷ fps when both are available
            duration: Optional[float] = None
            if fps and fps > 0 and no_frames and no_frames > 0:
                duration = round(no_frames / fps, 4)

            codec: str = _decode_fourcc(raw_fourcc)

            # ── Populate MediaFile ─────────────────────────────────────
            media_file.fps           = fps
            media_file.no_frames     = no_frames
            media_file.duration      = duration
            media_file.height        = height
            media_file.width         = width
            media_file.codec         = codec
            media_file.is_video_valid = True
            media_file.video_error   = None

            # ── Sample frames for scene classification ─────────────────
            n_samples = self.config.scene_sample_frames
            if self.config.enable_scene_classification and n_samples > 0:
                sampled_frames = _sample_frames(
                    cap=cap,
                    total_frames=raw_frame_count,
                    n_samples=n_samples,
                    fallback_first_frame=first_frame,
                )
                if not sampled_frames:
                    logger.warning(
                        "VideoAnalyzer: No frames could be sampled from '%s'. "
                        "Scene classification will be skipped.",
                        path.name,
                    )

            logger.debug(
                "VideoAnalyzer: '%s' — fps=%.3f, frames=%s, duration=%ss, "
                "%sx%s, codec=%r, samples=%d",
                path.name,
                fps or 0.0,
                no_frames,
                duration,
                width,
                height,
                codec,
                len(sampled_frames),
            )

        except Exception as exc:
            err = f"Unexpected exception during analysis of '{path.name}': {type(exc).__name__}: {exc}"
            logger.error("VideoAnalyzer: %s", err, exc_info=True)
            media_file.is_video_valid = False
            media_file.video_error    = err
            sampled_frames = []

        finally:
            if cap is not None:
                cap.release()

        return sampled_frames


# ── Module-level helpers ──────────────────────────────────────────────────────

def _sample_frames(
    cap: cv2.VideoCapture,
    total_frames: int,
    n_samples: int,
    fallback_first_frame: Optional[np.ndarray] = None,
) -> List[np.ndarray]:
    """
    Sample up to n_samples frames evenly distributed across the video.

    Seeks to each target frame position and reads. Positions that fail to
    read are silently skipped — we never raise inside a sampling operation.

    Args:
        cap:                  Open VideoCapture (caller owns release).
        total_frames:         CAP_PROP_FRAME_COUNT from the container metadata.
                              May be 0 for some containers — handled below.
        n_samples:            Desired number of sample frames.
        fallback_first_frame: If total_frames == 0, use this pre-read frame
                              as the single sample (avoids re-opening).

    Returns:
        List of BGR frames as np.ndarray. May be shorter than n_samples
        if seeks or reads fail. Empty list only if nothing could be read.
    """
    if n_samples <= 0:
        return []

    # Fallback: if frame count is unknown, return the first frame only
    if total_frames <= 0:
        if fallback_first_frame is not None:
            return [fallback_first_frame]
        # Try to read frame 0 directly
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()
        return [frame] if ret and frame is not None else []

    # Compute evenly-spaced sample positions within [0, total_frames)
    if total_frames <= n_samples:
        # Fewer frames than requested samples — sample every frame
        positions = list(range(total_frames))
    else:
        # Spread n_samples positions across the full duration
        step = total_frames / n_samples
        positions = [int(i * step) for i in range(n_samples)]

    frames: List[np.ndarray] = []
    for pos in positions:
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ret, frame = cap.read()
        if ret and frame is not None:
            frames.append(frame)

    return frames


def _decode_fourcc(fourcc_int: int) -> str:
    """
    Decode an OpenCV fourcc integer to a 4-character codec string.

    Args:
        fourcc_int: Integer returned by cap.get(cv2.CAP_PROP_FOURCC).

    Returns:
        4-character string (e.g. "avc1", "mp4v"). Empty string if decoding
        produces non-printable characters.
    """
    chars = []
    for i in range(4):
        byte = (fourcc_int >> (8 * i)) & 0xFF
        chars.append(chr(byte) if 32 <= byte < 127 else "?")
    return "".join(chars)


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
