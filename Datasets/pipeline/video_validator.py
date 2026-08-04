"""
Video file validation and metadata extraction utilities.

Provides:
- File existence checks
- Single-frame OpenCV integrity probe
- Combined probe + FPS extraction (single file open)
- SHA-256 file hashing (delegated to utils.compute_file_hash)
- Duplicate detection across a hash column
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import pandas as pd


def check_video_exists(path: Path) -> bool:
    """Check if a video file exists and is a regular file."""
    return path.exists() and path.is_file()


def probe_video(path: Path, extract_fps: bool = True) -> Dict:
    """
    Combined video integrity probe and metadata extraction.

    Opens the file once with OpenCV, reads the first frame, and
    optionally extracts FPS. This avoids opening the file twice.

    Args:
        path: Absolute path to the video file.
        extract_fps: Whether to extract FPS from the video.

    Returns:
        Dictionary with keys:
            is_valid (bool): Whether the video can be opened and decoded.
            error (str): Error message if invalid, empty if valid.
            fps (float): Frames per second (0.0 if not extracted or invalid).
            frame_count (int): Frame count from container metadata.
            codec (str): 4-char codec identifier.
    """
    result = {
        "is_valid": False,
        "error": "",
        "fps": 0.0,
        "frame_count": 0,
        "codec": "",
    }

    if not check_video_exists(path):
        result["error"] = f"File does not exist: {path}"
        return result

    cap = None
    try:
        cap = cv2.VideoCapture(str(path))

        if not cap.isOpened():
            result["error"] = f"cv2.VideoCapture failed to open: {path.name}"
            return result

        # Read first frame to verify decode works
        ret, frame = cap.read()
        if not ret or frame is None:
            result["error"] = f"Failed to decode first frame: {path.name}"
            return result

        result["is_valid"] = True

        # Extract metadata
        if extract_fps:
            result["fps"] = round(cap.get(cv2.CAP_PROP_FPS), 3)
        result["frame_count"] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        result["codec"] = _decode_fourcc(int(cap.get(cv2.CAP_PROP_FOURCC)))

        return result

    except Exception as e:
        result["error"] = f"Exception during probe: {type(e).__name__}: {e}"
        return result

    finally:
        if cap is not None:
            cap.release()


def get_file_size(path: Path) -> int:
    """Return file size in bytes, or 0 if the file doesn't exist."""
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


def find_duplicates(df: pd.DataFrame, hash_column: str = "file_hash") -> pd.DataFrame:
    """
    Detect duplicate files based on hash values.

    Args:
        df: DataFrame with a hash column.
        hash_column: Name of the column containing file hashes.

    Returns:
        DataFrame of duplicate groups with columns:
            file_hash, count, video_ids, original_paths
        Only includes groups with count > 1.
    """
    # Filter out empty/missing hashes
    valid = df[df[hash_column].astype(str).str.len() > 0].copy()

    if valid.empty:
        return pd.DataFrame(columns=["file_hash", "count", "video_ids", "original_paths"])

    groups = (
        valid
        .groupby(hash_column)
        .agg(
            count=("video_id", "size"),
            video_ids=("video_id", list),
            original_paths=("original_path", list),
        )
        .reset_index()
    )

    duplicates = groups[groups["count"] > 1].copy()
    return duplicates.sort_values("count", ascending=False).reset_index(drop=True)


def _decode_fourcc(fourcc_int: int) -> str:
    """Decode OpenCV fourcc int to a 4-character codec string."""
    return "".join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)])
