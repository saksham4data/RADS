from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np


SAFE_MARGIN_FRAC = 0.15


def probe_reported_frame_count(video_path: Path) -> int:
    """Probe the container-reported frame count via OpenCV."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return max(0, count)


def read_frame_with_fallback(
    video_path: Path,
    frame_idx: int,
) -> Optional[np.ndarray]:
    """Read a specific frame using robust seek and sequential fallback."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    actual_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

    if actual_idx == frame_idx:
        ret, frame = cap.read()
        if ret and frame is not None:
            cap.release()
            return frame
    elif 0 <= actual_idx < frame_idx:
        success = True
        for _ in range(frame_idx - actual_idx):
            ret = cap.grab()
            if not ret:
                success = False
                break
        if success:
            ret, frame = cap.read()
            if ret and frame is not None:
                cap.release()
                return frame

        cap.release()
        return None

    cap.release()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    success = True
    for _ in range(frame_idx):
        ret = cap.grab()
        if not ret:
            success = False
            break

    frame = None
    if success:
        ret, frame = cap.read()
        if not ret:
            frame = None

    cap.release()
    return frame


def probe_decodable_frame_count(
    video_path: Path,
    reported_frame_count: int = 0,
    *,
    frame_reader: Optional[Callable[[Path, int], Optional[np.ndarray]]] = None,
) -> int:
    """Return the length of the contiguous decodable frame prefix."""
    upper_bound = reported_frame_count if reported_frame_count > 0 else probe_reported_frame_count(video_path)
    if upper_bound <= 0:
        return 0

    reader = frame_reader or read_frame_with_fallback
    if reader(video_path, 0) is None:
        return 0

    lo, hi = 1, upper_bound
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if reader(video_path, mid - 1) is not None:
            lo = mid
        else:
            hi = mid - 1
    return lo


def safe_range(total_frames: int) -> Tuple[int, int]:
    """Return the [lo, hi) safe frame range excluding boundary frames."""
    margin = max(1, int(total_frames * SAFE_MARGIN_FRAC))
    lo = min(margin, total_frames - 1)
    hi = max(lo + 1, total_frames - margin)
    return lo, hi


def compute_sample_indices(
    total_frames: int,
    frames_per_video: int,
    strategy: str,
) -> List[int]:
    """Compute which frame indices to sample from a decodable frame range."""
    n = min(frames_per_video, total_frames)

    if strategy == "uniform":
        lo, hi = safe_range(total_frames)
        safe_count = hi - lo
        if n == 1:
            return [lo + safe_count // 2]
        return [
            lo + int(i * (safe_count - 1) / (n - 1))
            for i in range(n)
        ]
    if strategy == "random":
        lo, hi = safe_range(total_frames)
        safe_count = hi - lo
        n = min(n, safe_count)
        rng = np.random.default_rng()
        return sorted(
            (lo + rng.choice(safe_count, size=n, replace=False)).tolist()
        )
    if strategy == "first":
        return list(range(n))
    if strategy == "middle":
        mid = total_frames // 2
        start = max(0, mid - n // 2)
        return list(range(start, min(start + n, total_frames)))
    raise ValueError(f"Unknown sampling strategy: '{strategy}'")
