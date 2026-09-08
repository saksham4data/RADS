import cv2
import os
from typing import Iterator, Tuple
import numpy as np

class VideoReader:
    """Reads video files and yields frames with metadata."""

    def __init__(self, video_path: str):
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        
        if not self.cap.isOpened():
            raise IOError(f"Failed to open video file: {video_path}")

        self._fps = self.cap.get(cv2.CAP_PROP_FPS)
        self._width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def resolution(self) -> Tuple[int, int]:
        return (self._width, self._height)

    @property
    def total_frames(self) -> int:
        return self._total_frames

    @property
    def duration_seconds(self) -> float:
        if self._fps > 0:
            return self._total_frames / self._fps
        return 0.0

    def get_frames(self) -> Iterator[Tuple[int, float, np.ndarray]]:
        """Yields (frame_index, timestamp_seconds, frame)"""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_idx = 0
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            timestamp = frame_idx / self._fps if self._fps > 0 else 0.0
            yield (frame_idx, timestamp, frame)
            frame_idx += 1

    def release(self):
        self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
