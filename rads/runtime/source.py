import abc
import logging
import os
import time

import cv2

logger = logging.getLogger(__name__)


class SourceExhaustedError(Exception):
    pass


class FrameSource(abc.ABC):
    def __init__(self, source_uri):
        self._source_uri = str(source_uri)
        self._cap = None
        self._fps = 0.0
        self._width = 0
        self._height = 0
        self._frame_index = 0

    @abc.abstractmethod
    def open(self):
        raise NotImplementedError

    @abc.abstractmethod
    def read(self):
        raise NotImplementedError

    def close(self):
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def is_open(self):
        return self._cap is not None and self._cap.isOpened()

    @property
    def fps(self):
        return self._fps

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    @property
    def source_uri(self):
        return self._source_uri

    def _store_metadata(self, default_fps=None):
        fps = float(self._cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if fps <= 0 and default_fps is not None:
            fps = float(default_fps)
        self._fps = fps
        self._width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        self._height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    def _read_once(self):
        if not self.is_open():
            return None
        ret, frame = self._cap.read()
        if not ret or frame is None:
            return None
        timestamp = self._frame_index / self._fps if self._fps > 0 else 0.0
        item = (self._frame_index, timestamp, frame)
        self._frame_index += 1
        return item


class FileSource(FrameSource):
    def __init__(self, path):
        if path and not os.path.exists(path):
            raise FileNotFoundError(f"Video file not found: {path}")
        super().__init__(path)

    def open(self):
        if not self._source_uri or not os.path.exists(self._source_uri):
            raise FileNotFoundError(f"Video file not found: {self._source_uri}")
        self.close()
        self._cap = cv2.VideoCapture(self._source_uri)
        if not self._cap.isOpened():
            raise IOError(f"Failed to open video file: {self._source_uri}")
        self._store_metadata()

    def read(self):
        return self._read_once()


class WebcamSource(FrameSource):
    def __init__(self, device_index):
        self._device_index = int(device_index)
        super().__init__(device_index)

    def open(self):
        self.close()
        self._cap = cv2.VideoCapture(self._device_index)
        if not self._cap.isOpened():
            raise IOError(f"Failed to open webcam: {self._device_index}")
        self._store_metadata(default_fps=30)

    def read(self):
        return self._read_once()


class RTSPSource(FrameSource):
    def __init__(self, url, reconnect_interval_s, max_reconnect_attempts):
        super().__init__(url)
        self._reconnect_interval_s = reconnect_interval_s
        self._max_reconnect_attempts = max_reconnect_attempts
        self._reconnect_count = 0

    def open(self):
        self.close()
        self._cap = cv2.VideoCapture(self._source_uri)
        if not self._cap.isOpened():
            raise IOError(f"Failed to open RTSP source: {self._source_uri}")
        self._store_metadata()

    def read(self):
        while True:
            item = self._read_once()
            if item is not None:
                return item
            self._reconnect()

    def _reconnect(self):
        if self._max_reconnect_attempts >= 0 and self._reconnect_count >= self._max_reconnect_attempts:
            raise SourceExhaustedError(self._source_uri)
        self._reconnect_count += 1
        logger.warning(
            "RTSP reconnect attempt %d for %s",
            self._reconnect_count,
            self._source_uri,
        )
        self.close()
        time.sleep(self._reconnect_interval_s)
        self._cap = cv2.VideoCapture(self._source_uri)
        if self._cap.isOpened():
            self._store_metadata()


def create_source(config):
    kind = config.runtime_source
    uri = config.runtime_source_uri
    if kind == "file":
        return FileSource(uri)
    if kind == "webcam":
        return WebcamSource(uri)
    if kind == "rtsp":
        return RTSPSource(
            uri,
            config.runtime_reconnect_interval_s,
            config.runtime_max_reconnect_attempts,
        )
    raise ValueError(f"Unknown runtime source: {kind}")
