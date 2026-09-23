import os
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from rads.config.config_loader import ConfigLoader
from rads.runtime.source import FileSource, RTSPSource, SourceExhaustedError, create_source

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
CONFIG_PATH = os.path.join(REPO_ROOT, 'rads', 'config', 'pipeline_config.yaml')
VIDEO = os.path.join(
    REPO_ROOT,
    'Datasets', 'processed', 'picek_sorted', 'trimmed', 'positive', 'real', '-PpBteU0p3Q_00.mp4',
)


class _FakeCapture:
    def __init__(self, frames, opened=True):
        self._frames = list(frames)
        self._opened = opened

    def isOpened(self):
        return self._opened

    def read(self):
        if not self._frames:
            return False, None
        return self._frames.pop(0)

    def get(self, prop):
        import cv2
        if prop == cv2.CAP_PROP_FPS:
            return 25.0
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return 640
        if prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return 480
        return 0

    def release(self):
        self._opened = False


class TestFileSource(unittest.TestCase):
    def test_open_and_read_real_video(self):
        source = FileSource(VIDEO)
        source.open()
        self.addCleanup(source.close)
        self.assertTrue(source.is_open())
        self.assertGreater(source.fps, 0)
        self.assertGreater(source.width, 0)
        self.assertGreater(source.height, 0)
        first = source.read()
        self.assertIsNotNone(first)
        index, timestamp, frame = first
        self.assertEqual(index, 0)
        self.assertGreaterEqual(timestamp, 0)
        self.assertEqual(frame.ndim, 3)


class TestCreateSource(unittest.TestCase):
    def test_file_source_from_config(self):
        config = ConfigLoader(CONFIG_PATH)
        config.config['runtime']['source'] = 'file'
        config.config['runtime']['source_uri'] = VIDEO
        source = create_source(config)
        self.assertIsInstance(source, FileSource)
        self.assertEqual(source.source_uri, VIDEO)


class TestRTSPReconnect(unittest.TestCase):
    def test_reconnects_after_read_failure(self):
        frame = np.zeros((4, 4, 3), dtype=np.uint8)
        captures = [
            _FakeCapture([(False, None)]),
            _FakeCapture([(True, frame)]),
        ]

        def factory(url):
            return captures.pop(0)

        with patch('rads.runtime.source.cv2.VideoCapture', side_effect=factory), \
                patch('rads.runtime.source.time.sleep') as sleep, \
                self.assertLogs('rads.runtime.source', level='WARNING') as logs:
            source = RTSPSource('rtsp://camera/stream', 0.5, 3)
            source.open()
            index, timestamp, got = source.read()

        self.assertEqual(index, 0)
        self.assertIs(got, frame)
        self.assertEqual(timestamp, 0.0)
        sleep.assert_called_once_with(0.5)
        self.assertTrue(any('reconnect' in line for line in logs.output))

    def test_raises_when_reconnect_attempts_exhausted(self):
        def factory(url):
            return _FakeCapture([])

        with patch('rads.runtime.source.cv2.VideoCapture', side_effect=factory), \
                patch('rads.runtime.source.time.sleep') as sleep:
            source = RTSPSource('rtsp://camera/stream', 0.5, 1)
            source.open()
            with self.assertRaises(SourceExhaustedError):
                source.read()

        self.assertEqual(sleep.call_count, 1)
