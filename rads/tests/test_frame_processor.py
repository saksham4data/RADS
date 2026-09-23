import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from rads.config.config_loader import ConfigLoader
from rads.runtime.frame_processor import FrameProcessor

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
CONFIG_PATH = os.path.join(REPO_ROOT, 'rads', 'config', 'pipeline_config.yaml')


class TestFrameProcessor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.processor = FrameProcessor(ConfigLoader(CONFIG_PATH))

    def setUp(self):
        self.processor.reset()

    def test_initializes(self):
        state = self.processor.get_state()
        self.assertEqual(state['active_track_count'], 0)
        self.assertEqual(state['frame_count'], 0)
        self.assertIsNone(state['last_reasoning_result'])

    def test_process_frame_accepts_synthetic_frame(self):
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        result = self.processor.process_frame(frame, 0, 0.0)
        self.assertTrue(result is None or isinstance(result, dict))
        self.assertEqual(self.processor.get_state()['frame_count'], 1)

    def test_reset_clears_state(self):
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        self.processor.process_frame(frame, 0, 0.0)
        self.processor.reset()
        state = self.processor.get_state()
        self.assertEqual(state['frame_count'], 0)
        self.assertEqual(state['active_track_count'], 0)
        self.assertIsNone(state['last_reasoning_result'])

    def test_window_pruning_removes_old_tracks(self):
        self.processor.track_history.update([{
            'track_id': 99999,
            'bbox_xyxy': [0, 0, 10, 10],
            'class_name': 'car',
            'confidence': 0.9,
        }], 0, 0.0)
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        self.processor.process_frame(frame, 10, 100.0)
        self.assertNotIn(99999, self.processor.track_history.history)
