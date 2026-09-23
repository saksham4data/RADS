import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from rads.runtime.engine import RuntimeEngine

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
CONFIG_PATH = os.path.join(REPO_ROOT, 'rads', 'config', 'pipeline_config.yaml')


class TestRuntimeEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = RuntimeEngine(CONFIG_PATH)

    def test_construction(self):
        self.assertIsNotNone(self.engine.config)
        self.assertIsNotNone(self.engine.source)
        self.assertIsNotNone(self.engine.processor)

    def test_register_handler_stores_callback(self):
        def callback(event):
            return None
        self.engine.register_handler(callback)
        self.assertIs(self.engine._handlers[-1], callback)

    def test_stop_sets_flag(self):
        self.engine._running = True
        self.engine.stop()
        self.assertFalse(self.engine._running)
