import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from rads.runtime.engine import RuntimeEngine
from rads.runtime.health import HealthMonitor

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


class TestHealthMonitor(unittest.TestCase):
    def test_get_health_returns_valid_dict(self):
        monitor = HealthMonitor()
        monitor.update(4, 1)
        health = monitor.get_health()
        self.assertEqual(health['frames_processed'], 4)
        self.assertEqual(health['events_detected'], 1)
        self.assertIsNotNone(health['last_event_time'])
        for key in (
            'start_time',
            'frames_processed',
            'events_detected',
            'last_event_time',
            'source_status',
            'current_fps',
        ):
            self.assertIn(key, health)

    def test_uptime_calculation(self):
        monitor = HealthMonitor()
        monitor._start_time = 1000.0
        with patch('rads.runtime.health.time.time', return_value=1002.5):
            self.assertEqual(monitor.get_uptime_s(), 2.5)
