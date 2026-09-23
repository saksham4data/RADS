import datetime
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from rads.runtime.event_lifecycle import EventLifecycleManager


def _detection(confidence, track_ids, impact_time):
    return {
        'accident': True,
        'confidence': confidence,
        'event': {
            'start_time': impact_time - 1.0,
            'impact_time': impact_time,
            'end_time': None,
        },
        'objects_involved': [{'id': track_id, 'class': 'car'} for track_id in track_ids],
        'severity': 'LOW',
        'severity_detail': {'score': 1.0, 'evidence': []},
        'accident_type': 'unknown',
    }


class TestEventLifecycle(unittest.TestCase):
    def test_creation_and_id(self):
        manager = EventLifecycleManager(2.0, 10.0)
        event = manager.submit_detection(_detection(0.6, [3], 5.0))
        today = datetime.date.today().strftime('%Y%m%d')
        self.assertEqual(event['status'], 'candidate')
        self.assertEqual(event['event_id'], f'RADS-{today}-00001')
        second = manager.submit_detection(_detection(0.6, [8], 6.0))
        self.assertEqual(second['event_id'], f'RADS-{today}-00002')

    def test_status_transitions(self):
        manager = EventLifecycleManager(2.0, 10.0)
        manager.active_track_ids = [1, 2]
        created = manager.submit_detection(_detection(0.4, [1, 2], 5.0))
        self.assertEqual(created['status'], 'candidate')
        detected = manager.submit_detection(_detection(0.9, [1, 2], 5.2))
        self.assertEqual(detected['status'], 'detected')
        manager.tick(0.0)
        manager.tick(2.0)
        self.assertEqual(manager.get_active_events()[0]['status'], 'confirmed')
        manager.active_track_ids = []
        manager.tick(2.1)
        self.assertEqual(manager.get_active_events(), [])
        self.assertEqual(manager.get_recent_events(1)[0]['status'], 'resolved')

    def test_timeout_resolution(self):
        manager = EventLifecycleManager(1.0, 5.0)
        manager.active_track_ids = [4]
        manager.submit_detection(_detection(0.2, [4], 1.0))
        manager.submit_detection(_detection(0.7, [4], 1.0))
        manager.tick(10.0)
        manager.tick(11.0)
        self.assertEqual(manager.get_active_events()[0]['status'], 'confirmed')
        manager.tick(16.0)
        self.assertEqual(manager.get_active_events(), [])
        self.assertEqual(manager.get_recent_events(1)[0]['status'], 'resolved')

    def test_ring_buffer_count(self):
        manager = EventLifecycleManager(2.0, 10.0, buffer_size=50)
        for track_id in range(5):
            manager.submit_detection(_detection(0.5, [track_id], 0.0))
        self.assertEqual(len(manager.get_recent_events(3)), 3)
        self.assertEqual(len(manager.get_recent_events(10)), 5)
