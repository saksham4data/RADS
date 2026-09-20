"""Severity engine contract: ordering, non-empty evidence, and traceable factors."""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from rads.config.config_loader import ConfigLoader
from rads.motion.trajectory import TrackHistory
from rads.severity.severity_engine import estimate_severity

CONFIG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'config', 'pipeline_config.yaml'))
FPS = 30.0
BOX = 40.0

def build_history(tracks):
    """tracks maps track_id -> (class_name, [(cx, cy), ...] indexed by frame)."""
    history = TrackHistory()
    frame_count = max(len(points) for _, points in tracks.values())
    for frame_idx in range(frame_count):
        objects = []
        for track_id, (class_name, points) in tracks.items():
            if frame_idx >= len(points):
                continue
            cx, cy = points[frame_idx]
            objects.append({
                'track_id': track_id,
                'bbox_xyxy': [cx - BOX / 2, cy - BOX / 2, cx + BOX / 2, cy + BOX / 2],
                'class_name': class_name,
                'confidence': 0.9
            })
        history.update(objects, frame_idx, frame_idx / FPS)
    return history

class TestSeverityEngine(unittest.TestCase):

    def setUp(self):
        self.config = ConfigLoader(CONFIG_PATH)

    def _major_event(self):
        # Three vehicles, high closing speed, large post-impact displacement
        tracks = {
            1: ('car', [(100.0 + 8.0 * i, 200.0) for i in range(60)]),
            2: ('car', [(400.0 - 8.0 * i, 200.0) for i in range(60)]),
            3: ('motorcycle', [(250.0, 100.0 + 6.0 * i) for i in range(60)]),
        }
        result = {
            "accident": True,
            "involved_object_ids": [1, 2, 3],
            "kinematics": {
                "peak_relative_velocity": 120.0,
                "max_velocity_drop": 90.0,
                "impact_frame": 20
            }
        }
        return result, build_history(tracks)

    def _minor_event(self):
        # Two cars, low closing speed, barely moving after the interaction
        tracks = {
            1: ('car', [(100.0 + 0.4 * i, 200.0) for i in range(60)]),
            2: ('car', [(160.0 + 0.2 * i, 200.0) for i in range(60)]),
        }
        result = {
            "accident": True,
            "involved_object_ids": [1, 2],
            "kinematics": {
                "peak_relative_velocity": 8.0,
                "max_velocity_drop": 3.0,
                "impact_frame": 20
            }
        }
        return result, build_history(tracks)

    def test_major_event_scores_above_minor_event(self):
        major_result, major_history = self._major_event()
        minor_result, minor_history = self._minor_event()

        major = estimate_severity(major_result, major_history, self.config)
        minor = estimate_severity(minor_result, minor_history, self.config)

        self.assertGreater(major['score'], minor['score'])
        self.assertEqual(major['severity'], 'HIGH')
        self.assertEqual(minor['severity'], 'LOW')

    def test_evidence_is_non_empty_and_traceable(self):
        result, history = self._major_event()
        severity = estimate_severity(result, history, self.config)

        self.assertTrue(severity['evidence'])
        expected_factors = {
            'num_objects_involved',
            'vulnerable_class_involved',
            'peak_relative_velocity',
            'velocity_change',
            'post_impact_displacement'
        }
        factors = {entry['factor'] for entry in severity['evidence']}
        self.assertEqual(factors, expected_factors)

        for entry in severity['evidence']:
            self.assertIn('value', entry)
            self.assertIn('contribution', entry)
            self.assertGreaterEqual(entry['contribution'], 0.0)

        contribution_sum = sum(entry['contribution'] for entry in severity['evidence'])
        self.assertAlmostEqual(contribution_sum, severity['score'], places=3)

    def test_no_accident_returns_null_severity(self):
        _, history = self._minor_event()
        severity = estimate_severity({"accident": False}, history, self.config)

        self.assertIsNone(severity['severity'])
        self.assertEqual(severity['score'], 0.0)
        self.assertEqual(severity['evidence'], [])

    def test_vulnerable_class_raises_severity(self):
        tracks = {
            1: ('car', [(100.0 + 5.0 * i, 200.0) for i in range(60)]),
            2: ('person', [(300.0, 200.0) for _ in range(60)]),
        }
        history = build_history(tracks)
        kinematics = {"peak_relative_velocity": 30.0, "max_velocity_drop": 20.0, "impact_frame": 20}

        with_vulnerable = estimate_severity(
            {"accident": True, "involved_object_ids": [1, 2], "kinematics": kinematics},
            history, self.config)

        tracks[2] = ('car', tracks[2][1])
        without_vulnerable = estimate_severity(
            {"accident": True, "involved_object_ids": [1, 2], "kinematics": kinematics},
            build_history(tracks), self.config)

        self.assertGreater(with_vulnerable['score'], without_vulnerable['score'])

if __name__ == '__main__':
    unittest.main()
