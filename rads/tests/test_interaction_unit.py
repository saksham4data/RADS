"""Phase 4 verification: interaction detection on synthetic tracks.

Named test_interaction_unit to avoid colliding with the clip runner
rads/interaction/test_interactions.py.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from rads.config.config_loader import ConfigLoader
from rads.motion.trajectory import TrackHistory
from rads.interaction.pairwise import compute_pairwise_metrics, enrich_with_motion
from rads.interaction.interaction_engine import detect_interactions

CONFIG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'config', 'pipeline_config.yaml'))
FPS = 10.0
BOX = 40.0

def build_history(tracks, fps=FPS):
    """tracks maps track_id -> list of (cx, cy), one entry per frame."""
    history = TrackHistory()
    frame_count = max(len(pts) for pts in tracks.values())
    for frame_idx in range(frame_count):
        objects = []
        for track_id, points in tracks.items():
            if frame_idx >= len(points):
                continue
            cx, cy = points[frame_idx]
            objects.append({
                'track_id': track_id,
                'bbox_xyxy': [cx - BOX / 2, cy - BOX / 2, cx + BOX / 2, cy + BOX / 2],
                'class_name': 'car',
                'confidence': 0.9
            })
        history.update(objects, frame_idx, frame_idx / fps)
    return history

def run_interactions(history):
    config = ConfigLoader(CONFIG_PATH)
    pairwise = compute_pairwise_metrics(history)
    enrich_with_motion(pairwise)
    return detect_interactions(pairwise, config)

class TestInteractionDetection(unittest.TestCase):

    def test_collision_course_yields_candidate(self):
        # Head-on approach closing 20 px per frame, ending overlapped
        frames = 20
        track_a = [(100.0 + 10.0 * i, 200.0) for i in range(frames)]
        track_b = [(400.0 - 10.0 * i, 200.0) for i in range(frames)]

        candidates = run_interactions(build_history({1: track_a, 2: track_b}))

        self.assertEqual(len(candidates), 1)
        candidate = candidates[0]
        self.assertEqual({candidate['object_a_id'], candidate['object_b_id']}, {1, 2})
        self.assertGreater(candidate['max_relative_velocity'], 0.0)
        self.assertTrue(candidate['evidence_list'])

    def test_parallel_tracks_yield_no_candidate(self):
        # Same heading, same speed, constant 300 px lateral separation
        frames = 20
        track_a = [(100.0 + 10.0 * i, 150.0) for i in range(frames)]
        track_b = [(100.0 + 10.0 * i, 450.0) for i in range(frames)]

        candidates = run_interactions(build_history({1: track_a, 2: track_b}))

        self.assertEqual(candidates, [])

if __name__ == '__main__':
    unittest.main()
