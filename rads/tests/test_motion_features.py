"""Phase 3 verification: motion features on synthetic trajectories with known positions."""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from rads.motion.trajectory import TrackHistory
from rads.motion.motion_features import compute_motion_features
from rads.reasoning.accident_reasoner import get_velocities_around_frame

FPS = 10.0
DT = 1.0 / FPS
BOX = 40.0

def build_history(track_id, positions, fps=FPS, class_name='car'):
    """positions is a list of (cx, cy); frame i is sampled at i / fps seconds."""
    history = TrackHistory()
    for frame_idx, (cx, cy) in enumerate(positions):
        obj = {
            'track_id': track_id,
            'bbox_xyxy': [cx - BOX / 2, cy - BOX / 2, cx + BOX / 2, cy + BOX / 2],
            'class_name': class_name,
            'confidence': 0.9
        }
        history.update([obj], frame_idx, frame_idx / fps)
    return history

class TestMotionFeatures(unittest.TestCase):

    def test_left_to_right_velocity(self):
        # 10 px per frame at 10 fps is 100 px/s moving in +x
        positions = [(100.0 + 10.0 * i, 200.0) for i in range(20)]
        feats = compute_motion_features(build_history(1, positions))[1]

        self.assertAlmostEqual(feats['max_velocity'], 100.0, places=4)
        self.assertGreater(feats['mean_direction'][0], 0.99)
        self.assertAlmostEqual(feats['mean_direction'][1], 0.0, places=6)
        self.assertAlmostEqual(feats['net_displacement'], 190.0, places=4)
        self.assertAlmostEqual(feats['total_displacement'], 190.0, places=4)

    def test_right_to_left_velocity(self):
        positions = [(400.0 - 10.0 * i, 200.0) for i in range(20)]
        feats = compute_motion_features(build_history(1, positions))[1]

        self.assertAlmostEqual(feats['max_velocity'], 100.0, places=4)
        self.assertLess(feats['mean_direction'][0], -0.99)
        self.assertAlmostEqual(feats['net_displacement'], 190.0, places=4)

    def test_velocity_near_zero_after_stop(self):
        # Moves for 15 frames, then holds position for 20 frames
        moving = [(100.0 + 10.0 * i, 200.0) for i in range(15)]
        stopped = [(moving[-1][0], 200.0) for _ in range(20)]
        history = build_history(1, moving + stopped)
        traj = history.get_trajectory(1)

        pre_vel, post_vel, terminated = get_velocities_around_frame(traj, 15, max_frame=34, window=10)
        self.assertGreater(pre_vel, 90.0)
        self.assertLess(post_vel, 1e-6)
        self.assertFalse(terminated)

    def test_constant_velocity_has_no_sustained_acceleration(self):
        positions = [(100.0 + 10.0 * i, 200.0) for i in range(40)]
        feats = compute_motion_features(build_history(1, positions))[1]

        # Smoothing leaves an edge artefact at the trajectory ends, so only the mean is bounded
        self.assertLess(feats['mean_acceleration'], 60.0)

    def test_constant_acceleration_magnitude(self):
        # cx = 0.5 * a * t^2 with a = 200 px/s^2
        accel = 200.0
        positions = [(100.0 + 0.5 * accel * (i * DT) ** 2, 200.0) for i in range(40)]
        feats = compute_motion_features(build_history(1, positions))[1]

        self.assertAlmostEqual(feats['mean_acceleration'], accel, delta=0.2 * accel)
        self.assertGreater(feats['max_velocity'], 700.0)

if __name__ == '__main__':
    unittest.main()
