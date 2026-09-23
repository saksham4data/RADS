"""B1 regression: one Pipeline instance must give the same result for a video regardless of
whether another video was processed first.

Requires ultralytics, the yolo11n.pt weights and two dataset clips; skipped if absent.
frame_skip is raised to keep the test to a few tracking passes.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
CONFIG_PATH = os.path.join(REPO_ROOT, 'rads', 'config', 'pipeline_config.yaml')
WEIGHTS = os.path.join(REPO_ROOT, 'models', 'yolo11n.pt')
CLIP_DIR = os.path.join(REPO_ROOT, 'Datasets', 'processed', 'picek_sorted', 'trimmed', 'positive', 'real')
VIDEO_A = os.path.join(CLIP_DIR, '-PpBteU0p3Q_00.mp4')
VIDEO_B = os.path.join(CLIP_DIR, '-Qt5bDJNT84_00.mp4')
FRAME_SKIP = 3

def _missing():
    try:
        import ultralytics  # noqa: F401
        import torch  # noqa: F401
    except ImportError as exc:
        return str(exc)
    for path in (WEIGHTS, VIDEO_A, VIDEO_B, CONFIG_PATH):
        if not os.path.exists(path):
            return f"missing {path}"
    return None

MISSING = _missing()

@unittest.skipIf(MISSING is not None, f"pipeline prerequisites unavailable: {MISSING}")
class TestPipelineIsolation(unittest.TestCase):

    def _make_pipeline(self):
        from rads.pipeline.pipeline import Pipeline
        pipeline = Pipeline(CONFIG_PATH)
        pipeline.config.config['pipeline']['frame_skip'] = FRAME_SKIP
        return pipeline

    @staticmethod
    def _signature(result):
        return {
            'accident': result['accident'],
            'confidence': result['confidence'],
            'score': result['score'],
            'track_ids': sorted(result['tracks_summary'].keys())
        }

    def test_video_result_is_independent_of_preceding_video(self):
        standalone = self._signature(self._make_pipeline().run(VIDEO_B, visualize=False))

        pipeline = self._make_pipeline()
        pipeline.run(VIDEO_A, visualize=False)
        after_other_video = self._signature(pipeline.run(VIDEO_B, visualize=False))

        self.assertEqual(standalone['accident'], after_other_video['accident'])
        self.assertEqual(standalone['confidence'], after_other_video['confidence'])
        self.assertEqual(standalone['score'], after_other_video['score'])
        self.assertEqual(standalone['track_ids'], after_other_video['track_ids'])

    def test_reset_uses_the_predictor_tracker_path(self):
        pipeline = self._make_pipeline()
        pipeline.run(VIDEO_A, visualize=False)
        # First reset has no predictor yet; after one run the live trackers must be reachable
        self.assertEqual(pipeline.tracker.reset(), 'predictor_trackers')

if __name__ == '__main__':
    unittest.main()
