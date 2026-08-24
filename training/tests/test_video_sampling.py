import unittest
from pathlib import Path

from training.datasets.video_sampling import (
    compute_sample_indices,
    probe_decodable_frame_count,
)


class TestVideoSampling(unittest.TestCase):
    def test_probe_decodable_frame_count_uses_contiguous_prefix(self):
        video_path = Path("dummy.mov")

        def fake_reader(_path: Path, frame_idx: int):
            return object() if frame_idx < 31 else None

        decodable = probe_decodable_frame_count(
            video_path,
            reported_frame_count=52,
            frame_reader=fake_reader,
        )

        self.assertEqual(decodable, 31)

    def test_probe_decodable_frame_count_returns_zero_when_first_frame_fails(self):
        video_path = Path("dummy.mov")

        def fake_reader(_path: Path, _frame_idx: int):
            return None

        decodable = probe_decodable_frame_count(
            video_path,
            reported_frame_count=52,
            frame_reader=fake_reader,
        )

        self.assertEqual(decodable, 0)

    def test_uniform_sampling_uses_decodable_range(self):
        indices = compute_sample_indices(31, 8, "uniform")
        self.assertEqual(indices, [4, 7, 10, 13, 16, 19, 22, 26])


if __name__ == "__main__":
    unittest.main()
