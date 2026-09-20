"""Integrity tests for the held-out evaluation split and its source-id extraction."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rads.config.create_splits import (  # noqa: E402
    TUNING_SOURCES,
    extract_source_id,
)

SPLIT_CSV = REPO_ROOT / "rads" / "config" / "picek_heldout_split_v2.csv"

SOURCE_ID_CASES = {
    "GnpwLP2THmM_2_00.mp4": "GnpwLP2THmM",
    "AuQz_-J2kzc_5_00.mp4": "AuQz_-J2kzc",
    "651RQwxB3WA_12_00.mp4": "651RQwxB3WA",
    "c_u8oPMBGEI_01.mp4": "c_u8oPMBGEI",
    "__WFqm4i3vE_00.mp4": "__WFqm4i3vE",
    "_KmJLS_TJKY_00.mp4": "_KmJLS_TJKY",
    "russia23_00.mp4": "russia23",
    "video13_00.mp4": "video13",
    "s80RaghtMRk_02.mp4": "s80RaghtMRk",
    "3p_-1RjvCmw_1_00.mp4": "3p_-1RjvCmw",
    "-dVU8qW4ik8_00.mp4": "-dVU8qW4ik8",
    "xKT7khciy-c_14_00.mp4": "xKT7khciy-c",
    "Vtp5G_Z5BZQ_01.mp4": "Vtp5G_Z5BZQ",
}


class TestExtractSourceId(unittest.TestCase):
    def test_known_mappings(self):
        for filename, expected in SOURCE_ID_CASES.items():
            with self.subTest(filename=filename):
                self.assertEqual(extract_source_id(filename), expected)

    def test_full_path_input(self):
        self.assertEqual(
            extract_source_id(
                "Datasets/processed/picek_sorted/trimmed/positive/real/GnpwLP2THmM_2_00.mp4"
            ),
            "GnpwLP2THmM",
        )

    def test_clips_of_one_source_share_an_id(self):
        clips = [
            "GnpwLP2THmM_0_00.mp4",
            "GnpwLP2THmM_2_00.mp4",
            "GnpwLP2THmM_5_00.mp4",
            "GnpwLP2THmM_10_00.mp4",
        ]
        self.assertEqual({extract_source_id(c) for c in clips}, {"GnpwLP2THmM"})


class TestHeldOutSplit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not SPLIT_CSV.is_file():
            raise unittest.SkipTest(
                f"{SPLIT_CSV} not found; run python rads/config/create_splits.py first"
            )
        cls.df = pd.read_csv(SPLIT_CSV)

    def test_required_columns_present(self):
        required = {"video_id", "binary_label", "processed_path", "split_in_distribution"}
        self.assertTrue(required.issubset(set(self.df.columns)))

    def test_no_duplicate_video_id(self):
        duplicates = self.df["video_id"][self.df["video_id"].duplicated()].tolist()
        self.assertEqual(duplicates, [])

    def test_class_balance_is_exact(self):
        positives = int((self.df["binary_label"] == 1).sum())
        negatives = int((self.df["binary_label"] == 0).sum())
        self.assertEqual(positives, negatives)
        self.assertEqual(positives + negatives, len(self.df))

    def test_every_source_contributes_one_pair(self):
        counts = self.df.groupby("source_video_id")["binary_label"].agg(list)
        for source, labels in counts.items():
            with self.subTest(source=source):
                self.assertEqual(sorted(labels), [0, 1])

    def test_video_id_prefixes_match_labels(self):
        for _, row in self.df.iterrows():
            expected = "picek_pos_" if row["binary_label"] == 1 else "picek_neg_"
            with self.subTest(video_id=row["video_id"]):
                self.assertTrue(str(row["video_id"]).startswith(expected))

    def test_no_tuning_source_contamination(self):
        sources = {extract_source_id(p) for p in self.df["processed_path"]}
        self.assertEqual(sorted(sources & set(TUNING_SOURCES)), [])

    def test_processed_paths_are_repo_relative_and_exist(self):
        for path in self.df["processed_path"]:
            with self.subTest(path=path):
                self.assertFalse(Path(path).is_absolute())
                self.assertTrue(str(path).startswith("Datasets/processed/"))
                self.assertTrue((REPO_ROOT / path).is_file())


if __name__ == "__main__":
    unittest.main()
