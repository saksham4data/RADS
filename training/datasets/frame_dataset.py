# ─────────────────────────────────────────────────────────────
# Generic Video Frame Dataset
# ─────────────────────────────────────────────────────────────
"""
Dataset-agnostic PyTorch Dataset that loads video frames from
any dataset registered in ``global_master_metadata.csv``.

Switching datasets requires only a YAML config change — no code
modifications.  Supports TUDAT, Picek, and any future dataset.

Frame sampling strategies:
    uniform  — evenly spaced frames across the video
    random   — randomly sampled frames
    first    — first N frames
    middle   — N frames centred around the video midpoint
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from training.configs.config import TrainingConfig
from training.datasets.video_sampling import (
    SAFE_MARGIN_FRAC,
    compute_sample_indices,
    probe_decodable_frame_count,
    probe_reported_frame_count,
    read_frame_with_fallback,
    safe_range,
)

logger = logging.getLogger(__name__)


class VideoFrameDataset(Dataset):
    """Generic video frame dataset for classification.

    Loads ``global_master_metadata.csv``, filters by dataset name
    and split, then samples frames from each video.

    Parameters
    ----------
    config : TrainingConfig
        Training configuration.
    split : str
        One of ``"train"``, ``"val"``, ``"test"``.
    transform : callable, optional
        Transform applied to each frame (numpy BGR → tensor).
    class_mapping : dict[str, int], optional
        Explicit label → index mapping.  If ``None``, auto-detected
        from the data.

    Usage::

        ds = VideoFrameDataset(config, split="train", transform=train_tf)
        frame, label = ds[0]
        assert frame.shape == (3, 224, 224)
    """

    def __init__(
        self,
        config: TrainingConfig,
        split: str,
        *,
        transform: Optional[Any] = None,
        class_mapping: Optional[Dict[str, int]] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.split = split
        self.transform = transform

        # ── Load and filter metadata ──
        self._df = self._load_metadata()
        self._df = self._filter_dataset(self._df)
        self._df = self._assign_or_filter_split(self._df)

        # ── Build class mapping ──
        self._class_mapping = class_mapping or config.class_mapping
        if self._class_mapping is None:
            self._class_mapping = self._auto_class_mapping(self._df)
        self._idx_to_class = {v: k for k, v in self._class_mapping.items()}
        self._num_classes = len(self._class_mapping)

        # ── Build sample index (video_idx, frame_idx) ──
        self._samples: List[Tuple[int, int]] = []
        self._video_paths: List[Path] = []
        self._video_labels: List[int] = []
        self._build_sample_index()

        logger.info(
            "VideoFrameDataset[%s] — %d videos → %d samples | "
            "%d classes %s",
            self.split,
            len(self._df),
            len(self._samples),
            self._num_classes,
            list(self._class_mapping.keys()),
        )

    # ── Metadata loading ────────────────────────────────────

    def _load_metadata(self) -> pd.DataFrame:
        """Load the global master metadata CSV."""
        path = self.config.metadata_path
        if not path.is_file():
            raise FileNotFoundError(
                f"Metadata file not found: {path}\n"
                f"Expected at: {path.resolve()}"
            )
        return pd.read_csv(path, low_memory=False)

    def _filter_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter to rows matching ``config.data.dataset_name``."""
        dataset_name = self.config.dataset_name
        mask = df["dataset_name"] == dataset_name
        filtered = df[mask].copy().reset_index(drop=True)

        if filtered.empty:
            available = df["dataset_name"].unique().tolist()
            raise ValueError(
                f"No rows found for dataset '{dataset_name}'. "
                f"Available datasets: {available}"
            )

        logger.debug(
            "Filtered to dataset '%s': %d rows",
            dataset_name, len(filtered),
        )
        return filtered

    def _normalize_label(self, raw_label: Any) -> str:
        """Map raw dataset labels into the configured label space."""
        label = str(raw_label).strip()
        if self.config.label_mode == "binary" and label == "challenging":
            return "accident"
        return label

    def _assign_or_filter_split(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter by split column, or generate splits if absent.

        If the configured split column exists and has values for
        this dataset, filter by it.  Otherwise, generate stratified
        splits using the configured ratios and seed.
        """
        split_col = self.config.split_column
        label_col = self.config.label_column

        # Check if split column exists and has non-null values
        if split_col in df.columns and df[split_col].notna().any():
            # Normalize split names
            df[split_col] = df[split_col].astype(str).str.lower().str.strip()
            split_mask = df[split_col] == self.split
            filtered = df[split_mask].copy().reset_index(drop=True)

            if filtered.empty:
                available = df[split_col].dropna().unique().tolist()
                logger.warning(
                    "No rows for split '%s' in column '%s'. "
                    "Available splits: %s. Generating new splits.",
                    self.split, split_col, available,
                )
                return self._generate_splits(df)
            return filtered

        # No split column — generate stratified splits
        logger.info(
            "Split column '%s' not available for dataset '%s'. "
            "Generating stratified splits.",
            split_col, self.config.dataset_name,
        )
        return self._generate_splits(df)

    def _generate_splits(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate stratified train/val/test splits.

        Uses ``sklearn.model_selection.train_test_split`` with
        the configured ratios and seed.
        """
        from sklearn.model_selection import train_test_split

        label_col = self.config.label_column
        ratios = self.config.split_ratios
        seed = self.config.seed

        # Validate label column
        if label_col not in df.columns:
            raise ValueError(
                f"Label column '{label_col}' not found in metadata. "
                f"Available columns: {df.columns.tolist()}"
            )

        # Filter out rows with missing labels
        df = df[df[label_col].notna()].copy().reset_index(drop=True)

        # First split: train vs (val + test)
        val_test_ratio = ratios["val"] + ratios["test"]
        labels = df[label_col].map(self._normalize_label)

        # Handle classes with very few samples
        class_counts = labels.value_counts()
        small_classes = class_counts[class_counts < 3].index.tolist()
        if small_classes:
            logger.warning(
                "Classes with < 3 samples (stratification may be imperfect): %s",
                small_classes,
            )

        try:
            train_df, val_test_df = train_test_split(
                df,
                test_size=val_test_ratio,
                random_state=seed,
                stratify=labels,
            )
        except ValueError:
            logger.warning(
                "Stratified split failed (class too small). "
                "Falling back to non-stratified split."
            )
            train_df, val_test_df = train_test_split(
                df,
                test_size=val_test_ratio,
                random_state=seed,
            )

        # Second split: val vs test
        if ratios["test"] > 0 and len(val_test_df) > 1:
            test_fraction = ratios["test"] / val_test_ratio
            vt_labels = val_test_df[label_col].map(self._normalize_label)
            try:
                val_df, test_df = train_test_split(
                    val_test_df,
                    test_size=test_fraction,
                    random_state=seed,
                    stratify=vt_labels,
                )
            except ValueError:
                val_df, test_df = train_test_split(
                    val_test_df,
                    test_size=test_fraction,
                    random_state=seed,
                )
        else:
            val_df = val_test_df
            test_df = pd.DataFrame(columns=df.columns)

        split_map = {"train": train_df, "val": val_df, "test": test_df}

        logger.info(
            "Generated splits — train: %d, val: %d, test: %d",
            len(train_df), len(val_df), len(test_df),
        )

        result = split_map.get(self.split)
        if result is None or result.empty:
            raise ValueError(
                f"Split '{self.split}' produced 0 rows. "
                f"Available: {list(split_map.keys())}"
            )
        return result.reset_index(drop=True)

    # ── Class mapping ───────────────────────────────────────

    def _auto_class_mapping(self, df: pd.DataFrame) -> Dict[str, int]:
        """Auto-detect class mapping from label column values."""
        label_col = self.config.label_column
        unique_labels = sorted(
            {
                self._normalize_label(label)
                for label in df[label_col].dropna().tolist()
            }
        )
        mapping = {label: idx for idx, label in enumerate(unique_labels)}
        logger.info("Auto-detected class mapping: %s", mapping)
        return mapping

    # ── Sample index ────────────────────────────────────────

    def _build_sample_index(self) -> None:
        """Build the flat index of (video_idx, frame_idx) tuples.

        For each video row, determine the frame indices to sample,
        and create an entry for each.
        """
        label_col = self.config.label_column
        frames_per_video = self.config.frames_per_video
        strategy = self.config.sampling_strategy

        for row_idx, row in self._df.iterrows():
            original_path = row.get("original_path", "")
            video_path = (
                self.config.raw_base_dir
                / self.config.dataset_name
                / str(original_path)
            )

            if not video_path.is_file():
                # Fallback: check under Final_videos/ subfolder (e.g., TUDAT structure)
                fallback_path = (
                    self.config.raw_base_dir
                    / self.config.dataset_name
                    / "Final_videos"
                    / str(original_path)
                )
                if fallback_path.is_file():
                    video_path = fallback_path

            if not video_path.is_file():
                logger.warning(
                    "Video not found, skipping: %s", video_path,
                )
                continue

            # ── Get label ──
            label_str = self._normalize_label(row[label_col])
            if label_str not in self._class_mapping:
                logger.warning(
                    "Unknown label '%s' in row %d, skipping.",
                    label_str, row_idx,
                )
                continue

            label_idx = self._class_mapping[label_str]

            # ── Probe decodable frame range ──
            total_frames = int(row.get("no_frames", 0))
            total_frames = self._probe_frame_count(video_path, total_frames)

            if total_frames <= 0:
                logger.warning(
                    "Could not determine frame count for %s, skipping.",
                    video_path,
                )
                continue

            # ── Compute frame indices ──
            frame_indices = self._compute_frame_indices(
                total_frames, frames_per_video, strategy,
            )

            # ── Add to sample index ──
            vid_idx = len(self._video_paths)
            self._video_paths.append(video_path)
            self._video_labels.append(label_idx)

            for fi in frame_indices:
                self._samples.append((vid_idx, fi))

    @staticmethod
    def _probe_frame_count(video_path: Path, reported_frame_count: int = 0) -> int:
        """Probe the actually decodable frame count for the video."""
        upper_bound = reported_frame_count if reported_frame_count > 0 else probe_reported_frame_count(video_path)
        return probe_decodable_frame_count(video_path, upper_bound)

    # Boundary margin: avoid the first/last 15% of frames in a video.
    # OpenCV cannot reliably decode frames near the end of many
    # compressed .mov containers (metadata frame counts overreport
    # the actual decodable range).  Sampling within [15%, 85%] of
    # the total frame range eliminates virtually all seek/decode
    # failures while preserving good temporal coverage.
    _SAFE_MARGIN_FRAC = SAFE_MARGIN_FRAC

    @staticmethod
    def _safe_range(total_frames: int) -> Tuple[int, int]:
        """Return the [lo, hi) safe frame range excluding boundary frames.

        For very short videos (< 20 frames) the margin is clamped to
        at most 1 frame on each side so we don't exclude too much.
        """
        return safe_range(total_frames)

    @staticmethod
    def _compute_frame_indices(
        total_frames: int,
        frames_per_video: int,
        strategy: str,
    ) -> List[int]:
        """Compute which frame indices to sample.

        Parameters
        ----------
        total_frames : int
            Total frames in the video.
        frames_per_video : int
            Number of frames to sample.
        strategy : str
            Sampling strategy: uniform, random, first, middle.

        Notes
        -----
        For ``uniform`` and ``random`` strategies, sampling is
        restricted to a safe inner range (15 %–85 % of total frames)
        to avoid the unreliable first/last frames in compressed .mov
        containers.  ``first`` and ``middle`` are left unchanged.
        """
        return compute_sample_indices(total_frames, frames_per_video, strategy)

    # ── Dataset interface ───────────────────────────────────

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Optional[Tuple[torch.Tensor, int]]:
        vid_idx, frame_idx = self._samples[idx]
        video_path = self._video_paths[vid_idx]
        label = self._video_labels[vid_idx]

        try:
            # Read frame
            frame = self._read_frame(video_path, frame_idx)
            if frame is None:
                raise IOError("Frame extraction failed after both seeking and sequential decoding attempts")

            # Apply transform
            if self.transform is not None:
                frame = self.transform(frame)
            else:
                # Default: convert BGR → RGB, HWC → CHW, scale to [0, 1]
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0

            return frame, label
        except Exception as exc:
            logger.warning(
                "Failed to load frame %d from %s: %s. Skipping sample.",
                frame_idx, video_path, exc,
            )
            return None

    @staticmethod
    def _read_frame(video_path: Path, frame_idx: int) -> Optional[np.ndarray]:
        """Read a specific frame using the shared robust decoder."""
        return read_frame_with_fallback(video_path, frame_idx)

    # ── Properties ──────────────────────────────────────────

    @property
    def class_mapping(self) -> Dict[str, int]:
        """Label → index mapping."""
        return self._class_mapping.copy()

    @property
    def class_names(self) -> List[str]:
        """Ordered list of class names."""
        return [
            self._idx_to_class[i]
            for i in range(self._num_classes)
        ]

    @property
    def num_classes(self) -> int:
        """Number of classes."""
        return self._num_classes

    @property
    def class_counts(self) -> Dict[str, int]:
        """Number of samples per class."""
        counts: Dict[str, int] = {name: 0 for name in self._class_mapping}
        for vid_idx, _ in self._samples:
            label_idx = self._video_labels[vid_idx]
            class_name = self._idx_to_class[label_idx]
            counts[class_name] += 1
        return counts

    @property
    def class_weights(self) -> torch.Tensor:
        """Inverse-frequency class weights for loss balancing.

        Returns a tensor of shape ``[num_classes]``.
        """
        counts = self.class_counts
        total = sum(counts.values())
        weights = []
        for i in range(self._num_classes):
            class_name = self._idx_to_class[i]
            count = max(counts.get(class_name, 0), 1)
            weights.append(total / (self._num_classes * count))
        return torch.tensor(weights, dtype=torch.float32)
