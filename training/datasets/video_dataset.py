# ─────────────────────────────────────────────────────────────
# Video Sequence Dataset (Temporal Modeling)
# ─────────────────────────────────────────────────────────────
"""
PyTorch Dataset that returns **per-video frame sequences** instead
of individual frames.

Each ``__getitem__`` call returns a tensor of shape ``[T, 3, H, W]``
containing ``T`` temporally-ordered frames from a single video,
along with the video-level label.

This is the temporal counterpart to
:class:`~training.datasets.frame_dataset.VideoFrameDataset`, which
returns individual frames as independent samples.

The frame sampling logic (uniform/random/first/middle, safe-range
margins) is reused from ``video_sampling.py`` so that the same
frame indices are selected — the only difference is that they are
returned as an ordered sequence rather than flattened into
independent samples.
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
from training.datasets.transforms import get_train_transforms, get_val_transforms
from training.datasets.video_sampling import (
    SAFE_MARGIN_FRAC,
    compute_sample_indices,
    probe_decodable_frame_count,
    probe_reported_frame_count,
    read_frame_with_fallback,
    safe_range,
)

logger = logging.getLogger(__name__)


class VideoSequenceDataset(Dataset):
    """Dataset that yields ordered frame sequences per video.

    Parameters
    ----------
    config : TrainingConfig
        Training configuration.
    split : str
        One of ``"train"``, ``"val"``, ``"test"``.
    transform : callable, optional
        Per-frame transform (numpy BGR -> tensor).
    class_mapping : dict[str, int], optional
        Explicit label -> index mapping.

    Returns
    -------
    tuple[torch.Tensor, int]
        ``(frames, label)`` where ``frames`` has shape
        ``[T, 3, H, W]`` and ``label`` is the integer class index.
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
        if transform is None:
            self.transform = (
                get_train_transforms(self.config.image_size, self.config.augmentation_config)
                if split == "train" and self.config.augmentation_enabled
                else get_val_transforms(self.config.image_size)
            )
        else:
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

        # ── Build video-level index ──
        # Each entry: (video_path, label_idx, [frame_indices])
        self._videos: List[Tuple[Path, int, List[int]]] = []
        self._build_video_index()

        logger.info(
            "VideoSequenceDataset[%s] — %d videos | T=%d | "
            "%d classes %s",
            self.split,
            len(self._videos),
            config.frames_per_video,
            self._num_classes,
            list(self._class_mapping.keys()),
        )

    # ── Metadata loading ────────────────────────────────────
    # These methods mirror VideoFrameDataset exactly.

    def _load_metadata(self) -> pd.DataFrame:
        path = self.config.metadata_path
        if not path.is_file():
            raise FileNotFoundError(
                f"Metadata file not found: {path}\n"
                f"Expected at: {path.resolve()}"
            )
        return pd.read_csv(path, low_memory=False)

    def _filter_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        dataset_name = self.config.dataset_name
        mask = df["dataset_name"] == dataset_name
        filtered = df[mask].copy().reset_index(drop=True)
        if filtered.empty:
            available = df["dataset_name"].unique().tolist()
            raise ValueError(
                f"No rows found for dataset '{dataset_name}'. "
                f"Available datasets: {available}"
            )
        return filtered

    def _normalize_label(self, raw_label: Any) -> str:
        label = str(raw_label).strip()
        if self.config.label_mode == "binary" and label == "challenging":
            return "accident"
        return label

    def _assign_or_filter_split(self, df: pd.DataFrame) -> pd.DataFrame:
        split_col = self.config.split_column
        if split_col not in df.columns:
            raise KeyError(
                f"Configured split column '{split_col}' not found in metadata. "
                f"Available columns: {df.columns.tolist()}"
            )
        mask = df[split_col] == self.split
        split_df = df[mask].copy().reset_index(drop=True)
        if split_df.empty:
            logger.warning(
                "Split '%s' in column '%s' has 0 records.",
                self.split, split_col,
            )
        return split_df

    def _auto_class_mapping(self, df: pd.DataFrame) -> Dict[str, int]:
        label_col = self.config.label_column
        unique_labels = sorted(
            self._normalize_label(lbl)
            for lbl in df[label_col].unique()
        )
        return {lbl: idx for idx, lbl in enumerate(unique_labels)}

    # ── Video index ─────────────────────────────────────────

    def _build_video_index(self) -> None:
        """Build the per-video index with precomputed frame indices."""
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
                fallback_picekl = (
                    self.config.raw_base_dir
                    / "picekl"
                    / str(original_path)
                )
                if fallback_picekl.is_file():
                    video_path = fallback_picekl

            if not video_path.is_file():
                fallback_path = (
                    self.config.raw_base_dir
                    / self.config.dataset_name
                    / "Final_videos"
                    / str(original_path)
                )
                if fallback_path.is_file():
                    video_path = fallback_path

            if not video_path.is_file():
                fallback_direct = self.config.raw_base_dir / str(original_path)
                if fallback_direct.is_file():
                    video_path = fallback_direct

            if not video_path.is_file():
                logger.warning("Video not found, skipping: %s", video_path)
                continue

            # ── Label ──
            label_str = self._normalize_label(row[label_col])
            if label_str not in self._class_mapping:
                logger.warning(
                    "Unknown label '%s' in row %d, skipping.",
                    label_str, row_idx,
                )
                continue
            label_idx = self._class_mapping[label_str]

            # ── Frame count ──
            total_frames = int(row.get("no_frames", 0))
            if total_frames <= 0:
                upper_bound = probe_reported_frame_count(video_path)
                decodable = probe_decodable_frame_count(video_path, upper_bound)
            else:
                decodable = total_frames

            if decodable <= 0:
                logger.warning(
                    "Could not determine frame count for %s, skipping.",
                    video_path,
                )
                continue

            # ── Frame indices (sorted for temporal order) ──
            frame_indices = compute_sample_indices(
                decodable, frames_per_video, strategy,
            )

            self._videos.append((video_path, label_idx, frame_indices))

    # ── Dataset interface ───────────────────────────────────

    def __len__(self) -> int:
        return len(self._videos)

    def __getitem__(self, idx: int) -> Optional[Tuple[torch.Tensor, int]]:
        video_path, label, frame_indices = self._videos[idx]

        frames = []
        for fi in frame_indices:
            raw_frame = read_frame_with_fallback(video_path, fi)
            if raw_frame is None:
                logger.warning(
                    "Failed to decode frame %d from %s",
                    fi, video_path,
                )
                # Create a black frame as fallback to maintain sequence length
                raw_frame = np.zeros((224, 224, 3), dtype=np.uint8)

            raw_frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
            if self.transform is not None:
                frame_tensor = self.transform(raw_frame)
            else:
                frame_tensor = (
                    torch.from_numpy(raw_frame).permute(2, 0, 1).float() / 255.0
                )

            frames.append(frame_tensor)

        # Stack: [T, C, H, W]
        sequence = torch.stack(frames, dim=0)
        return sequence, label

    # ── Properties ──────────────────────────────────────────

    @property
    def class_mapping(self) -> Dict[str, int]:
        return self._class_mapping.copy()

    @property
    def class_names(self) -> List[str]:
        return [
            self._idx_to_class[i]
            for i in range(self._num_classes)
        ]

    @property
    def num_classes(self) -> int:
        return self._num_classes

    @property
    def class_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {name: 0 for name in self._class_mapping}
        for _, label_idx, _ in self._videos:
            class_name = self._idx_to_class[label_idx]
            counts[class_name] += 1
        return counts

    @property
    def class_weights(self) -> torch.Tensor:
        """Inverse-frequency class weights for loss balancing."""
        counts = self.class_counts
        total = sum(counts.values())
        weights = []
        for i in range(self._num_classes):
            class_name = self._idx_to_class[i]
            count = max(counts.get(class_name, 0), 1)
            weights.append(total / (self._num_classes * count))
        return torch.tensor(weights, dtype=torch.float32)

    @property
    def sequence_length(self) -> int:
        """Number of frames per video sequence (T)."""
        return self.config.frames_per_video
