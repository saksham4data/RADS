# ─────────────────────────────────────────────────────────────
# DataLoader Factory
# ─────────────────────────────────────────────────────────────
"""
Creates train / val / test DataLoaders with proper shuffling,
class-weighted sampling, and multiprocessing configuration.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from training.configs.config import TrainingConfig
from training.datasets.frame_dataset import VideoFrameDataset
from training.datasets.transforms import get_train_transforms, get_val_transforms

logger = logging.getLogger(__name__)


def create_dataloaders(
    config: TrainingConfig,
    *,
    use_weighted_sampler: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """Build train and validation DataLoaders from configuration.

    Parameters
    ----------
    config : TrainingConfig
        Full training configuration.
    use_weighted_sampler : bool
        If ``True``, use ``WeightedRandomSampler`` for the training
        set to handle class imbalance.

    Returns
    -------
    tuple[DataLoader, DataLoader]
        ``(train_loader, val_loader)``
    """
    image_size = config.image_size

    # ── Build datasets ──
    train_dataset = VideoFrameDataset(
        config,
        split="train",
        transform=get_train_transforms(image_size, aug_config=config.augmentation_config),
    )
    val_dataset = VideoFrameDataset(
        config,
        split="val",
        transform=get_val_transforms(image_size),
    )

    # ── Training sampler (class-weighted) ──
    sampler: Optional[WeightedRandomSampler] = None
    shuffle = True

    if use_weighted_sampler and len(train_dataset) > 0:
        class_weights = train_dataset.class_weights
        # Assign per-sample weight based on its class
        sample_weights = []
        for idx in range(len(train_dataset)):
            vid_idx, _ = train_dataset._samples[idx]
            label = train_dataset._video_labels[vid_idx]
            sample_weights.append(float(class_weights[label]))

        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        shuffle = False  # Mutually exclusive with sampler

        logger.info(
            "WeightedRandomSampler enabled — class weights: %s",
            {name: f"{w:.3f}" for name, w in
             zip(train_dataset.class_names, class_weights.tolist())},
        )

    # ── Build DataLoaders ──
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=False,
        collate_fn=_safe_collate,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=False,
        collate_fn=_safe_collate,
    )

    logger.info(
        "DataLoaders created — train: %d batches (%d samples), "
        "val: %d batches (%d samples)",
        len(train_loader), len(train_dataset),
        len(val_loader), len(val_dataset),
    )

    return train_loader, val_loader


def create_test_dataloader(
    config: TrainingConfig,
) -> DataLoader:
    """Build a test DataLoader.

    Parameters
    ----------
    config : TrainingConfig
        Full training configuration.

    Returns
    -------
    DataLoader
        Test data loader (no shuffling, no augmentation).
    """
    image_size = config.image_size

    test_dataset = VideoFrameDataset(
        config,
        split="test",
        transform=get_val_transforms(image_size),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=False,
        collate_fn=_safe_collate,
    )

    logger.info(
        "Test DataLoader created — %d batches (%d samples)",
        len(test_loader), len(test_dataset),
    )

    return test_loader


def _safe_collate(batch):
    """Collate function that filters out ``None`` entries.

    If a sample fails to load (e.g. corrupt video), it returns
    ``None``.  This collate function skips those entries.
    """
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return torch.tensor([]), torch.tensor([])
    return torch.utils.data.dataloader.default_collate(batch)
