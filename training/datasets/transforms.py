# ─────────────────────────────────────────────────────────────
# Train / Val / Test Augmentation Pipelines
# ─────────────────────────────────────────────────────────────
"""
Provides composable ``torchvision.transforms`` pipelines for
training, validation, and test sets.

All pipelines normalise to ImageNet statistics since the
baseline models are pretrained on ImageNet.
"""

from __future__ import annotations

from typing import Tuple

from torchvision import transforms


# ── ImageNet statistics ─────────────────────────────────────

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms(
    image_size: Tuple[int, int] = (224, 224),
) -> transforms.Compose:
    """Build the training augmentation pipeline.

    Applies random augmentations for regularisation:
    Resize → RandomHorizontalFlip → ColorJitter → ToTensor → Normalize.

    Parameters
    ----------
    image_size : tuple[int, int]
        Target ``(height, width)`` for resizing.

    Returns
    -------
    transforms.Compose
        Composed transform pipeline.
    """
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1,
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_val_transforms(
    image_size: Tuple[int, int] = (224, 224),
) -> transforms.Compose:
    """Build the validation / test transform pipeline.

    No random augmentation — deterministic processing:
    Resize → CenterCrop → ToTensor → Normalize.

    Parameters
    ----------
    image_size : tuple[int, int]
        Target ``(height, width)`` for resizing.

    Returns
    -------
    transforms.Compose
        Composed transform pipeline.
    """
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
