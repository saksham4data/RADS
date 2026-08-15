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
    aug_config: dict | None = None,
) -> transforms.Compose:
    """Build the training augmentation pipeline.

    Applies optional random augmentations based on configuration.
    Base pipeline: Resize → [Augmentations] → ToTensor → Normalize.

    Parameters
    ----------
    image_size : tuple[int, int]
        Target ``(height, width)`` for resizing.
    aug_config : dict, optional
        Augmentation configuration block from TrainingConfig.

    Returns
    -------
    transforms.Compose
        Composed transform pipeline.
    """
    transform_list = [
        transforms.ToPILImage(),
        transforms.Resize(image_size),
    ]

    if aug_config and aug_config.get("enabled", False):
        preset = aug_config.get("preset", "none")
        
        if preset == "conservative_v1":
            transform_list.extend([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=5),
                transforms.ColorJitter(brightness=0.15, contrast=0.15),
            ])
        elif preset == "custom":
            if aug_config.get("horizontal_flip", {}).get("enabled", False):
                transform_list.append(transforms.RandomHorizontalFlip(
                    p=aug_config["horizontal_flip"].get("p", 0.5)
                ))
            
            if aug_config.get("rotation", {}).get("enabled", False):
                transform_list.append(transforms.RandomRotation(
                    degrees=aug_config["rotation"].get("degrees", 5)
                ))
                
            bc_config = aug_config.get("brightness_contrast", {})
            sat_config = aug_config.get("saturation", {})
            hue_config = aug_config.get("hue", {})
            
            brightness = bc_config.get("brightness", 0.0) if bc_config.get("enabled", False) else 0.0
            contrast = bc_config.get("contrast", 0.0) if bc_config.get("enabled", False) else 0.0
            saturation = sat_config.get("value", 0.0) if sat_config.get("enabled", False) else 0.0
            hue = hue_config.get("value", 0.0) if hue_config.get("enabled", False) else 0.0
            
            if any(v > 0 for v in [brightness, contrast, saturation, hue]):
                transform_list.append(transforms.ColorJitter(
                    brightness=brightness,
                    contrast=contrast,
                    saturation=saturation,
                    hue=hue
                ))
            
            blur_config = aug_config.get("gaussian_blur", {})
            if blur_config.get("enabled", False):
                transform_list.append(transforms.GaussianBlur(
                    kernel_size=blur_config.get("kernel_size", 3),
                    sigma=blur_config.get("sigma", (0.1, 0.5))
                ))

    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    return transforms.Compose(transform_list)


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
