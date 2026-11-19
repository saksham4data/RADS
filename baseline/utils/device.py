# ─────────────────────────────────────────────────────────────
# Device Detection \u0026 Info
# ─────────────────────────────────────────────────────────────
"""
Auto-detects the best available compute device (CUDA GPU or CPU)
and logs hardware information for reproducibility.
"""

from __future__ import annotations

import logging
from typing import Dict, Any

import torch

logger = logging.getLogger(__name__)


def get_device() -> torch.device:
    """Auto-detect the best available compute device.

    Returns ``torch.device("cuda")`` if a CUDA GPU is available,
    otherwise ``torch.device("cpu")``.  Logs device information.

    Returns
    -------
    torch.device
        The selected compute device.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_mem / (1024 ** 3)
        logger.info(
            "Device: CUDA — %s (%.1f GB)", gpu_name, gpu_memory,
        )
    else:
        device = torch.device("cpu")
        logger.info("Device: CPU (no CUDA GPU detected)")

    return device


def get_device_info() -> Dict[str, Any]:
    """Return a dict of device information for logging / W&B.

    Returns
    -------
    dict
        Keys: ``device``, ``cuda_available``, ``gpu_name``,
        ``gpu_memory_gb``, ``cuda_version``.
    """
    info: Dict[str, Any] = {
        "cuda_available": torch.cuda.is_available(),
    }

    if torch.cuda.is_available():
        info["device"] = "cuda"
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_memory_gb"] = round(
            torch.cuda.get_device_properties(0).total_mem / (1024 ** 3), 2,
        )
        info["cuda_version"] = torch.version.cuda or "unknown"
        info["gpu_count"] = torch.cuda.device_count()
    else:
        info["device"] = "cpu"
        info["gpu_name"] = None
        info["gpu_memory_gb"] = None
        info["cuda_version"] = None
        info["gpu_count"] = 0

    info["pytorch_version"] = torch.__version__

    return info
