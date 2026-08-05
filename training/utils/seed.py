# ─────────────────────────────────────────────────────────────
# Global Seed Setter
# ─────────────────────────────────────────────────────────────
"""
Sets deterministic seeds across Python, NumPy, and PyTorch
for fully reproducible training runs.
"""

from __future__ import annotations

import logging
import random
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def set_global_seed(seed: int = 42) -> None:
    """Set deterministic seeds for reproducibility.

    Seeds Python's ``random``, NumPy, and PyTorch (CPU + CUDA).
    Also sets cuDNN to deterministic mode.

    Parameters
    ----------
    seed : int
        Seed value.  Default 42.
    """
    # Python built-in
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch (import lazily to avoid hard dependency at module level)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        logger.info(
            "Global seed set: %d (Python + NumPy + PyTorch + cuDNN deterministic)",
            seed,
        )
    except ImportError:
        logger.info(
            "Global seed set: %d (Python + NumPy only — PyTorch not installed)",
            seed,
        )
