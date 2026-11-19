# ─────────────────────────────────────────────────────────────
# RADS — Training Infrastructure Module (Pipeline 4)
# ─────────────────────────────────────────────────────────────
"""
Modular, reusable training infrastructure for the RADS project.

This module validates the complete ML workflow using a lightweight
baseline model before moving to object detection.  It reuses the
existing configuration, logging, W&B, and monitoring systems
established by the EDA module (Pipeline 3).

It does NOT modify any existing pipeline code or data.
"""

__version__ = "1.0.0"
