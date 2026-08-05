# ─────────────────────────────────────────────────────────────
# System Resource Monitor — Re-export
# ─────────────────────────────────────────────────────────────
"""
Re-exports ``SystemMonitor`` from ``eda.utils.system_monitor``
so the training module can use it without duplicating code.
"""

from eda.utils.system_monitor import SystemMonitor

__all__ = ["SystemMonitor"]
