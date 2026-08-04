# ─────────────────────────────────────────────────────────────
# System Resource Monitor
# ─────────────────────────────────────────────────────────────
"""
Collects CPU, RAM, disk, GPU, and software environment
snapshots for reproducibility and performance tracking.
"""

from __future__ import annotations

import platform
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import psutil


def _get_gpu_info() -> Optional[List[Dict[str, Any]]]:
    """Attempt to read GPU information via GPUtil or pynvml."""
    # Try GPUtil first
    try:
        import GPUtil  # type: ignore[import-untyped]
        gpus = GPUtil.getGPUs()
        if gpus:
            return [
                {
                    "id": g.id,
                    "name": g.name,
                    "memory_total_mb": g.memoryTotal,
                    "memory_used_mb": g.memoryUsed,
                    "memory_free_mb": g.memoryFree,
                    "load_percent": round(g.load * 100, 1),
                    "temperature_c": g.temperature,
                }
                for g in gpus
            ]
    except Exception:
        pass

    # Fallback: pynvml
    try:
        import pynvml  # type: ignore[import-untyped]
        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        gpus = []
        for i in range(count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8")
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpus.append({
                "id": i,
                "name": name,
                "memory_total_mb": round(mem.total / 1048576),
                "memory_used_mb": round(mem.used / 1048576),
                "memory_free_mb": round(mem.free / 1048576),
                "load_percent": util.gpu,
            })
        pynvml.nvmlShutdown()
        if gpus:
            return gpus
    except Exception:
        pass

    return None


def _get_package_versions() -> Dict[str, str]:
    """Return versions of key packages."""
    packages = [
        "pandas", "numpy", "matplotlib", "seaborn",
        "wandb", "psutil", "opencv-python", "pyyaml",
        "jupyter", "ipykernel", "tabulate",
    ]
    versions: Dict[str, str] = {}
    for pkg in packages:
        try:
            # Handle packages whose import name differs from pip name
            import_name = pkg.replace("-", "_")
            mod = __import__(import_name)
            versions[pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            versions[pkg] = "not installed"
    return versions


class SystemMonitor:
    """Captures system resource snapshots at start and end of execution.

    Usage::

        monitor = SystemMonitor()
        start = monitor.snapshot("start")
        # ... do work ...
        end = monitor.snapshot("end")
        delta = monitor.compute_delta(start, end)
    """

    def snapshot(self, label: str = "snapshot") -> Dict[str, Any]:
        """Capture a point-in-time system resource snapshot."""
        cpu_percent = psutil.cpu_percent(interval=0.5)
        mem = psutil.virtual_memory()
        disk = psutil.disk_usage("/") if platform.system() != "Windows" else psutil.disk_usage("C:\\")

        snap: Dict[str, Any] = {
            "label": label,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "epoch": time.time(),
            "cpu": {
                "count_logical": psutil.cpu_count(logical=True),
                "count_physical": psutil.cpu_count(logical=False),
                "percent": cpu_percent,
            },
            "memory": {
                "total_gb": round(mem.total / (1024 ** 3), 2),
                "available_gb": round(mem.available / (1024 ** 3), 2),
                "used_gb": round(mem.used / (1024 ** 3), 2),
                "percent": mem.percent,
            },
            "disk": {
                "total_gb": round(disk.total / (1024 ** 3), 2),
                "used_gb": round(disk.used / (1024 ** 3), 2),
                "free_gb": round(disk.free / (1024 ** 3), 2),
                "percent": disk.percent,
            },
            "software": {
                "python_version": platform.python_version(),
                "os": platform.system(),
                "os_version": platform.version(),
                "platform": platform.platform(),
                "packages": _get_package_versions(),
            },
        }

        # GPU (optional, graceful)
        gpu_info = _get_gpu_info()
        if gpu_info:
            snap["gpu"] = gpu_info
        else:
            snap["gpu"] = None

        return snap

    @staticmethod
    def compute_delta(
        start: Dict[str, Any],
        end: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Compute resource usage deltas between two snapshots."""
        duration = end["epoch"] - start["epoch"]
        mem_delta = (
            end["memory"]["used_gb"] - start["memory"]["used_gb"]
        )
        return {
            "duration_seconds": round(duration, 2),
            "memory_delta_gb": round(mem_delta, 3),
            "cpu_start_percent": start["cpu"]["percent"],
            "cpu_end_percent": end["cpu"]["percent"],
            "memory_start_percent": start["memory"]["percent"],
            "memory_end_percent": end["memory"]["percent"],
        }


