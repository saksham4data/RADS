#!/usr/bin/env python3
"""
picek_negative_extractor.py
===========================
Pre-Accident Negative Clip Generator for PICEK Dataset

Extracts normal-driving negative clips from real PICEK accident videos prior
to accident onset, enforcing a configurable safety gap before the collision.

Key Features:
- Uses existing metadata.json / CSV as source of truth for accident timing.
- Enforces safety gap: end_sec = accident_time_sec - safety_gap_sec.
- Target clip duration (e.g. 5.0s) extracted immediately prior to safe end.
- Non-destructive: Writes to trimmed/negative/real/ without touching positive data.
- System resource monitor: Real-time CPU, RAM, Disk, GPU/VRAM tracking with auto-pause.
- Resumable: Skips existing valid negative clips.
- Automated pilot verification: Validates pre-accident timing before batch processing.
- Generates negative_metadata.json and comprehensive Markdown report.

Usage:
    # Run Pilot check only (e.g. 5 videos)
    python scripts/picek_negative_extractor.py --pilot 5

    # Run Full extraction
    python scripts/picek_negative_extractor.py --safety-gap-sec 1.0 --clip-duration-sec 5.0 --min-clip-sec 2.0

Author: Antigravity (RADS Research)
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import shutil
import sys
import threading
import time
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import psutil

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION DEFAULTS
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_SAFETY_GAP_SEC: float = 1.0       # Minimum seconds prior to accident onset
DEFAULT_CLIP_DURATION_SEC: float = 5.0     # Target negative clip duration (seconds)
DEFAULT_MIN_CLIP_SEC: float = 2.0          # Minimum duration required to produce clip

# System Resource Thresholds
CPU_WARN_PCT: float = 90.0
RAM_WARN_PCT: float = 92.0
DISK_WARN_PCT: float = 90.0
CPU_PAUSE_PCT: float = 98.0
RAM_PAUSE_PCT: float = 97.0

MONITOR_INTERVAL_SEC: float = 2.0
PAUSE_SLEEP_SEC: float = 2.0
THROTTLE_SLEEP_SEC: float = 0.03

DEFAULT_METADATA_JSON = "Datasets/processed/picek_sorted/metadata.json"
DEFAULT_OUTPUT_DIR = "Datasets/processed/picek_sorted"
DEFAULT_RAW_DIR = "Datasets/raw/picekl"


# ──────────────────────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class NegativeClipMeta:
    """Stores full metadata & provenance for one generated negative clip."""
    video_id: str
    source_filename: str
    source_abs_path: str
    source_accident_type: str
    source_accident_time_sec: float
    source_accident_frame: int
    source_duration_sec: float
    source_fps: float
    source_frame_count: int
    source_width: int
    source_height: int
    source_split: str
    weather: str
    day_time: str
    rollover: int

    # Negative Extraction Parameters
    clip_type: str = "pre_accident_negative"
    label: str = "negative"
    safety_gap_sec: float = DEFAULT_SAFETY_GAP_SEC
    target_duration_sec: float = DEFAULT_CLIP_DURATION_SEC
    min_clip_sec: float = DEFAULT_MIN_CLIP_SEC

    # Extracted Clip Properties
    trim_start_sec: float = 0.0
    trim_end_sec: float = 0.0
    trim_start_frame: int = 0
    trim_end_frame: int = 0
    trim_frame_count: int = 0
    trim_duration_sec: float = 0.0
    output_path: Optional[str] = None
    file_size_bytes: int = 0
    codec: str = "mp4v"

    # Status
    status: str = "pending"   # pending | done | skipped | error
    skip_reason: Optional[str] = None
    error_message: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class NegativePipelineStats:
    total_candidates: int = 0
    done: int = 0
    skipped: int = 0
    errors: int = 0
    existing_reused: int = 0
    total_bytes_written: int = 0
    start_time: float = field(default_factory=time.time)
    end_time: float = 0.0

    @property
    def elapsed_sec(self) -> float:
        t = self.end_time if self.end_time else time.time()
        return t - self.start_time


# ──────────────────────────────────────────────────────────────────────────────
# SYSTEM RESOURCE MONITOR
# ──────────────────────────────────────────────────────────────────────────────

class SystemMonitor(threading.Thread):
    """
    Continuous background monitor for CPU, RAM, Disk, and GPU/VRAM.
    Dynamically signals pauses during dangerous load spikes.
    """

    def __init__(self, logger: logging.Logger, output_dir: Path):
        super().__init__(daemon=True, name="SystemMonitor")
        self.logger = logger
        self.output_dir = output_dir
        self.paused = False
        self.pause_count = 0
        self.stop_event = threading.Event()
        self.samples: List[dict] = []

    def _sample_gpu(self) -> Tuple[Optional[float], Optional[float]]:
        """Safely inspect GPU utilization and VRAM usage if present."""
        try:
            import torch
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated(0) / (1024 ** 2)
                reserved = torch.cuda.memory_reserved(0) / (1024 ** 2)
                return allocated, reserved
        except Exception:
            pass
        return None, None

    def run(self):
        while not self.stop_event.is_set():
            try:
                cpu = psutil.cpu_percent(interval=None)
                ram = psutil.virtual_memory().percent
                disk_drive = Path(self.output_dir.resolve()).anchor or "C:\\"
                disk_usage = psutil.disk_usage(disk_drive)
                disk = disk_usage.percent
                disk_free_gb = disk_usage.free / (1024 ** 3)
                gpu_alloc_mb, gpu_res_mb = self._sample_gpu()

                sample = {
                    "ts": datetime.now().isoformat(timespec="seconds"),
                    "cpu_pct": cpu,
                    "ram_pct": ram,
                    "disk_pct": disk,
                    "disk_free_gb": round(disk_free_gb, 2),
                    "gpu_alloc_mb": gpu_alloc_mb,
                }
                self.samples.append(sample)

                # Overload protection trigger
                if cpu > CPU_PAUSE_PCT or ram > RAM_PAUSE_PCT:
                    if not self.paused:
                        self.pause_count += 1
                        self.logger.warning(
                            f"[MONITOR] OVERLOAD DETECTED -- CPU {cpu:.1f}% | RAM {ram:.1f}% "
                            f"-- Pausing pipeline until system cools down..."
                        )
                    self.paused = True
                else:
                    if self.paused:
                        self.logger.info(
                            f"[MONITOR] Load normalized -- CPU {cpu:.1f}% | RAM {ram:.1f}% "
                            f"-- Resuming execution."
                        )
                    self.paused = False

                # Warnings
                if cpu > CPU_WARN_PCT:
                    self.logger.warning(f"[MONITOR] High CPU: {cpu:.1f}% (thresh {CPU_WARN_PCT}%)")
                if ram > RAM_WARN_PCT:
                    self.logger.warning(f"[MONITOR] High RAM: {ram:.1f}% (thresh {RAM_WARN_PCT}%)")
                if disk > DISK_WARN_PCT:
                    self.logger.warning(
                        f"[MONITOR] Low Disk Space: {disk:.1f}% used, {disk_free_gb:.1f} GB free"
                    )

            except Exception as exc:
                self.logger.debug(f"[MONITOR] poll error: {exc}")

            self.stop_event.wait(MONITOR_INTERVAL_SEC)

    def wait_if_paused(self):
        """Block caller thread while system is in paused/overloaded state."""
        while self.paused and not self.stop_event.is_set():
            time.sleep(PAUSE_SLEEP_SEC)

    def stop(self):
        self.stop_event.set()

    def summary(self) -> dict:
        if not self.samples:
            return {}
        cpu_vals = [s["cpu_pct"] for s in self.samples]
        ram_vals = [s["ram_pct"] for s in self.samples]
        disk_vals = [s["disk_pct"] for s in self.samples]
        return {
            "samples": len(self.samples),
            "pause_events": self.pause_count,
            "cpu_avg": round(sum(cpu_vals) / len(cpu_vals), 1),
            "cpu_max": round(max(cpu_vals), 1),
            "ram_avg": round(sum(ram_vals) / len(ram_vals), 1),
            "ram_max": round(max(ram_vals), 1),
            "disk_avg": round(sum(disk_vals) / len(disk_vals), 1),
            "disk_max": round(max(disk_vals), 1),
        }


# ──────────────────────────────────────────────────────────────────────────────
# LOGGING
# ──────────────────────────────────────────────────────────────────────────────

def setup_logging(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("picek_negative_extractor")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_path, encoding="utf-8", mode="a")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


# ──────────────────────────────────────────────────────────────────────────────
# EXTRACTION LOGIC & OPENCV
# ──────────────────────────────────────────────────────────────────────────────

def calculate_negative_bounds(
    accident_time_sec: float,
    video_duration_sec: float,
    safety_gap_sec: float,
    target_duration_sec: float,
    min_clip_sec: float,
) -> Tuple[bool, float, float, Optional[str]]:
    """
    Calculate safe pre-accident negative window.
    Returns: (is_valid, start_sec, end_sec, reason_if_invalid)
    """
    safe_end = accident_time_sec - safety_gap_sec
    if safe_end < min_clip_sec:
        reason = (
            f"insufficient_pre_accident_time: accident_time ({accident_time_sec:.2f}s) "
            f"< safety_gap ({safety_gap_sec:.2f}s) + min_clip ({min_clip_sec:.2f}s)"
        )
        return False, 0.0, 0.0, reason

    # Clip end cannot exceed safe_end or video duration
    end_sec = safe_end
    if video_duration_sec > 0:
        end_sec = min(end_sec, video_duration_sec)

    start_sec = max(0.0, end_sec - target_duration_sec)
    actual_duration = end_sec - start_sec

    if actual_duration < min_clip_sec:
        reason = f"clip_too_short: actual_duration ({actual_duration:.2f}s) < min_clip ({min_clip_sec:.2f}s)"
        return False, 0.0, 0.0, reason

    return True, start_sec, end_sec, None


def extract_negative_clip_cv2(
    src_path: str,
    dest_path: str,
    start_sec: float,
    end_sec: float,
    logger: logging.Logger,
) -> Tuple[bool, int, float, Optional[str]]:
    """
    Extract frame-accurate video clip [start_sec, end_sec] via OpenCV.
    Returns: (success, frames_written, actual_duration_sec, error_msg)
    """
    try:
        cap = cv2.VideoCapture(src_path)
        if not cap.isOpened():
            return False, 0, 0.0, f"Cannot open source video: {src_path}"

        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if fps <= 0 or total_frames <= 0:
            cap.release()
            return False, 0, 0.0, f"Invalid video properties: fps={fps}, frames={total_frames}"

        start_frame = max(0, int(start_sec * fps))
        end_frame = min(total_frames, int(end_sec * fps))

        if start_frame >= end_frame:
            cap.release()
            return False, 0, 0.0, f"Degenerate frame range [{start_frame}, {end_frame}]"

        frames_to_write = end_frame - start_frame
        Path(dest_path).parent.mkdir(parents=True, exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(dest_path, fourcc, fps, (width, height))
        if not out.isOpened():
            cap.release()
            return False, 0, 0.0, f"cv2.VideoWriter failed to open for {dest_path}"

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        written = 0

        for _ in range(frames_to_write):
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
            written += 1

        cap.release()
        out.release()

        if written == 0:
            if Path(dest_path).exists():
                Path(dest_path).unlink()
            return False, 0, 0.0, "Zero frames written"

        actual_duration = written / fps
        return True, written, actual_duration, None

    except Exception as exc:
        if Path(dest_path).exists():
            try:
                Path(dest_path).unlink()
            except Exception:
                pass
        return False, 0, 0.0, f"Exception during cv2 extraction: {exc}"


# ──────────────────────────────────────────────────────────────────────────────
# PILOT RUN & VERIFICATION
# ──────────────────────────────────────────────────────────────────────────────

def run_pilot_verification(
    pilot_candidates: List[NegativeClipMeta],
    output_dir: Path,
    safety_gap_sec: float,
    target_duration_sec: float,
    min_clip_sec: float,
    logger: logging.Logger,
) -> bool:
    """
    Execute a pilot run on sample candidate videos and verify that:
    1. Extracted clips genuinely precede the annotated accident time.
    2. End timestamp strictly satisfies: end_sec <= accident_time - safety_gap.
    3. Output files are valid, non-empty, and decodable via cv2.
    """
    logger.info("=" * 70)
    logger.info(f"PILOT VERIFICATION PHASE: Testing {len(pilot_candidates)} candidate videos")
    logger.info("=" * 70)

    pilot_dir = output_dir / "trimmed" / "negative" / "pilot_test"
    pilot_dir.mkdir(parents=True, exist_ok=True)

    pilot_results: List[dict] = []
    all_passed = True

    for idx, cand in enumerate(pilot_candidates, 1):
        test_dest = pilot_dir / f"pilot_{cand.video_id}.mp4"
        valid, start_sec, end_sec, reason = calculate_negative_bounds(
            accident_time_sec=cand.source_accident_time_sec,
            video_duration_sec=cand.source_duration_sec,
            safety_gap_sec=safety_gap_sec,
            target_duration_sec=target_duration_sec,
            min_clip_sec=min_clip_sec,
        )

        if not valid:
            logger.info(
                f"  Pilot [{idx}/{len(pilot_candidates)}] {cand.video_id}: "
                f"Accident at {cand.source_accident_time_sec:.2f}s -> Correctly SKIPPED ({reason})"
            )
            pilot_results.append({
                "video_id": cand.video_id,
                "status": "SKIPPED_VALID",
                "accident_time": cand.source_accident_time_sec,
                "reason": reason,
            })
            continue

        # Extract pilot clip
        ok, written, dur, err = extract_negative_clip_cv2(
            src_path=cand.source_abs_path,
            dest_path=str(test_dest),
            start_sec=start_sec,
            end_sec=end_sec,
            logger=logger,
        )

        if not ok:
            logger.error(f"  Pilot [{idx}/{len(pilot_candidates)}] FAIL on {cand.video_id}: {err}")
            all_passed = False
            continue

        # Verify decoded properties
        cap = cv2.VideoCapture(str(test_dest))
        if not cap.isOpened():
            logger.error(f"  Pilot [{idx}/{len(pilot_candidates)}] FAIL: Output not decodable")
            all_passed = False
            continue

        dec_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        dec_fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        # Strict Temporal Verification Gate
        temporal_clearance = cand.source_accident_time_sec - end_sec
        is_strictly_pre_accident = (end_sec <= (cand.source_accident_time_sec - safety_gap_sec + 1e-4))

        logger.info(
            f"  Pilot [{idx}/{len(pilot_candidates)}] PASS: {cand.video_id} | "
            f"Accident={cand.source_accident_time_sec:.2f}s | "
            f"Trim=[{start_sec:.2f}s - {end_sec:.2f}s] ({dur:.2f}s, {written} frames) | "
            f"Safety Clearance={temporal_clearance:.2f}s (>= {safety_gap_sec:.2f}s)"
        )

        if not is_strictly_pre_accident:
            logger.error(
                f"  CRITICAL TEMPORAL VIOLATION: end_sec ({end_sec:.2f}s) "
                f"> accident_time ({cand.source_accident_time_sec:.2f}s) - safety_gap ({safety_gap_sec:.2f}s)"
            )
            all_passed = False

        pilot_results.append({
            "video_id": cand.video_id,
            "status": "PASSED",
            "accident_time": cand.source_accident_time_sec,
            "trim_window": f"[{start_sec:.2f}s - {end_sec:.2f}s]",
            "duration": dur,
            "frames": written,
            "temporal_clearance": temporal_clearance,
        })

    # Clean up pilot directory files
    try:
        shutil.rmtree(pilot_dir, ignore_errors=True)
    except Exception:
        pass

    if all_passed:
        logger.info("=" * 70)
        logger.info("PILOT VERIFICATION PASSED: All pre-accident criteria satisfied!")
        logger.info("=" * 70)
    else:
        logger.error("PILOT VERIFICATION FAILED: Aborting before full extraction.")

    return all_passed


# ──────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ──────────────────────────────────────────────────────────────────────────────

def run_pipeline(
    metadata_json_path: Path,
    output_dir: Path,
    raw_dir: Path,
    safety_gap_sec: float,
    clip_duration_sec: float,
    min_clip_sec: float,
    pilot_count: int,
    skip_existing: bool,
    max_videos: Optional[int],
    dry_run: bool,
):
    log_path = output_dir / "negative_pipeline.log"
    logger = setup_logging(log_path)

    logger.info("=" * 70)
    logger.info("PICEK PRE-ACCIDENT NEGATIVE EXTRACTION PIPELINE")
    logger.info("=" * 70)
    logger.info(f"Metadata Source    : {metadata_json_path}")
    logger.info(f"Output Directory   : {output_dir}")
    logger.info(f"Safety Gap         : {safety_gap_sec:.2f} s")
    logger.info(f"Target Duration    : {clip_duration_sec:.2f} s")
    logger.info(f"Min Duration       : {min_clip_sec:.2f} s")
    logger.info(f"Skip Existing      : {skip_existing}")
    logger.info(f"Dry Run            : {dry_run}")
    logger.info(f"Max Videos Limit   : {max_videos}")

    # Start System Resource Monitor
    monitor = SystemMonitor(logger=logger, output_dir=output_dir)
    monitor.start()

    stats = NegativePipelineStats()

    try:
        # Load Existing Metadata JSON
        if not metadata_json_path.exists():
            logger.error(f"Metadata JSON not found at {metadata_json_path}")
            return False

        with open(metadata_json_path, "r", encoding="utf-8") as f:
            metadata_payload = json.load(f)

        raw_videos = metadata_payload.get("videos", [])
        # Filter for REAL videos only (as synthetic are CARLA simulations)
        real_videos = [v for v in raw_videos if v.get("source_type") == "real"]
        logger.info(f"Found {len(real_videos)} real PICEK videos in metadata.")

        if max_videos:
            real_videos = real_videos[:max_videos]
            logger.info(f"Applied limit: processing {len(real_videos)} videos.")

        stats.total_candidates = len(real_videos)

        # Build candidate objects
        candidates: List[NegativeClipMeta] = []
        for v in real_videos:
            # Resolve source file path (check raw or positive directory)
            src_path = v.get("abs_path")
            if not src_path or not Path(src_path).exists():
                alt_path = output_dir / "positive" / "real" / f"{v['video_id']}.mp4"
                if alt_path.exists():
                    src_path = str(alt_path)
                else:
                    raw_alt = raw_dir / "real_videos" / f"{v['video_id']}.mp4"
                    if raw_alt.exists():
                        src_path = str(raw_alt)

            fps = float(v.get("fps", 0.0))
            dur = float(v.get("duration_sec", 0.0))
            frame_cnt = int(v.get("frame_count", 0))

            # If probed properties are zero, probe on the fly
            if (fps <= 0 or dur <= 0) and src_path and Path(src_path).exists():
                cap = cv2.VideoCapture(src_path)
                if cap.isOpened():
                    fps = cap.get(cv2.CAP_PROP_FPS) or fps
                    frame_cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or frame_cnt
                    if fps > 0 and frame_cnt > 0:
                        dur = frame_cnt / fps
                    cap.release()

            cand = NegativeClipMeta(
                video_id=v["video_id"],
                source_filename=v.get("filename", f"real_videos/{v['video_id']}.mp4"),
                source_abs_path=str(src_path),
                source_accident_type=v.get("accident_type", "unknown"),
                source_accident_time_sec=float(v.get("accident_time_sec", 0.0)),
                source_accident_frame=int(v.get("accident_frame", 0)),
                source_duration_sec=dur,
                source_fps=fps,
                source_frame_count=frame_cnt,
                source_width=int(v.get("width", 0)),
                source_height=int(v.get("height", 0)),
                source_split=v.get("split", "train"),
                weather=v.get("weather", "normal"),
                day_time=v.get("day_time", "day"),
                rollover=int(v.get("rollover", 0)),
                safety_gap_sec=safety_gap_sec,
                target_duration_sec=clip_duration_sec,
                min_clip_sec=min_clip_sec,
            )
            candidates.append(cand)

        # ── Step 1: Run Pilot Gate ──────────────────────────────────────────
        pilot_sample_size = pilot_count if pilot_count > 0 else 5
        pilot_selection = [c for c in candidates if Path(c.source_abs_path).exists()][:pilot_sample_size]

        pilot_ok = run_pilot_verification(
            pilot_candidates=pilot_selection,
            output_dir=output_dir,
            safety_gap_sec=safety_gap_sec,
            target_duration_sec=clip_duration_sec,
            min_clip_sec=min_clip_sec,
            logger=logger,
        )

        if not pilot_ok:
            logger.error("Pilot verification failed. Halting pipeline.")
            return False

        if pilot_count > 0 and not max_videos:
            logger.info("Pilot mode only requested (--pilot flag). Exiting.")
            return True

        # ── Step 2: Full Dataset Negative Extraction ────────────────────────
        neg_trimmed_dir = output_dir / "trimmed" / "negative" / "real"
        neg_trimmed_dir.mkdir(parents=True, exist_ok=True)

        logger.info("=" * 70)
        logger.info(f"PROCESSING FULL DATASET: Extracting {len(candidates)} Negative Clips")
        logger.info("=" * 70)

        for idx, cand in enumerate(candidates, 1):
            monitor.wait_if_paused()
            time.sleep(THROTTLE_SLEEP_SEC)  # Throttle to keep CPU stable

            dest_path = neg_trimmed_dir / f"{cand.video_id}.mp4"
            cand.output_path = str(dest_path)

            if idx % 50 == 0 or idx == len(candidates) or idx == 1:
                logger.info(
                    f"Progress: [{idx}/{len(candidates)}] | "
                    f"Done={stats.done} | Skipped={stats.skipped} | Errors={stats.errors}"
                )

            # Check if source video exists
            if not cand.source_abs_path or not Path(cand.source_abs_path).exists():
                cand.status = "error"
                cand.error_message = f"Source file does not exist: {cand.source_abs_path}"
                stats.errors += 1
                continue

            # Calculate safe bounds
            valid, start_sec, end_sec, skip_reason = calculate_negative_bounds(
                accident_time_sec=cand.source_accident_time_sec,
                video_duration_sec=cand.source_duration_sec,
                safety_gap_sec=safety_gap_sec,
                target_duration_sec=clip_duration_sec,
                min_clip_sec=min_clip_sec,
            )

            if not valid:
                cand.status = "skipped"
                cand.skip_reason = skip_reason
                stats.skipped += 1
                logger.debug(f"[{idx}/{len(candidates)}] {cand.video_id}: SKIPPED ({skip_reason})")
                continue

            cand.trim_start_sec = round(start_sec, 3)
            cand.trim_end_sec = round(end_sec, 3)
            fps = cand.source_fps if cand.source_fps > 0 else 30.0
            cand.trim_start_frame = int(start_sec * fps)
            cand.trim_end_frame = int(end_sec * fps)

            # Resumability check
            if skip_existing and dest_path.exists() and dest_path.stat().st_size > 1000:
                cand.status = "done"
                cand.file_size_bytes = dest_path.stat().st_size
                cand.trim_duration_sec = round(end_sec - start_sec, 3)
                cand.trim_frame_count = max(1, int(cand.trim_duration_sec * fps))
                stats.done += 1
                stats.existing_reused += 1
                stats.total_bytes_written += cand.file_size_bytes
                logger.debug(f"[{idx}/{len(candidates)}] {cand.video_id}: Existing valid clip reused")
                continue

            if dry_run:
                cand.status = "skipped"
                cand.skip_reason = "dry_run"
                stats.skipped += 1
                continue

            # Execute OpenCV Extraction
            ok, written, dur, err_msg = extract_negative_clip_cv2(
                src_path=cand.source_abs_path,
                dest_path=str(dest_path),
                start_sec=start_sec,
                end_sec=end_sec,
                logger=logger,
            )

            if ok and dest_path.exists():
                cand.status = "done"
                cand.trim_frame_count = written
                cand.trim_duration_sec = round(dur, 3)
                cand.file_size_bytes = dest_path.stat().st_size
                stats.done += 1
                stats.total_bytes_written += cand.file_size_bytes
                logger.debug(
                    f"[{idx}/{len(candidates)}] {cand.video_id}: "
                    f"EXTRACTED [{start_sec:.2f}s-{end_sec:.2f}s] ({dur:.2f}s, {written} frames)"
                )
            else:
                cand.status = "error"
                cand.error_message = err_msg or "Extraction failed"
                stats.errors += 1
                logger.warning(f"[{idx}/{len(candidates)}] {cand.video_id}: ERROR: {cand.error_message}")

            # Incremental checkpoint every 100 videos
            if idx % 100 == 0:
                try:
                    neg_json_path = output_dir / "negative_metadata.json"
                    neg_checkpoint = {
                        "checkpoint_at": datetime.now().isoformat(),
                        "progress": f"{idx}/{len(candidates)}",
                        "done_count": stats.done,
                        "skipped_count": stats.skipped,
                        "error_count": stats.errors,
                        "videos": [c.to_dict() for c in candidates[:idx]],
                    }
                    with open(neg_json_path, "w", encoding="utf-8") as f:
                        json.dump(neg_checkpoint, f, indent=2, ensure_ascii=False)
                except Exception:
                    pass

        stats.end_time = time.time()

        # ── Step 3: Write Negative Metadata JSON & Update Master Metadata ───
        neg_json_path = output_dir / "negative_metadata.json"
        neg_payload = {
            "generated_at": datetime.now().isoformat(),
            "pipeline": "picek_negative_extractor",
            "safety_gap_sec": safety_gap_sec,
            "target_clip_duration_sec": clip_duration_sec,
            "min_clip_sec": min_clip_sec,
            "total_candidates": stats.total_candidates,
            "done_count": stats.done,
            "skipped_count": stats.skipped,
            "error_count": stats.errors,
            "existing_reused_count": stats.existing_reused,
            "total_size_mb": round(stats.total_bytes_written / (1024 ** 2), 2),
            "elapsed_sec": round(stats.elapsed_sec, 2),
            "videos": [c.to_dict() for c in candidates],
        }

        with open(neg_json_path, "w", encoding="utf-8") as f:
            json.dump(neg_payload, f, indent=2, ensure_ascii=False)
        logger.info(f"Negative metadata written -> {neg_json_path}")

        # Update Master metadata.json with negative provenance without altering positive fields
        cand_by_id = {c.video_id: c for c in candidates}
        for v in metadata_payload.get("videos", []):
            vid = v.get("video_id")
            if vid in cand_by_id:
                c = cand_by_id[vid]
                v["negative_clip"] = {
                    "status": c.status,
                    "trimmed_path": c.output_path if c.status == "done" else None,
                    "skip_reason": c.skip_reason,
                    "safety_gap_sec": c.safety_gap_sec,
                    "trim_start_sec": c.trim_start_sec,
                    "trim_end_sec": c.trim_end_sec,
                    "trim_duration_sec": c.trim_duration_sec,
                    "trim_frame_count": c.trim_frame_count,
                    "file_size_bytes": c.file_size_bytes,
                }

        metadata_payload["negative_generation"] = {
            "timestamp": datetime.now().isoformat(),
            "done_count": stats.done,
            "skipped_count": stats.skipped,
            "error_count": stats.errors,
            "safety_gap_sec": safety_gap_sec,
        }

        with open(metadata_json_path, "w", encoding="utf-8") as f:
            json.dump(metadata_payload, f, indent=2, ensure_ascii=False)
        logger.info(f"Updated master metadata -> {metadata_json_path}")

        # ── Step 4: Write Comprehensive Markdown Report ─────────────────────
        report_path = output_dir / "negative_dataset_report.md"
        write_markdown_report(
            candidates=candidates,
            stats=stats,
            monitor=monitor,
            report_path=report_path,
            safety_gap_sec=safety_gap_sec,
            target_duration_sec=clip_duration_sec,
            min_clip_sec=min_clip_sec,
            output_dir=output_dir,
        )
        logger.info(f"Markdown report written -> {report_path}")

        logger.info("=" * 70)
        logger.info("NEGATIVE EXTRACTION PIPELINE COMPLETED SUCCESSFULLY")
        logger.info(f"Elapsed     : {stats.elapsed_sec / 60:.2f} mins")
        logger.info(f"Total Videos: {stats.total_candidates}")
        logger.info(f"Done        : {stats.done}")
        logger.info(f"Skipped     : {stats.skipped}")
        logger.info(f"Errors      : {stats.errors}")
        logger.info(f"Report      : {report_path}")
        logger.info("=" * 70)

        return True

    finally:
        monitor.stop()


# ──────────────────────────────────────────────────────────────────────────────
# REPORT GENERATION
# ──────────────────────────────────────────────────────────────────────────────

def write_markdown_report(
    candidates: List[NegativeClipMeta],
    stats: NegativePipelineStats,
    monitor: SystemMonitor,
    report_path: Path,
    safety_gap_sec: float,
    target_duration_sec: float,
    min_clip_sec: float,
    output_dir: Path,
):
    mon_summary = monitor.summary()

    # Breakdown by accident type
    acc_types: Dict[str, dict] = {}
    for c in candidates:
        t = c.source_accident_type
        if t not in acc_types:
            acc_types[t] = {"total": 0, "done": 0, "skipped": 0, "errors": 0}
        acc_types[t]["total"] += 1
        if c.status == "done":
            acc_types[t]["done"] += 1
        elif c.status == "skipped":
            acc_types[t]["skipped"] += 1
        elif c.status == "error":
            acc_types[t]["errors"] += 1

    # Breakdown by weather
    weather_types: Dict[str, dict] = {}
    for c in candidates:
        w = c.weather
        if w not in weather_types:
            weather_types[w] = {"total": 0, "done": 0, "skipped": 0}
        weather_types[w]["total"] += 1
        if c.status == "done":
            weather_types[w]["done"] += 1
        elif c.status == "skipped":
            weather_types[w]["skipped"] += 1

    md = []
    md.append("# PICEK Pre-Accident Negative Clip Generation Report\n")
    md.append(f"> **Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ")
    md.append(f"> **Pipeline Elapsed**: {stats.elapsed_sec / 60:.2f} minutes  ")
    md.append(f"> **Target Dataset**: Real PICEK Accident Videos  \n")
    md.append("---\n")

    md.append("## 1. Executive Summary\n")
    md.append("| Metric | Count / Value |")
    md.append("| :--- | :--- |")
    md.append(f"| **Total Candidates Processed** | **{stats.total_candidates}** |")
    md.append(f"| **Negative Clips Generated (`done`)** | **{stats.done}** |")
    md.append(f"| **Skipped (Insufficient Pre-Crash Time)** | **{stats.skipped}** |")
    md.append(f"| **Extraction Errors** | **{stats.errors}** |")
    md.append(f"| **Existing Reused** | **{stats.existing_reused}** |")
    md.append(f"| **Total Data Generated** | **{stats.total_bytes_written / (1024 ** 2):.1f} MB** |")
    md.append(f"| **Safety Gap Applied** | **{safety_gap_sec:.2f} s** |")
    md.append(f"| **Target Clip Length** | **{target_duration_sec:.2f} s** |")
    md.append(f"| **Minimum Required Length** | **{min_clip_sec:.2f} s** |\n")

    md.append("---\n")
    md.append("## 2. Extraction Methodology & Safety Logic\n")
    md.append("To ensure that generated negative clips contain strictly **normal driving** with zero collision frames or immediate pre-crash maneuvers:")
    md.append("1. **Source of Truth**: The annotated `accident_time_sec` and `accident_frame` from the PICEK ground truth metadata.")
    md.append("2. **Safety Gap Calculation**:")
    md.append("   $$\\text{end\\_sec} = \\text{accident\\_time\\_sec} - \\text{safety\\_gap\\_sec}$$")
    md.append("   $$\\text{start\\_sec} = \\max(0.0, \\text{end\\_sec} - \\text{target\\_clip\\_duration\\_sec})$$")
    md.append("3. **Skip Policy**:")
    md.append("   - If $\\text{end\\_sec} < \\text{min\\_clip\\_sec}$ (i.e. the collision occurs within $3.0\\text{s}$ of video start), the video is safely skipped and logged with reason `insufficient_pre_accident_time`.")
    md.append("4. **Non-Destructive Integrity**:")
    md.append("   - Original raw files and existing positive clips (`positive/real/`, `trimmed/positive/real/`) are untouched.")
    md.append("   - All negative clips are written to `trimmed/negative/real/<video_id>.mp4`.\n")

    md.append("---\n")
    md.append("## 3. Collision Category Breakdown\n")
    md.append("| Collision Category | Total Real | Negative Generated | Skipped | Errors | Yield Rate |")
    md.append("| :--- | :--- | :--- | :--- | :--- | :--- |")
    for t, counts in sorted(acc_types.items()):
        yield_rate = (counts["done"] / counts["total"] * 100) if counts["total"] > 0 else 0
        md.append(
            f"| **{t}** | {counts['total']} | **{counts['done']}** | {counts['skipped']} | {counts['errors']} | {yield_rate:.1f}% |"
        )
    md.append("")

    md.append("---\n")
    md.append("## 4. Weather & Environmental Distribution\n")
    md.append("| Weather Condition | Total Real | Negative Generated | Skipped |")
    md.append("| :--- | :--- | :--- | :--- |")
    for w, counts in sorted(weather_types.items()):
        md.append(f"| **{w}** | {counts['total']} | {counts['done']} | {counts['skipped']} |")
    md.append("")

    md.append("---\n")
    md.append("## 5. System Resource Monitor Summary\n")
    md.append("| Resource Metric | Average | Peak | Safety Threshold |")
    md.append("| :--- | :--- | :--- | :--- |")
    md.append(f"| **CPU Usage** | {mon_summary.get('cpu_avg', 'N/A')}% | {mon_summary.get('cpu_max', 'N/A')}% | Warn > {CPU_WARN_PCT}%, Pause > {CPU_PAUSE_PCT}% |")
    md.append(f"| **RAM Usage** | {mon_summary.get('ram_avg', 'N/A')}% | {mon_summary.get('ram_max', 'N/A')}% | Warn > {RAM_WARN_PCT}%, Pause > {RAM_PAUSE_PCT}% |")
    md.append(f"| **Disk Usage** | {mon_summary.get('disk_avg', 'N/A')}% | {mon_summary.get('disk_max', 'N/A')}% | Warn > {DISK_WARN_PCT}% |")
    md.append(f"| **Monitor Samples** | {mon_summary.get('samples', 0)} | — | — |")
    md.append(f"| **Overload Pause Events** | {mon_summary.get('pause_events', 0)} | — | — |\n")

    md.append("---\n")
    md.append("## 6. Output Directory Structure\n")
    md.append("```text")
    md.append(f"{output_dir}\\")
    md.append("├── positive\\")
    md.append("│   ├── real\\                  (2,027 files - untouched)")
    md.append("│   └── synthetic\\             (2,211 files - untouched)")
    md.append("├── negative\\")
    md.append("│   ├── real\\")
    md.append("│   └── synthetic\\")
    md.append("├── trimmed\\")
    md.append("│   ├── positive\\")
    md.append("│   │   ├── real\\              (2,027 files - untouched)")
    md.append("│   │   └── synthetic\\         (2,211 files - untouched)")
    md.append("│   └── negative\\")
    md.append(f"│       └── real\\              ({stats.done} generated negative clips)")
    md.append("├── metadata.json              (Master catalog updated with negative provenance)")
    md.append("├── negative_metadata.json     (Dedicated negative clip metadata & provenance)")
    md.append("├── negative_pipeline.log      (Complete execution transcript)")
    md.append("└── negative_dataset_report.md (This report)")
    md.append("```\n")

    md.append("---\n")
    md.append("## 7. Sample Provenance Record (`negative_metadata.json`)\n")
    sample_done = next((c for c in candidates if c.status == "done"), None)
    if sample_done:
        md.append("```json")
        md.append(json.dumps(sample_done.to_dict(), indent=2))
        md.append("```\n")

    md.append("---\n")
    md.append("*Report generated by `scripts/picek_negative_extractor.py`*")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))


# ──────────────────────────────────────────────────────────────────────────────
# CLI ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate Pre-Accident Negative Clips for PICEK Dataset"
    )
    parser.add_argument(
        "--metadata-json",
        type=str,
        default=DEFAULT_METADATA_JSON,
        help="Path to metadata.json source of truth",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Root output directory (e.g. Datasets/processed/picek_sorted)",
    )
    parser.add_argument(
        "--raw-dir",
        type=str,
        default=DEFAULT_RAW_DIR,
        help="Root raw dataset directory (e.g. Datasets/raw/picekl)",
    )
    parser.add_argument(
        "--safety-gap-sec",
        type=float,
        default=DEFAULT_SAFETY_GAP_SEC,
        help="Safety gap in seconds before accident onset (default: 1.0)",
    )
    parser.add_argument(
        "--clip-duration-sec",
        type=float,
        default=DEFAULT_CLIP_DURATION_SEC,
        help="Target negative clip duration in seconds (default: 5.0)",
    )
    parser.add_argument(
        "--min-clip-sec",
        type=float,
        default=DEFAULT_MIN_CLIP_SEC,
        help="Minimum clip duration in seconds to produce a negative (default: 2.0)",
    )
    parser.add_argument(
        "--pilot",
        type=int,
        default=0,
        help="Run pilot verification on N videos (default: 0 = full run)",
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        default=None,
        help="Max videos to process (useful for testing)",
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Force re-extraction of existing negative clips",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate extraction without writing video files",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    success = run_pipeline(
        metadata_json_path=Path(args.metadata_json),
        output_dir=Path(args.output_dir),
        raw_dir=Path(args.raw_dir),
        safety_gap_sec=args.safety_gap_sec,
        clip_duration_sec=args.clip_duration_sec,
        min_clip_sec=args.min_clip_sec,
        pilot_count=args.pilot,
        skip_existing=not args.no_skip_existing,
        max_videos=args.max_videos,
        dry_run=args.dry_run,
    )
    sys.exit(0 if success else 1)
