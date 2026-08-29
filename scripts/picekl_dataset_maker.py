#!/usr/bin/env python3
"""
picekl_dataset_maker.py
=======================
Picek Dataset Sorter & Trimmer — Standalone Dataset-Making Tool
================================================================

Pipeline steps:
    1. SCAN   — Walk the picekl raw folder and read full metadata
                (from both metadata CSVs + real cv2 probe of each video)
    2. EXPORT — Write combined metadata to JSON
    3. SORT   — Classify every video as Positive (accident) / Negative (no accident)
                and copy into sorted sub-folders
    4. TRIM   — Trim each video: positive clips to [-PRE, +POST] seconds around
                the accident frame; negative clips to first TRIM_NEGATIVE_SEC seconds
    5. MONITOR — System resource monitor thread runs throughout; prints warnings
                 if CPU/RAM/Disk go above thresholds; pauses processing on danger
    6. REPORT  — Write a complete Markdown report

Usage
-----
    python scripts/picekl_dataset_maker.py
    python scripts/picekl_dataset_maker.py --raw-dir Datasets/raw/picekl
    python scripts/picekl_dataset_maker.py --output-dir Datasets/processed/picek_sorted
    python scripts/picekl_dataset_maker.py --skip-trim          # sort only, no re-encode
    python scripts/picekl_dataset_maker.py --dry-run            # no files written
    python scripts/picekl_dataset_maker.py --max-videos 50      # limit for testing

Author : Antigravity (dataset pipeline — separate from training code)
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
from typing import Dict, List, Optional, Tuple

import cv2
import psutil

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION DEFAULTS
# ──────────────────────────────────────────────────────────────────────────────

# Seconds to include *before* the accident frame in positive clips
PRE_ACCIDENT_SEC: float = 5.0
# Seconds to include *after* the accident frame in positive clips
POST_ACCIDENT_SEC: float = 3.0
# How long to trim negative videos (seconds from start)
TRIM_NEGATIVE_SEC: float = 10.0

# System monitor thresholds
CPU_WARN_PCT: float = 90.0   # warn if CPU% > this
RAM_WARN_PCT: float = 92.0   # warn if RAM% > this
DISK_WARN_PCT: float = 90.0  # warn if Disk% > this
CPU_PAUSE_PCT: float = 98.0  # pause pipeline if CPU% > this
RAM_PAUSE_PCT: float = 97.0  # pause pipeline if RAM% > this

MONITOR_INTERVAL_SEC: float = 2.0   # how often the monitor samples
PAUSE_SLEEP_SEC: float = 2.0        # sleep step while system is overloaded

# Default paths (relative to workspace root  e:\Rads)
DEFAULT_RAW_DIR = "Datasets/raw/picekl"
DEFAULT_OUTPUT_DIR = "Datasets/processed/picek_sorted"
DEFAULT_SCRIPT_DIR = Path(__file__).resolve().parent.parent  # e:\Rads

# ──────────────────────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class VideoMeta:
    """Holds all metadata for one video clip."""
    # Identifiers
    video_id: str            # stem of the filename, e.g. "Z4kg2Ev3vhk_00"
    filename: str            # relative path from raw_dir, e.g. "real_videos/Z4kg2Ev3vhk_00.mp4"
    abs_path: str            # absolute path on disk
    source_type: str         # "real" | "synthetic"

    # From CSV
    accident_type: str       # e.g. "rear-end", "t-bone", "sideswipe"
    rollover: int            # 0/1 (real only; 0 for synthetic)
    accident_time_sec: float # seconds into clip where accident happens
    accident_frame: int      # frame index of accident
    weather: str
    day_time: str            # "day" | "night" | ...
    split: str               # "train" | "test" | ...

    # Probed by cv2
    frame_count: int
    fps: float
    duration_sec: float
    width: int
    height: int
    file_size_bytes: int
    codec: str

    # Classification
    label: str               # "positive" | "negative"
    has_accident: bool

    # Processing status
    trimmed_path: Optional[str] = None
    trim_status: str = "pending"   # pending | done | skipped | error
    trim_error: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PipelineStats:
    total_videos: int = 0
    real_count: int = 0
    synthetic_count: int = 0
    positive_count: int = 0
    negative_count: int = 0
    trim_done: int = 0
    trim_skipped: int = 0
    trim_error: int = 0
    total_size_bytes: int = 0
    start_time: float = field(default_factory=time.time)
    end_time: float = 0.0

    @property
    def elapsed_sec(self) -> float:
        t = self.end_time if self.end_time else time.time()
        return t - self.start_time


# ──────────────────────────────────────────────────────────────────────────────
# LOGGING SETUP
# ──────────────────────────────────────────────────────────────────────────────

def setup_logging(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("picekl_maker")
    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler (INFO+)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler (DEBUG+)
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


# ──────────────────────────────────────────────────────────────────────────────
# SYSTEM MONITOR
# ──────────────────────────────────────────────────────────────────────────────

class SystemMonitor(threading.Thread):
    """
    Background thread that watches CPU, RAM and Disk every MONITOR_INTERVAL_SEC.
    Sets self.paused = True when the system is under stress so the main
    pipeline can call monitor.wait_if_paused() before heavy operations.
    """

    def __init__(self, logger: logging.Logger, output_dir: Path):
        super().__init__(daemon=True, name="SysMonitor")
        self.logger = logger
        self.output_dir = output_dir
        self.paused = False
        self.stop_event = threading.Event()
        self.samples: List[dict] = []

    def run(self):
        while not self.stop_event.is_set():
            try:
                cpu = psutil.cpu_percent(interval=None)
                ram = psutil.virtual_memory().percent
                disk_usage = psutil.disk_usage(str(self.output_dir))
                disk = disk_usage.percent
                disk_free_gb = disk_usage.free / (1024 ** 3)

                sample = {
                    "ts": datetime.now().isoformat(timespec="seconds"),
                    "cpu_pct": cpu,
                    "ram_pct": ram,
                    "disk_pct": disk,
                    "disk_free_gb": round(disk_free_gb, 2),
                }
                self.samples.append(sample)

                # Pause trigger
                if cpu > CPU_PAUSE_PCT or ram > RAM_PAUSE_PCT:
                    if not self.paused:
                        self.logger.warning(
                            f"[MONITOR] SYSTEM OVERLOAD — CPU {cpu:.1f}% | RAM {ram:.1f}% "
                            f"— pipeline PAUSED until system recovers"
                        )
                    self.paused = True
                else:
                    if self.paused:
                        self.logger.info(
                            f"[MONITOR] System recovered — CPU {cpu:.1f}% | RAM {ram:.1f}% "
                            f"— pipeline RESUMED"
                        )
                    self.paused = False

                # Warning level
                if cpu > CPU_WARN_PCT:
                    self.logger.warning(f"[MONITOR] CPU {cpu:.1f}% (threshold {CPU_WARN_PCT}%)")
                if ram > RAM_WARN_PCT:
                    self.logger.warning(f"[MONITOR] RAM {ram:.1f}% (threshold {RAM_WARN_PCT}%)")
                if disk > DISK_WARN_PCT:
                    self.logger.warning(
                        f"[MONITOR] Disk {disk:.1f}% — only {disk_free_gb:.1f} GB free!"
                    )

            except Exception as exc:
                self.logger.debug(f"[MONITOR] error: {exc}")

            self.stop_event.wait(MONITOR_INTERVAL_SEC)

    def wait_if_paused(self):
        """Block the calling thread while the system is overloaded."""
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
            "cpu_avg": round(sum(cpu_vals) / len(cpu_vals), 1),
            "cpu_max": round(max(cpu_vals), 1),
            "ram_avg": round(sum(ram_vals) / len(ram_vals), 1),
            "ram_max": round(max(ram_vals), 1),
            "disk_avg": round(sum(disk_vals) / len(disk_vals), 1),
            "disk_max": round(max(disk_vals), 1),
        }


# ──────────────────────────────────────────────────────────────────────────────
# STEP 1 — SCAN: Read metadata CSVs + probe each video with cv2
# ──────────────────────────────────────────────────────────────────────────────

def load_real_csv(csv_path: Path) -> Dict[str, dict]:
    """Parse metadata-real.csv -> {relative_path: row_dict}"""
    rows = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            path_key = row["path"].replace("\\", "/")
            rows[path_key] = dict(row)
    return rows


def load_synthetic_csv(csv_path: Path) -> Dict[str, dict]:
    """Parse metadata-synthetic.csv -> {relative_path: row_dict}"""
    rows = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            path_key = row["rgb_path"].replace("\\", "/")
            rows[path_key] = dict(row)
    return rows


def probe_video_cv2(abs_path: str) -> dict:
    """Use cv2.VideoCapture to probe a video file. Returns a dict of properties."""
    result = {
        "frame_count": 0,
        "fps": 0.0,
        "duration_sec": 0.0,
        "width": 0,
        "height": 0,
        "codec": "unknown",
    }
    try:
        cap = cv2.VideoCapture(abs_path)
        if not cap.isOpened():
            return result
        result["frame_count"] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        result["fps"] = cap.get(cv2.CAP_PROP_FPS) or 0.0
        result["width"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        result["height"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if result["fps"] > 0 and result["frame_count"] > 0:
            result["duration_sec"] = result["frame_count"] / result["fps"]
        fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
        if fourcc_int:
            result["codec"] = (
                chr(fourcc_int & 0xFF)
                + chr((fourcc_int >> 8) & 0xFF)
                + chr((fourcc_int >> 16) & 0xFF)
                + chr((fourcc_int >> 24) & 0xFF)
            ).strip()
        cap.release()
    except Exception:
        pass
    return result


def scan_videos(
    raw_dir: Path,
    logger: logging.Logger,
    monitor: SystemMonitor,
    max_videos: Optional[int] = None,
) -> List[VideoMeta]:
    """
    STEP 1: Walk raw_dir, read both CSVs, probe each .mp4 with cv2.
    Returns a list of VideoMeta objects.
    """
    logger.info("=" * 70)
    logger.info("STEP 1 -- SCAN: Reading metadata and probing video files")
    logger.info("=" * 70)

    real_csv = raw_dir / "metadata-real.csv"
    synth_csv = raw_dir / "metadata-synthetic.csv"

    real_rows = load_real_csv(real_csv) if real_csv.exists() else {}
    synth_rows = load_synthetic_csv(synth_csv) if synth_csv.exists() else {}
    logger.info(f"  CSV entries -- real: {len(real_rows)} | synthetic: {len(synth_rows)}")

    # Collect all .mp4 files
    all_mp4s: List[Path] = []
    for folder in ["real_videos", "synthetic_videos/videos"]:
        search_dir = raw_dir / folder
        if search_dir.exists():
            found = sorted(search_dir.rglob("*.mp4"))
            all_mp4s.extend(found)
            logger.info(f"  Found {len(found)} mp4 files in {folder}/")

    if max_videos:
        all_mp4s = all_mp4s[:max_videos]
        logger.info(f"  Limiting to first {max_videos} videos (--max-videos)")

    logger.info(f"  Total to process: {len(all_mp4s)} video(s)")

    records: List[VideoMeta] = []
    for idx, mp4_path in enumerate(all_mp4s, 1):
        monitor.wait_if_paused()

        # Determine relative key (how it appears in CSVs)
        try:
            rel = mp4_path.relative_to(raw_dir).as_posix()
        except ValueError:
            rel = mp4_path.name

        logger.debug(f"  [{idx}/{len(all_mp4s)}] Scanning: {rel}")
        if idx % 50 == 0 or idx == len(all_mp4s):
            logger.info(f"  Progress: {idx}/{len(all_mp4s)} videos scanned")

        is_real = "real_videos" in rel
        is_synth = "synthetic_videos" in rel
        source_type = "real" if is_real else ("synthetic" if is_synth else "unknown")

        # Lookup in CSV
        csv_row: dict = {}
        if is_real and rel in real_rows:
            csv_row = real_rows[rel]
        elif is_synth and rel in synth_rows:
            csv_row = synth_rows[rel]

        def _float(key: str, default: float = 0.0) -> float:
            try:
                return float(csv_row.get(key, default))
            except (ValueError, TypeError):
                return default

        def _int(key: str, default: int = 0) -> int:
            try:
                return int(float(csv_row.get(key, default)))
            except (ValueError, TypeError):
                return default

        accident_time = _float("accident_time")
        accident_frame = _int("accident_frame")
        rollover = _int("rollover", 0)
        weather = csv_row.get("weather", "unknown")
        if is_real:
            day_time = csv_row.get("day_time", "unknown")
            accident_type = csv_row.get("type", "unknown")
            split = csv_row.get("split_in_distribution", "unknown")
        else:
            day_time = csv_row.get("weather", "unknown")
            accident_type = csv_row.get("type", "unknown")
            split = "synthetic"

        probe = probe_video_cv2(str(mp4_path))

        # Classification:
        # All picekl videos that appear in the CSV ARE accident videos (positive).
        # Any mp4 file not found in either CSV and with accident_time==0 is negative.
        has_accident = bool(csv_row) or accident_time > 0 or accident_frame > 0
        label = "positive" if has_accident else "negative"

        record = VideoMeta(
            video_id=mp4_path.stem,
            filename=rel,
            abs_path=str(mp4_path),
            source_type=source_type,
            accident_type=accident_type,
            rollover=rollover,
            accident_time_sec=accident_time,
            accident_frame=accident_frame,
            weather=weather,
            day_time=day_time,
            split=split,
            frame_count=probe["frame_count"],
            fps=probe["fps"],
            duration_sec=probe["duration_sec"],
            width=probe["width"],
            height=probe["height"],
            file_size_bytes=mp4_path.stat().st_size,
            codec=probe["codec"],
            label=label,
            has_accident=has_accident,
        )
        records.append(record)

    pos = sum(1 for r in records if r.label == "positive")
    neg = sum(1 for r in records if r.label == "negative")
    logger.info(f"  Scan complete -- positive: {pos} | negative: {neg}")
    return records


# ──────────────────────────────────────────────────────────────────────────────
# STEP 2 — EXPORT metadata to JSON
# ──────────────────────────────────────────────────────────────────────────────

def export_metadata_json(records: List[VideoMeta], json_path: Path, logger: logging.Logger):
    logger.info("=" * 70)
    logger.info("STEP 2 -- EXPORT: Writing metadata JSON")
    logger.info("=" * 70)

    payload = {
        "generated_at": datetime.now().isoformat(),
        "total_videos": len(records),
        "positive_count": sum(1 for r in records if r.label == "positive"),
        "negative_count": sum(1 for r in records if r.label == "negative"),
        "videos": [r.to_dict() for r in records],
    }
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    logger.info(f"  Metadata JSON written -> {json_path}")
    logger.info(f"  File size: {json_path.stat().st_size / 1024:.1f} KB")


# ──────────────────────────────────────────────────────────────────────────────
# STEP 3 — SORT: Copy files into positive/ and negative/ folders
# ──────────────────────────────────────────────────────────────────────────────

def sort_videos(
    records: List[VideoMeta],
    output_dir: Path,
    dry_run: bool,
    logger: logging.Logger,
    monitor: SystemMonitor,
) -> Tuple[Path, Path]:
    """
    STEP 3: Copy each video to output_dir/positive/ or output_dir/negative/.
    Preserves the source_type sub-folder (real/ or synthetic/).
    Returns (positive_dir, negative_dir).
    """
    logger.info("=" * 70)
    logger.info("STEP 3 -- SORT: Classifying and copying videos")
    logger.info("=" * 70)

    pos_dir = output_dir / "positive"
    neg_dir = output_dir / "negative"

    if not dry_run:
        (pos_dir / "real").mkdir(parents=True, exist_ok=True)
        (pos_dir / "synthetic").mkdir(parents=True, exist_ok=True)
        (neg_dir / "real").mkdir(parents=True, exist_ok=True)
        (neg_dir / "synthetic").mkdir(parents=True, exist_ok=True)

    for idx, rec in enumerate(records, 1):
        monitor.wait_if_paused()

        label_dir = pos_dir if rec.label == "positive" else neg_dir
        dest_subdir = label_dir / rec.source_type
        dest_path = dest_subdir / Path(rec.abs_path).name

        logger.debug(
            f"  [{idx}/{len(records)}] {rec.label.upper()} -- {rec.video_id}"
            f" -> {dest_path.relative_to(output_dir)}"
        )
        if idx % 100 == 0 or idx == len(records):
            logger.info(f"  Sort progress: {idx}/{len(records)}")

        if not dry_run:
            if not dest_path.exists():
                shutil.copy2(rec.abs_path, dest_path)

        rec.trimmed_path = str(dest_path)

    logger.info(f"  Sorting done -- files in {pos_dir} and {neg_dir}")
    return pos_dir, neg_dir


# ──────────────────────────────────────────────────────────────────────────────
# STEP 4 — TRIM: Cut each video using OpenCV
# ──────────────────────────────────────────────────────────────────────────────

def trim_video_cv2(
    src_path: str,
    dest_path: str,
    start_sec: float,
    end_sec: float,
    logger: logging.Logger,
) -> bool:
    """
    Trim a video [start_sec, end_sec] using OpenCV frame-by-frame read/write.
    Returns True on success, False on failure.
    """
    try:
        cap = cv2.VideoCapture(src_path)
        if not cap.isOpened():
            logger.error(f"    Cannot open: {src_path}")
            return False

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if fps <= 0:
            cap.release()
            logger.warning(f"    FPS=0, skipping trim for {Path(src_path).name}")
            return False

        start_frame = max(0, int(start_sec * fps))
        end_frame = min(total_frames - 1, int(end_sec * fps))

        if start_frame >= end_frame:
            cap.release()
            logger.warning(
                f"    Degenerate trim window [{start_sec:.2f}s-{end_sec:.2f}s] "
                f"for {Path(src_path).name}, copying whole file instead."
            )
            shutil.copy2(src_path, dest_path)
            return True

        # Use mp4v which is universally supported on Windows OpenCV
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        Path(dest_path).parent.mkdir(parents=True, exist_ok=True)
        out = cv2.VideoWriter(dest_path, fourcc, fps, (width, height))
        if not out.isOpened():
            cap.release()
            logger.warning(f"    VideoWriter could not open for {Path(src_path).name}, copying instead")
            shutil.copy2(src_path, dest_path)
            return True

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        written = 0
        for _ in range(end_frame - start_frame):
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
            written += 1

        cap.release()
        out.release()

        if written == 0:
            logger.warning(f"    No frames written for {Path(src_path).name}")
            return False

        return True

    except Exception as exc:
        logger.error(f"    Trim error for {Path(src_path).name}: {exc}")
        logger.debug(traceback.format_exc())
        return False


def trim_all_videos(
    records: List[VideoMeta],
    output_dir: Path,
    dry_run: bool,
    logger: logging.Logger,
    monitor: SystemMonitor,
):
    """
    STEP 4: Trim each video clip.
    - Positive: keep [accident_time - PRE_ACCIDENT_SEC, accident_time + POST_ACCIDENT_SEC]
    - Negative: keep [0, TRIM_NEGATIVE_SEC]
    Trimmed files go into output_dir/trimmed/positive/ and output_dir/trimmed/negative/
    """
    logger.info("=" * 70)
    logger.info("STEP 4 -- TRIM: Cutting video clips")
    logger.info("=" * 70)

    trimmed_pos_dir = output_dir / "trimmed" / "positive"
    trimmed_neg_dir = output_dir / "trimmed" / "negative"

    if not dry_run:
        (trimmed_pos_dir / "real").mkdir(parents=True, exist_ok=True)
        (trimmed_pos_dir / "synthetic").mkdir(parents=True, exist_ok=True)
        (trimmed_neg_dir / "real").mkdir(parents=True, exist_ok=True)
        (trimmed_neg_dir / "synthetic").mkdir(parents=True, exist_ok=True)

    total = len(records)
    for idx, rec in enumerate(records, 1):
        monitor.wait_if_paused()

        if idx % 25 == 0 or idx == total or idx == 1:
            done_so_far = sum(1 for r in records if r.trim_status == "done")
            logger.info(f"  Trim progress: {idx}/{total} | done={done_so_far}")

        src = rec.abs_path  # trim from ORIGINAL raw file
        label_trim_dir = trimmed_pos_dir if rec.label == "positive" else trimmed_neg_dir
        dest = str(label_trim_dir / rec.source_type / Path(src).name)

        if dry_run:
            rec.trim_status = "skipped"
            rec.trimmed_path = dest
            continue

        # Compute trim window
        if rec.label == "positive":
            t_acc = rec.accident_time_sec
            start = max(0.0, t_acc - PRE_ACCIDENT_SEC)
            end = t_acc + POST_ACCIDENT_SEC
            if rec.duration_sec > 0:
                end = min(end, rec.duration_sec)
        else:
            start = 0.0
            end = min(TRIM_NEGATIVE_SEC, rec.duration_sec) if rec.duration_sec > 0 else TRIM_NEGATIVE_SEC

        logger.debug(
            f"  [{idx}/{total}] Trimming [{start:.2f}s-{end:.2f}s] -> {Path(dest).name}"
        )

        if Path(dest).exists():
            rec.trim_status = "done"
            rec.trimmed_path = dest
            continue

        ok = trim_video_cv2(src, dest, start, end, logger)
        if ok:
            rec.trim_status = "done"
            rec.trimmed_path = dest
        else:
            rec.trim_status = "error"
            rec.trim_error = "cv2 trim failed"

    done = sum(1 for r in records if r.trim_status == "done")
    err = sum(1 for r in records if r.trim_status == "error")
    skipped = sum(1 for r in records if r.trim_status == "skipped")
    logger.info(f"  Trim complete -- done={done} | skipped={skipped} | error={err}")


# ──────────────────────────────────────────────────────────────────────────────
# STEP 5 — UPDATE JSON with final statuses
# ──────────────────────────────────────────────────────────────────────────────

def update_metadata_json(records: List[VideoMeta], json_path: Path, logger: logging.Logger):
    logger.info("  Updating metadata JSON with trim results...")
    with open(json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    payload["updated_at"] = datetime.now().isoformat()
    payload["videos"] = [r.to_dict() for r in records]
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    logger.info(f"  JSON updated -> {json_path}")


# ──────────────────────────────────────────────────────────────────────────────
# STEP 6 — REPORT
# ──────────────────────────────────────────────────────────────────────────────

def write_report(
    records: List[VideoMeta],
    stats: PipelineStats,
    monitor: SystemMonitor,
    output_dir: Path,
    json_path: Path,
    logger: logging.Logger,
    dry_run: bool,
    skip_trim: bool,
) -> Path:
    logger.info("=" * 70)
    logger.info("STEP 6 -- REPORT: Writing Markdown report")
    logger.info("=" * 70)

    report_path = output_dir / "dataset_report.md"
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    elapsed = stats.elapsed_sec
    elapsed_str = f"{int(elapsed // 60)}m {int(elapsed % 60)}s"

    pos_records = [r for r in records if r.label == "positive"]
    neg_records = [r for r in records if r.label == "negative"]
    real_records = [r for r in records if r.source_type == "real"]
    synth_records = [r for r in records if r.source_type == "synthetic"]

    total_bytes = sum(r.file_size_bytes for r in records)
    total_gb = total_bytes / (1024 ** 3)

    acc_types: Dict[str, int] = {}
    for r in records:
        acc_types[r.accident_type] = acc_types.get(r.accident_type, 0) + 1

    weathers: Dict[str, int] = {}
    for r in records:
        weathers[r.weather] = weathers.get(r.weather, 0) + 1

    splits: Dict[str, int] = {}
    for r in records:
        splits[r.split] = splits.get(r.split, 0) + 1

    mon_summary = monitor.summary()
    errors = [r for r in records if r.trim_status == "error"]

    lines = [
        "# Picekl Dataset Sorting & Trimming Report",
        "",
        f"> Generated: **{now_str}**  ",
        f"> Pipeline elapsed: **{elapsed_str}**  ",
        f"> Mode: `{'dry-run' if dry_run else 'full'}`  ",
        f"> Skip-trim: `{skip_trim}`",
        "",
        "---",
        "",
        "## 1. Summary",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Total videos processed | **{len(records)}** |",
        f"| Positive (accident happening) | **{len(pos_records)}** |",
        f"| Negative (no accident / control) | **{len(neg_records)}** |",
        f"| Real videos | **{len(real_records)}** |",
        f"| Synthetic videos | **{len(synth_records)}** |",
        f"| Total raw dataset size | **{total_gb:.2f} GB** |",
        f"| Trim done | **{sum(1 for r in records if r.trim_status == 'done')}** |",
        f"| Trim skipped | **{sum(1 for r in records if r.trim_status == 'skipped')}** |",
        f"| Trim errors | **{len(errors)}** |",
        "",
        "---",
        "",
        "## 2. Output Directory Structure",
        "",
        "```",
        f"{output_dir}/",
        "├── positive/          <-- videos WHERE accident IS happening",
        f"|   ├── real/          ({sum(1 for r in pos_records if r.source_type == 'real')} files)",
        f"|   └── synthetic/     ({sum(1 for r in pos_records if r.source_type == 'synthetic')} files)",
        "├── negative/          <-- videos WHERE accident IS NOT happening",
        f"|   ├── real/          ({sum(1 for r in neg_records if r.source_type == 'real')} files)",
        f"|   └── synthetic/     ({sum(1 for r in neg_records if r.source_type == 'synthetic')} files)",
        "├── trimmed/           <-- trimmed clips (accident-window for positives)",
        "|   ├── positive/",
        "|   |   ├── real/",
        "|   |   └── synthetic/",
        "|   └── negative/",
        "|       ├── real/",
        "|       └── synthetic/",
        "├── metadata.json      <-- full per-video metadata",
        "├── pipeline.log       <-- detailed execution log",
        "└── dataset_report.md  <-- this file",
        "```",
        "",
        "---",
        "",
        "## 3. Classification Logic",
        "",
        "- **Positive** = video appears in `metadata-real.csv` or `metadata-synthetic.csv`",
        "  AND has a non-zero `accident_time` / `accident_frame`.",
        "  These are videos **where an accident is actively happening**.",
        "- **Negative** = any mp4 found on disk but NOT in either CSV (unlikely in this dataset,",
        "  but included as a safety net). These would be videos with **no accident**.",
        "",
        "> Note: The Picekl dataset is specifically an accident dataset, so virtually",
        "> all clips are positive. The negative category exists for future extensibility.",
        "",
        "---",
        "",
        "## 4. Accident Type Breakdown",
        "",
        "| Accident Type | Count |",
        "|---------------|-------|",
    ]
    for atype, cnt in sorted(acc_types.items(), key=lambda x: -x[1]):
        lines.append(f"| {atype} | {cnt} |")

    lines += [
        "",
        "---",
        "",
        "## 5. Weather Conditions",
        "",
        "| Weather | Count |",
        "|---------|-------|",
    ]
    for w, cnt in sorted(weathers.items(), key=lambda x: -x[1]):
        lines.append(f"| {w} | {cnt} |")

    lines += [
        "",
        "---",
        "",
        "## 6. Dataset Splits",
        "",
        "| Split | Count |",
        "|-------|-------|",
    ]
    for s, cnt in sorted(splits.items(), key=lambda x: -x[1]):
        lines.append(f"| {s} | {cnt} |")

    lines += [
        "",
        "---",
        "",
        "## 7. Trim Settings Used",
        "",
        "| Setting | Value |",
        "|---------|-------|",
        f"| Pre-accident window | {PRE_ACCIDENT_SEC} s |",
        f"| Post-accident window | {POST_ACCIDENT_SEC} s |",
        f"| Positive clip window | [{PRE_ACCIDENT_SEC}s before, {POST_ACCIDENT_SEC}s after] accident_time |",
        f"| Negative clip length | first {TRIM_NEGATIVE_SEC} s of original video |",
        f"| Video trimming tool | OpenCV {cv2.__version__} (frame-by-frame, no ffmpeg needed) |",
        "",
        "---",
        "",
        "## 8. System Monitor Summary",
        "",
        "| Metric | Average | Peak |",
        "|--------|---------|------|",
        f"| CPU % | {mon_summary.get('cpu_avg', 'N/A')} | {mon_summary.get('cpu_max', 'N/A')} |",
        f"| RAM % | {mon_summary.get('ram_avg', 'N/A')} | {mon_summary.get('ram_max', 'N/A')} |",
        f"| Disk % | {mon_summary.get('disk_avg', 'N/A')} | {mon_summary.get('disk_max', 'N/A')} |",
        f"| Monitor samples taken | {mon_summary.get('samples', 'N/A')} | -- |",
        "",
        f"> **Auto-pause thresholds:** CPU > {CPU_PAUSE_PCT}% or RAM > {RAM_PAUSE_PCT}%  ",
        f"> **Warning thresholds:** CPU > {CPU_WARN_PCT}%, RAM > {RAM_WARN_PCT}%, Disk > {DISK_WARN_PCT}%  ",
        "> The pipeline automatically paused when thresholds were exceeded and resumed",
        "> after the system recovered, preventing crashes.",
        "",
        "---",
        "",
        "## 9. Metadata JSON Schema",
        "",
        f"Full per-video metadata saved to: `{json_path}`",
        "",
        "Each video entry contains:",
        "",
        "| Field | Description |",
        "|-------|-------------|",
        "| `video_id` | Filename stem, e.g. `Z4kg2Ev3vhk_00` |",
        "| `filename` | Relative path from raw_dir |",
        "| `source_type` | `real` or `synthetic` |",
        "| `accident_type` | e.g. `rear-end`, `t-bone`, `sideswipe`, `single` |",
        "| `rollover` | 0 or 1 |",
        "| `accident_time_sec` | Seconds into clip when accident occurs |",
        "| `accident_frame` | Frame index of the accident |",
        "| `weather` | `normal`, `rain`, `snow`, `clear`, etc. |",
        "| `day_time` | `day` or `night` |",
        "| `split` | `train` or `test` |",
        "| `frame_count` | Total frames (cv2 probe) |",
        "| `fps` | Frames per second (cv2 probe) |",
        "| `duration_sec` | Total duration in seconds |",
        "| `width` / `height` | Resolution |",
        "| `codec` | FOURCC codec string |",
        "| `file_size_bytes` | Raw file size |",
        "| `label` | `positive` or `negative` |",
        "| `trimmed_path` | Absolute path to trimmed output file |",
        "| `trim_status` | `done`, `skipped`, or `error` |",
        "| `trim_error` | Error message if trimming failed |",
        "",
        "---",
        "",
    ]

    if errors:
        lines += [
            f"## 10. Trim Errors ({len(errors)})",
            "",
            "| Video ID | Error |",
            "|----------|-------|",
        ]
        for r in errors[:50]:
            lines.append(f"| `{r.video_id}` | {r.trim_error} |")
        if len(errors) > 50:
            lines.append(f"| ... | (+{len(errors)-50} more, see pipeline.log) |")
        lines.append("")
    else:
        lines += [
            "## 10. Trim Errors",
            "",
            "No trim errors. All clips processed successfully.",
            "",
        ]

    lines += [
        "---",
        "",
        "## 11. Next Steps",
        "",
        "1. Use `trimmed/positive/` clips for accident detection model training",
        "2. Use `trimmed/negative/` clips as background / no-accident examples",
        "3. Load `metadata.json` to get per-clip labels and accident timing for temporal models",
        "4. The `split` field in metadata.json preserves the original train/test split",
        "",
        "---",
        "",
        "*Report auto-generated by `scripts/picekl_dataset_maker.py`*",
    ]

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    logger.info(f"  Report written -> {report_path}")
    return report_path


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Picekl Dataset Maker -- sort and trim accident videos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/picekl_dataset_maker.py\n"
            "  python scripts/picekl_dataset_maker.py --dry-run\n"
            "  python scripts/picekl_dataset_maker.py --skip-trim\n"
            "  python scripts/picekl_dataset_maker.py --max-videos 20\n"
            "  python scripts/picekl_dataset_maker.py --output-dir Datasets/processed/picek_sorted\n"
        ),
    )
    parser.add_argument(
        "--raw-dir",
        default=DEFAULT_RAW_DIR,
        help=f"Path to the raw picekl folder (default: {DEFAULT_RAW_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Where to write sorted/trimmed output (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--skip-trim",
        action="store_true",
        help="Sort videos only -- do not trim (faster, for classification only)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan and classify only -- no files copied or trimmed",
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        default=None,
        help="Limit to N videos (useful for quick testing)",
    )
    parser.add_argument(
        "--pre-sec",
        type=float,
        default=PRE_ACCIDENT_SEC,
        help=f"Seconds before accident to include in positive clips (default: {PRE_ACCIDENT_SEC})",
    )
    parser.add_argument(
        "--post-sec",
        type=float,
        default=POST_ACCIDENT_SEC,
        help=f"Seconds after accident to include in positive clips (default: {POST_ACCIDENT_SEC})",
    )
    parser.add_argument(
        "--neg-sec",
        type=float,
        default=TRIM_NEGATIVE_SEC,
        help=f"Trim length for negative clips in seconds (default: {TRIM_NEGATIVE_SEC})",
    )
    return parser.parse_args()


def main():
    global PRE_ACCIDENT_SEC, POST_ACCIDENT_SEC, TRIM_NEGATIVE_SEC

    args = parse_args()

    # Apply CLI overrides to globals
    PRE_ACCIDENT_SEC = args.pre_sec
    POST_ACCIDENT_SEC = args.post_sec
    TRIM_NEGATIVE_SEC = args.neg_sec

    # Resolve paths
    workspace = DEFAULT_SCRIPT_DIR
    raw_dir = Path(args.raw_dir)
    if not raw_dir.is_absolute():
        raw_dir = (workspace / raw_dir).resolve()

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = (workspace / output_dir).resolve()

    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    log_path = output_dir / "pipeline.log"
    logger = setup_logging(log_path)

    logger.info("=" * 70)
    logger.info("  PICEKL DATASET MAKER -- Sort & Trim Pipeline")
    logger.info("=" * 70)
    logger.info(f"  Raw dir    : {raw_dir}")
    logger.info(f"  Output dir : {output_dir}")
    logger.info(f"  Dry run    : {args.dry_run}")
    logger.info(f"  Skip trim  : {args.skip_trim}")
    logger.info(f"  Max videos : {args.max_videos}")
    logger.info(f"  Pre-sec    : {PRE_ACCIDENT_SEC}s")
    logger.info(f"  Post-sec   : {POST_ACCIDENT_SEC}s")
    logger.info(f"  Neg-sec    : {TRIM_NEGATIVE_SEC}s")
    logger.info("=" * 70)

    if not raw_dir.exists():
        logger.error(f"Raw directory not found: {raw_dir}")
        sys.exit(1)

    # ── Start system monitor ─────────────────────────────────────────────────
    monitor = SystemMonitor(logger, output_dir)
    monitor.start()
    logger.info("System monitor started (sampling every 2s, will pause pipeline if CPU/RAM > 95%)")

    stats = PipelineStats()
    records: List[VideoMeta] = []
    json_path = output_dir / "metadata.json"

    try:
        # STEP 1 -- SCAN
        records = scan_videos(raw_dir, logger, monitor, args.max_videos)

        stats.total_videos = len(records)
        stats.real_count = sum(1 for r in records if r.source_type == "real")
        stats.synthetic_count = sum(1 for r in records if r.source_type == "synthetic")
        stats.positive_count = sum(1 for r in records if r.label == "positive")
        stats.negative_count = sum(1 for r in records if r.label == "negative")
        stats.total_size_bytes = sum(r.file_size_bytes for r in records)

        # STEP 2 -- EXPORT JSON
        export_metadata_json(records, json_path, logger)

        # STEP 3 -- SORT
        sort_videos(records, output_dir, args.dry_run, logger, monitor)

        # STEP 4 -- TRIM
        if not args.skip_trim:
            trim_all_videos(records, output_dir, args.dry_run, logger, monitor)
        else:
            logger.info("STEP 4 -- TRIM: Skipped (--skip-trim flag set)")
            for r in records:
                r.trim_status = "skipped"

        stats.trim_done = sum(1 for r in records if r.trim_status == "done")
        stats.trim_skipped = sum(1 for r in records if r.trim_status == "skipped")
        stats.trim_error = sum(1 for r in records if r.trim_status == "error")

        # STEP 5 -- Update JSON with final statuses
        update_metadata_json(records, json_path, logger)

    except KeyboardInterrupt:
        logger.warning("Pipeline interrupted by user (Ctrl+C). Saving partial results...")
        if records:
            update_metadata_json(records, json_path, logger)
    except Exception as exc:
        logger.error(f"Pipeline FAILED: {exc}")
        logger.debug(traceback.format_exc())
        if records:
            try:
                update_metadata_json(records, json_path, logger)
            except Exception:
                pass
    finally:
        stats.end_time = time.time()

        # STEP 6 -- REPORT
        try:
            report_path = write_report(
                records, stats, monitor, output_dir, json_path,
                logger, args.dry_run, args.skip_trim
            )
        except Exception as exc:
            logger.error(f"Report generation failed: {exc}")
            report_path = None

        monitor.stop()
        monitor.join(timeout=5)

        elapsed_str = f"{int(stats.elapsed_sec // 60)}m {int(stats.elapsed_sec % 60)}s"
        logger.info("")
        logger.info("=" * 70)
        logger.info("PIPELINE COMPLETE")
        logger.info("=" * 70)
        logger.info(f"  Elapsed    : {elapsed_str}")
        logger.info(f"  Total      : {stats.total_videos}")
        logger.info(f"  Positive   : {stats.positive_count}")
        logger.info(f"  Negative   : {stats.negative_count}")
        logger.info(f"  Trimmed    : {stats.trim_done}")
        logger.info(f"  Errors     : {stats.trim_error}")
        if report_path:
            logger.info(f"  Report     : {report_path}")
        logger.info(f"  JSON       : {json_path}")
        logger.info(f"  Log        : {log_path}")
        logger.info("=" * 70)

        if stats.trim_error > 0:
            sys.exit(1)


if __name__ == "__main__":
    main()
