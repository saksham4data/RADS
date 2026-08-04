"""
Picek dataset processor — concrete implementation of BaseDatasetProcessor.

Handles the specific column mappings, directory layouts, and quirks of the
Picek traffic accident video dataset (real dashcam + CARLA synthetic videos).

Implements all v2 features:
- Configurable video mode (reference/copy/symlink)
- SHA-256 hashing + duplicate detection
- FPS extraction
- Preprocessing status state machine
- Standardized train/val/test splits
- Processing manifest for reproducibility
"""

import os
import shutil
from pathlib import Path
from typing import List, Optional

import pandas as pd

from pipeline.base_processor import BaseDatasetProcessor, ProcessingResult
from pipeline.config import PipelineConfig
from pipeline.id_generator import generate_video_id
from pipeline.metadata_validator import validate_row
from pipeline.video_validator import (
    check_video_exists,
    find_duplicates,
    get_file_size,
    probe_video,
)
from pipeline.report_generator import (
    generate_manifest,
    generate_report,
    generate_statistics,
    write_manifest,
    write_report,
    write_statistics,
)
from pipeline.utils import (
    MASTER_COLUMN_ORDER,
    PreprocessingStatus,
    ProcessingLogger,
    ProcessingStatus,
    ValidationStatus,
    assign_deterministic_split,
    compute_file_hash,
    ensure_dir,
    now_iso,
    resolve_raw_path,
)

# tqdm for progress bars (graceful fallback)
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


def _progress(iterable, total=None, desc="", disable=False):
    """Wrap an iterable with tqdm if available, otherwise pass through."""
    if tqdm is not None and not disable:
        return tqdm(iterable, total=total, desc=desc, unit="file", ncols=90)
    return iterable


class PicekProcessor(BaseDatasetProcessor):
    """
    Preprocessor for the Picek traffic accident video dataset.

    Raw layout expected:
        raw/picekl/
            real_videos/          (flat directory of .mp4 files)
            synthetic_videos/
                videos/<type>/    (nested by accident type)
                annotations/      (json.gz annotation dirs)
            metadata-real.csv
            metadata-synthetic.csv
            annotation_classes.yaml
    """

    def __init__(self, config: PipelineConfig):
        super().__init__(config)
        self.log = ProcessingLogger(
            log_file=self.output_reports / "processing_log.txt",
            name="picek",
            console_level=config.console_log_level,
            file_level=config.file_log_level,
        )
        self.result = ProcessingResult(dataset_name="picek")
        self._duplicate_groups: Optional[pd.DataFrame] = None

    # ── Step 1: Load Metadata ────────────────────────────────────────────

    def load_metadata(self) -> pd.DataFrame:
        """Read both CSV files, normalize column names, and tag source types."""
        self.log.section("Step 1: Loading Raw Metadata")

        # ── Real metadata ────────────────────────────────────────────
        real_csv = self.raw_dir / "metadata-real.csv"
        self.log.info(f"Reading {real_csv.name}...")
        df_real = pd.read_csv(real_csv)
        self.log.info(f"  Loaded {len(df_real)} real rows")

        df_real = df_real.rename(columns={"path": "original_path"})
        df_real["source_type"] = "real"
        df_real["dataset_name"] = self.config.dataset_name

        # ── Synthetic metadata ───────────────────────────────────────
        synth_csv = self.raw_dir / "metadata-synthetic.csv"
        self.log.info(f"Reading {synth_csv.name}...")
        df_synth = pd.read_csv(synth_csv)
        self.log.info(f"  Loaded {len(df_synth)} synthetic rows")

        df_synth = df_synth.rename(columns={"rgb_path": "original_path"})
        df_synth["source_type"] = "synthetic"
        df_synth["dataset_name"] = self.config.dataset_name

        # ── Combine ──────────────────────────────────────────────────
        df = pd.concat([df_real, df_synth], ignore_index=True, sort=False)
        self.log.info(f"  Combined: {len(df)} total rows")

        # Initialize preprocessing_status to 'raw'
        df["preprocessing_status"] = PreprocessingStatus.RAW.value

        return df

    # ── Step 2: Validate Metadata ────────────────────────────────────────

    def validate_metadata(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate each row's fields and populate 'validation_status'."""
        self.log.section("Step 2: Validating Metadata Fields")

        if not self.config.validate_metadata:
            self.log.info("  Metadata validation disabled by config — marking all as valid")
            df["validation_status"] = ValidationStatus.VALID.value
            return df

        statuses: List[str] = []
        issue_count = 0

        for idx, row in df.iterrows():
            source_type = row.get("source_type", "")
            is_valid, issues = validate_row(row.to_dict(), source_type, self.config)

            if is_valid:
                statuses.append(ValidationStatus.VALID.value)
            else:
                statuses.append(ValidationStatus.INVALID_METADATA.value)
                issue_count += 1
                vid_path = row.get("original_path", f"row_{idx}")
                self.log.debug(f"  INVALID [{vid_path}]: {'; '.join(issues)}")

        df["validation_status"] = statuses

        valid_count = len(df) - issue_count
        self.log.info(f"  Valid metadata: {valid_count}/{len(df)}")
        self.log.info(f"  Invalid metadata: {issue_count}/{len(df)}")

        # Update preprocessing_status for valid rows
        mask_valid = df["validation_status"] == ValidationStatus.VALID.value
        df.loc[mask_valid, "preprocessing_status"] = PreprocessingStatus.VALIDATED.value
        df.loc[~mask_valid, "preprocessing_status"] = PreprocessingStatus.INVALID.value

        return df

    # ── Steps 3-4: Verify Video Files, Probe Integrity, Hash ────────────

    def verify_and_probe_videos(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        For rows with valid metadata, check file existence, probe integrity,
        extract FPS, compute hash, and record file size.
        """
        self.log.section("Step 3-4: Verifying Videos, Probing, Hashing")

        file_sizes: List[int] = []
        fps_values: List[float] = []
        hash_values: List[str] = []
        annotation_available: List[bool] = []

        missing_count = 0
        corrupt_count = 0
        probed_count = 0

        rows_to_check = list(df.iterrows())

        for idx, row in _progress(rows_to_check, total=len(df), desc="Verifying"):
            original_path = str(row.get("original_path", ""))
            abs_path = resolve_raw_path(self.raw_dir, original_path)
            source_type = str(row.get("source_type", ""))

            # Skip already-invalid rows
            if row.get("validation_status") != ValidationStatus.VALID.value:
                file_sizes.append(0)
                fps_values.append(0.0)
                hash_values.append("")
                annotation_available.append(False)
                continue

            # Check file existence
            if not check_video_exists(abs_path):
                df.at[idx, "validation_status"] = ValidationStatus.MISSING_VIDEO.value
                df.at[idx, "preprocessing_status"] = PreprocessingStatus.INVALID.value
                file_sizes.append(0)
                fps_values.append(0.0)
                hash_values.append("")
                annotation_available.append(False)
                missing_count += 1
                self.log.debug(f"  MISSING: {original_path}")
                continue

            # Probe video integrity + extract FPS
            if self.config.probe_video_integrity:
                probe_result = probe_video(abs_path, extract_fps=self.config.extract_fps)
                if not probe_result["is_valid"]:
                    df.at[idx, "validation_status"] = ValidationStatus.CORRUPTED_VIDEO.value
                    df.at[idx, "preprocessing_status"] = PreprocessingStatus.INVALID.value
                    file_sizes.append(get_file_size(abs_path))
                    fps_values.append(0.0)
                    hash_values.append("")
                    annotation_available.append(False)
                    corrupt_count += 1
                    self.log.debug(f"  CORRUPT: {original_path} — {probe_result['error']}")
                    continue
                fps_values.append(probe_result["fps"])
            else:
                fps_values.append(0.0)

            # File size
            file_sizes.append(get_file_size(abs_path))

            # Compute hash
            if self.config.compute_hashes:
                h = compute_file_hash(abs_path, self.config.hash_algorithm)
                hash_values.append(h)
            else:
                hash_values.append("")

            # Annotation availability
            if source_type == "synthetic":
                ann_path = str(row.get("annotations_path", ""))
                if ann_path:
                    ann_abs = resolve_raw_path(self.raw_dir, ann_path)
                    annotation_available.append(ann_abs.exists())
                else:
                    annotation_available.append(False)
            else:
                annotation_available.append(False)

            probed_count += 1

        df["file_size_bytes"] = file_sizes
        df["fps"] = fps_values
        df["file_hash"] = hash_values
        df["annotation_available"] = annotation_available

        self.log.info(
            f"  Verified: {probed_count} valid, "
            f"{missing_count} missing, {corrupt_count} corrupted"
        )

        return df

    # ── Step 5: Duplicate Detection ──────────────────────────────────────

    def detect_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect duplicate videos by file hash and flag them."""
        self.log.section("Step 5: Detecting Duplicates")

        if not self.config.detect_duplicates or not self.config.compute_hashes:
            self.log.info("  Duplicate detection disabled by config")
            df["is_duplicate"] = False
            df["duplicate_group_id"] = ""
            return df

        # Need video_id for grouping — generate temporary IDs if not yet set
        if "video_id" not in df.columns:
            df["video_id"] = df.apply(
                lambda r: generate_video_id(
                    self.config.dataset_name,
                    str(r.get("source_type", "")),
                    Path(str(r.get("original_path", ""))).name,
                ),
                axis=1,
            )

        dup_groups = find_duplicates(df, hash_column="file_hash")
        self._duplicate_groups = dup_groups

        # Initialize columns
        df["is_duplicate"] = False
        df["duplicate_group_id"] = ""

        if len(dup_groups) > 0:
            self.log.info(f"  Found {len(dup_groups)} duplicate group(s)")
            # Mark duplicate rows
            for gid, row in dup_groups.iterrows():
                group_hash = row["file_hash"]
                group_id = f"dup_{gid:04d}"
                mask = df["file_hash"] == group_hash
                df.loc[mask, "is_duplicate"] = True
                df.loc[mask, "duplicate_group_id"] = group_id
                self.log.debug(
                    f"  Group {group_id}: {row['count']} files, "
                    f"hash={group_hash[:16]}..."
                )
            self.result.duplicates_found = len(dup_groups)
        else:
            self.log.info("  No duplicates found")

        return df

    # ── Steps 6-7: Generate IDs & Organize Videos ───────────────────────

    def process_videos(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate video IDs, handle files according to video_mode config,
        and set processing_status + processed_path + preprocessing_status.
        """
        self.log.section(f"Step 6-7: Processing Videos (mode={self.config.video_mode})")

        # Create output directories (even in reference mode, for metadata/reports)
        ensure_dir(self.output_metadata)
        ensure_dir(self.output_reports)
        ensure_dir(self.output_statistics)

        if self.config.video_mode in ("copy", "symlink"):
            ensure_dir(self.output_real)
            ensure_dir(self.output_synthetic)

        video_ids: List[str] = []
        processed_paths: List[str] = []
        processing_statuses: List[str] = []
        processed_at_values: List[str] = []

        processed = 0
        skipped = 0
        failed = 0

        rows_to_process = list(df.iterrows())

        for idx, row in _progress(rows_to_process, total=len(df), desc="Processing"):
            source_type = str(row.get("source_type", ""))
            original_path = str(row.get("original_path", ""))
            validation = str(row.get("validation_status", ""))
            filename = Path(original_path).name

            # Generate deterministic ID
            vid_id = generate_video_id(self.config.dataset_name, source_type, filename)
            video_ids.append(vid_id)

            # Skip invalid entries
            if validation != ValidationStatus.VALID.value:
                processed_paths.append("")
                processing_statuses.append(ProcessingStatus.SKIPPED.value)
                processed_at_values.append(now_iso())
                skipped += 1
                continue

            src_abs = resolve_raw_path(self.raw_dir, original_path)

            # Handle based on video_mode
            if self.config.video_mode == "reference":
                # Store path relative to raw_dir — no file I/O
                processed_paths.append(original_path)
                processing_statuses.append(ProcessingStatus.SUCCESS.value)
                processed += 1

            elif self.config.video_mode == "copy":
                dest_dir = self.output_real if source_type == "real" else self.output_synthetic
                dest_path = dest_dir / filename
                try:
                    if not dest_path.exists():
                        shutil.copy2(str(src_abs), str(dest_path))
                    processed_paths.append(str(Path(source_type) / filename))
                    processing_statuses.append(ProcessingStatus.SUCCESS.value)
                    processed += 1
                except Exception as e:
                    processed_paths.append("")
                    processing_statuses.append(ProcessingStatus.FAILED.value)
                    failed += 1
                    self.log.error(f"  COPY FAILED [{original_path}]: {e}")

            elif self.config.video_mode == "symlink":
                dest_dir = self.output_real if source_type == "real" else self.output_synthetic
                dest_path = dest_dir / filename
                try:
                    if not dest_path.exists():
                        os.symlink(str(src_abs), str(dest_path))
                    processed_paths.append(str(Path(source_type) / filename))
                    processing_statuses.append(ProcessingStatus.SUCCESS.value)
                    processed += 1
                except Exception as e:
                    processed_paths.append("")
                    processing_statuses.append(ProcessingStatus.FAILED.value)
                    failed += 1
                    self.log.error(f"  SYMLINK FAILED [{original_path}]: {e}")

            processed_at_values.append(now_iso())

        df["video_id"] = video_ids
        df["processed_path"] = processed_paths
        df["processing_status"] = processing_statuses
        df["processed_at"] = processed_at_values
        df["video_path_mode"] = self.config.video_mode

        # Advance preprocessing_status for successfully processed samples
        mask_success = df["processing_status"] == ProcessingStatus.SUCCESS.value
        df.loc[mask_success, "preprocessing_status"] = PreprocessingStatus.PROCESSED.value

        # Add version columns
        df["dataset_version"] = self.config.dataset_version
        df["pipeline_version"] = self.config.pipeline_version

        self.log.info(f"  Processed: {processed}, Skipped: {skipped}, Failed: {failed}")

        self.result.processed_samples = processed
        self.result.skipped_samples = skipped
        self.result.failed_samples = failed

        return df

    # ── Step 8: Assign Splits ────────────────────────────────────────────

    def assign_splits(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Assign standardized train/validation/test splits.

        For real videos: use existing split_in_distribution column, but hold out
        15% of train as validation (deterministic, based on video_id).
        For synthetic videos: assign 70/15/15 (deterministic, based on video_id).
        """
        self.log.section("Step 8: Assigning Standardized Splits")

        if not self.config.generate_standardized_splits:
            self.log.info("  Standardized splits disabled by config")
            df["split"] = ""
            return df

        splits: List[str] = []

        for _, row in df.iterrows():
            source_type = str(row.get("source_type", ""))
            vid_id = str(row.get("video_id", ""))

            if source_type == "real":
                original_split = str(row.get("split_in_distribution", "")).strip().lower()
                if original_split == "test":
                    splits.append("test")
                elif original_split == "train":
                    # Deterministic: hold out ~18% of train as validation
                    # (so overall ratio is ~70/15/15 accounting for test)
                    sub_split = assign_deterministic_split(
                        vid_id,
                        train_ratio=0.82,
                        validation_ratio=0.18,
                        seed=self.config.split_seed,
                    )
                    if sub_split == "validation":
                        splits.append("validation")
                    else:
                        splits.append("train")
                else:
                    # No original split info — use full ratio
                    splits.append(assign_deterministic_split(
                        vid_id,
                        self.config.train_ratio,
                        self.config.validation_ratio,
                        self.config.split_seed,
                    ))
            else:
                # Synthetic: no original split info — use config ratios
                splits.append(assign_deterministic_split(
                    vid_id,
                    self.config.train_ratio,
                    self.config.validation_ratio,
                    self.config.split_seed,
                ))

        df["split"] = splits

        # Log distribution
        split_counts = df["split"].value_counts()
        for s, c in split_counts.items():
            self.log.info(f"  {s}: {c} ({c / len(df) * 100:.1f}%)")

        return df

    # ── Orchestrator ─────────────────────────────────────────────────────

    def run(self) -> ProcessingResult:
        """Execute the full Picek preprocessing pipeline."""
        self.log.section("PICEK DATASET PREPROCESSING PIPELINE v2")
        self.log.info(f"Raw directory:    {self.raw_dir}")
        self.log.info(f"Output directory: {self.output_dir}")
        self.log.info(f"Video mode:       {self.config.video_mode}")
        self.log.info(f"Pipeline version: {self.config.pipeline_version}")
        self.log.info(f"Dataset version:  {self.config.dataset_version}")

        try:
            # Step 1: Load
            df = self.load_metadata()

            # Step 2: Validate metadata
            df = self.validate_metadata(df)

            # Steps 3-4: Verify, probe, hash
            df = self.verify_and_probe_videos(df)

            # Step 5: Detect duplicates
            df = self.detect_duplicates(df)

            # Steps 6-7: Generate IDs & process
            df = self.process_videos(df)

            # Step 8: Assign splits
            df = self.assign_splits(df)

            # ── Step 9: Save master metadata ─────────────────────────
            self.log.section("Step 9: Saving Master Metadata")
            self._save_master_metadata(df)

            # ── Step 10: Generate statistics ─────────────────────────
            self.log.section("Step 10: Generating Statistics")
            stats = generate_statistics(
                df, self.config,
                duplicates_found=self.result.duplicates_found,
            )
            stats_path = self.output_statistics / "dataset_statistics.json"
            write_statistics(stats, stats_path)
            self.log.info(f"  Saved: {stats_path}")
            self.result.statistics_path = stats_path

            # ── Step 11: Generate report ─────────────────────────────
            self.log.section("Step 11: Generating Report")
            report_md = generate_report(
                df, stats, self.config,
                duplicate_groups=self._duplicate_groups,
            )
            report_path = self.output_reports / "dataset_report.md"
            write_report(report_md, report_path)
            self.log.info(f"  Saved: {report_path}")
            self.result.report_path = report_path

            # ── Step 12: Generate manifest ───────────────────────────
            self.log.section("Step 12: Generating Processing Manifest")
            output_files = {
                "master_metadata.csv": self.result.master_metadata_path,
                "dataset_statistics.json": stats_path,
                "dataset_report.md": report_path,
            }
            manifest = generate_manifest(
                self.config, df, stats,
                duplicates_found=self.result.duplicates_found,
                output_files=output_files,
            )
            manifest_path = self.output_reports / "processing_manifest.json"
            write_manifest(manifest, manifest_path)
            self.log.info(f"  Saved: {manifest_path}")
            self.result.manifest_path = manifest_path

            # ── Finalize result ──────────────────────────────────────
            self.result.total_samples = len(df)
            self.result.valid_samples = int(
                (df["validation_status"] == ValidationStatus.VALID.value).sum()
            )
            self.result.invalid_samples = (
                self.result.total_samples - self.result.valid_samples
            )
            self.result.log_path = self.output_reports / "processing_log.txt"

            self.log.section("Pipeline Complete")
            self.log.info(self.result.summary())

        except Exception as e:
            self.log.error(f"PIPELINE FAILED: {type(e).__name__}: {e}")
            import traceback
            self.log.error(traceback.format_exc())
            raise

        return self.result

    # ── Private Helpers ──────────────────────────────────────────────────

    def _save_master_metadata(self, df: pd.DataFrame) -> None:
        """Reorder columns to the canonical 42-column schema and save."""
        # Ensure all expected columns exist
        for col in MASTER_COLUMN_ORDER:
            if col not in df.columns:
                df[col] = ""

        df_out = df[MASTER_COLUMN_ORDER].copy()

        master_path = self.output_metadata / "master_metadata.csv"
        ensure_dir(master_path.parent)
        df_out.to_csv(master_path, index=False)
        self.log.info(
            f"  Saved: {master_path} "
            f"({len(df_out)} rows × {len(df_out.columns)} columns)"
        )
        self.result.master_metadata_path = master_path
