"""
Pipeline 2: Global Metadata Aggregation.

Responsible ONLY for discovering, validating, and merging metadata files
produced by previous pipelines into a single global master metadata CSV.
Uses pipeline_config.yaml for dataset definitions.
"""

from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd

from pipeline.config import PipelineConfig
from pipeline.utils import ProcessingLogger, ensure_dir


class MasterMetadataMerger:
    """
    Merges metadata files across defined datasets.
    Validates schemas (required/optional), deduplicates using a composite key,
    and produces a single global CSV and a summary report.
    """

    def __init__(
        self,
        config: PipelineConfig,
        output_file: Path,
        report_file: Path,
        logger: ProcessingLogger
    ):
        self.config = config
        self.output_file = Path(output_file)
        self.report_file = Path(report_file)
        self.logger = logger
        self.found_files: Dict[str, Path] = {}
        self.dataset_stats: List[Dict[str, any]] = []

    def discover_files(self) -> None:
        """
        Identify metadata files based on pipeline_config.yaml.
        Prefers processed/ over raw/ for each dataset.
        """
        self.logger.section("Discovering Metadata Files")
        self.found_files = {}

        for ds_config in self.config.aggregation_datasets:
            dataset_name = ds_config.name
            
            # 1. Look in processed directory first
            if ds_config.processed_dir:
                master_csv = ds_config.processed_dir / "metadata" / "master_metadata.csv"
                if master_csv.exists():
                    self.found_files[dataset_name] = master_csv
                    self.logger.info(f"Selected PROCESSED metadata for '{dataset_name}': {master_csv}")
                    continue
                else:
                    self.logger.debug(f"Processed metadata not found at {master_csv}")
            
            # 2. Look in raw directory as fallback
            if ds_config.raw_dir:
                raw_csv = ds_config.raw_dir / "metadata.csv"
                if raw_csv.exists():
                    self.found_files[dataset_name] = raw_csv
                    self.logger.info(f"Selected RAW metadata for '{dataset_name}': {raw_csv}")
                    continue
                else:
                    self.logger.debug(f"Raw metadata not found at {raw_csv}")
                    
            self.logger.warning(f"Could not find any metadata file for dataset '{dataset_name}'.")

    def validate_schema(self, df: pd.DataFrame, dataset_name: str) -> Tuple[bool, List[str], List[str]]:
        """
        Validate schema for required and optional columns.
        Returns:
            (is_valid, missing_required, missing_optional)
        """
        missing_required = [col for col in self.config.aggregation_required_columns if col not in df.columns]
        missing_optional = [col for col in self.config.aggregation_optional_columns if col not in df.columns]
        
        is_valid = len(missing_required) == 0
        return is_valid, missing_required, missing_optional

    def generate_report(self, total_raw: int, total_merged: int, deduplicated_count: int) -> None:
        """Generates a summary report of the aggregation process."""
        self.logger.section("Generating Summary Report")
        
        lines = [
            f"{'='*50}",
            f"GLOBAL METADATA AGGREGATION REPORT",
            f"{'='*50}",
            f"Output File:     {self.output_file}",
            f"Total Datasets:  {len(self.dataset_stats)}",
            f"Total Raw Rows:  {total_raw}",
            f"Duplicates:      {deduplicated_count}",
            f"Final Row Count: {total_merged}",
            f"{'='*50}",
            "DATASET BREAKDOWN:"
        ]
        
        for stat in self.dataset_stats:
            lines.append(f"\n- Dataset: {stat['name']}")
            lines.append(f"  Source:  {stat['source']}")
            lines.append(f"  Rows:    {stat['rows']}")
            if stat['missing_optional']:
                lines.append(f"  Missing optional columns: {', '.join(stat['missing_optional'])}")
        
        report_text = "\n".join(lines)
        ensure_dir(self.report_file.parent)
        self.report_file.write_text(report_text, encoding="utf-8")
        self.logger.info(f"Summary report generated at {self.report_file}")

    def run(self) -> None:
        """Execute the metadata aggregation pipeline."""
        self.discover_files()
        
        if not self.found_files:
            self.logger.error("Aggregation aborted: no metadata files found.")
            return

        self.logger.section("Loading and Merging Metadata")
        dataframes = []
        total_initial_rows = 0

        for dataset_name, file_path in self.found_files.items():
            self.logger.info(f"Loading {dataset_name} from {file_path}")
            try:
                df = pd.read_csv(file_path, low_memory=False)
            except Exception as e:
                self.logger.error(f"Failed to read {file_path}: {e}")
                continue
            
            # Auto-inject dataset_name if missing before validation
            if "dataset_name" not in df.columns:
                df["dataset_name"] = dataset_name
                
            is_valid, missing_req, missing_opt = self.validate_schema(df, dataset_name)
            
            if not is_valid:
                self.logger.error(f"Dataset '{dataset_name}' missing REQUIRED columns: {missing_req}. Skipping.")
                continue
                
            if missing_opt:
                self.logger.warning(f"Dataset '{dataset_name}' missing OPTIONAL columns: {missing_opt}.")
            
            rows = len(df)
            self.logger.info(f"  -> {rows} rows loaded.")
            total_initial_rows += rows
            
            self.dataset_stats.append({
                "name": dataset_name,
                "source": str(file_path),
                "rows": rows,
                "missing_optional": missing_opt
            })
            
            dataframes.append(df)

        if not dataframes:
            self.logger.error("No valid dataframes to merge after validation.")
            return

        # Concatenate
        merged_df = pd.concat(dataframes, ignore_index=True)
        self.logger.info(f"Successfully concatenated. Total raw rows: {len(merged_df)}")

        # Deduplication using composite key
        self.logger.section("Deduplication")
        dedupe_keys = self.config.aggregation_dedupe_keys
        
        # Verify all dedupe keys exist in the merged dataframe
        missing_keys = [k for k in dedupe_keys if k not in merged_df.columns]
        deduplicated_count = 0
        
        if missing_keys:
            self.logger.error(f"Cannot deduplicate using {dedupe_keys}. Missing columns: {missing_keys}")
            self.logger.warning("Falling back to full-row deduplication.")
            initial_count = len(merged_df)
            merged_df = merged_df.drop_duplicates()
            deduplicated_count = initial_count - len(merged_df)
        else:
            initial_count = len(merged_df)
            # Find rows where all components of the composite key are non-null
            valid_keys_mask = merged_df[dedupe_keys].notna().all(axis=1)
            
            # Duplicates among valid keys
            duplicates_mask = merged_df.duplicated(subset=dedupe_keys, keep="first") & valid_keys_mask
            
            deduplicated_count = duplicates_mask.sum()
            if deduplicated_count > 0:
                merged_df = merged_df[~duplicates_mask]
                self.logger.info(f"Removed {deduplicated_count} duplicate records based on composite key: {dedupe_keys}")
            else:
                self.logger.info(f"No duplicates found based on composite key: {dedupe_keys}")

        # Save result
        self.logger.section("Saving Global Metadata")
        ensure_dir(self.output_file.parent)
        merged_df.to_csv(self.output_file, index=False)
        self.logger.info(f"Global master metadata saved to: {self.output_file}")
        
        # Summary Report
        self.generate_report(total_initial_rows, len(merged_df), deduplicated_count)
        
        self.logger.info("Pipeline 2 complete.")
