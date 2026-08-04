#!/usr/bin/env python3
"""
Entry point for the Metadata Aggregation Pipeline (Pipeline 2).

Usage:
    python run_master_metadata.py
    python run_master_metadata.py --config pipeline_config.yaml
"""

import argparse
import sys
from pathlib import Path

from pipeline.config import PipelineConfig
from pipeline.master_metadata_merger import MasterMetadataMerger
from pipeline.utils import ProcessingLogger


def main():
    parser = argparse.ArgumentParser(
        description="Global Metadata Aggregation Pipeline (Pipeline 2)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML configuration file (default: pipeline_config.yaml)",
    )
    
    args = parser.parse_args()

    # Resolve config path
    script_dir = Path(__file__).resolve().parent

    config_path = None
    if args.config:
        config_path = Path(args.config).resolve()
    else:
        default_config = script_dir / "pipeline_config.yaml"
        if default_config.exists():
            config_path = default_config

    if not config_path or not config_path.exists():
        print(f"ERROR: Config file not found.")
        sys.exit(1)

    # Load config
    try:
        config = PipelineConfig.from_yaml(config_path)
    except Exception as e:
        print(f"ERROR loading config: {e}")
        sys.exit(1)
        
    # Resolve relative paths in config based on script directory
    for ds_config in config.aggregation_datasets:
        if not ds_config.raw_dir.is_absolute():
            ds_config.raw_dir = (script_dir / ds_config.raw_dir).resolve()
        if ds_config.processed_dir and not ds_config.processed_dir.is_absolute():
            ds_config.processed_dir = (script_dir / ds_config.processed_dir).resolve()

    output_dir = (script_dir / "processed").resolve()
    output_file = output_dir / "global_master_metadata.csv"
    report_file = output_dir / "global_metadata_summary.txt"
    log_file = output_dir / "pipeline_2_aggregation.log"

    # Print banner
    print(f"+{'=' * 58}+")
    print(f"|  GLOBAL METADATA AGGREGATION PIPELINE (PIPELINE 2)       |")
    print(f"+{'=' * 58}+")
    print(f"|  Output:   {str(output_file):<47}|")
    if config_path:
        print(f"|  Config:   {str(config_path.name):<47}|")
    print(f"+{'=' * 58}+")
    print()

    # Setup logger
    logger = ProcessingLogger(
        log_file=log_file,
        name="aggregation_pipeline",
        console_level=config.console_log_level,
        file_level=config.file_log_level,
    )

    # Run pipeline
    merger = MasterMetadataMerger(
        config=config,
        output_file=output_file,
        report_file=report_file,
        logger=logger,
    )
    merger.run()

if __name__ == "__main__":
    main()
