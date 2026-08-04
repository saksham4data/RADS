"""
CLI Runner for the Metadata Generation Pipeline (Pipeline 0).

Parses command-line arguments, applies configuration overrides, configures
console logging, and executes the MetadataAssembler.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

from metadata_gen.config import MetadataGenConfig
from metadata_gen.assembler import MetadataAssembler


logger = logging.getLogger("metadata_gen")


def _setup_console_logging(level_name: str) -> None:
    """Configure the root logger for console output."""
    level = getattr(logging, level_name.upper(), logging.INFO)
    
    # Configure the metadata_gen logger (not the root logger)
    root = logging.getLogger("metadata_gen")
    root.setLevel(logging.DEBUG)  # Let handlers filter
    
    # Clear existing console handlers
    for h in root.handlers[:]:
        if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler):
            root.removeHandler(h)
            
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    
    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
    )
    console_handler.setFormatter(fmt)
    root.addHandler(console_handler)


def main(args: Optional[List[str]] = None) -> int:
    """
    Main entry point for Pipeline 0 CLI.

    Args:
        args: Command-line arguments. Defaults to sys.argv[1:].

    Returns:
        Exit code: 0 for success, 1 for partial failure (some files failed),
        2 for fatal error.
    """
    parser = argparse.ArgumentParser(
        description="Metadata Generation Pipeline (Pipeline 0)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("metadata_gen_config.yaml"),
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="Override dataset name (e.g. tudat, kaggle).",
    )
    parser.add_argument(
        "--raw-dir",
        type=str,
        help="Override raw media directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Override directory where metadata.csv is saved.",
    )
    parser.add_argument(
        "--media-type",
        type=str,
        choices=["video", "image"],
        help="Override media type (video or image).",
    )

    parsed = parser.parse_args(args)

    try:
        # Load config with overrides
        config = MetadataGenConfig.from_overrides(
            yaml_path=parsed.config,
            dataset_name=parsed.dataset,
            raw_dir=parsed.raw_dir,
            output_dir=parsed.output_dir,
            media_type=parsed.media_type,
        )
        
        # Setup console logging
        _setup_console_logging(config.console_log_level)
        
        # Execute assembly
        assembler = MetadataAssembler(config)
        result = assembler.run()
        
        if result.failed_files > 0:
            logger.warning("Pipeline completed with %d failed files.", result.failed_files)
            return 1
            
        return 0

    except Exception as exc:
        # Fallback logging if something fails during startup
        logging.basicConfig(level=logging.ERROR)
        logging.error("Fatal error in Metadata Generation Pipeline: %s: %s", type(exc).__name__, exc)
        return 2
