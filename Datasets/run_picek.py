"""
Entry point for the Picek dataset preprocessing pipeline v2.

Usage:
    python run_picek.py
    python run_picek.py --config pipeline_config.yaml
    python run_picek.py --video-mode copy
    python run_picek.py --raw-dir path/to/raw --output-dir path/to/out

This script never modifies the raw dataset.
"""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Picek Dataset Preprocessing Pipeline v2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python run_picek.py\n"
            "  python run_picek.py --config pipeline_config.yaml\n"
            "  python run_picek.py --video-mode copy\n"
            "  python run_picek.py --raw-dir ./raw/picekl --output-dir ./processed/picek\n"
        ),
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML configuration file (default: pipeline_config.yaml)",
    )
    parser.add_argument(
        "--raw-dir",
        type=str,
        default=None,
        help="Override: path to raw Picek dataset directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override: path to processed output directory",
    )
    parser.add_argument(
        "--video-mode",
        type=str,
        choices=["reference", "copy", "symlink"],
        default=None,
        help="Override: how to handle video files (default: reference)",
    )

    args = parser.parse_args()

    # ── Resolve config path ──────────────────────────────────────────────
    script_dir = Path(__file__).resolve().parent

    config_path = None
    if args.config:
        config_path = Path(args.config).resolve()
    else:
        # Auto-detect config in script directory
        default_config = script_dir / "pipeline_config.yaml"
        if default_config.exists():
            config_path = default_config

    # ── Load config with overrides ───────────────────────────────────────
    from pipeline.config import PipelineConfig

    config = PipelineConfig.from_overrides(
        yaml_path=config_path,
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        video_mode=args.video_mode,
    )

    # Resolve relative paths against script directory
    if not config.raw_dir.is_absolute():
        config.raw_dir = (script_dir / config.raw_dir).resolve()
    if not config.output_dir.is_absolute():
        config.output_dir = (script_dir / config.output_dir).resolve()
    config.__post_init__()  # Re-validate

    # ── Validate inputs ──────────────────────────────────────────────────
    if not config.raw_dir.exists():
        print(f"ERROR: Raw directory does not exist: {config.raw_dir}")
        sys.exit(1)

    real_csv = config.raw_dir / "metadata-real.csv"
    synth_csv = config.raw_dir / "metadata-synthetic.csv"
    if not real_csv.exists() or not synth_csv.exists():
        print(f"ERROR: Required CSV files not found in {config.raw_dir}")
        print(f"  Expected: metadata-real.csv, metadata-synthetic.csv")
        sys.exit(1)

    # ── Print banner ─────────────────────────────────────────────────────
    print(f"+{'=' * 58}+")
    print(f"|  PICEK DATASET PREPROCESSING PIPELINE v2                 |")
    print(f"+{'=' * 58}+")
    print(f"|  Pipeline: {config.pipeline_version:<47}|")
    print(f"|  Dataset:  {config.dataset_version:<47}|")
    print(f"|  Mode:     {config.video_mode:<47}|")
    print(f"|  Raw:      {str(config.raw_dir):<47}|")
    print(f"|  Output:   {str(config.output_dir):<47}|")
    if config_path:
        print(f"|  Config:   {str(config_path.name):<47}|")
    print(f"+{'=' * 58}+")
    print()

    # ── Run pipeline ─────────────────────────────────────────────────────
    from pipeline.picek_processor import PicekProcessor

    processor = PicekProcessor(config)
    result = processor.run()

    # ── Print summary ────────────────────────────────────────────────────
    print()
    print(result.summary())
    print()
    print("Output artifacts:")
    print(f"  Master metadata: {result.master_metadata_path}")
    print(f"  Statistics:      {result.statistics_path}")
    print(f"  Report:          {result.report_path}")
    print(f"  Manifest:        {result.manifest_path}")
    print(f"  Log:             {result.log_path}")
    print()

    if result.failed_samples > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
