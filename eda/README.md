# RADS — Exploratory Data Analysis Module

> **Version:** 1.0.0
> **Status:** Isolated, read-only EDA module
> **Data Source:** `Datasets/processed/global_master_metadata.csv`

## Overview

This module provides comprehensive exploratory data analysis for the RADS
project. It is completely isolated from the production pipelines and operates
in **read-only** mode — it never modifies existing data, pipelines, or project
architecture.

## Structure

```
eda/
├── config/
│   └── eda_config.yaml          # Central configuration
├── notebooks/
│   ├── 01_dataset_overview.ipynb
│   ├── 02_class_distribution.ipynb
│   ├── 03_video_statistics.ipynb
│   ├── 04_annotation_quality.ipynb
│   ├── 05_split_analysis.ipynb
│   └── 06_final_eda_summary.ipynb
├── utils/
│   ├── config.py                # YAML config loader
│   ├── logger.py                # Dual logging system
│   ├── wandb_manager.py         # W&B lifecycle manager
│   ├── metadata_loader.py       # Read-only metadata access
│   ├── plotting.py              # Dark theme plotting engine
│   ├── report_generator.py      # Markdown + JSON reports
│   ├── system_monitor.py        # System resource tracker
│   ├── output_manager.py        # Timestamped output versioning
│   └── statistics/
│       ├── base_analyzer.py     # Abstract base class
│       ├── overview_analyzer.py
│       ├── class_analyzer.py
│       ├── video_analyzer.py
│       ├── bbox_analyzer.py
│       ├── split_analyzer.py
│       └── quality_scorer.py    # Composite quality score
├── figures/                     # Generated plots (PNG + SVG)
├── reports/                     # Markdown reports + summary.json
├── exports/                     # CSV exports
├── logs/                        # Execution logs
└── run_history.json             # Execution audit trail
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set W&B API Key

```bash
export WANDB_API_KEY=your_key_here       # Linux/Mac
set WANDB_API_KEY=your_key_here          # Windows CMD
$env:WANDB_API_KEY="your_key_here"       # PowerShell
```

### 3. Run Notebooks

Open any notebook in Jupyter and execute from top to bottom.
Notebooks are designed to be run sequentially (01 → 06) but each
can run independently.

```bash
jupyter notebook eda/notebooks/
```

## Configuration

Edit `eda/config/eda_config.yaml` to change:

- **Data source path** — point to a different metadata CSV
- **W&B settings** — project name, entity, tags
- **Visualization** — DPI, formats, video sampling settings
- **Quality score weights** — adjust component weights
- **Output versioning** — enable/disable timestamps, manifests

## Reproducibility

Every run captures:

| Metadata | Purpose |
|---|---|
| EDA Version | Track which code version produced results |
| Dataset Version | Track which dataset was analyzed |
| Metadata Hash (SHA-256) | Verify exact input file identity |
| Git Commit | Link results to source code state |
| Random Seed | Ensure deterministic sampling |

## Output Organisation

Outputs are stored in timestamped subdirectories:

```
figures/2026-07-28_23-45/
├── missing_values_heatmap.png
├── missing_values_heatmap.svg
└── manifest.json
```

A `latest/` directory always points to the most recent run.
The `run_history.json` file maintains a full audit trail.

## W&B Integration

All notebooks log to the `RADS` W&B project with:

- **Metrics** — key statistics via `wandb.log()`
- **Tables** — DataFrames as interactive tables
- **Images** — generated figures
- **Artifacts** — complete output directories (reports, figures, exports)

Runs are tagged with `job_type="eda"` and organised under
`group="exploratory-analysis"`.

## Design Principles

1. **Isolation** — zero modifications to existing pipelines
2. **Read-only** — never mutates source data
3. **Thin notebooks** — all logic in `eda/utils/`
4. **Reproducible** — fixed seeds, metadata hashes, version tracking
5. **Extensible** — analyzer class hierarchy, YAML config
