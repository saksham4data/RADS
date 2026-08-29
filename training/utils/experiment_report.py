from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from training.configs.config import TrainingConfig


def generate_experiment_report(
    run_dir: Path,
    *,
    config: Optional[TrainingConfig] = None,
) -> Path:
    """Generate an experiment report for a completed run directory."""
    run_dir = run_dir.resolve()
    manifest = _load_json(run_dir / "manifest.json")
    if manifest is None:
        raise FileNotFoundError(f"manifest.json not found in {run_dir}")

    is_training_run = manifest.get("execution", {}).get("status") == "success"
    checkpoint_info = manifest.get("checkpoint", {}) if isinstance(manifest.get("checkpoint"), dict) else {}
    checkpoint_path = checkpoint_info.get("path")
    if is_training_run and not checkpoint_path:
        if (run_dir / "checkpoints" / "best.pt").is_file():
            checkpoint_path = str((run_dir / "checkpoints" / "best.pt").resolve())
        elif (run_dir / "checkpoints" / "last.pt").is_file():
            checkpoint_path = str((run_dir / "checkpoints" / "last.pt").resolve())

    checkpoint_run_dir = _resolve_checkpoint_run_dir(checkpoint_path)
    if is_training_run and not checkpoint_run_dir:
        checkpoint_run_dir = run_dir

    checkpoint_manifest = _load_json(checkpoint_run_dir / "manifest.json") if checkpoint_run_dir else None
    training_history = _load_json(checkpoint_run_dir / "metrics" / "training_history.json") if checkpoint_run_dir else None

    run_metrics_path = run_dir / "metrics" / "test_metrics.json"
    if not run_metrics_path.is_file():
        run_metrics_path = run_dir / "metrics" / "val_metrics.json"
    run_metrics = _load_json(run_metrics_path)

    predictions_path = run_dir / "predictions" / "test_predictions.json"
    if not predictions_path.is_file():
        predictions_path = run_dir / "predictions" / "val_predictions.json"
    predictions = _load_json(predictions_path)

    confusion_path = run_dir / "predictions" / "test_confusion_matrix.json"
    if not confusion_path.is_file():
        confusion_path = run_dir / "predictions" / "val_confusion_matrix.json"
    confusion_data = _load_json(confusion_path)

    source_files = _collect_source_files(
        run_dir=run_dir,
        manifest=manifest,
        checkpoint_path=checkpoint_path,
        checkpoint_run_dir=checkpoint_run_dir,
        config=config,
    )

    metadata_info = _load_dataset_info(config)
    derived_metrics = _derive_prediction_metrics(predictions)
    consistency_rows = _build_consistency_rows(
        manifest=manifest,
        checkpoint_manifest=checkpoint_manifest,
        checkpoint_info=checkpoint_info,
        predictions=predictions,
        confusion_data=confusion_data,
        metadata_info=metadata_info,
        config=config,
        is_training_run=is_training_run,
    )

    report_lines: List[str] = []
    report_lines.append(f"# Experiment Report: {manifest.get('run_name', run_dir.name)}")
    report_lines.append("")
    report_lines.append("## Artifact Inventory")
    report_lines.append("")
    report_lines.append("| Artifact | Path | Present |")
    report_lines.append("|---|---|---|")
    for label, path in source_files.items():
        report_lines.append(
            f"| {label} | `{_display_path(path)}` | {'YES' if path and Path(path).is_file() else 'NO'} |"
        )

    report_lines.append("")
    report_lines.append("## 1. Experiment Information")
    report_lines.append("")
    report_lines.extend(
        _experiment_information_section(
            manifest=manifest,
            checkpoint_manifest=checkpoint_manifest,
            config=config,
            checkpoint_info=checkpoint_info,
            checkpoint_path=checkpoint_path,
            is_training_run=is_training_run,
        )
    )

    report_lines.append("")
    report_lines.append("## 2. Dataset And Split Information")
    report_lines.append("")
    report_lines.extend(_dataset_information_section(metadata_info, config))

    report_lines.append("")
    report_lines.append("## 3. Complete Training History")
    report_lines.append("")
    report_lines.extend(_training_history_section(training_history))

    report_lines.append("")
    report_lines.append("## 4. Checkpoint Information")
    report_lines.append("")
    report_lines.extend(
        _checkpoint_information_section(
            manifest=manifest,
            checkpoint_manifest=checkpoint_manifest,
            config=config,
            checkpoint_path=checkpoint_path,
            checkpoint_info=checkpoint_info,
            is_training_run=is_training_run,
        )
    )

    report_lines.append("")
    report_lines.append("## 5. Test Results")
    report_lines.append("")
    report_lines.extend(
        _test_results_section(
            manifest=manifest,
            predictions=predictions,
            derived_metrics=derived_metrics,
            run_metrics=run_metrics,
            is_training_run=is_training_run,
        )
    )

    report_lines.append("")
    report_lines.append("## 6. Confusion Matrix")
    report_lines.append("")
    report_lines.extend(_confusion_section(confusion_data, is_training_run=is_training_run))

    report_lines.append("")
    report_lines.append("## 7. Prediction Analysis")
    report_lines.append("")
    report_lines.extend(
        _prediction_analysis_section(
            predictions=predictions,
            derived_metrics=derived_metrics,
            is_training_run=is_training_run,
        )
    )

    report_lines.append("")
    report_lines.append("## 8. Cross-Check / Consistency Audit")
    report_lines.append("")
    report_lines.extend(_consistency_section(consistency_rows))

    report_lines.append("")
    report_lines.append("## 9. What This Experiment Actually Tells Us")
    report_lines.append("")
    report_lines.extend(
        _interpretation_section(
            derived_metrics=derived_metrics,
            manifest=manifest,
            checkpoint_manifest=checkpoint_manifest,
            checkpoint_path=checkpoint_path,
            is_training_run=is_training_run,
        )
    )

    report_lines.append("")
    report_lines.append("## 10. Experiment Verdict")
    report_lines.append("")
    report_lines.extend(
        _verdict_section(
            manifest=manifest,
            checkpoint_manifest=checkpoint_manifest,
            consistency_rows=consistency_rows,
            is_training_run=is_training_run,
            run_metrics=run_metrics,
        )
    )

    report_lines.append("")
    report_lines.append("## 11. Next Actions")
    report_lines.append("")
    report_lines.extend(_next_actions_section(consistency_rows, is_training_run=is_training_run, run_metrics=run_metrics))

    report_path = run_dir / "experiment_report.md"
    report_path.write_text("\n".join(report_lines).rstrip() + "\n", encoding="utf-8")
    return report_path


def _collect_source_files(
    run_dir: Path,
    manifest: Dict[str, Any],
    checkpoint_path: Optional[str],
    checkpoint_run_dir: Optional[Path],
    config: Optional[TrainingConfig],
) -> Dict[str, Optional[str]]:
    cfg_path = manifest.get("config_path")
    if not cfg_path and config and config.config_path:
        cfg_path = str(Path(config.config_path).resolve())

    training_log = str(run_dir.parents[2] / "logs" / "training" / "training.log")

    return {
        "Run manifest": str(run_dir / "manifest.json"),
        "Prediction file": str(run_dir / "predictions" / "test_predictions.json")
        if (run_dir / "predictions" / "test_predictions.json").is_file()
        else (
            str(run_dir / "predictions" / "val_predictions.json")
            if (run_dir / "predictions" / "val_predictions.json").is_file()
            else None
        ),
        "Confusion matrix": str(run_dir / "predictions" / "test_confusion_matrix.json")
        if (run_dir / "predictions" / "test_confusion_matrix.json").is_file()
        else (
            str(run_dir / "predictions" / "val_confusion_matrix.json")
            if (run_dir / "predictions" / "val_confusion_matrix.json").is_file()
            else None
        ),
        "Checkpoint file": checkpoint_path,
        "Checkpoint source manifest": str(checkpoint_run_dir / "manifest.json") if checkpoint_run_dir and (checkpoint_run_dir / "manifest.json").is_file() else None,
        "Training history": str(checkpoint_run_dir / "metrics" / "training_history.json") if checkpoint_run_dir and (checkpoint_run_dir / "metrics" / "training_history.json").is_file() else None,
        "Run metrics": str(run_dir / "metrics" / "test_metrics.json")
        if (run_dir / "metrics" / "test_metrics.json").is_file()
        else (
            str(run_dir / "metrics" / "val_metrics.json")
            if (run_dir / "metrics" / "val_metrics.json").is_file()
            else None
        ),
        "Configuration file": cfg_path,
        "Root training log": training_log if Path(training_log).is_file() else None,
    }


def _load_dataset_info(config: Optional[TrainingConfig]) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "metadata_path": None,
        "integrity_report_path": None,
        "counts": None,
        "split_class_counts": None,
        "dataset_version": None,
        "duplicate_info": [],
        "notes": [],
    }
    if config is None:
        return info

    metadata_path = config.metadata_path.resolve()
    info["metadata_path"] = str(metadata_path)
    if metadata_path.is_file():
        with metadata_path.open(encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh))
        split_col = config.split_column
        alt_split_col = "split" if split_col != "split" else "split_in_distribution"
        counts = Counter()
        split_class_counts: Dict[str, Counter[str]] = {}
        dataset_version = Counter()
        for row in rows:
            split_value = (row.get(split_col) or row.get(alt_split_col) or "").strip()
            counts["total"] += 1
            if split_value:
                counts[split_value] += 1
                split_class_counts.setdefault(split_value, Counter())[row.get("type", "")] += 1
            dataset_version[str(row.get("dataset_version", ""))] += 1
        info["counts"] = dict(counts)
        info["split_class_counts"] = {k: dict(v) for k, v in split_class_counts.items()}
        info["dataset_version"] = dict(dataset_version)

    integrity_report_path = metadata_path.parent / "tudat_v2_integrity_report.md"
    if integrity_report_path.is_file():
        info["integrity_report_path"] = str(integrity_report_path)
        text = integrity_report_path.read_text(encoding="utf-8")
        excluded_match = re.search(r"(\d+) records were excluded from v2", text)
        if excluded_match:
            info["notes"].append(f"{excluded_match.group(1)} records were excluded from v2.")
        if "No cross-split leakage." in text:
            info["notes"].append("Integrity report states no cross-split leakage.")
    return info


def _derive_prediction_metrics(predictions: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(predictions, dict):
        return {}

    y_pred = np.asarray(predictions.get("predictions", []), dtype=int)
    y_true = np.asarray(predictions.get("targets", []), dtype=int)
    if y_pred.size == 0 or y_true.size == 0:
        return {}

    class_names = predictions.get("class_names") or []
    confidences = predictions.get("confidences")
    metrics: Dict[str, Any] = {
        "num_predictions": int(y_pred.size),
        "num_correct": int((y_true == y_pred).sum()),
        "num_incorrect": int((y_true != y_pred).sum()),
        "predicted_distribution": {str(k): int(v) for k, v in Counter(y_pred.tolist()).items()},
        "actual_distribution": {str(k): int(v) for k, v in Counter(y_true.tolist()).items()},
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(range(len(class_names) or 2))).tolist(),
    }

    per_class: Dict[str, Dict[str, float]] = {}
    label_count = len(class_names) if class_names else int(max(y_true.max(initial=0), y_pred.max(initial=0)) + 1)
    for idx in range(label_count):
        name = class_names[idx] if idx < len(class_names) else str(idx)
        per_class[name] = {
            "precision": float(precision_score(y_true, y_pred, labels=[idx], average="macro", zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, labels=[idx], average="macro", zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, labels=[idx], average="macro", zero_division=0)),
        }
    metrics["per_class"] = per_class

    if isinstance(confidences, list) and confidences:
        conf = np.asarray(confidences, dtype=float)
        if conf.ndim == 2 and conf.shape[0] == y_true.size:
            max_conf = conf.max(axis=1)
            metrics["confidence_stats"] = {
                "mean_max_confidence": float(max_conf.mean()),
                "min_max_confidence": float(max_conf.min()),
                "max_max_confidence": float(max_conf.max()),
            }
            if conf.shape[1] >= 2:
                try:
                    metrics["auroc"] = float(roc_auc_score(y_true, conf[:, 1]))
                except ValueError:
                    pass
    return metrics


def _build_consistency_rows(
    *,
    manifest: Dict[str, Any],
    checkpoint_manifest: Optional[Dict[str, Any]],
    checkpoint_info: Dict[str, Any],
    predictions: Optional[Dict[str, Any]],
    confusion_data: Optional[Dict[str, Any]],
    metadata_info: Dict[str, Any],
    config: Optional[TrainingConfig],
    is_training_run: bool,
) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []

    def add(source: str, metric: str, value: Any, related: Any, consistent: str, explanation: str) -> None:
        rows.append({
            "Source": source,
            "Metric": metric,
            "Value": _stringify(value),
            "Expected/Related Value": _stringify(related),
            "Consistent?": consistent,
            "Explanation": explanation,
        })

    if is_training_run:
        add(
            "Training manifest",
            "Run name",
            manifest.get("run_name"),
            manifest.get("run_name"),
            "YES",
            "Training run manifest is self-consistent.",
        )
        add(
            "Training manifest",
            "W&B run ID",
            manifest.get("wandb_run_id"),
            manifest.get("wandb_run_id"),
            "YES" if manifest.get("wandb_run_id") else "UNKNOWN",
            "W&B run ID recorded for training run." if manifest.get("wandb_run_id") else "No W&B run ID recorded.",
        )
    else:
        source_name = checkpoint_info.get("source_run_name") or (checkpoint_manifest.get("run_name") if checkpoint_manifest else None)
        eval_run_name = manifest.get("run_name")
        runs_aligned = bool(source_name and (source_name == eval_run_name or source_name in str(eval_run_name) or str(eval_run_name) in source_name))
        add(
            "Test manifest vs checkpoint source manifest",
            "Run name",
            eval_run_name,
            source_name if source_name else "Missing",
            "YES" if runs_aligned else ("NO" if checkpoint_manifest else "UNKNOWN"),
            "The evaluation run references a checkpoint from the expected source training run."
            if runs_aligned
            else (
                "The completed test run is labeled differently from the source training run used by its checkpoint."
                if checkpoint_manifest
                else "Checkpoint source manifest unavailable."
            ),
        )

        source_wb = checkpoint_info.get("source_wandb_run_id") or (checkpoint_manifest.get("wandb_run_id") if checkpoint_manifest else None)
        add(
            "Test manifest vs checkpoint source manifest",
            "W&B run ID",
            source_wb,
            checkpoint_manifest.get("wandb_run_id") if checkpoint_manifest else source_wb,
            "YES" if source_wb else "UNKNOWN",
            "Source training W&B run ID is tracked in checkpoint lineage." if source_wb else "No source W&B run ID found.",
        )

        cp_epoch = checkpoint_info.get("training_epoch")
        training_best_epoch = checkpoint_info.get("source_best_epoch")
        if training_best_epoch is None and checkpoint_manifest:
            training_best_epoch = checkpoint_manifest.get("execution", {}).get("best_epoch")

        is_epoch_consistent = (
            cp_epoch is not None and training_best_epoch is not None and (cp_epoch == training_best_epoch + 1 or cp_epoch == training_best_epoch)
        )
        add(
            "Test manifest vs checkpoint source manifest",
            "Checkpoint epoch",
            cp_epoch,
            training_best_epoch,
            "YES" if is_epoch_consistent else ("PARTIAL" if cp_epoch is not None and training_best_epoch is not None else "UNKNOWN"),
            f"Checkpoint training epoch ({cp_epoch}) matches source training run best_epoch (index {training_best_epoch})."
            if is_epoch_consistent
            else "One of the checkpoint epoch sources is missing or mismatched.",
        )

    if isinstance(predictions, dict) and isinstance(confusion_data, dict):
        derived_cm = _derive_prediction_metrics(predictions).get("confusion_matrix")
        add(
            "Prediction file vs confusion matrix file",
            "Confusion matrix",
            derived_cm,
            confusion_data.get("matrix"),
            "YES" if derived_cm == confusion_data.get("matrix") else "NO",
            "Derived confusion matrix from raw predictions matches the recorded confusion matrix file."
            if derived_cm == confusion_data.get("matrix")
            else "Mismatch between raw prediction confusion matrix and recorded confusion matrix.",
        )

    if config is not None:
        add(
            "Config vs metadata",
            "Configured split column",
            config.split_column,
            "split_in_distribution populated in metadata"
            if metadata_info.get("counts")
            else "Metadata unavailable",
            "YES" if config.split_column == "split_in_distribution" and metadata_info.get("counts") else "UNKNOWN",
            "The current config points at split_in_distribution, which is the populated split column in the TUDAT v2 metadata file.",
        )

    git_commit = manifest.get("git_commit")
    source_git = checkpoint_info.get("source_git_commit") or (checkpoint_manifest.get("git_commit") if checkpoint_manifest else None)
    if is_training_run:
        add(
            "Training manifest",
            "Git commit",
            git_commit,
            git_commit,
            "YES" if git_commit else "UNKNOWN",
            "Git commit recorded in training manifest." if git_commit else "No git commit recorded.",
        )
    else:
        add(
            "Test manifest vs training source manifest",
            "Git commit",
            git_commit or source_git,
            source_git,
            "YES" if (git_commit and source_git and git_commit == source_git) or (source_git and not git_commit) else ("NO" if git_commit and source_git and git_commit != source_git else "UNKNOWN"),
            "Git commit hash is tracked and consistent with training source."
            if (git_commit and source_git and git_commit == source_git) or (source_git and not git_commit)
            else "Checkpoint source manifest git commit unavailable or mismatched.",
        )

    return rows


def _experiment_information_section(
    manifest: Dict[str, Any],
    checkpoint_manifest: Optional[Dict[str, Any]],
    config: Optional[TrainingConfig],
    checkpoint_info: Dict[str, Any],
    checkpoint_path: Optional[str],
    is_training_run: bool,
) -> List[str]:
    cfg_path = manifest.get("config_path")
    if not cfg_path and config and config.config_path:
        cfg_path = str(Path(config.config_path).resolve())

    source_run_name = checkpoint_info.get("source_run_name") or (checkpoint_manifest.get("run_name") if checkpoint_manifest else ("Current training run" if is_training_run else None))
    source_wandb_id = checkpoint_info.get("source_wandb_run_id") or (checkpoint_manifest.get("wandb_run_id") if checkpoint_manifest else manifest.get("wandb_run_id"))
    source_wandb_url = checkpoint_info.get("source_wandb_run_url") or (checkpoint_manifest.get("wandb_run_url") if checkpoint_manifest else manifest.get("wandb_run_url"))
    source_duration = checkpoint_manifest.get("execution", {}).get("duration_seconds") if checkpoint_manifest else (manifest.get("execution", {}).get("duration_seconds") if is_training_run else None)

    training_epoch = checkpoint_info.get("training_epoch")
    if training_epoch is None and is_training_run:
        best_ep = manifest.get("execution", {}).get("best_epoch")
        training_epoch = (best_ep + 1) if isinstance(best_ep, int) else None

    lines = [
        "| Field | Value | Source |",
        "|---|---|---|",
        f"| Experiment/run name | {_stringify(manifest.get('run_name'))} | Run manifest |",
        f"| Timestamp | {_stringify(manifest.get('timestamp'))} | Run manifest |",
        f"| Git commit | {_stringify(manifest.get('git_commit', checkpoint_info.get('source_git_commit', 'Not recorded')))} | Run manifest |",
        f"| Model architecture | {_stringify(manifest.get('model'))} | Run manifest |",
    ]
    if config:
        lines.extend([
            f"| Temporal architecture | {_stringify(config.temporal_architecture)} | Config |",
            f"| Temporal hidden dim | {_stringify(config.temporal_hidden_dim)} | Config |",
            f"| Optimizer | {_stringify(config.optimizer_name)} | Config |",
            f"| Learning rate | {_stringify(config.learning_rate)} | Config |",
            f"| Weight decay | {_stringify(config.weight_decay)} | Config |",
            f"| Batch size | {_stringify(config.batch_size)} | Config |",
        ])
    lines.extend([
        f"| Dataset | {_stringify(manifest.get('dataset'))} | Run manifest |",
        f"| Training version | {_stringify(manifest.get('training_version'))} | Run manifest |",
        f"| Seed | {_stringify(manifest.get('seed'))} | Run manifest |",
        f"| Execution status | {_stringify(manifest.get('execution', {}).get('status', 'success'))} | Run manifest |",
        f"| Duration (s) | {_stringify(manifest.get('execution', {}).get('duration_seconds', 'Not recorded'))} | Run manifest |",
        f"| Checkpoint used | `{_display_path(checkpoint_path)}` | {'Training output' if is_training_run else 'Run manifest / evaluation argument'} |",
        f"| Checkpoint training epoch | {_stringify(training_epoch)} | Checkpoint metadata |",
        f"| Source training run name | {_stringify(source_run_name)} | Checkpoint source manifest |",
        f"| Source training W&B run ID | {_stringify(source_wandb_id)} | Checkpoint source manifest |",
        f"| Source training W&B run URL | {_stringify(source_wandb_url)} | Checkpoint source manifest |",
        f"| Source training duration (s) | {_stringify(source_duration)} | Checkpoint source manifest |",
        f"| Configuration file inspected | `{_display_path(cfg_path)}` | Manifest / Config loader |",
    ])
    return lines


def _dataset_information_section(metadata_info: Dict[str, Any], config: Optional[TrainingConfig]) -> List[str]:
    lines: List[str] = []
    counts = metadata_info.get("counts") or {}
    split_class_counts = metadata_info.get("split_class_counts") or {}
    lines.extend([
        "| Field | Value | Source |",
        "|---|---|---|",
        f"| Metadata file | `{_display_path(metadata_info.get('metadata_path'))}` | Metadata CSV |",
        f"| Integrity report | `{_display_path(metadata_info.get('integrity_report_path'))}` | Integrity report |",
        f"| Total samples | {_stringify(counts.get('total', 'Not recorded'))} | Metadata CSV |",
        f"| Training samples | {_stringify(counts.get('train', 'Not recorded'))} | Metadata CSV |",
        f"| Validation samples | {_stringify(counts.get('val', 'Not recorded'))} | Metadata CSV |",
        f"| Test samples | {_stringify(counts.get('test', 'Not recorded'))} | Metadata CSV |",
        f"| Classes | `accident`, `non-accident` | Metadata CSV / config label mapping |",
        f"| Dataset version values | {_stringify(metadata_info.get('dataset_version', 'Not recorded'))} | Metadata CSV |",
    ])
    if config is not None:
        lines.append(f"| Split column configured | `{config.split_column}` | Config file |")
    lines.append("")
    lines.append("### Split Class Distribution")
    lines.append("")
    lines.append("| Split | accident | non-accident | Total | Source |")
    lines.append("|---|---|---|---|---|")
    for split_name in ("train", "val", "test"):
        class_counts = split_class_counts.get(split_name, {})
        total = sum(class_counts.values()) if class_counts else "Not recorded"
        lines.append(
            f"| {split_name} | {_stringify(class_counts.get('accident', 'Not recorded'))} | "
            f"{_stringify(class_counts.get('non-accident', 'Not recorded'))} | {_stringify(total)} | Metadata CSV |"
        )
    if metadata_info.get("notes"):
        lines.append("")
        lines.append("### Recorded Notes")
        lines.append("")
        for note in metadata_info["notes"]:
            lines.append(f"- {note}")
    lines.append("")
    lines.append("### Important Distinction")
    lines.append("")
    lines.append("- The metadata split counts above are video-level counts.")
    lines.append("- Frame-level predictions are generated during evaluation for frame-based or temporal sequences.")
    return lines


def _training_history_section(training_history: Optional[Dict[str, Any]]) -> List[str]:
    if not isinstance(training_history, dict) or not training_history:
        return ["Training history JSON was not found. No epoch table can be produced from recorded artifacts."]

    metric_order = [
        "train_loss",
        "val_loss",
        "train_accuracy",
        "val_accuracy",
        "train/top1_accuracy",
        "val/top1_accuracy",
        "train/balanced_accuracy",
        "val/balanced_accuracy",
        "train/macro_f1",
        "val/macro_f1",
        "train/precision_accident",
        "val/precision_accident",
        "train/recall_accident",
        "val/recall_accident",
        "train/f1_accident",
        "val/f1_accident",
        "train/precision_non_accident",
        "val/precision_non_accident",
        "train/recall_non_accident",
        "val/recall_non_accident",
        "train/f1_non_accident",
        "val/f1_non_accident",
        "train/auroc",
        "val/auroc",
        "lr",
    ]
    extra_metrics = [key for key in training_history.keys() if key not in metric_order]
    metric_order.extend(sorted(extra_metrics))
    epoch_count = max((len(v) for v in training_history.values() if isinstance(v, list)), default=0)

    lines = ["| Epoch | " + " | ".join(metric_order) + " |"]
    lines.append("|---|" + "|".join(["---"] * len(metric_order)) + "|")
    for idx in range(epoch_count):
        row = [str(idx + 1)]
        for metric in metric_order:
            values = training_history.get(metric)
            if isinstance(values, list) and idx < len(values):
                row.append(_stringify(values[idx]))
            else:
                row.append("Not recorded")
        lines.append("| " + " | ".join(row) + " |")
    return lines


def _checkpoint_information_section(
    manifest: Dict[str, Any],
    checkpoint_manifest: Optional[Dict[str, Any]],
    config: Optional[TrainingConfig],
    checkpoint_path: Optional[str],
    checkpoint_info: Dict[str, Any],
    is_training_run: bool,
) -> List[str]:
    training_epoch = checkpoint_info.get("training_epoch")
    if training_epoch is None and is_training_run:
        best_ep = manifest.get("execution", {}).get("best_epoch")
        training_epoch = (best_ep + 1) if isinstance(best_ep, int) else None

    monitored_metric = checkpoint_info.get("monitored_metric") or (config.monitor_metric if config else None)
    monitor_mode = checkpoint_info.get("monitor_mode") or (config.monitor_mode if config else None)

    best_epoch = checkpoint_info.get("source_best_epoch")
    if best_epoch is None:
        best_epoch = checkpoint_manifest.get("execution", {}).get("best_epoch") if checkpoint_manifest else manifest.get("execution", {}).get("best_epoch")

    best_metric = checkpoint_info.get("best_metric") or checkpoint_info.get("source_best_metric")
    if best_metric is None:
        best_metric = checkpoint_manifest.get("execution", {}).get("best_metric") if checkpoint_manifest else manifest.get("execution", {}).get("best_metric")

    lines = [
        f"- Selected checkpoint path: `{_display_path(checkpoint_path)}`",
        f"- Checkpoint training epoch: {_stringify(training_epoch)}",
        f"- Monitored metric: `{_stringify(monitored_metric)}` (mode: `{_stringify(monitor_mode)}`)",
        f"- Best metric value: `{_stringify(best_metric)}`",
    ]
    if checkpoint_info.get("training_metrics"):
        lines.append(f"- Checkpoint-associated validation metrics: `{_stringify(checkpoint_info.get('training_metrics'))}`")

    if checkpoint_manifest:
        lines.append(f"- Source training manifest run name: `{_stringify(checkpoint_manifest.get('run_name'))}`")
        lines.append(f"- Source training W&B run ID: `{_stringify(checkpoint_manifest.get('wandb_run_id'))}`")
        lines.append(f"- Source training best_epoch: {_stringify(best_epoch)}")
        lines.append(f"- Source training best_metric: {_stringify(best_metric)}")

    lines.append("")
    lines.append("| Source | Value |")
    lines.append("|---|---|")
    lines.append(f"| Checkpoint training_epoch | {_stringify(training_epoch)} |")
    lines.append(f"| Source training manifest best_epoch | {_stringify(best_epoch)} |")
    lines.append(f"| Source training manifest best_metric | {_stringify(best_metric)} |")
    lines.append(f"| Source training W&B run ID | {_stringify(checkpoint_info.get('source_wandb_run_id') or (checkpoint_manifest.get('wandb_run_id') if checkpoint_manifest else manifest.get('wandb_run_id')))} |")
    return lines


def _test_results_section(
    manifest: Dict[str, Any],
    predictions: Optional[Dict[str, Any]],
    derived_metrics: Dict[str, Any],
    run_metrics: Optional[Dict[str, Any]] = None,
    is_training_run: bool = False,
) -> List[str]:
    if is_training_run and not run_metrics and not derived_metrics:
        return [
            "This report is for a training run. Test evaluation metrics are generated and recorded when running the test pipeline (`training/test.py`)."
        ]

    prefix = "test" if isinstance(run_metrics, dict) and any(str(k).startswith("test/") for k in run_metrics.keys()) else "val"
    metrics_source = "Run metrics JSON" if run_metrics else "No test metrics JSON found in run directory"
    precision_map = {k: v for k, v in (run_metrics or {}).items() if "/precision_" in str(k)}
    recall_map = {k: v for k, v in (run_metrics or {}).items() if "/recall_" in str(k)}
    f1_map = {k: v for k, v in (run_metrics or {}).items() if "/f1_" in str(k)}
    lines = [
        "### Recorded Metrics",
        "",
        "| Metric | Recorded value | Source |",
        "|---|---|---|",
        f"| Test loss | {_stringify((run_metrics or {}).get(f'{prefix}/loss', 'Not recorded'))} | {metrics_source} |",
        f"| Test accuracy / top-1 accuracy | {_stringify((run_metrics or {}).get(f'{prefix}/top1_accuracy', 'Not recorded'))} | {metrics_source} |",
        f"| Test balanced accuracy | {_stringify((run_metrics or {}).get(f'{prefix}/balanced_accuracy', 'Not recorded'))} | {metrics_source} |",
        f"| Test macro F1 | {_stringify((run_metrics or {}).get(f'{prefix}/macro_f1', 'Not recorded'))} | {metrics_source} |",
        f"| Test precision (per class) | {_stringify(precision_map or 'Not recorded')} | {metrics_source} |",
        f"| Test recall (per class) | {_stringify(recall_map or 'Not recorded')} | {metrics_source} |",
        f"| Test F1 (per class) | {_stringify(f1_map or 'Not recorded')} | {metrics_source} |",
        f"| Test AUROC | {_stringify((run_metrics or {}).get(f'{prefix}/auroc', 'Not recorded'))} | {metrics_source} |",
    ]
    if derived_metrics:
        lines.extend([
            "",
            "### Independently Derived Metrics From Raw Predictions",
            "",
            "| Metric | Derived value | Basis |",
            "|---|---|---|",
            f"| Accuracy | {_stringify(derived_metrics.get('accuracy'))} | `predictions` vs `targets` |",
            f"| Balanced accuracy | {_stringify(derived_metrics.get('balanced_accuracy'))} | `predictions` vs `targets` |",
            f"| Macro F1 | {_stringify(derived_metrics.get('macro_f1'))} | `predictions` vs `targets` |",
            f"| AUROC | {_stringify(derived_metrics.get('auroc', 'Not derived'))} | `confidences[:, 1]` vs `targets` |",
        ])
        for class_name, class_metrics in (derived_metrics.get("per_class") or {}).items():
            lines.append(
                f"| {class_name}: precision / recall / F1 | "
                f"{_stringify(class_metrics.get('precision'))} / {_stringify(class_metrics.get('recall'))} / {_stringify(class_metrics.get('f1'))} | "
                "`predictions` vs `targets` |"
            )
    return lines


def _confusion_section(confusion_data: Optional[Dict[str, Any]], is_training_run: bool = False) -> List[str]:
    if not isinstance(confusion_data, dict):
        if is_training_run:
            return ["Confusion matrix is recorded during evaluation runs (`training/test.py` or `training/validate.py`)."]
        return ["Confusion matrix JSON was not found."]
    class_names = confusion_data.get("class_names") or []
    matrix = confusion_data.get("matrix") or []
    lines = [
        f"- Recorded class ordering: `{_stringify(class_names)}`",
        f"- Recorded matrix: `{_stringify(matrix)}`",
        "",
        "| Actual \\ Predicted | " + " | ".join(class_names) + " |",
        "|---|" + "|".join(["---"] * len(class_names)) + "|",
    ]
    for row_name, row in zip(class_names, matrix):
        lines.append("| " + str(row_name) + " | " + " | ".join(_stringify(v) for v in row) + " |")
    return lines


def _prediction_analysis_section(
    predictions: Optional[Dict[str, Any]],
    derived_metrics: Dict[str, Any],
    is_training_run: bool = False,
) -> List[str]:
    if not isinstance(predictions, dict):
        if is_training_run:
            return ["Prediction artifacts are recorded during evaluation runs (`training/test.py`)."]
        return ["Prediction file was not found."]
    lines = [
        f"- Number of predictions: {_stringify(derived_metrics.get('num_predictions', 'Not recorded'))}",
        f"- Number of correct predictions: {_stringify(derived_metrics.get('num_correct', 'Not derived'))}",
        f"- Number of incorrect predictions: {_stringify(derived_metrics.get('num_incorrect', 'Not derived'))}",
        f"- Predicted class distribution: `{_stringify(derived_metrics.get('predicted_distribution', 'Not derived'))}`",
        f"- Actual class distribution in prediction file: `{_stringify(derived_metrics.get('actual_distribution', 'Not derived'))}`",
        f"- Confidence statistics: `{_stringify(derived_metrics.get('confidence_stats', 'Not derived'))}`",
        "",
        "### Per-Class Prediction Behavior",
        "",
    ]
    per_class = derived_metrics.get("per_class") or {}
    if per_class:
        lines.append("| Class | Precision | Recall | F1 | Basis |")
        lines.append("|---|---|---|---|---|")
        for class_name, class_metrics in per_class.items():
            lines.append(
                f"| {class_name} | {_stringify(class_metrics.get('precision'))} | {_stringify(class_metrics.get('recall'))} | {_stringify(class_metrics.get('f1'))} | Independently derived from raw predictions |"
            )
    else:
        lines.append("Per-class derived metrics could not be computed.")
    return lines


def _consistency_section(rows: List[Dict[str, str]]) -> List[str]:
    lines = ["| Source | Metric | Value | Expected/Related Value | Consistent? | Explanation |", "|---|---|---|---|---|---|"]
    for row in rows:
        lines.append(
            f"| {row['Source']} | {row['Metric']} | {row['Value']} | {row['Expected/Related Value']} | {row['Consistent?']} | {row['Explanation']} |"
        )
    return lines


def _interpretation_section(
    derived_metrics: Dict[str, Any],
    manifest: Dict[str, Any],
    checkpoint_manifest: Optional[Dict[str, Any]],
    checkpoint_path: Optional[str],
    is_training_run: bool,
) -> List[str]:
    if is_training_run:
        total_ep = manifest.get("execution", {}).get("total_epochs")
        best_ep = manifest.get("execution", {}).get("best_epoch")
        best_val = manifest.get("execution", {}).get("best_metric")
        return [
            "### Fact",
            "",
            f"- Completed training run with `{total_ep}` epochs.",
            f"- Best checkpoint achieved at epoch index `{best_ep}` with monitored metric value `{best_val}`.",
            f"- Checkpoint saved to `{_display_path(checkpoint_path)}`.",
            f"- W&B run ID: `{manifest.get('wandb_run_id')}`.",
            "",
            "### Interpretation",
            "",
            "- The training session successfully completed and registered its checkpoint and metric lineage.",
            "- Downstream evaluation runs (`training/test.py`) reference this checkpoint directly to establish end-to-end evaluation metrics.",
        ]

    lines = [
        "### Fact",
        "",
        f"- The evaluation run produced `{_stringify(derived_metrics.get('num_predictions', 'Not recorded'))}` predictions.",
        f"- The independently derived accuracy from saved predictions is `{_stringify(derived_metrics.get('accuracy', 'Not derived'))}`.",
        f"- The evaluation manifest references checkpoint `{_display_path(checkpoint_path)}`.",
    ]
    if checkpoint_manifest:
        lines.append(
            f"- The source training manifest for that checkpoint is named `{_stringify(checkpoint_manifest.get('run_name'))}`."
        )
    return lines


def _verdict_section(
    manifest: Dict[str, Any],
    checkpoint_manifest: Optional[Dict[str, Any]],
    consistency_rows: List[Dict[str, str]],
    is_training_run: bool,
    run_metrics: Optional[Dict[str, Any]],
) -> List[str]:
    inconsistent = any(row["Consistent?"] == "NO" for row in consistency_rows)
    if inconsistent:
        return [
            "**INCONCLUSIVE**",
            "",
            "The run contains inconsistencies between artifacts or lineage records.",
        ]

    if is_training_run:
        return [
            "**VALID TRAINING RUN**",
            "",
            "The training run artifacts, checkpoints, metrics history, and lineage tracking are self-consistent.",
        ]

    if run_metrics is not None:
        return [
            "**VALID**",
            "",
            "The evaluation run artifacts, checkpoint lineage, predictions, and recorded test metrics are self-consistent.",
        ]

    return [
        "**VALID WITH CONCERNS**",
        "",
        "The saved artifacts are mostly self-consistent, but recorded test metrics are pending or partial.",
    ]


def _next_actions_section(
    consistency_rows: List[Dict[str, str]],
    is_training_run: bool,
    run_metrics: Optional[Dict[str, Any]],
) -> List[str]:
    if is_training_run:
        return [
            "1. Run `python training/test.py --config <config.yaml> --checkpoint <path_to_best.pt>` to evaluate on held-out test split.",
            "2. Verify evaluation artifact lineage records the exact checkpoint and training W&B run ID.",
        ]

    return [
        "1. Verify recorded test metrics and confusion matrix in the generated experiment report.",
        "2. Check consistency with baseline experiments (e.g. E07) under controlled evaluation settings.",
    ]


def _resolve_checkpoint_run_dir(checkpoint_path: Optional[str]) -> Optional[Path]:
    if not checkpoint_path:
        return None
    path = Path(checkpoint_path)
    try:
        candidate = path.parent.parent
        if (candidate / "manifest.json").is_file():
            return candidate
    except Exception:
        pass
    if path.is_file():
        try:
            return path.parent.parent
        except IndexError:
            return None
    return None


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _display_path(path: Any) -> str:
    if not path:
        return "Not recorded"
    return str(path)


def _stringify(value: Any) -> str:
    if value is None:
        return "Not recorded"
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return str(value)
        return repr(value)
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)

