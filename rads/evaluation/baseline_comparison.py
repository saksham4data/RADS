import csv
import sys
import json
import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASELINE_METRICS = REPO_ROOT / "training/outputs/2026-08-31_19-55-16/metrics/test_metrics.json"
DEFAULT_BASELINE_PREDICTIONS = REPO_ROOT / "training/outputs/2026-08-31_19-55-16/predictions/test_predictions.json"
DEFAULT_SPLIT_CSV = REPO_ROOT / "rads/config/p02_test_split.csv"

# Fallback for clean clones: training/outputs/ is gitignored. Values as recorded in
# docs/architecture/MASTER_SPEC.md section 15 and docs/architecture/MVP.md section 15.
SPEC_BASELINE = {
    'accuracy': 0.459,
    'balanced_accuracy': 0.459,
    'macro_f1': 0.407,
    'f1_accident': 0.583,
    'f1_normal': 0.231,
    'auroc': 0.535,
    'confusion_matrix': [[6, 31], [9, 28]],
}


def compute_metrics(y_true, y_pred, y_prob=None):
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    f1_normal = f1_score(y_true, y_pred, pos_label=0, zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    else:
        tn, fp, fn, tp = 0, 0, 0, 0
        fpr, fnr = 0.0, 0.0

    auroc = None
    if y_prob is not None and len(np.unique(y_true)) > 1:
        try:
            auroc = roc_auc_score(y_true, y_prob)
        except ValueError:
            pass

    return {
        'accuracy': acc,
        'balanced_accuracy': bal_acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'macro_f1': macro_f1,
        'f1_accident': f1,
        'f1_normal': f1_normal,
        'confusion_matrix': {'TN': int(tn), 'FP': int(fp), 'FN': int(fn), 'TP': int(tp)},
        'fpr': fpr,
        'fnr': fnr,
        'auroc': auroc
    }


def metrics_from_confusion(cm):
    """cm is [[TN, FP], [FN, TP]] with rows actual normal/accident."""
    tn, fp = int(cm[0][0]), int(cm[0][1])
    fn, tp = int(cm[1][0]), int(cm[1][1])
    return {
        'confusion_matrix': {'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp},
        'precision': tp / (tp + fp) if (tp + fp) > 0 else 0.0,
        'recall': tp / (tp + fn) if (tp + fn) > 0 else 0.0,
        'fpr': fp / (fp + tn) if (fp + tn) > 0 else 0.0,
        'fnr': fn / (fn + tp) if (fn + tp) > 0 else 0.0,
    }


def load_baseline_metrics(path=None):
    """Return (metrics, source_description). Falls back to the spec-recorded values."""
    candidate = Path(path) if path else DEFAULT_BASELINE_METRICS
    if candidate.exists():
        with open(candidate, 'r') as f:
            raw = json.load(f)
        derived = metrics_from_confusion(raw['test/confusion_matrix'])
        metrics = {
            'accuracy': raw['test/top1_accuracy'],
            'balanced_accuracy': raw['test/balanced_accuracy'],
            'precision': raw['test/precision_accident'],
            'recall': raw['test/recall_accident'],
            'f1': raw['test/f1_accident'],
            'macro_f1': raw['test/macro_f1'],
            'f1_accident': raw['test/f1_accident'],
            'f1_normal': raw['test/f1_normal'],
            'confusion_matrix': derived['confusion_matrix'],
            'fpr': derived['fpr'],
            'fnr': derived['fnr'],
            'auroc': raw['test/auroc'],
        }
        source = f"recorded artifact {candidate.as_posix()}"
        return metrics, source

    derived = metrics_from_confusion(SPEC_BASELINE['confusion_matrix'])
    metrics = {
        'accuracy': SPEC_BASELINE['accuracy'],
        'balanced_accuracy': SPEC_BASELINE['balanced_accuracy'],
        'precision': derived['precision'],
        'recall': derived['recall'],
        'f1': SPEC_BASELINE['f1_accident'],
        'macro_f1': SPEC_BASELINE['macro_f1'],
        'f1_accident': SPEC_BASELINE['f1_accident'],
        'f1_normal': SPEC_BASELINE['f1_normal'],
        'confusion_matrix': derived['confusion_matrix'],
        'fpr': derived['fpr'],
        'fnr': derived['fnr'],
        'auroc': SPEC_BASELINE['auroc'],
    }
    source = (
        "spec fallback (MASTER_SPEC section 15 / MVP section 15); "
        f"artifact {candidate.as_posix()} not present. "
        "Precision, recall, FPR and FNR derived from the spec confusion matrix"
    )
    return metrics, source


def load_split_labels(path):
    labels = []
    with open(path, 'r', newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            labels.append(int(row['binary_label']))
    return labels


def check_alignment(predictions_path, split_csv):
    """Assert baseline targets are index-aligned with the split CSV. Returns a status string."""
    pred_path = Path(predictions_path)
    split_path = Path(split_csv)
    if not pred_path.exists():
        return f"SKIPPED: per-video baseline predictions not found at {pred_path.as_posix()}"
    if not split_path.exists():
        return f"SKIPPED: split CSV not found at {split_path.as_posix()}"

    with open(pred_path, 'r') as f:
        targets = json.load(f)['targets']
    labels = load_split_labels(split_path)

    if len(targets) != len(labels):
        raise SystemExit(
            f"Alignment check failed: {len(targets)} baseline targets vs {len(labels)} split rows "
            f"({pred_path.as_posix()} vs {split_path.as_posix()})"
        )
    mismatches = [i for i, (t, l) in enumerate(zip(targets, labels)) if t != l]
    if mismatches:
        raise SystemExit(
            f"Alignment check failed: {len(mismatches)} of {len(labels)} rows disagree, "
            f"first at index {mismatches[0]} (baseline target {targets[mismatches[0]]}, "
            f"split binary_label {labels[mismatches[0]]}). Refusing to print a misaligned table."
        )
    return f"PASSED: {len(labels)}/{len(labels)} rows, baseline targets equal split binary_label"


def load_rads_results(path):
    """Return (y_true, y_pred, y_prob, report). Rows with an error status or no prediction are skipped."""
    y_true, y_pred, y_prob = [], [], []
    total = 0
    skipped = {'error_status': 0, 'missing_prediction': 0, 'missing_ground_truth': 0, 'unparseable': 0}

    with open(path, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            total += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                skipped['unparseable'] += 1
                continue
            if str(record.get('status', '')).lower() == 'error' or record.get('error'):
                skipped['error_status'] += 1
                continue
            if record.get('prediction') is None:
                skipped['missing_prediction'] += 1
                continue
            if record.get('ground_truth') is None:
                skipped['missing_ground_truth'] += 1
                continue
            y_true.append(int(record['ground_truth']))
            y_pred.append(int(record['prediction']))
            y_prob.append(record.get('confidence'))

    if any(p is None for p in y_prob):
        y_prob = None

    report = {'rows_read': total, 'rows_used': len(y_true), 'skipped': skipped,
              'skipped_total': sum(skipped.values())}
    return y_true, y_pred, y_prob, report


def fmt(val):
    if val is None:
        return "N/A"
    return f"{val:.4f}" if isinstance(val, float) else str(val)


METRIC_ROWS = [
    ('Accuracy', 'accuracy'),
    ('Balanced Acc', 'balanced_accuracy'),
    ('Precision', 'precision'),
    ('Recall', 'recall'),
    ('F1 Score', 'f1'),
    ('Macro F1', 'macro_f1'),
    ('Accident F1', 'f1_accident'),
    ('Normal F1', 'f1_normal'),
    ('FPR', 'fpr'),
    ('FNR', 'fnr'),
    ('AUROC', 'auroc'),
]


def render_text(rads, baseline, ctx):
    rads_col = f"RADS ({ctx['rads_label']})"
    lines = []
    lines.append("=" * 64)
    lines.append(f"P02 test split comparison ({ctx['n_rads']} RADS rows, {ctx['n_baseline']} baseline rows)")
    lines.append(f"RADS results:    {ctx['jsonl']}  [{ctx['rads_label']}]")
    lines.append(f"Baseline source: {ctx['baseline_source']}")
    lines.append(f"Alignment check: {ctx['alignment']}")
    if ctx['note']:
        lines.append(f"Note: {ctx['note']}")
    lines.append(f"JSONL rows read: {ctx['report']['rows_read']}, used: {ctx['report']['rows_used']}, "
                 f"skipped: {ctx['report']['skipped_total']} {ctx['report']['skipped']}")
    lines.append("=" * 64)
    lines.append(f"{'Metric':<16} | {rads_col:<22} | {ctx['baseline_name']:<15}")
    lines.append("-" * 64)
    for name, key in METRIC_ROWS:
        lines.append(f"{name:<16} | {fmt(rads.get(key)):<22} | {fmt(baseline.get(key)):<15}")
    lines.append("=" * 64)
    for label, m in ((f"RADS ({ctx['rads_label']})", rads), (ctx['baseline_name'], baseline)):
        cm = m['confusion_matrix']
        lines.append(f"{label} confusion matrix:")
        lines.append(f"  TN: {cm['TN']:<5} FP: {cm['FP']:<5}")
        lines.append(f"  FN: {cm['FN']:<5} TP: {cm['TP']:<5}")
    lines.append("=" * 64)
    return "\n".join(lines)


def render_markdown(rads, baseline, ctx):
    rads_col = f"RADS ({ctx['rads_label']})"
    lines = []
    lines.append("# P02 baseline comparison")
    lines.append("")
    lines.append(f"RADS results file: `{ctx['jsonl']}`")
    lines.append(f"RADS run label: {ctx['rads_label']}")
    lines.append(f"Baseline: {ctx['baseline_name']}")
    lines.append(f"Baseline source: {ctx['baseline_source']}")
    lines.append(f"Alignment check (baseline targets vs `p02_test_split.csv` binary_label): {ctx['alignment']}")
    lines.append(f"JSONL rows read: {ctx['report']['rows_read']}; used: {ctx['report']['rows_used']}; "
                 f"skipped: {ctx['report']['skipped_total']} ({ctx['report']['skipped']})")
    lines.append(f"Compared on {ctx['n_rads']} RADS rows and {ctx['n_baseline']} baseline rows.")
    if ctx['note']:
        lines.append("")
        lines.append(ctx['note'])
    lines.append("")
    lines.append(f"| Metric | {rads_col} | {ctx['baseline_name']} |")
    lines.append("|---|---|---|")
    for name, key in METRIC_ROWS:
        lines.append(f"| {name} | {fmt(rads.get(key))} | {fmt(baseline.get(key))} |")
    lines.append("")
    lines.append("## Confusion matrices")
    lines.append("")
    lines.append("| System | TN | FP | FN | TP |")
    lines.append("|---|---|---|---|---|")
    for label, m in ((rads_col, rads), (ctx['baseline_name'], baseline)):
        cm = m['confusion_matrix']
        lines.append(f"| {label} | {cm['TN']} | {cm['FP']} | {cm['FN']} | {cm['TP']} |")
    lines.append("")
    lines.append("## Sources")
    lines.append("")
    lines.append("- Baseline metrics: `training/outputs/2026-08-31_19-55-16/metrics/test_metrics.json`. "
                 "That path is gitignored, so on a clean clone the script falls back to the values recorded in "
                 "`docs/architecture/MASTER_SPEC.md` section 15 and `docs/architecture/MVP.md` section 15. "
                 "The source actually used is stated above.")
    lines.append("- Per-video baseline predictions for the alignment check: "
                 "`training/outputs/2026-08-31_19-55-16/predictions/test_predictions.json` (also gitignored).")
    lines.append("- Correction: `docs/architecture/IMPLEMENTATION_AUDIT_AND_ORDER.md` line 504 names "
                 "`training/p02_prediction.json` as the P02 results source. That file holds a single entry for "
                 "`Datasets/processed/picek_sorted/trimmed/positive/real/jD8ybdMZOU8_00.mp4` and is a single-video "
                 "demo dump, not test-set predictions. It is not used here.")
    lines.append("- Metric set follows the comparison protocol in "
                 "`docs/architecture/IMPLEMENTATION_AUDIT_AND_ORDER.md` section 5 and MVP sections 17, 18 and 24.")
    lines.append("")
    lines.append(f"Generated by: `{ctx['command']}`")
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Compare RADS evaluation results against the P02 baseline.")
    parser.add_argument("--jsonl", type=str, required=True, help="Path to RADS evaluator JSONL output")
    parser.add_argument("--baseline-name", type=str, default="ResNet18+GRU", help="Name of the baseline to compare against")
    parser.add_argument("--baseline-metrics", type=str, default=None,
                        help=f"Baseline metrics JSON (default: {DEFAULT_BASELINE_METRICS.as_posix()})")
    parser.add_argument("--baseline-predictions", type=str, default=str(DEFAULT_BASELINE_PREDICTIONS),
                        help="Per-video baseline predictions JSON, used for the index-alignment check")
    parser.add_argument("--split-csv", type=str, default=str(DEFAULT_SPLIT_CSV),
                        help="Test split CSV providing binary_label in file order")
    parser.add_argument("--rads-label", type=str, default="unlabelled",
                        help="Provenance label for the RADS column, for example PRE-FIX or POST-FIX")
    parser.add_argument("--note", type=str, default=None,
                        help="Provenance sentence recorded in the printed output and the markdown artifact")
    parser.add_argument("--out", type=str, default=None,
                        help="Write the same comparison as markdown to this path")
    args = parser.parse_args()

    alignment = check_alignment(args.baseline_predictions, args.split_csv)
    baseline, baseline_source = load_baseline_metrics(args.baseline_metrics)
    y_true, y_pred, y_prob, report = load_rads_results(args.jsonl)

    if not y_true:
        print("No usable predictions found in the JSONL file.")
        return

    rads = compute_metrics(y_true, y_pred, y_prob)
    cm = baseline['confusion_matrix']
    ctx = {
        'jsonl': Path(args.jsonl).as_posix(),
        'rads_label': args.rads_label,
        'baseline_name': args.baseline_name,
        'baseline_source': baseline_source,
        'alignment': alignment,
        'note': args.note,
        'command': " ".join(
            f'"{a}"' if " " in a else a for a in [Path(sys.argv[0]).name] + sys.argv[1:]
        ),
        'report': report,
        'n_rads': len(y_true),
        'n_baseline': cm['TN'] + cm['FP'] + cm['FN'] + cm['TP'],
    }

    print(render_text(rads, baseline, ctx))

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(render_markdown(rads, baseline, ctx), encoding='utf-8')
        print(f"Wrote {out_path.as_posix()}")


if __name__ == "__main__":
    main()
