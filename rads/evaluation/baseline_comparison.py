import json
import argparse
import numpy as np
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score
)

def compute_metrics(y_true, y_pred, y_prob=None):
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
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
        'confusion_matrix': {'TN': int(tn), 'FP': int(fp), 'FN': int(fn), 'TP': int(tp)},
        'fpr': fpr,
        'fnr': fnr,
        'auroc': auroc
    }

def print_comparison_table(rads_metrics, baseline_name="ResNet18+GRU"):
    print("="*60)
    print(f"{'Metric':<20} | {'RADS Pipeline':<15} | {baseline_name:<15}")
    print("-" * 60)
    
    def fmt(val):
        return f"{val:.4f}" if isinstance(val, float) else str(val)
        
    metrics = [
        ('Accuracy', rads_metrics['accuracy']),
        ('Balanced Acc', rads_metrics['balanced_accuracy']),
        ('Precision', rads_metrics['precision']),
        ('Recall', rads_metrics['recall']),
        ('F1 Score', rads_metrics['f1']),
        ('Macro F1', rads_metrics['macro_f1']),
        ('FPR', rads_metrics['fpr']),
        ('FNR', rads_metrics['fnr']),
        ('AUROC', rads_metrics['auroc'] if rads_metrics['auroc'] is not None else "N/A")
    ]
    
    for name, rad_val in metrics:
        # Placeholder 'TBD' for baseline as we don't have the exact numbers loaded here
        print(f"{name:<20} | {fmt(rad_val):<15} | {'TBD':<15}")
        
    print("="*60)
    cm = rads_metrics['confusion_matrix']
    print(f"RADS Confusion Matrix:")
    print(f"  TN: {cm['TN']:<5} FP: {cm['FP']:<5}")
    print(f"  FN: {cm['FN']:<5} TP: {cm['TP']:<5}")
    print("="*60)

def main():
    parser = argparse.ArgumentParser(description="Compare RADS evaluation results against baseline.")
    parser.add_argument("--jsonl", type=str, required=True, help="Path to RADS evaluator JSONL output")
    parser.add_argument("--baseline-name", type=str, default="ResNet18+GRU", help="Name of the baseline to compare against")
    args = parser.parse_args()
    
    y_true = []
    y_pred = []
    y_prob = []
    
    with open(args.jsonl, 'r') as f:
        for line in f:
            if not line.strip(): continue
            record = json.loads(line)
            y_true.append(record['ground_truth'])
            y_pred.append(record['prediction'])
            y_prob.append(record['confidence'])
            
    if not y_true:
        print("No predictions found in the JSONL file.")
        return
        
    metrics = compute_metrics(y_true, y_pred, y_prob)
    print_comparison_table(metrics, args.baseline_name)

if __name__ == "__main__":
    main()
