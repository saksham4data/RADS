"""Re-runs the 10-clip sample from experiments/track_loss_reasoning_fix/regression_summary.md
at frame_skip=1, once with the B11/B12 flags enabled and once with them disabled.

One Pipeline instance is reused across all clips, matching how evaluator.py runs a split, so
the B1 tracker reset is exercised.
"""

import argparse
import json
import os
import sys
import time

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
sys.path.insert(0, REPO_ROOT)

from rads.pipeline.pipeline import Pipeline

CONFIG_PATH = os.path.join(REPO_ROOT, 'rads', 'config', 'pipeline_config.yaml')
POSITIVE_DIR = os.path.join(REPO_ROOT, 'Datasets', 'processed', 'picek_sorted', 'trimmed', 'positive', 'real')
NEGATIVE_DIR = os.path.join(REPO_ROOT, 'Datasets', 'processed', 'picek_sorted', 'trimmed', 'negative', 'real')

# (clip, ground_truth_label, recorded_baseline_prediction) from regression_summary.md
SAMPLE = [
    ('-FQxK6HdxNU_00.mp4', 'accident', 'normal'),
    ('-NgnSm_oEB4_00.mp4', 'accident', 'accident'),
    ('-PpBteU0p3Q_00.mp4', 'accident', 'accident'),
    ('-PpjzmhI_PE_00.mp4', 'accident', 'accident'),
    ('-Qt5bDJNT84_00.mp4', 'accident', 'accident'),
    ('-NgnSm_oEB4_00.mp4', 'normal', 'accident'),
    ('-PpBteU0p3Q_00.mp4', 'normal', 'accident'),
    ('-RE3XseZINA_00.mp4', 'normal', 'normal'),
    ('-RrDtLjWsT4_00.mp4', 'normal', 'normal'),
    ('-SNFUobKjoM_00.mp4', 'normal', 'accident'),
]

def clip_path(clip, ground_truth):
    return os.path.join(POSITIVE_DIR if ground_truth == 'accident' else NEGATIVE_DIR, clip)

def run_configuration(flags_enabled: bool):
    pipeline = Pipeline(CONFIG_PATH)
    pipeline.config.config['pipeline']['frame_skip'] = 1
    pipeline.config.config.setdefault('reasoning', {})
    pipeline.config.config['reasoning'].setdefault('clustering', {})['require_own_evidence'] = flags_enabled
    pipeline.config.config['reasoning'].setdefault('confidence', {})['bounded_transform'] = flags_enabled

    rows = []
    for clip, ground_truth, baseline_prediction in SAMPLE:
        path = clip_path(clip, ground_truth)
        if not os.path.exists(path):
            raise FileNotFoundError(path)

        start = time.perf_counter()
        result = pipeline.run(path, visualize=False)
        runtime = time.perf_counter() - start

        prediction = 'accident' if result.get('accident') else 'normal'
        rows.append({
            'clip': clip,
            'ground_truth': ground_truth,
            'baseline_prediction': baseline_prediction,
            'prediction': prediction,
            'correct': prediction == ground_truth,
            'matches_baseline': prediction == baseline_prediction,
            'confidence': result.get('confidence'),
            'score': result.get('score'),
            'severity': result.get('severity'),
            'severity_score': (result.get('severity_detail') or {}).get('score'),
            'severity_evidence': (result.get('severity_detail') or {}).get('evidence', []),
            'impact_time': (result.get('event') or {}).get('impact_time'),
            'evidence_list': result.get('evidence_list', []),
            'objects_involved': result.get('objects_involved', []),
            'interaction_candidate_count': len(result.get('interaction_candidates', [])),
            'num_tracks': len(result.get('tracks_summary', {})),
            'kinematics': result.get('kinematics', {}),
            'runtime_seconds': round(runtime, 2),
        })
    return rows

def summarize(rows):
    correct = sum(1 for r in rows if r['correct'])
    false_positives = sum(1 for r in rows if r['ground_truth'] == 'normal' and r['prediction'] == 'accident')
    false_negatives = sum(1 for r in rows if r['ground_truth'] == 'accident' and r['prediction'] == 'normal')
    baseline_matches = sum(1 for r in rows if r['matches_baseline'])
    return {
        'clips': len(rows),
        'correct': correct,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'label_matches_baseline': baseline_matches,
        'identical_to_baseline': baseline_matches == len(rows),
        'total_runtime_seconds': round(sum(r['runtime_seconds'] for r in rows), 2),
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default=os.path.join(os.path.dirname(__file__), 'gate_results.json'))
    args = parser.parse_args()

    payload = {
        'frame_skip': 1,
        'config_path': os.path.relpath(CONFIG_PATH, REPO_ROOT).replace('\\', '/'),
        'recorded_baseline': {'correct': 6, 'false_positives': 3, 'false_negatives': 1},
        'acceptance': {'min_correct': 6, 'max_false_positives': 3},
        'configurations': {}
    }

    for name, flags_enabled in (('flags_enabled', True), ('flags_disabled', False)):
        print(f"=== configuration: {name} ===")
        rows = run_configuration(flags_enabled)
        payload['configurations'][name] = {
            'b11_require_own_evidence': flags_enabled,
            'b12_bounded_confidence': flags_enabled,
            'summary': summarize(rows),
            'rows': rows,
        }

    enabled = payload['configurations']['flags_enabled']['summary']
    payload['gate_passed'] = (enabled['correct'] >= 6 and enabled['false_positives'] <= 3)
    payload['behaviour_preserving'] = payload['configurations']['flags_disabled']['summary']['identical_to_baseline']

    with open(args.out, 'w') as f:
        json.dump(payload, f, indent=2)

    print(json.dumps({
        'flags_enabled': enabled,
        'flags_disabled': payload['configurations']['flags_disabled']['summary'],
        'gate_passed': payload['gate_passed'],
        'behaviour_preserving': payload['behaviour_preserving'],
    }, indent=2))

if __name__ == '__main__':
    main()
