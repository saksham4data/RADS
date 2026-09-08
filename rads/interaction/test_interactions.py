import os
import json
import argparse
from pathlib import Path

from rads.pipeline.pipeline import Pipeline

def run_tests(pipeline_config: str, positive_dir: str, negative_dir: str):
    pipeline = Pipeline(pipeline_config)
    
    pos_files = list(Path(positive_dir).glob("*.mp4"))[:3]  # Just take a few for validation
    neg_files = list(Path(negative_dir).glob("*.mp4"))[:3]
    
    print(f"Validating against {len(pos_files)} positive and {len(neg_files)} negative clips.")
    
    for label, files in [("POSITIVE", pos_files), ("NEGATIVE", neg_files)]:
        for video_path in files:
            print(f"\n[{label}] Processing {video_path.name}")
            try:
                result = pipeline.run(str(video_path), visualize=False)
                candidates = result.get('interaction_candidates', [])
                print(f"  -> Found {len(candidates)} interaction candidates.")
                for i, c in enumerate(candidates):
                    print(f"    Candidate {i+1}: obj {c['object_a_id']} & obj {c['object_b_id']}")
                    print(f"      Times: {c['start_time']:.2f}s - {c['end_time']:.2f}s (peak: {c['peak_time']:.2f}s)")
                    print(f"      Min Prox: {c['min_distance']:.2f}, Max Rel Vel: {c['max_relative_velocity']:.2f}, Peak IoU: {c['peak_iou']:.2f}")
                    print(f"      Evidence: {c['evidence_list']}")
            except Exception as e:
                print(f"  -> Error processing {video_path.name}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="rads/config/pipeline_config.yaml")
    parser.add_argument("--pos-dir", default="Datasets/processed/picek_sorted/positive/real")
    parser.add_argument("--neg-dir", default="Datasets/processed/picek_sorted/negative/real")
    args = parser.parse_args()
    
    run_tests(args.config, args.pos_dir, args.neg_dir)
