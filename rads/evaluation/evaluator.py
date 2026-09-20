import os
import re
import json
import pandas as pd
from tqdm import tqdm
import argparse

import sys
# Ensure rads is in the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from rads.pipeline.pipeline import Pipeline

def evaluate_dataset(csv_path: str, output_jsonl: str, config_path: str, dump_dir: str = None):
    """
    Evaluates the RADS pipeline on a dataset specified in a CSV file.
    Saves predictions to a JSONL file and supports resuming from interrupted runs.

    dump_dir, when given, also writes the full Pipeline.run() dict to
    <dump_dir>/<video_id>.json per successful video. The JSONL row is a flat summary and
    drops objects_involved, severity_detail, kinematics, tracks_summary and
    interaction_candidates, which downstream consumers need.
    """
    print(f"Loading split from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Initialize the pipeline
    print(f"Initializing pipeline with config {config_path}...")
    pipeline = Pipeline(config_path)
    # Ensure we use frame_skip=1 per Phase 8 instructions
    pipeline.config.config['pipeline']['frame_skip'] = 1

    if dump_dir:
        os.makedirs(dump_dir, exist_ok=True)
        print(f"Per-clip full results will be written to {dump_dir}")
    
    # Checkpoint recovery: read existing processed videos
    processed_videos = set()
    if os.path.exists(output_jsonl):
        print(f"Found existing output at {output_jsonl}. Recovering checkpoint...")
        with open(output_jsonl, 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    # Error rows are not a result: leave them out so the video is retried
                    if record.get('status', 'ok') == 'ok':
                        processed_videos.add(record['video_id'])
        print(f"Recovered {len(processed_videos)} previously processed videos.")
    
    # Filter the dataframe
    videos_to_process = []
    for _, row in df.iterrows():
        vid_id = row['video_id']
        if vid_id not in processed_videos:
            videos_to_process.append(row)
            
    if not videos_to_process:
        print("All videos in the dataset have already been processed.")
        _report_completeness(output_jsonl, len(df))
        return

    print(f"Starting evaluation on {len(videos_to_process)} remaining videos...")
    
    # Open the file in append mode
    with open(output_jsonl, 'a') as f_out:
        # Use tqdm for progress tracking
        with tqdm(total=len(videos_to_process), desc="Evaluating") as pbar:
            for row in videos_to_process:
                vid_id = row['video_id']
                # P02 metadata has 'original_path' or 'processed_path' depending on the CSV
                # For the new splits, we used 'processed_path' which is absolute or relative
                video_path = row['processed_path']
                
                # Ensure path is absolute if it's relative
                if not os.path.isabs(video_path):
                    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
                    if video_path.startswith('processed'):
                        video_path = os.path.join(base_dir, 'Datasets', video_path)
                    else:
                        video_path = os.path.join(base_dir, video_path)
                
                if not os.path.exists(video_path):
                    print(f"\nWARNING: Video not found at {video_path}")
                    _write_record(f_out, {
                        "video_id": vid_id,
                        "ground_truth": int(row['binary_label']),
                        "status": "error",
                        "error": f"video not found: {video_path}"
                    })
                    pbar.update(1)
                    continue
                
                try:
                    # Run the pipeline (disable visualization for evaluation)
                    # We capture stdout to avoid cluttering tqdm if needed, but for now we let it print
                    result = pipeline.run(video_path, visualize=False)
                    
                    # Create the JSON record
                    record = {
                        "video_id": vid_id,
                        "ground_truth": int(row['binary_label']),
                        "status": "ok",
                        "prediction": 1 if result.get("accident", False) else 0,
                        "confidence": result.get("confidence", 0.0),
                        "score": result.get("score"),
                        "severity": result.get("severity", "unknown"),
                        "severity_score": (result.get("severity_detail") or {}).get("score"),
                        "event": result.get("event", {}),
                        "evidence_list": result.get("evidence_list", []),
                        "interaction_candidate_count": len(result.get("interaction_candidates", [])),
                        "num_tracks": len(result.get("tracks_summary", {}))
                    }
                    
                    _write_record(f_out, record)

                    if dump_dir:
                        _dump_full_result(dump_dir, vid_id, result)

                except Exception as e:
                    print(f"\nError processing {vid_id}: {e}")
                    _write_record(f_out, {
                        "video_id": vid_id,
                        "ground_truth": int(row['binary_label']),
                        "status": "error",
                        "error": f"{type(e).__name__}: {e}"
                    })
                
                pbar.update(1)
                
    print("\nEvaluation complete.")
    _report_completeness(output_jsonl, len(df))

def _write_record(f_out, record: dict):
    """Appends one JSONL row and flushes, so a killed run still has a usable checkpoint."""
    f_out.write(json.dumps(record) + '\n')
    f_out.flush()

def _dump_full_result(dump_dir: str, video_id: str, result: dict):
    """Writes the complete pipeline result for one video. Filename is derived from video_id,
    which is unique per split, so dumps cannot collide."""
    safe_id = re.sub(r'[^A-Za-z0-9._-]', '_', str(video_id))
    path = os.path.join(dump_dir, f"{safe_id}.json")
    with open(path, 'w') as f:
        json.dump(result, f, default=str)

def _report_completeness(output_jsonl: str, expected_rows: int):
    """Compares successful rows against the CSV row count and fails loudly on a shortfall."""
    ok_ids = set()
    error_ids = []
    if os.path.exists(output_jsonl):
        with open(output_jsonl, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                # Rows written before B5 carry no status field and were all successful
                if record.get('status', 'ok') == 'ok':
                    ok_ids.add(record.get('video_id'))
                else:
                    error_ids.append(record.get('video_id'))

    print(f"Completeness: {len(ok_ids)} successful rows / {expected_rows} expected from CSV, "
          f"{len(error_ids)} error rows.")
    if len(ok_ids) != expected_rows:
        print(f"RUN INCOMPLETE: {expected_rows - len(ok_ids)} videos did not produce a successful row.")
        if error_ids:
            print(f"Error rows: {error_ids}")
    else:
        print("Run complete: every CSV row has a successful result.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RADS Phase 8 Evaluator")
    parser.add_argument("--csv", type=str, required=True, help="Path to the dataset CSV split")
    parser.add_argument("--output", type=str, required=True, help="Path to the output JSONL file")
    parser.add_argument("--config", type=str, default="rads/config/pipeline_config.yaml", help="Path to the pipeline config")
    parser.add_argument("--dump-dir", type=str, default=None,
                        help="Optional directory for one full-result JSON per video")
    
    args = parser.parse_args()
    evaluate_dataset(args.csv, args.output, args.config, args.dump_dir)
