import os
import json
import pandas as pd
from tqdm import tqdm
import argparse

import sys
# Ensure rads is in the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from rads.pipeline.pipeline import Pipeline

def evaluate_dataset(csv_path: str, output_jsonl: str, config_path: str):
    """
    Evaluates the RADS pipeline on a dataset specified in a CSV file.
    Saves predictions to a JSONL file and supports resuming from interrupted runs.
    """
    print(f"Loading split from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Initialize the pipeline
    print(f"Initializing pipeline with config {config_path}...")
    pipeline = Pipeline(config_path)
    # Ensure we use frame_skip=1 per Phase 8 instructions
    pipeline.config.config['pipeline']['frame_skip'] = 1
    
    # Checkpoint recovery: read existing processed videos
    processed_videos = set()
    if os.path.exists(output_jsonl):
        print(f"Found existing output at {output_jsonl}. Recovering checkpoint...")
        with open(output_jsonl, 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        record = json.loads(line)
                        processed_videos.add(record['video_id'])
                    except json.JSONDecodeError:
                        continue
        print(f"Recovered {len(processed_videos)} previously processed videos.")
    
    # Filter the dataframe
    videos_to_process = []
    for _, row in df.iterrows():
        vid_id = row['video_id']
        if vid_id not in processed_videos:
            videos_to_process.append(row)
            
    if not videos_to_process:
        print("All videos in the dataset have already been processed.")
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
                    print(f"\nWARNING: Video not found at {video_path}, skipping.")
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
                        "prediction": 1 if result.get("accident", False) else 0,
                        "confidence": result.get("confidence", 0.0),
                        "severity": result.get("severity", "unknown"),
                        "event": result.get("event", {}),
                        "num_tracks": len(result.get("track_summaries", {}))
                    }
                    
                    # Write to JSONL
                    f_out.write(json.dumps(record) + '\n')
                    f_out.flush() # Ensure it's written immediately for safe checkpointing
                    
                except Exception as e:
                    print(f"\nError processing {vid_id}: {e}")
                
                pbar.update(1)
                
    print("\nEvaluation complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RADS Phase 8 Evaluator")
    parser.add_argument("--csv", type=str, required=True, help="Path to the dataset CSV split")
    parser.add_argument("--output", type=str, required=True, help="Path to the output JSONL file")
    parser.add_argument("--config", type=str, default="rads/config/pipeline_config.yaml", help="Path to the pipeline config")
    
    args = parser.parse_args()
    evaluate_dataset(args.csv, args.output, args.config)
