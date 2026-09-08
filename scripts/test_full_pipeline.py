import argparse
import os
import json
from rads.pipeline.pipeline import Pipeline

def run_test(video_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    
    base = os.path.basename(video_path)
    name, ext = os.path.splitext(base)
    output_video_path = os.path.join(output_dir, f"{name}_annotated{ext}")
    output_json_path = os.path.join(output_dir, f"{name}_result.json")
    
    pipeline = Pipeline("rads/config/pipeline_config.yaml")
    result = pipeline.run(video_path, visualize=True, output_video_path=output_video_path)
    
    with open(output_json_path, 'w') as f:
        json.dump(result, f, indent=2)
        
    print(f"Result for {base}:")
    print(f"  Accident: {result.get('accident')}")
    print(f"  Confidence: {result.get('confidence')}")
    print(f"  Severity: {result.get('severity')}")
    print(f"  Event Window: {result.get('event')}")
    
    # Check if evidence clip was created
    evidence_clip_path = output_video_path.replace(".mp4", "_evidence.mp4")
    if os.path.exists(evidence_clip_path):
        print(f"  Evidence clip generated: {evidence_clip_path}")
    else:
        print(f"  No evidence clip generated.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="Datasets/processed/picek_sorted/positive/real/-6SQSDj8cYU_00.mp4")
    parser.add_argument("--out", default="output/tests")
    args = parser.parse_args()
    
    run_test(args.video, args.out)
