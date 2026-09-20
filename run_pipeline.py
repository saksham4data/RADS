import argparse
import json
import os
import sys

from rads.pipeline.pipeline import Pipeline

def _ensure_parent_dir(path: str):
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)

def main():
    parser = argparse.ArgumentParser(description="Run RADS Pipeline")
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument("--config", required=True, help="Path to pipeline configuration YAML")
    parser.add_argument("--output", required=True, help="Path to save result JSON")
    parser.add_argument("--visualize", action="store_true", help="Generate an annotated output video")
    parser.add_argument("--output-video", type=str, default="output_videos/output.mp4", help="Path to save the annotated video (if --visualize is set)")
    
    args = parser.parse_args()
    
    try:
        # cv2.VideoWriter fails silently if the directory is missing, so create both up front
        _ensure_parent_dir(args.output)
        if args.visualize:
            _ensure_parent_dir(args.output_video)
        
        # Initialize and run pipeline
        pipeline = Pipeline(args.config)
        result = pipeline.run(args.video, visualize=args.visualize, output_video_path=args.output_video)
        
        # Save output
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=4)
            
        print(f"Successfully wrote results to {args.output}")
        if args.visualize:
            print(f"Successfully wrote annotated video to {args.output_video}")
            
    except Exception as e:
        print(f"Error executing pipeline: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
