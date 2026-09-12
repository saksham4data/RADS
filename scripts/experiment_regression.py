import os
import json
import time
import glob
from pathlib import Path
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rads.pipeline.pipeline import Pipeline

def run_regression_test():
    output_dir = Path("rads/evaluation/experiments/track_loss_reasoning_fix")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all positive and negative videos
    all_pos_files = sorted(glob.glob(r"e:\Rads\Datasets\processed\picek_sorted\trimmed\positive\real\*.mp4"))
    all_neg_files = sorted(glob.glob(r"e:\Rads\Datasets\processed\picek_sorted\trimmed\negative\real\*.mp4"))
    
    # Original 5 pos and 5 neg were the first 5 in the list (index 0 to 4)
    # For the regression test, we select the NEXT 5 positive and NEXT 5 negative (index 5 to 9)
    new_pos_files = all_pos_files[5:10]
    new_neg_files = all_neg_files[5:10]
    
    videos = [(f, "accident") for f in new_pos_files] + [(f, "normal") for f in new_neg_files]
    config_path = "rads/config/pipeline_config.yaml"
    
    results = {}
    total_time = 0
    
    for video_path, gt_label in videos:
        video_name = os.path.basename(video_path)
        print(f"\n{'='*50}\nTesting {video_name} (GT: {gt_label})\n{'='*50}")
        
        results[video_path] = {"gt": gt_label}
        
        # Only run with frame_skip=1
        pipeline = Pipeline(config_path)
        pipeline.config.config['pipeline']['frame_skip'] = 1
        
        start_time = time.perf_counter()
        res = pipeline.run(video_path, visualize=False)
        runtime = time.perf_counter() - start_time
        
        total_time += runtime
            
        is_accident = res.get("accident", False)
        conf = res.get("confidence", 0.0)
        
        event_time = "N/A"
        if is_accident and res.get("event") and res["event"].get("impact_time") is not None:
            event_time = f"{res['event']['impact_time']:.2f}s"
        
        results[video_path]["skip_1"] = {
            "prediction": "accident" if is_accident else "normal",
            "confidence": f"{conf:.2f}",
            "event_time": event_time,
            "runtime": f"{runtime:.2f}s"
        }
            
    # Generate Markdown Summary
    md_lines = [
        "# Reasoning Improvement Regression Test",
        "",
        "## New 10-Video Sample (frame_skip=1)",
        "",
        "| Video | Ground Truth | Prediction | Confidence | Event Time | Runtime |",
        "|---|---|---|---|---|---|"
    ]
    
    correct = 0
    fp = 0
    fn = 0
    
    for video_path, gt_label in videos:
        vname = os.path.basename(video_path)
        d = results[video_path]
        s1 = d["skip_1"]
        
        gt = d["gt"]
        p1 = s1["prediction"]
        
        if p1 == gt: correct += 1
        if p1 == "accident" and gt == "normal": fp += 1
        if p1 == "normal" and gt == "accident": fn += 1
            
        md_lines.append(f"| {vname} | {gt} | {p1} | {s1['confidence']} | {s1['event_time']} | {s1['runtime']} |")
        
    md_lines.extend([
        "",
        "## Summary Statistics",
        f"- **Total Runtime:** {total_time:.2f}s",
        f"- **Correct Predictions:** {correct}/10",
        f"- **False Positives:** {fp}",
        f"- **False Negatives:** {fn}",
    ])
        
    with open(output_dir / "regression_summary.md", "w") as f:
        f.write("\n".join(md_lines))
        
    print(f"\nRegression test complete. Check {output_dir}")

if __name__ == "__main__":
    run_regression_test()
