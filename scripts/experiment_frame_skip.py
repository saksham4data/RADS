import os
import json
import time
import glob
from pathlib import Path

# Fix python path if running from scripts directory
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rads.pipeline.pipeline import Pipeline

def run_experiment():
    output_dir = Path("rads/evaluation/experiments/frame_skip_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    pos_files = glob.glob(r"e:\Rads\Datasets\processed\picek_sorted\trimmed\positive\real\*.mp4")[:5]
    neg_files = glob.glob(r"e:\Rads\Datasets\processed\picek_sorted\trimmed\negative\real\*.mp4")[:5]
    
    videos = [(f, "accident") for f in pos_files] + [(f, "normal") for f in neg_files]
    config_path = "rads/config/pipeline_config.yaml"
    
    results = {}
    
    # We will measure total runtime explicitly
    total_time_skip1 = 0
    total_time_skip3 = 0
    
    for video_path, gt_label in videos:
        video_name = os.path.basename(video_path)
        print(f"\n{'='*50}\nTesting {video_name} (GT: {gt_label})\n{'='*50}")
        
        results[video_path] = {"gt": gt_label}
        
        for skip in [1, 3]:
            print(f"\n--- Running with frame_skip={skip} ---")
            # We instantiate a fresh pipeline to ensure no state leakage (like tracker state)
            pipeline = Pipeline(config_path)
            pipeline.config.config['pipeline']['frame_skip'] = skip
            
            start_time = time.perf_counter()
            # redirect stdout to avoid massive logs unless we need them? No, let's keep them
            res = pipeline.run(video_path, visualize=False)
            runtime = time.perf_counter() - start_time
            
            if skip == 1:
                total_time_skip1 += runtime
            else:
                total_time_skip3 += runtime
                
            is_accident = res.get("accident", False)
            conf = res.get("confidence", 0.0)
            severity = res.get("severity", "NONE")
            
            event_time = "N/A"
            if is_accident and res.get("event") and res["event"].get("impact_time") is not None:
                event_time = f"{res['event']['impact_time']:.2f}s"
            
            results[video_path][f"skip_{skip}"] = {
                "prediction": "accident" if is_accident else "normal",
                "confidence": f"{conf:.2f}",
                "event_time": event_time,
                "severity": severity,
                "runtime": f"{runtime:.2f}s"
            }
            
            # Save raw results incrementally
            with open(output_dir / "raw_results.json", "w") as f:
                json.dump(results, f, indent=4)
                
    # Generate Markdown Summary
    md_lines = [
        "# Frame Skip Comparison: 1 vs 3",
        "",
        "| Video | Ground Truth | Skip 1 Prediction | Skip 3 Prediction | Skip 1 Event Time | Skip 3 Event Time | Skip 1 Runtime | Skip 3 Runtime | Notes |",
        "|---|---|---|---|---|---|---|---|---|"
    ]
    
    correct_1 = 0
    correct_3 = 0
    fp_1 = 0
    fp_3 = 0
    fn_1 = 0
    fn_3 = 0
    diff_event = 0
    missed_by_3 = 0
    
    for video_path, gt_label in videos:
        vname = os.path.basename(video_path)
        d = results[video_path]
        s1 = d["skip_1"]
        s3 = d["skip_3"]
        
        gt = d["gt"]
        p1 = s1["prediction"]
        p3 = s3["prediction"]
        
        # Stats skip 1
        if p1 == gt: correct_1 += 1
        if p1 == "accident" and gt == "normal": fp_1 += 1
        if p1 == "normal" and gt == "accident": fn_1 += 1
        
        # Stats skip 3
        if p3 == gt: correct_3 += 1
        if p3 == "accident" and gt == "normal": fp_3 += 1
        if p3 == "normal" and gt == "accident": fn_3 += 1
        
        if p1 == "accident" and p3 == "normal": missed_by_3 += 1
        if s1["event_time"] != s3["event_time"] and s1["event_time"] != "N/A" and s3["event_time"] != "N/A":
            diff_event += 1
            
        md_lines.append(f"| {vname} | {gt} | {p1} | {p3} | {s1['event_time']} | {s3['event_time']} | {s1['runtime']} | {s3['runtime']} |  |")
        
    speed_imp = total_time_skip1 / total_time_skip3 if total_time_skip3 > 0 else 0
    
    md_lines.extend([
        "",
        "## Summary Statistics",
        f"- **Total Runtime (Skip=1):** {total_time_skip1:.2f}s",
        f"- **Total Runtime (Skip=3):** {total_time_skip3:.2f}s",
        f"- **Speed Improvement:** {speed_imp:.2f}x faster",
        "",
        f"- **Correct Predictions:** Skip 1 = {correct_1}/10 | Skip 3 = {correct_3}/10",
        f"- **False Positives:** Skip 1 = {fp_1} | Skip 3 = {fp_3}",
        f"- **False Negatives:** Skip 1 = {fn_1} | Skip 3 = {fn_3}",
        f"- **Missed by Skip=3 (detected by Skip=1):** {missed_by_3}",
        f"- **Event Localization Differences:** {diff_event} cases where timing materially differed",
        "",
        "## Recommendation",
        ""
    ])
    
    # Recommendation logic
    if missed_by_3 == 0 and fp_3 <= fp_1 and diff_event <= 1:
        md_lines.append("A. frame_skip=3 is acceptable for the eventual Phase 8 benchmark")
    elif missed_by_3 >= 2 or fn_3 > fn_1:
        md_lines.append("B. frame_skip=3 is risky, so use frame_skip=1")
    else:
        md_lines.append("C. results are inconclusive, so test a larger sample")
        
    with open(output_dir / "comparison_summary.md", "w") as f:
        f.write("\n".join(md_lines))
        
    print(f"\nExperiment complete. Check {output_dir}")

if __name__ == "__main__":
    run_experiment()
