import time
import os

jsonl_file = "rads/evaluation/picek_500_results.jsonl"
dashboard_file = "rads/evaluation/dashboard.md"
total_videos = 500

print("Starting dashboard updater...")

while True:
    try:
        if os.path.exists(jsonl_file):
            with open(jsonl_file, "r") as f:
                lines = sum(1 for line in f if line.strip())
        else:
            lines = 0
            
        progress_pct = (lines / total_videos) * 100
        
        with open(dashboard_file, "w", encoding="utf-8") as f:
            f.write("# RADS Phase 8 Evaluation Dashboard\n\n")
            f.write(f"> **Status**: {'Completed' if lines >= total_videos else 'Running'}\n\n")
            f.write(f"**Progress**: {lines} / {total_videos} videos processed ({progress_pct:.1f}%)\n\n")
            
            # Text progress bar
            bar_len = 40
            filled = int(bar_len * lines // total_videos)
            bar = "█" * filled + "-" * (bar_len - filled)
            f.write(f"`[{bar}]`\n\n")
            
            f.write("---\n")
            f.write("*(This file updates automatically every 15 seconds. Keep it open in your IDE to watch the progress!)*\n")
            
        if lines >= total_videos:
            print("Target reached. Exiting dashboard updater.")
            break
            
    except Exception as e:
        print(f"Error updating dashboard: {e}")
        
    time.sleep(15)
