import argparse
import json
import os
import time

def count_rows(jsonl_file: str) -> tuple:
    """Returns (successful_rows, error_rows). Rows written before B5 carry no status field."""
    if not os.path.exists(jsonl_file):
        return 0, 0

    ok = 0
    errors = 0
    with open(jsonl_file, "r") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("status", "ok") == "ok":
                ok += 1
            else:
                errors += 1
    return ok, errors

def main():
    parser = argparse.ArgumentParser(description="RADS evaluation progress dashboard")
    parser.add_argument("--jsonl", default="rads/evaluation/picek_500_results.jsonl",
                        help="Path to the evaluator JSONL output")
    parser.add_argument("--total", type=int, default=500,
                        help="Expected number of videos in the split")
    parser.add_argument("--dashboard", default="rads/evaluation/dashboard.md",
                        help="Path to the markdown dashboard to write")
    parser.add_argument("--interval", type=float, default=15.0,
                        help="Seconds between refreshes")
    args = parser.parse_args()

    print("Starting dashboard updater...")

    while True:
        try:
            ok, errors = count_rows(args.jsonl)
            progress_pct = (ok / args.total) * 100 if args.total else 0.0

            with open(args.dashboard, "w", encoding="utf-8") as f:
                f.write("# RADS Phase 8 Evaluation Dashboard\n\n")
                f.write(f"> **Status**: {'Completed' if ok >= args.total else 'Running'}\n\n")
                f.write(f"**Source**: `{args.jsonl}`\n\n")
                f.write(f"**Progress**: {ok} / {args.total} videos processed ({progress_pct:.1f}%)\n\n")
                f.write(f"**Error rows**: {errors}\n\n")

                bar_len = 40
                filled = int(bar_len * ok // args.total) if args.total else 0
                bar = "#" * filled + "-" * (bar_len - filled)
                f.write(f"`[{bar}]`\n\n")

                f.write("---\n")
                f.write(f"*(Refreshes every {args.interval:.0f} seconds.)*\n")

            if ok >= args.total:
                print("Target reached. Exiting dashboard updater.")
                break

        except Exception as e:
            print(f"Error updating dashboard: {e}")

        time.sleep(args.interval)

if __name__ == "__main__":
    main()
