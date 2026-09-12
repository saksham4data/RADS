"""
Forensic diagnostic script: runs the pipeline on the 4 failure videos
and dumps detailed intermediate state at every stage.

DOES NOT MODIFY any production code. Read-only forensics.
"""
import os, sys, json, time
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rads.pipeline.pipeline import Pipeline
from rads.config.config_loader import ConfigLoader
from rads.video.video_reader import VideoReader
from rads.tracking.tracker import Tracker
from rads.motion.trajectory import TrackHistory
from rads.motion.motion_features import compute_motion_features
from rads.interaction.pairwise import compute_pairwise_metrics, enrich_with_motion
from rads.interaction.interaction_engine import detect_interactions
from rads.reasoning.accident_reasoner import evaluate_accident, get_velocities_around_frame
from rads.severity.severity_engine import estimate_severity
from pathlib import Path

OUTPUT_DIR = Path("rads/evaluation/experiments/frame_skip_comparison/forensics")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# The 4 failure videos (frame_skip=1 results)
FAILURES = [
    # False Negatives
    (r"e:\Rads\Datasets\processed\picek_sorted\trimmed\positive\real\-2UPLUV7JLg_00.mp4", "accident", "FN"),
    (r"e:\Rads\Datasets\processed\picek_sorted\trimmed\positive\real\-9oifpjUxxM_00.mp4", "accident", "FN"),
    # False Positives
    (r"e:\Rads\Datasets\processed\picek_sorted\trimmed\negative\real\-6SQSDj8cYU_00.mp4", "normal", "FP"),
    (r"e:\Rads\Datasets\processed\picek_sorted\trimmed\negative\real\-dmYsQc-odI_00.mp4", "normal", "FP"),
]

config_path = "rads/config/pipeline_config.yaml"

for video_path, gt, failure_type in FAILURES:
    video_name = os.path.basename(video_path)
    safe_name = video_name.replace(".mp4", "")
    print(f"\n{'='*60}")
    print(f"FORENSICS: {video_name} | GT={gt} | Type={failure_type}")
    print(f"{'='*60}")
    
    config = ConfigLoader(config_path)
    tracker = Tracker(
        model_path=config.detector_model,
        tracker_type=config.tracker_type,
        confidence=config.detector_confidence,
        iou=config.detector_iou,
        classes=config.detector_classes
    )
    
    track_history = TrackHistory()
    track_summaries = {}
    frame_skip = config.frame_skip
    
    # Pass 1: Detection & Tracking
    with VideoReader(video_path) as video:
        total_frames = video.total_frames
        fps = video.fps
        duration = video.duration_seconds
        
        for frame_idx, timestamp, frame in video.get_frames():
            if frame_idx % frame_skip != 0:
                continue
            tracked_objects = tracker.track(frame, frame_idx, timestamp)
            track_history.update(tracked_objects, frame_idx, timestamp)
            
            for obj in tracked_objects:
                t_id = obj['track_id']
                if t_id not in track_summaries:
                    track_summaries[t_id] = {
                        "class_name": obj['class_name'],
                        "first_frame": frame_idx,
                        "last_frame": frame_idx,
                        "frame_count": 0
                    }
                track_summaries[t_id]["last_frame"] = frame_idx
                track_summaries[t_id]["frame_count"] += 1
    
    # Motion features
    motion_features = compute_motion_features(track_history)
    
    # Pairwise metrics
    pairwise_data = compute_pairwise_metrics(track_history)
    enrich_with_motion(pairwise_data)
    
    # Interaction detection
    interaction_candidates = detect_interactions(pairwise_data, config)
    
    # Accident reasoning
    accident_result = evaluate_accident(interaction_candidates, track_history)
    severity = estimate_severity(accident_result, track_history)
    
    # --- DUMP EVERYTHING ---
    report = {
        "video_name": video_name,
        "video_path": video_path,
        "ground_truth": gt,
        "failure_type": failure_type,
        "video_meta": {
            "total_frames": total_frames,
            "fps": fps,
            "duration_seconds": round(duration, 2)
        },
        "tracking": {
            "total_unique_tracks": len(track_summaries),
            "track_summaries": {}
        },
        "motion_features": {},
        "pairwise_pairs_count": len(pairwise_data),
        "pairwise_detail": {},
        "interaction_candidates_count": len(interaction_candidates),
        "interaction_candidates": interaction_candidates,
        "accident_result": accident_result,
        "severity": severity
    }
    
    # Track summaries with class + lifespan
    for t_id, summary in track_summaries.items():
        traj = track_history.get_trajectory(t_id)
        report["tracking"]["track_summaries"][str(t_id)] = {
            "class_name": summary["class_name"],
            "first_frame": summary["first_frame"],
            "last_frame": summary["last_frame"],
            "frame_count": summary["frame_count"],
            "lifespan_frames": summary["last_frame"] - summary["first_frame"],
        }
    
    # Motion features per track
    for t_id, feats in motion_features.items():
        report["motion_features"][str(t_id)] = feats
    
    # Pairwise detail - for each pair dump the per-frame metrics
    for (id_a, id_b), frames_data in pairwise_data.items():
        pair_key_str = f"{id_a}_{id_b}"
        sorted_frames = sorted(frames_data.keys())
        frames_summary = []
        for f_idx in sorted_frames:
            d = frames_data[f_idx]
            frames_summary.append({
                "frame": f_idx,
                "timestamp": round(d['timestamp'], 3),
                "distance": round(d['distance'], 1),
                "norm_prox": round(d['normalized_proximity'], 3),
                "iou": round(d['iou'], 4),
                "rel_vel": round(d.get('relative_velocity', 0), 2),
                "conv_angle": round(d.get('convergence_angle', 0), 1)
            })
        report["pairwise_detail"][pair_key_str] = {
            "id_a": id_a,
            "id_b": id_b,
            "co_existing_frames": len(sorted_frames),
            "frames": frames_summary
        }
    
    # For interaction candidates, also dump the velocity context used by the reasoner
    if interaction_candidates:
        all_trajs = [track_history.get_trajectory(tid) for tid in track_history.get_all_track_ids()]
        max_frame = max([t[-1]['frame_index'] for t in all_trajs if t], default=0)
        
        for i, cand in enumerate(interaction_candidates):
            traj_a = track_history.get_trajectory(cand['object_a_id'])
            traj_b = track_history.get_trajectory(cand['object_b_id'])
            end_frame = cand.get('end_frame', 0)
            pre_a, post_a, term_a = get_velocities_around_frame(traj_a, end_frame, max_frame, window=15)
            pre_b, post_b, term_b = get_velocities_around_frame(traj_b, end_frame, max_frame, window=15)
            
            cand[f"_forensic_velocity_context"] = {
                "end_frame": end_frame,
                "max_frame": max_frame,
                "obj_a": {"pre_vel": round(pre_a, 2), "post_vel": round(post_a, 2), "terminated": term_a},
                "obj_b": {"pre_vel": round(pre_b, 2), "post_vel": round(post_b, 2), "terminated": term_b},
                "passing_filter_would_apply": (
                    not term_a and not term_b and
                    pre_a > 0 and pre_b > 0 and
                    post_a > pre_a * 0.5 and post_b > pre_b * 0.5
                )
            }
    
    out_path = OUTPUT_DIR / f"{failure_type}_{safe_name}.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"Saved forensics to {out_path}")

print("\n\nForensics complete.")
