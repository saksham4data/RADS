import json
import time
import os
from typing import Dict, Any

from rads.config.config_loader import ConfigLoader
from rads.video.video_reader import VideoReader
from rads.tracking.tracker import Tracker
from rads.motion.trajectory import TrackHistory
from rads.motion.motion_features import compute_motion_features
from rads.interaction.pairwise import compute_pairwise_metrics, enrich_with_motion
from rads.interaction.interaction_engine import detect_interactions
from rads.reasoning.accident_reasoner import evaluate_accident
from rads.severity.severity_engine import estimate_severity
from rads.output.event_schema import get_stub_event_result
from rads.output.visualizer import Visualizer, extract_evidence_clip

class Pipeline:
    """Orchestrates the RADS processing pipeline."""

    def __init__(self, config_path: str):
        self.config = ConfigLoader(config_path)
        self.tracker = Tracker(
            model_path=self.config.detector_model,
            tracker_type=self.config.tracker_type,
            confidence=self.config.detector_confidence,
            iou=self.config.detector_iou,
            classes=self.config.detector_classes
        )
        
    def run(self, video_path: str, visualize: bool = False, output_video_path: str = None) -> Dict[str, Any]:
        """Runs the pipeline on a video and returns the event result."""
        
        print(f"Starting pipeline on {video_path}")
        total_start_time = time.perf_counter()
        
        track_summaries = {}
        frame_skip = self.config.frame_skip
        
        track_history = TrackHistory()
        
        # Pass 1: Tracking
        with VideoReader(video_path) as video:
            for frame_idx, timestamp, frame in video.get_frames():
                if frame_idx % frame_skip != 0:
                    continue
                
                frame_start_time = time.perf_counter()
                
                tracked_objects = self.tracker.track(frame, frame_idx, timestamp)
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
                
                frame_end_time = time.perf_counter()
                if frame_idx % 30 == 0:
                    frame_ms = (frame_end_time - frame_start_time) * 1000
                    print(f"Processed frame {frame_idx}/{video.total_frames} ({frame_ms:.1f} ms)")

        # Phase 3: compute motion features
        print("Computing motion features...")
        motion_features = compute_motion_features(track_history)
        
        total_end_time = time.perf_counter()
        pipeline_time = total_end_time - total_start_time
        processed_frames = video.total_frames // frame_skip
        avg_fps = processed_frames / pipeline_time if pipeline_time > 0 else 0
        
        print(f"Finished processing {video_path}")
        print(f"Pipeline Time: {pipeline_time:.2f}s | Processed Frames: {processed_frames} | Average FPS: {avg_fps:.2f}")
        print(f"Tracked {len(track_summaries)} unique objects.")
        
        for t_id, feats in motion_features.items():
            if t_id in track_summaries:
                track_summaries[t_id]["motion_features"] = feats
                
        # Phase 4: pairwise metrics and interaction detection
        print("Computing pairwise metrics...")
        pairwise_data = compute_pairwise_metrics(track_history)
        enrich_with_motion(pairwise_data)
        
        print("Detecting interactions...")
        interaction_candidates = detect_interactions(pairwise_data, self.config)
        print(f"Found {len(interaction_candidates)} interaction candidates.")
        
        # Phase 5 & 6: Reasoning and Severity
        print("Evaluating accident likelihood...")
        accident_result = evaluate_accident(interaction_candidates)
        severity = estimate_severity(accident_result, track_history)
        accident_result['severity'] = severity
        
        # Generate the structured output
        result = get_stub_event_result(video_path, track_summaries)
        result["interaction_candidates"] = interaction_candidates
        
        # Update event result with our reasoning
        result["accident"] = accident_result["accident"]
        result["confidence"] = accident_result["confidence"]
        result["event"] = {
            "start_time": accident_result.get("start_time"),
            "impact_time": accident_result.get("impact_time"),
            "end_time": accident_result.get("end_time")
        }
        result["objects_involved"] = accident_result.get("involved_object_ids", [])
        result["severity"] = severity
        result["status"] = "PHASE_5_COMPLETE"
        
        # Pass 2: Visualization (Phase 7)
        if visualize and output_video_path:
            print("Running visualization pass...")
            with VideoReader(video_path) as video2:
                with Visualizer(output_video_path, video2.fps / frame_skip, video2.resolution) as visualizer:
                    for frame_idx, timestamp, frame in video2.get_frames():
                        if frame_idx % frame_skip != 0:
                            continue
                        visualizer.draw_frame(frame, frame_idx, timestamp, track_history, result)
            
            if result.get("accident"):
                base, ext = os.path.splitext(output_video_path)
                clip_path = f"{base}_evidence{ext}"
                print(f"Extracting evidence clip to {clip_path}...")
                # We extract the clip from the generated annotated video so it includes overlays
                extract_evidence_clip(output_video_path, clip_path, result, pad_seconds=2.0)
                
        return result
