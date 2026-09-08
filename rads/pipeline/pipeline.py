import json
import time
from typing import Dict, Any

from rads.config.config_loader import ConfigLoader
from rads.video.video_reader import VideoReader
from rads.tracking.tracker import Tracker
from rads.motion.trajectory import TrackHistory
from rads.motion.motion_features import compute_motion_features
from rads.interaction.pairwise import compute_pairwise_metrics, enrich_with_motion
from rads.interaction.interaction_engine import detect_interactions
from rads.output.event_schema import get_stub_event_result
from rads.output.visualizer import Visualizer

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
        
        # We will collect a summary of tracks to populate our stub output
        track_summaries = {}
        frame_skip = self.config.frame_skip
        
        track_history = TrackHistory()
        
        with VideoReader(video_path) as video:
            visualizer = None
            if visualize and output_video_path:
                visualizer = Visualizer(output_video_path, video.fps / frame_skip, video.resolution)
            
            try:
                for frame_idx, timestamp, frame in video.get_frames():
                    if frame_idx % frame_skip != 0:
                        continue
                    
                    frame_start_time = time.perf_counter()
                    
                    # The Tracker handles both detection and tracking via ultralytics.track()
                    tracked_objects = self.tracker.track(frame, frame_idx, timestamp)
                    
                    track_history.update(tracked_objects, frame_idx, timestamp)
                    
                    # Update our summary
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
                        
                    # Visualize
                    if visualizer:
                        visualizer.draw_frame(frame, tracked_objects, track_history)
                    
                    frame_end_time = time.perf_counter()
                    
                    if frame_idx % 30 == 0:
                        frame_ms = (frame_end_time - frame_start_time) * 1000
                        print(f"Processed frame {frame_idx}/{video.total_frames} ({frame_ms:.1f} ms)")
            finally:
                if visualizer:
                    visualizer.release()

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
        
        # We append motion features to the summary to inspect it in the stub JSON
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
        
        # Generate the structured output (stub for Phase 1/4)
        result = get_stub_event_result(video_path, track_summaries)
        result["interaction_candidates"] = interaction_candidates
        return result
