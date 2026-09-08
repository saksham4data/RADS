import cv2
import numpy as np
from typing import List, Dict, Any, Tuple
from rads.motion.trajectory import TrackHistory

class Visualizer:
    """Renders pipeline output onto video frames and saves them."""

    def __init__(self, output_path: str, fps: float, resolution: Tuple[int, int]):
        self.output_path = output_path
        self.fps = fps
        self.resolution = resolution
        
        # Initialize OpenCV VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(output_path, fourcc, fps, resolution)

    def draw_frame(self, frame: np.ndarray, tracked_objects: List[Dict[str, Any]], track_history: TrackHistory = None):
        """Draws annotations for a single frame and writes to output."""
        
        annotated_frame = frame.copy()
        
        # 1. Draw trajectory trails if history is provided
        if track_history:
            for t_id in track_history.get_all_track_ids():
                traj = track_history.get_trajectory(t_id)
                # Only draw trail if it has at least 2 points
                if len(traj) >= 2:
                    color = self._get_color(t_id)
                    # Get up to the last 30 points (approx 1 second at 30 fps) to avoid clutter
                    recent_traj = traj[-30:]
                    pts = np.array([[pt['cx'], pt['cy']] for pt in recent_traj], np.int32)
                    pts = pts.reshape((-1, 1, 2))
                    cv2.polylines(annotated_frame, [pts], isClosed=False, color=color, thickness=2)

        # 2. Draw current bounding boxes
        for obj in tracked_objects:
            # Parse bounding box
            x1, y1, x2, y2 = map(int, obj['bbox_xyxy'])
            track_id = obj.get('track_id', -1)
            class_name = obj.get('class_name', 'unknown')
            
            # Generate a consistent color based on track_id
            color = self._get_color(track_id)
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{class_name} {track_id}"
            
            # Text background for readability
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
            )
            
            cv2.rectangle(
                annotated_frame, 
                (x1, y1 - text_height - 4), 
                (x1 + text_width, y1), 
                color, 
                -1
            )
            
            # Text
            cv2.putText(
                annotated_frame, 
                label, 
                (x1, y1 - 4), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                (255, 255, 255), 
                2
            )
            
        self.writer.write(annotated_frame)

    def _get_color(self, track_id: int) -> Tuple[int, int, int]:
        """Generate a consistent pseudo-random color for a given track ID."""
        if track_id == -1:
            return (128, 128, 128) # Gray for untracked objects
            
        np.random.seed(track_id)
        # Random pastel-ish color
        b = int(np.random.randint(50, 255))
        g = int(np.random.randint(50, 255))
        r = int(np.random.randint(50, 255))
        return (b, g, r)
        
    def release(self):
        """Release the video writer."""
        if self.writer:
            self.writer.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
