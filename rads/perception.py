import cv2
import os
from typing import Any, Dict, Iterator, List, Tuple
import numpy as np


class VideoReader:
    """Reads video files and yields frames with metadata."""

    def __init__(self, video_path: str):
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        
        if not self.cap.isOpened():
            raise IOError(f"Failed to open video file: {video_path}")

        self._fps = self.cap.get(cv2.CAP_PROP_FPS)
        self._width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def resolution(self) -> Tuple[int, int]:
        return (self._width, self._height)

    @property
    def total_frames(self) -> int:
        return self._total_frames

    @property
    def duration_seconds(self) -> float:
        if self._fps > 0:
            return self._total_frames / self._fps
        return 0.0

    def get_frames(self) -> Iterator[Tuple[int, float, np.ndarray]]:
        """Yields (frame_index, timestamp_seconds, frame)"""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_idx = 0
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            timestamp = frame_idx / self._fps if self._fps > 0 else 0.0
            yield (frame_idx, timestamp, frame)
            frame_idx += 1

    def release(self):
        self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

class Tracker:
    """Wraps ultralytics YOLO tracking capabilities (ByteTrack/BoT-SORT)."""

    def __init__(self, model_path: str = 'yolo11n.pt', tracker_type: str = 'bytetrack',
                 confidence: float = 0.25, iou: float = 0.45, classes: List[int] = None):
        if YOLO is None:
            raise ImportError("ultralytics package is required for tracking.")
        
        self.model_path = model_path
        self.model = YOLO(model_path)
        self.tracker_type = tracker_type
        
        # Configure tracker file based on type
        if tracker_type.lower() == 'bytetrack':
            self.tracker_config = 'bytetrack.yaml'
        elif tracker_type.lower() in ['botsort', 'bot-sort']:
            self.tracker_config = 'botsort.yaml'
        else:
            self.tracker_config = 'bytetrack.yaml' # default
            
        self.confidence = confidence
        self.iou = iou
        self.classes = classes
        self.last_reset_path = None

    def reset(self) -> str:
        """Clears per-video tracker state. Returns the path taken.

        track() runs with persist=True, so without this call track IDs and Kalman state
        from one video carry into the next. Paths, in preference order:
        'no_state' (nothing tracked yet), 'predictor_trackers' (ultralytics exposes live
        trackers), 'model_rebuild' (fallback for versions that do not).
        """
        predictor = getattr(self.model, 'predictor', None)
        if predictor is None:
            self.last_reset_path = 'no_state'
            return self.last_reset_path

        trackers = getattr(predictor, 'trackers', None)
        if trackers:
            try:
                for tracker in trackers:
                    tracker.reset()
                self.last_reset_path = 'predictor_trackers'
                return self.last_reset_path
            except AttributeError:
                pass

        self.model = YOLO(self.model_path)
        self.last_reset_path = 'model_rebuild'
        return self.last_reset_path

    def track(self, frame: np.ndarray, frame_index: int, timestamp: float) -> List[Dict[str, Any]]:
        """Runs detection and tracking on a single frame."""
        
        # We set persist=True to keep track IDs across frames. 
        # Note: In a real streaming pipeline, we pass frames sequentially.
        results = self.model.track(
            frame, 
            persist=True, 
            tracker=self.tracker_config,
            conf=self.confidence, 
            iou=self.iou, 
            classes=self.classes,
            verbose=False
        )
        
        tracked_objects = []
        
        if not results:
            return tracked_objects
            
        result = results[0]
        boxes = result.boxes
        
        if boxes is not None and boxes.id is not None:
            # ultralytics returns boxes.id only if objects are tracked
            for i, box in enumerate(boxes):
                xyxy = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = result.names[cls_id]
                
                # Check if this specific box has an ID
                # (sometimes detections occur but aren't assigned an ID yet)
                if len(boxes.id) > i:
                    track_id = int(boxes.id[i])
                    
                    center_x = (xyxy[0] + xyxy[2]) / 2.0
                    center_y = (xyxy[1] + xyxy[3]) / 2.0
                    
                    tracked_objects.append({
                        "track_id": track_id,
                        "class_id": cls_id,
                        "class_name": cls_name,
                        "confidence": conf,
                        "bbox_xyxy": xyxy,
                        "center_xy": [center_x, center_y],
                        "frame_index": frame_index,
                        "timestamp": timestamp
                    })
                
        return tracked_objects
