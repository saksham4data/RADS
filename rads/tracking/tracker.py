from typing import List, Dict, Any
import numpy as np

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
