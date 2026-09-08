from typing import List, Dict, Any
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    # Will fail at runtime if ultralytics is not installed, but allows module to import
    YOLO = None

class Detector:
    """Wraps YOLO for single-frame detection.
    Note: The tracking module usually handles detection intrinsically, 
    but this exists for architectural completeness if standalone detection is needed."""

    def __init__(self, model_path: str = 'yolo11n.pt', confidence: float = 0.25, 
                 iou: float = 0.45, classes: List[int] = None):
        if YOLO is None:
            raise ImportError("ultralytics package is required for detection.")
        
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.iou = iou
        self.classes = classes

    def detect(self, frame: np.ndarray, frame_index: int, timestamp: float) -> List[Dict[str, Any]]:
        """Runs detection on a single frame."""
        
        # Run inference
        results = self.model(frame, conf=self.confidence, iou=self.iou, 
                             classes=self.classes, verbose=False)
        
        detections = []
        
        if not results:
            return detections
            
        result = results[0]
        
        # Parse results
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                xyxy = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = result.names[cls_id]
                
                detections.append({
                    "class_id": cls_id,
                    "class_name": cls_name,
                    "confidence": conf,
                    "bbox_xyxy": xyxy,
                    "frame_index": frame_index,
                    "timestamp": timestamp
                })
                
        return detections
