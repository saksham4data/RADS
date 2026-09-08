from typing import Dict, List, Any, Tuple

class TrackHistory:
    """Stores temporal sequence of positions for all tracked objects."""

    def __init__(self):
        # Maps track_id -> list of state dicts
        # Each state dict: {'frame_index': int, 'timestamp': float, 'cx': float, 'cy': float, 'w': float, 'h': float, 'bbox_xyxy': list}
        self.history: Dict[int, List[Dict[str, Any]]] = {}

    def update(self, tracked_objects: List[Dict[str, Any]], frame_index: int, timestamp: float):
        """Append the new position of every tracked object for the current frame."""
        for obj in tracked_objects:
            t_id = obj['track_id']
            if t_id not in self.history:
                self.history[t_id] = []
                
            x1, y1, x2, y2 = obj['bbox_xyxy']
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            w = x2 - x1
            h = y2 - y1
            
            self.history[t_id].append({
                'frame_index': frame_index,
                'timestamp': timestamp,
                'cx': cx,
                'cy': cy,
                'w': w,
                'h': h,
                'bbox_xyxy': [x1, y1, x2, y2]
            })

    def get_trajectory(self, track_id: int) -> List[Dict[str, Any]]:
        """Get the full history for a specific track_id."""
        return self.history.get(track_id, [])

    def get_all_track_ids(self) -> List[int]:
        """Returns a list of all track_ids seen so far."""
        return list(self.history.keys())
