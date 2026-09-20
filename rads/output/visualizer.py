import cv2
import numpy as np
import os
from typing import List, Dict, Any, Tuple
from rads.motion.trajectory import TrackHistory
from rads.output.event_schema import involved_ids_from_result

def format_timestamp(seconds: float) -> str:
    """MM:SS.s per TECH_STACK.md section 16."""
    if seconds is None:
        return "--:--.-"
    seconds = max(float(seconds), 0.0)
    minutes = int(seconds // 60)
    return f"{minutes:02d}:{seconds - minutes * 60:04.1f}"

class Visualizer:
    """Renders pipeline output onto video frames in a post-processing pass."""

    def __init__(self, output_path: str, fps: float, resolution: Tuple[int, int]):
        self.output_path = output_path
        self.fps = fps
        self.resolution = resolution
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(output_path, fourcc, fps, resolution)

    def draw_frame(self, frame: np.ndarray, frame_idx: int, timestamp: float, track_history: TrackHistory, event_result: Dict[str, Any]):
        annotated_frame = frame.copy()
        
        # 1. Check if we are inside the accident window
        is_accident = event_result.get('accident', False)
        in_event_window = False
        if is_accident:
            event_data = event_result.get('event', {})
            start_time = event_data.get('start_time', 0)
            end_time = event_data.get('end_time', 0)
            if start_time is not None and end_time is not None:
                if start_time <= timestamp <= end_time:
                    in_event_window = True
                
        # 2. Draw trajectory trails
        for t_id in track_history.get_all_track_ids():
            traj = track_history.get_trajectory(t_id)
            # Find points up to current frame
            past_traj = [pt for pt in traj if pt['frame_index'] <= frame_idx]
            
            if len(past_traj) >= 2:
                color = self._get_color(t_id)
                recent_traj = past_traj[-30:]
                pts = np.array([[pt['cx'], pt['cy']] for pt in recent_traj], np.int32)
                pts = pts.reshape((-1, 1, 2))
                cv2.polylines(annotated_frame, [pts], isClosed=False, color=color, thickness=2)

        # 3. Draw current bounding boxes
        # objects_involved holds {id, class} objects, so membership needs an id set
        involved_ids = involved_ids_from_result(event_result)
        
        for t_id in track_history.get_all_track_ids():
            traj = track_history.get_trajectory(t_id)
            # Find point exactly at current frame
            current_pt = next((pt for pt in traj if pt['frame_index'] == frame_idx), None)
            if current_pt:
                x1, y1, x2, y2 = map(int, current_pt['bbox_xyxy'])
                class_name = current_pt.get('class_name', 'unknown')
                
                # Highlight involved objects during the event
                if t_id in involved_ids and in_event_window:
                    color = (0, 0, 255) # Red for involved objects
                    thickness = 3
                else:
                    color = self._get_color(t_id)
                    thickness = 2
                
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)
                
                label = f"{class_name} {t_id}"
                (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(annotated_frame, (x1, y1 - text_height - 4), (x1 + text_width, y1), color, -1)
                cv2.putText(annotated_frame, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # 4. Draw overlays
        if is_accident:
            conf = event_result.get('confidence', 0.0)
            sev = event_result.get('severity', 'UNKNOWN')
            text = f"ACCIDENT DETECTED | Conf: {conf:.2f} | Sev: {sev}"
            color = (0, 0, 255) if in_event_window else (0, 165, 255) # Red if inside window, Orange otherwise
            cv2.putText(annotated_frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            impact_time = (event_result.get('event') or {}).get('impact_time')
            cv2.putText(annotated_frame, f"TIME: {format_timestamp(impact_time)}", (20, 72),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        else:
            cv2.putText(annotated_frame, "NORMAL TRAFFIC", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        self.writer.write(annotated_frame)

    def write_summary_frame(self, event_result: Dict[str, Any], hold_seconds: float = 2.0):
        """Appends a held end-of-video frame carrying the structured result."""
        width, height = self.resolution
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        is_accident = event_result.get('accident', False)
        color = (0, 0, 255) if is_accident else (0, 255, 0)
        event = event_result.get('event') or {}
        severity_detail = event_result.get('severity_detail') or {}

        lines = [("ACCIDENT DETECTED" if is_accident else "NO ACCIDENT DETECTED", color)]
        if is_accident:
            lines.append((f"TIME: {format_timestamp(event.get('impact_time'))}", color))
            lines.append((f"WINDOW: {format_timestamp(event.get('start_time'))}"
                          f" - {format_timestamp(event.get('end_time'))}", (255, 255, 255)))
            lines.append((f"SEVERITY: {event_result.get('severity')}"
                          f" (score {severity_detail.get('score')})", (255, 255, 255)))
            lines.append((f"TYPE: {event_result.get('accident_type')}", (255, 255, 255)))
            for obj in event_result.get('objects_involved', []) or []:
                lines.append((f"  {obj.get('class', 'unknown').upper()} #{obj.get('id')}", (255, 255, 255)))
            evidence = event_result.get('evidence_list', []) or []
            lines.append(("EVIDENCE:", (255, 255, 255)))
            for item in evidence:
                lines.append((f"  - {item}", (255, 255, 255)))
        lines.append((f"CONFIDENCE: {event_result.get('confidence')}"
                      f" (raw score {event_result.get('score')})", (255, 255, 255)))
        lines.append((f"TRACKS: {len(event_result.get('tracks_summary', {}) or {})}", (255, 255, 255)))

        y = 60
        for text, line_color in lines:
            cv2.putText(frame, text, (40, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, line_color, 2)
            y += 34
            if y > height - 20:
                break

        for _ in range(max(1, int(self.fps * hold_seconds))):
            self.writer.write(frame)

    def _get_color(self, track_id: int) -> Tuple[int, int, int]:
        if track_id == -1:
            return (128, 128, 128)
        np.random.seed(track_id)
        b = int(np.random.randint(50, 255))
        g = int(np.random.randint(50, 255))
        r = int(np.random.randint(50, 255))
        return (b, g, r)

    def release(self):
        if self.writer:
            self.writer.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()

def extract_evidence_clip(video_path: str, output_path: str, event_result: Dict[str, Any], pad_seconds: float = 2.0):
    """Extracts a short clip around the entire accident event window."""
    if not event_result.get('accident'):
        return
        
    event = event_result.get('event', {})
    start_time = event.get('start_time')
    end_time = event.get('end_time')
    
    if start_time is None or end_time is None:
        impact_time = event.get('impact_time')
        if impact_time is None:
            return
        start_time = impact_time
        end_time = impact_time
        
    start_sec = max(0.0, start_time - pad_seconds)
    end_sec = end_time + pad_seconds
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    current_frame = start_frame
    
    while current_frame <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        writer.write(frame)
        current_frame += 1
        
    writer.release()
    cap.release()
