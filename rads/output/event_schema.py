from typing import Dict, Any, List

def get_stub_event_result(video_path: str, tracks_summary: Dict[int, Any] = None) -> Dict[str, Any]:
    """
    Returns a stub structured result dictionary.
    
    Args:
        video_path: Path to the processed video.
        tracks_summary: A summary dictionary of track_id to metadata.
                        e.g., {1: {"class_name": "car", "frame_count": 50}, ...}
    """
    if tracks_summary is None:
        tracks_summary = {}
        
    return {
        "video": video_path,
        "accident": None,            # Boolean (True/False) or None if not computed
        "confidence": None,          # Float [0, 1]
        "event": {
            "start_time": None,
            "impact_time": None,
            "end_time": None
        },
        "objects_involved": [],      # List of track_ids
        "interaction_candidates": [],
        "severity": None,            # 'LOW', 'MEDIUM', 'HIGH', or None
        "tracks_summary": tracks_summary,
        "status": "PHASE_4_STUB"     # Indicates this is an incomplete vertical slice
    }
