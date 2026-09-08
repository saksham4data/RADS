from typing import Dict, Any

def estimate_severity(accident_result: Dict[str, Any], track_history) -> str:
    """
    Estimates the severity (LOW, MEDIUM, HIGH) of a detected accident event.
    """
    if not accident_result.get('accident'):
        return None

    score = 0.0
    
    # 1. Number of objects involved
    num_objects = len(accident_result.get('involved_object_ids', []))
    if num_objects > 2:
        score += 2.0
    elif num_objects == 2:
        score += 1.0
        
    # We could look at object types from track history (e.g. if one is a pedestrian).
    has_vulnerable = False
    for obj_id in accident_result.get('involved_object_ids', []):
        traj = track_history.get_trajectory(obj_id)
        if traj and len(traj) > 0:
            class_name = traj[0].get('class_name', 'unknown')
            if class_name in ['person', 'bicycle', 'motorcycle']:
                has_vulnerable = True

    if has_vulnerable:
        score += 3.0
        
    if score >= 3.0:
        return 'HIGH'
    elif score >= 1.5:
        return 'MEDIUM'
    else:
        return 'LOW'
