import numpy as np
from typing import Dict, List, Any, Tuple
from rads.motion.trajectory import TrackHistory

def calculate_iou(boxA, boxB):
    # Determine the coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def compute_pairwise_metrics(track_history: TrackHistory) -> Dict[Tuple[int, int], Dict[int, Dict[str, Any]]]:
    """
    Computes pairwise metrics for all co-existing tracks.
    Returns:
        Dict mapping (track_id_a, track_id_b) -> {frame_index: metrics_dict}
        where track_id_a < track_id_b to avoid duplicates.
    """
    pairwise_data = {}
    track_ids = sorted(track_history.get_all_track_ids())
    
    # Pre-organize by frame to easily find co-existing tracks
    frames_dict = {}
    for t_id in track_ids:
        traj = track_history.get_trajectory(t_id)
        for pt in traj:
            f_idx = pt['frame_index']
            if f_idx not in frames_dict:
                frames_dict[f_idx] = {}
            frames_dict[f_idx][t_id] = pt

    # Calculate metrics per frame
    for f_idx, obj_dict in frames_dict.items():
        co_existing_ids = sorted(obj_dict.keys())
        n = len(co_existing_ids)
        for i in range(n):
            for j in range(i+1, n):
                id_a = co_existing_ids[i]
                id_b = co_existing_ids[j]
                
                pt_a = obj_dict[id_a]
                pt_b = obj_dict[id_b]
                
                # Distance
                dx = pt_a['cx'] - pt_b['cx']
                dy = pt_a['cy'] - pt_b['cy']
                distance = np.sqrt(dx**2 + dy**2)
                
                # Normalized proximity
                diag_a = np.sqrt(pt_a['w']**2 + pt_a['h']**2)
                diag_b = np.sqrt(pt_b['w']**2 + pt_b['h']**2)
                max_diag = max(diag_a, diag_b)
                # Avoid division by zero
                norm_proximity = distance / max_diag if max_diag > 0 else 999.0
                
                # Bbox overlap (IoU)
                iou = calculate_iou(pt_a['bbox_xyxy'], pt_b['bbox_xyxy'])
                
                pair_key = (id_a, id_b)
                if pair_key not in pairwise_data:
                    pairwise_data[pair_key] = {}
                    
                pairwise_data[pair_key][f_idx] = {
                    'timestamp': pt_a['timestamp'],
                    'distance': float(distance),
                    'normalized_proximity': float(norm_proximity),
                    'iou': float(iou),
                    'pt_a': pt_a,
                    'pt_b': pt_b
                }
                
    return pairwise_data

def enrich_with_motion(pairwise_data: Dict[Tuple[int, int], Dict[int, Dict[str, Any]]]):
    """
    Enrich pairwise data with temporal features like relative velocity and convergence angle.
    This requires looking at consecutive frames for a pair.
    """
    for pair_key, frames_data in pairwise_data.items():
        sorted_frames = sorted(frames_data.keys())
        
        for i in range(len(sorted_frames)):
            f_idx = sorted_frames[i]
            current_data = frames_data[f_idx]
            
            # Default values
            current_data['relative_velocity'] = 0.0
            current_data['convergence_angle'] = 0.0
            
            if i > 0:
                prev_idx = sorted_frames[i-1]
                prev_data = frames_data[prev_idx]
                
                dt = current_data['timestamp'] - prev_data['timestamp']
                if dt > 0.001: # Avoid division by zero
                    # Velocity of A
                    vx_a = (current_data['pt_a']['cx'] - prev_data['pt_a']['cx']) / dt
                    vy_a = (current_data['pt_a']['cy'] - prev_data['pt_a']['cy']) / dt
                    
                    # Velocity of B
                    vx_b = (current_data['pt_b']['cx'] - prev_data['pt_b']['cx']) / dt
                    vy_b = (current_data['pt_b']['cy'] - prev_data['pt_b']['cy']) / dt
                    
                    # Relative velocity (rate of change of distance)
                    dist_diff = current_data['distance'] - prev_data['distance']
                    relative_velocity = -dist_diff / dt  # Positive means they are getting closer
                    
                    # Convergence angle
                    # Dot product of motion vectors
                    mag_a = np.sqrt(vx_a**2 + vy_a**2)
                    mag_b = np.sqrt(vx_b**2 + vy_b**2)
                    
                    if mag_a > 0 and mag_b > 0:
                        dot_product = (vx_a * vx_b + vy_a * vy_b) / (mag_a * mag_b)
                        dot_product = np.clip(dot_product, -1.0, 1.0)
                        angle_rad = np.arccos(dot_product)
                        convergence_angle = np.degrees(angle_rad)
                    else:
                        convergence_angle = 0.0
                        
                    current_data['relative_velocity'] = float(relative_velocity)
                    current_data['convergence_angle'] = float(convergence_angle)
