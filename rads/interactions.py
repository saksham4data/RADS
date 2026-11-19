import numpy as np
from typing import Any, Dict, List, Tuple
from rads.tracks import TrackHistory
from rads.config import ConfigLoader


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


def detect_interactions(pairwise_data: Dict[Tuple[int, int], Dict[int, Dict[str, Any]]], config: ConfigLoader) -> List[Dict[str, Any]]:
    """
    Detect InteractionCandidate based on pairwise evidence.
    """
    candidates = []
    
    norm_prox_thresh = config.interaction_normalized_proximity
    iou_thresh = config.interaction_overlap_iou
    min_frames = config.interaction_min_persistence_frames
    rel_vel_thresh = config.interaction_relative_velocity
    
    for (id_a, id_b), frames_data in pairwise_data.items():
        sorted_frames = sorted(frames_data.keys())
        
        in_interaction = False
        interaction_start_frame = None
        interaction_start_time = None
        peak_iou = 0.0
        min_prox = 999.0
        max_rel_vel = 0.0
        evidence_frames = 0
        peak_time = None
        
        for f_idx in sorted_frames:
            data = frames_data[f_idx]
            
            # Evidence criteria
            is_close = data['normalized_proximity'] < norm_prox_thresh
            is_overlapping = data['iou'] > iou_thresh
            is_converging = data['relative_velocity'] > rel_vel_thresh
            
            # For an interaction, we want either overlap OR (close AND converging)
            has_evidence = is_overlapping or (is_close and is_converging)
            
            if has_evidence:
                if not in_interaction:
                    in_interaction = True
                    interaction_start_frame = f_idx
                    interaction_start_time = data['timestamp']
                    peak_iou = data['iou']
                    min_prox = data['normalized_proximity']
                    max_rel_vel = data['relative_velocity']
                    evidence_frames = 1
                    peak_time = data['timestamp']
                else:
                    if data['iou'] > peak_iou or (data['iou'] == peak_iou and data['normalized_proximity'] < min_prox):
                        peak_time = data['timestamp']
                    peak_iou = max(peak_iou, data['iou'])
                    min_prox = min(min_prox, data['normalized_proximity'])
                    max_rel_vel = max(max_rel_vel, data['relative_velocity'])
                    evidence_frames += 1
            else:
                if in_interaction:
                    if evidence_frames >= min_frames:
                        # Found a valid candidate
                        evidence_list = []
                        if peak_iou > iou_thresh:
                            evidence_list.append('overlap')
                        if min_prox < norm_prox_thresh and max_rel_vel > rel_vel_thresh:
                            evidence_list.append('proximity_and_convergence')
                            
                        candidates.append({
                            'object_a_id': id_a,
                            'object_b_id': id_b,
                            'start_frame': interaction_start_frame,
                            'end_frame': sorted_frames[sorted_frames.index(f_idx)-1],
                            'start_time': interaction_start_time,
                            'peak_time': peak_time,
                            'end_time': frames_data[sorted_frames[sorted_frames.index(f_idx)-1]]['timestamp'],
                            'min_distance': min_prox, # Normalized proximity
                            'max_relative_velocity': max_rel_vel,
                            'peak_iou': peak_iou,
                            'evidence_list': evidence_list
                        })
                    
                    in_interaction = False
                    evidence_frames = 0
                    
        # Check if interaction was ongoing at the end of the video
        if in_interaction and evidence_frames >= min_frames:
            last_idx = sorted_frames[-1]
            last_data = frames_data[last_idx]
            
            evidence_list = []
            if peak_iou > iou_thresh:
                evidence_list.append('overlap')
            if min_prox < norm_prox_thresh and max_rel_vel > rel_vel_thresh:
                evidence_list.append('proximity_and_convergence')
                
            candidates.append({
                'object_a_id': id_a,
                'object_b_id': id_b,
                'start_frame': interaction_start_frame,
                'end_frame': last_idx,
                'start_time': interaction_start_time,
                'peak_time': peak_time,
                'end_time': last_data['timestamp'],
                'min_distance': min_prox,
                'max_relative_velocity': max_rel_vel,
                'peak_iou': peak_iou,
                'evidence_list': evidence_list
            })
            
    return candidates
