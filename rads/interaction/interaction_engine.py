from typing import Dict, List, Any, Tuple
from rads.config.config_loader import ConfigLoader

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
