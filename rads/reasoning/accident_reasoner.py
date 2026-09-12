import numpy as np
from typing import List, Dict, Any

def get_velocities_around_frame(traj: List[Dict[str, Any]], frame_idx: int, max_frame: int, window: int = 15) -> tuple:
    """Returns (pre_vel, post_vel, track_terminated_soon) in pixels/second"""
    pre_pts = [p for p in traj if frame_idx - window <= p['frame_index'] < frame_idx]
    post_pts = [p for p in traj if frame_idx < p['frame_index'] <= frame_idx + window]
    
    def get_vel(pts):
        if len(pts) < 2: return None
        dx = pts[-1]['cx'] - pts[0]['cx']
        dy = pts[-1]['cy'] - pts[0]['cy']
        dt = pts[-1]['timestamp'] - pts[0]['timestamp']
        if dt <= 0.01: return 0.0
        return np.sqrt(dx**2 + dy**2) / dt

    pre_vel = get_vel(pre_pts) or 0.0
    post_vel = get_vel(post_pts)
    
    last_frame = traj[-1]['frame_index'] if traj else 0
    # A track is only considered terminated if it drops before the end of the video
    terminated_soon = (last_frame <= frame_idx + 5) and (last_frame < max_frame - 5)
    
    if post_vel is None:
        if terminated_soon:
            post_vel = 0.0
        else:
            post_vel = pre_vel
            
    return pre_vel, post_vel, terminated_soon

def evaluate_accident(interaction_candidates: List[Dict[str, Any]], track_history: Any) -> Dict[str, Any]:
    """
    Evaluates interaction candidates to determine if an accident occurred.
    Uses interaction metrics plus trajectory kinematics to filter false positives
    like passing vehicles.
    """
    if not interaction_candidates:
        return {
            "accident": False,
            "confidence": 0.0,
            "start_time": None,
            "impact_time": None,
            "end_time": None,
            "involved_object_ids": [],
            "evidence_list": []
        }

    best_candidate = None
    best_score = -1.0
    
    # Calculate the max frame in the whole video to avoid EOF triggering track loss
    all_trajs = [track_history.get_trajectory(tid) for tid in track_history.get_all_track_ids()]
    max_frame = max([t[-1]['frame_index'] for t in all_trajs if t], default=0)

    for cand in interaction_candidates:
        score = 0.0
        
        # 1. Base Signal
        rel_vel = cand.get('max_relative_velocity', 0.0)
        score += min(rel_vel / 50.0, 1.0) * 0.4  # Max 0.4 from velocity
        
        iou = cand.get('peak_iou', 0.0)
        if iou > 0.05:
            score += 0.3
            
        evidence = cand.get('evidence_list', [])
        if 'overlap' in evidence:
            score += 0.2

        # 2. Kinematic Anomaly & Post-Interaction Behavior
        obj_a_id = cand['object_a_id']
        obj_b_id = cand['object_b_id']
        end_frame = cand.get('end_frame', 0)
        
        traj_a = track_history.get_trajectory(obj_a_id)
        traj_b = track_history.get_trajectory(obj_b_id)
        
        pre_a, post_a, term_a = get_velocities_around_frame(traj_a, end_frame, max_frame, window=15)
        pre_b, post_b, term_b = get_velocities_around_frame(traj_b, end_frame, max_frame, window=15)
        
        anomaly_boost = 0.0
        
        # Trajectory disruption / termination (strong collision indicator)
        if term_a or term_b:
            track_loss_boost = 0.6
            
            # P1-A: Guard track_loss boost by checking the surviving object's behavior
            if term_a and not term_b:
                if not (pre_b > 15.0 and post_b < pre_b * 0.5):
                    track_loss_boost = 0.1
            elif term_b and not term_a:
                if not (pre_a > 15.0 and post_a < pre_a * 0.5):
                    track_loss_boost = 0.1

            anomaly_boost += track_loss_boost
            if 'track_loss' not in evidence:
                evidence.append('track_loss')
                
        # Significant deceleration (hit and stop/slow down)
        if not term_a and pre_a > 15.0 and post_a < pre_a * 0.3:
            anomaly_boost += 0.3
            if 'sudden_deceleration' not in evidence:
                evidence.append('sudden_deceleration')
                
        if not term_b and pre_b > 15.0 and post_b < pre_b * 0.3:
            anomaly_boost += 0.3
            if 'sudden_deceleration' not in evidence:
                evidence.append('sudden_deceleration')

        score += anomaly_boost
        
        # 3. Filters
        # Duplicate track filter: high IoU with low relative velocity
        if iou > 0.85 and rel_vel < 30.0:
            score *= 0.1
            
        # P1-B: Normal Passing Car Filter
        # If surviving objects maintained >50% of their pre-interaction velocity,
        # it is almost certainly a normal pass.
        if not term_a and not term_b:
            if pre_a > 0 and pre_b > 0:
                if post_a > pre_a * 0.5 and post_b > pre_b * 0.5:
                    score *= 0.1  # Heavy penalty for mutual passing
        elif term_a and not term_b:
            if pre_b > 0 and post_b > pre_b * 0.5:
                score *= 0.3  # Penalty for surviving object passing normally
        elif term_b and not term_a:
            if pre_a > 0 and post_a > pre_a * 0.5:
                score *= 0.3  # Penalty for surviving object passing normally
                    
        # Update evidence list
        cand['evidence_list'] = evidence

        if score > best_score:
            best_score = score
            best_candidate = cand

    # Threshold for reporting a confident accident
    if best_candidate and best_score >= 0.5:
        confidence = min(best_score, 1.0)
        
        # --- Dynamic Temporal Clustering ---
        # Find all candidates within +/- 3.0 seconds that share at least one ID
        cluster_cands = []
        target_ids = {best_candidate['object_a_id'], best_candidate['object_b_id']}
        peak_time = best_candidate.get('peak_time', 0.0)
        
        for cand in interaction_candidates:
            # We look for related interactions within a short temporal window
            if abs(cand.get('peak_time', 0.0) - peak_time) <= 3.0:
                cand_ids = {cand['object_a_id'], cand['object_b_id']}
                if cand_ids.intersection(target_ids):
                    cluster_cands.append(cand)
                    
        # Update event bounds based on the entire cluster of interactions
        start_time = min((c.get('start_time', best_candidate.get('start_time')) for c in cluster_cands))
        end_time = max((c.get('end_time', best_candidate.get('end_time')) for c in cluster_cands))
        
        # Include all uniquely involved objects from the cluster
        involved_ids = set()
        for c in cluster_cands:
            involved_ids.add(c['object_a_id'])
            involved_ids.add(c['object_b_id'])
            
        return {
            "accident": True,
            "confidence": round(confidence, 2),
            "start_time": start_time,
            "impact_time": best_candidate.get('peak_time'),
            "end_time": end_time,
            "involved_object_ids": list(involved_ids),
            "evidence_list": best_candidate.get('evidence_list', [])
        }
        
    return {
        "accident": False,
        "confidence": round(max(best_score, 0.0), 2),
        "start_time": None,
        "impact_time": None,
        "end_time": None,
        "involved_object_ids": [],
        "evidence_list": []
    }
