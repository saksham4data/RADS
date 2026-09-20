import math
import numpy as np
from typing import List, Dict, Any, Optional

from rads.config.config_loader import ConfigLoader

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

def evaluate_accident(interaction_candidates: List[Dict[str, Any]], track_history: Any,
                      config: Optional[ConfigLoader] = None) -> Dict[str, Any]:
    """
    Evaluates interaction candidates to determine if an accident occurred.
    Uses interaction metrics plus trajectory kinematics to filter false positives
    like passing vehicles. All weights and thresholds come from the 'reasoning'
    section of the pipeline config.
    """
    if config is None:
        config = ConfigLoader.defaults()

    threshold = config.reasoning_accident_threshold
    vel_window = config.reasoning_velocity_window_frames

    if not interaction_candidates:
        return {
            "accident": False,
            "score": 0.0,
            "confidence": 0.0,
            "start_time": None,
            "impact_time": None,
            "end_time": None,
            "involved_object_ids": [],
            "evidence_list": [],
            "kinematics": {}
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
        score += min(rel_vel / config.reasoning_relative_velocity_divisor, 1.0) * config.reasoning_weight_relative_velocity
        
        iou = cand.get('peak_iou', 0.0)
        if iou > config.reasoning_iou_gate:
            score += config.reasoning_weight_iou
            
        evidence = cand.get('evidence_list', [])
        if 'overlap' in evidence:
            score += config.reasoning_weight_overlap_evidence

        # 2. Kinematic Anomaly & Post-Interaction Behavior
        obj_a_id = cand['object_a_id']
        obj_b_id = cand['object_b_id']
        end_frame = cand.get('end_frame', 0)
        
        traj_a = track_history.get_trajectory(obj_a_id)
        traj_b = track_history.get_trajectory(obj_b_id)
        
        pre_a, post_a, term_a = get_velocities_around_frame(traj_a, end_frame, max_frame, window=vel_window)
        pre_b, post_b, term_b = get_velocities_around_frame(traj_b, end_frame, max_frame, window=vel_window)
        
        anomaly_boost = 0.0
        
        # Trajectory disruption / termination (strong collision indicator)
        if term_a or term_b:
            track_loss_boost = config.reasoning_track_loss_boost
            guard_pre = config.reasoning_track_loss_guard_pre_velocity
            guard_ratio = config.reasoning_track_loss_guard_post_ratio
            
            # P1-A: Guard track_loss boost by checking the surviving object's behavior
            if term_a and not term_b:
                if not (pre_b > guard_pre and post_b < pre_b * guard_ratio):
                    track_loss_boost = config.reasoning_track_loss_unguarded_boost
            elif term_b and not term_a:
                if not (pre_a > guard_pre and post_a < pre_a * guard_ratio):
                    track_loss_boost = config.reasoning_track_loss_unguarded_boost

            anomaly_boost += track_loss_boost
            if 'track_loss' not in evidence:
                evidence.append('track_loss')
                
        # Significant deceleration (hit and stop/slow down)
        decel_pre = config.reasoning_deceleration_pre_velocity
        decel_ratio = config.reasoning_deceleration_post_ratio
        if not term_a and pre_a > decel_pre and post_a < pre_a * decel_ratio:
            anomaly_boost += config.reasoning_deceleration_boost
            if 'sudden_deceleration' not in evidence:
                evidence.append('sudden_deceleration')
                
        if not term_b and pre_b > decel_pre and post_b < pre_b * decel_ratio:
            anomaly_boost += config.reasoning_deceleration_boost
            if 'sudden_deceleration' not in evidence:
                evidence.append('sudden_deceleration')

        score += anomaly_boost
        
        # 3. Filters
        # Duplicate track filter: high IoU with low relative velocity
        if iou > config.reasoning_duplicate_track_iou and rel_vel < config.reasoning_duplicate_track_rel_velocity:
            score *= config.reasoning_duplicate_track_scale
            
        # P1-B: Normal Passing Car Filter
        # If surviving objects maintained more than the retention ratio of their
        # pre-interaction velocity, it is almost certainly a normal pass.
        retention = config.reasoning_passing_velocity_retention
        if not term_a and not term_b:
            if pre_a > 0 and pre_b > 0:
                if post_a > pre_a * retention and post_b > pre_b * retention:
                    score *= config.reasoning_passing_both_survive_scale
        elif term_a and not term_b:
            if pre_b > 0 and post_b > pre_b * retention:
                score *= config.reasoning_passing_one_survives_scale
        elif term_b and not term_a:
            if pre_a > 0 and post_a > pre_a * retention:
                score *= config.reasoning_passing_one_survives_scale
                    
        # Update evidence list
        cand['evidence_list'] = evidence
        cand['score'] = score
        cand['kinematics'] = {
            "pre_velocity_a": float(pre_a),
            "post_velocity_a": float(post_a),
            "pre_velocity_b": float(pre_b),
            "post_velocity_b": float(post_b),
            "track_terminated_a": bool(term_a),
            "track_terminated_b": bool(term_b)
        }

        if score > best_score:
            best_score = score
            best_candidate = cand

    # Threshold for reporting a confident accident
    if best_candidate and best_score >= threshold:
        cluster_cands = _cluster_candidates(interaction_candidates, best_candidate, config)

        # Update event bounds based on the entire cluster of interactions
        start_time = min((c.get('start_time', best_candidate.get('start_time')) for c in cluster_cands))
        end_time = max((c.get('end_time', best_candidate.get('end_time')) for c in cluster_cands))
        
        involved_ids = _involved_ids(cluster_cands, best_candidate, config)

        return {
            "accident": True,
            "score": round(best_score, 4),
            "confidence": _confidence(best_score, config),
            "start_time": start_time,
            "impact_time": best_candidate.get('peak_time'),
            "end_time": end_time,
            "involved_object_ids": involved_ids,
            "evidence_list": best_candidate.get('evidence_list', []),
            "kinematics": _event_kinematics(cluster_cands, best_candidate)
        }
        
    return {
        "accident": False,
        "score": round(max(best_score, 0.0), 4),
        "confidence": _confidence(max(best_score, 0.0), config),
        "start_time": None,
        "impact_time": None,
        "end_time": None,
        "involved_object_ids": [],
        "evidence_list": [],
        "kinematics": {}
    }

def _confidence(score: float, config: ConfigLoader) -> float:
    """B12: bounded monotone transform of the raw score, so confidence does not saturate.

    The accident decision is taken on the raw score, never on this value.
    """
    score = max(score, 0.0)
    if not config.reasoning_confidence_bounded_transform:
        return round(min(score, 1.0), 2)
    scale = config.reasoning_confidence_transform_scale
    if scale <= 0:
        return round(min(score, 1.0), 2)
    return round(1.0 - math.exp(-score / scale), 4)

def _cluster_candidates(candidates: List[Dict[str, Any]], best: Dict[str, Any],
                        config: ConfigLoader) -> List[Dict[str, Any]]:
    """Candidates within the clustering window that share an id with the best candidate."""
    window = config.reasoning_clustering_window_seconds
    require_evidence = config.reasoning_clustering_require_own_evidence
    qualifying = set(config.reasoning_clustering_qualifying_evidence)

    target_ids = {best['object_a_id'], best['object_b_id']}
    peak_time = best.get('peak_time', 0.0)
    cluster = []

    for cand in candidates:
        if abs(cand.get('peak_time', 0.0) - peak_time) > window:
            continue
        cand_ids = {cand['object_a_id'], cand['object_b_id']}
        if not cand_ids.intersection(target_ids):
            continue
        # B11: sharing an id is not enough, the candidate must stand on its own evidence
        if require_evidence and cand is not best:
            if not qualifying.intersection(cand.get('evidence_list', [])):
                continue
        cluster.append(cand)

    return cluster if cluster else [best]

def _involved_ids(cluster: List[Dict[str, Any]], best: Dict[str, Any],
                  config: ConfigLoader) -> List[int]:
    """Unique ids across the cluster, capped to the strongest-scoring candidates."""
    ids = []
    for cand in sorted(cluster, key=lambda c: c.get('score', 0.0), reverse=True):
        for obj_id in (cand['object_a_id'], cand['object_b_id']):
            if obj_id not in ids:
                ids.append(obj_id)

    cap = config.reasoning_clustering_max_involved
    if config.reasoning_clustering_require_own_evidence and cap > 0:
        ids = ids[:max(cap, 2)]
    return ids

def _event_kinematics(cluster: List[Dict[str, Any]], best: Dict[str, Any]) -> Dict[str, Any]:
    """Quantities the severity engine needs, measured during scoring rather than recomputed."""
    peak_rel_vel = max((c.get('max_relative_velocity', 0.0) for c in cluster), default=0.0)
    peak_iou = max((c.get('peak_iou', 0.0) for c in cluster), default=0.0)

    velocity_drop = 0.0
    for cand in cluster:
        kin = cand.get('kinematics', {})
        for pre_key, post_key in (("pre_velocity_a", "post_velocity_a"), ("pre_velocity_b", "post_velocity_b")):
            pre = kin.get(pre_key, 0.0)
            post = kin.get(post_key, 0.0)
            velocity_drop = max(velocity_drop, pre - post)

    return {
        "peak_relative_velocity": float(peak_rel_vel),
        "peak_iou": float(peak_iou),
        "max_velocity_drop": float(velocity_drop),
        "impact_frame": best.get('end_frame'),
        "best_candidate": {
            "object_a_id": best['object_a_id'],
            "object_b_id": best['object_b_id'],
            "evidence_list": list(best.get('evidence_list', []))
        }
    }
