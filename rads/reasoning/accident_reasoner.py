from typing import List, Dict, Any

def evaluate_accident(interaction_candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Evaluates interaction candidates to determine if an accident occurred.
    For MVP, we select the candidate with the highest evidence score as the primary event.
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

    for cand in interaction_candidates:
        # Simple heuristic score:
        score = 0.0
        
        # High relative velocity adds to score
        rel_vel = cand.get('max_relative_velocity', 0.0)
        score += min(rel_vel / 50.0, 1.0) * 0.5  # Up to 0.5 for velocity
        
        # Overlap adds significantly
        iou = cand.get('peak_iou', 0.0)
        if iou > 0.05:
            score += 0.5
            
        # Evidence flags
        if 'overlap' in cand.get('evidence_list', []):
            score += 0.2
            
        if score > best_score:
            best_score = score
            best_candidate = cand

    if best_candidate and best_score >= 0.4:
        confidence = min(best_score, 1.0)
        return {
            "accident": True,
            "confidence": confidence,
            "start_time": best_candidate.get('start_time'),
            "impact_time": best_candidate.get('peak_time'),
            "end_time": best_candidate.get('end_time'),
            "involved_object_ids": [best_candidate['object_a_id'], best_candidate['object_b_id']],
            "evidence_list": best_candidate.get('evidence_list', [])
        }
        
    return {
        "accident": False,
        "confidence": max(best_score, 0.0),
        "start_time": None,
        "impact_time": None,
        "end_time": None,
        "involved_object_ids": [],
        "evidence_list": []
    }
