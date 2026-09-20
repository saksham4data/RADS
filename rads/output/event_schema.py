from typing import Dict, Any, List, Optional

def build_event_result(video_path: str,
                       tracks_summary: Dict[int, Any] = None,
                       accident_result: Dict[str, Any] = None,
                       severity_result: Dict[str, Any] = None,
                       interaction_candidates: List[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Builds the structured event result per SYSTEM.md section 21 and MASTER_SPEC.md section 11.

    Args:
        video_path: Path to the processed video.
        tracks_summary: track_id to metadata, e.g. {1: {"class_name": "car", "frame_count": 50}}.
        accident_result: output of rads.reasoning.accident_reasoner.evaluate_accident.
        severity_result: output of rads.severity.severity_engine.estimate_severity.
        interaction_candidates: candidates from the interaction stage.

    'severity' stays a plain string ('LOW'/'MEDIUM'/'HIGH' or None) because the evaluator,
    the visualizer and the existing JSONL consumers read it directly; the numeric score and
    factor breakdown live in 'severity_detail'.
    """
    tracks_summary = tracks_summary or {}
    accident_result = accident_result or {}
    severity_result = severity_result or {}
    interaction_candidates = interaction_candidates if interaction_candidates is not None else []

    involved_ids = accident_result.get("involved_object_ids", [])

    return {
        "video": video_path,
        "accident": accident_result.get("accident"),
        "confidence": accident_result.get("confidence"),
        "score": accident_result.get("score"),
        "event": {
            "start_time": accident_result.get("start_time"),
            "impact_time": accident_result.get("impact_time"),
            "end_time": accident_result.get("end_time")
        },
        "objects_involved": _objects_involved(involved_ids, tracks_summary),
        "accident_type": "unknown",   # post-MVP: classification is not implemented
        "evidence_list": accident_result.get("evidence_list", []),
        "kinematics": accident_result.get("kinematics", {}),
        "severity": severity_result.get("severity"),
        "severity_detail": {
            "score": severity_result.get("score"),
            "evidence": severity_result.get("evidence", [])
        },
        "interaction_candidates": interaction_candidates,
        "tracks_summary": tracks_summary
    }

def _objects_involved(involved_ids: List[int], tracks_summary: Dict[int, Any]) -> List[Dict[str, Any]]:
    objects = []
    for obj_id in involved_ids:
        summary = tracks_summary.get(obj_id) or {}
        objects.append({"id": int(obj_id), "class": summary.get("class_name", "unknown")})
    return objects

def involved_ids_from_result(event_result: Dict[str, Any]) -> set:
    """Track ids from 'objects_involved', which holds {id, class} objects."""
    ids = set()
    for obj in event_result.get("objects_involved", []) or []:
        if isinstance(obj, dict):
            if obj.get("id") is not None:
                ids.add(obj["id"])
        else:
            ids.add(obj)
    return ids
