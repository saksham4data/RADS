"""Rule-based severity scoring for a detected accident event.

This is an engineered heuristic. It has no ground-truth validation: the datasets in use
carry no severity labels, so the weights and cut-offs below are chosen for interpretability,
not fitted to data. Per MVP.md section 21 and TECH_STACK.md section 14, the output must not
be presented as a validated severity prediction.

score = w_count   * min(max(0, n_objects - 1), count_cap)
      + w_vuln    * [any involved object is person / bicycle / motorcycle]
      + w_relvel  * min(peak_relative_velocity / relvel_divisor, 1.0)
      + w_dvel    * min(max_velocity_drop / dvel_divisor, 1.0)
      + w_disp    * min(post_impact_displacement / disp_divisor, 1.0)

severity = HIGH if score >= thresholds.high, MEDIUM if score >= thresholds.medium, else LOW.

Defaults in rads/config/pipeline_config.yaml (section 'severity'): object_count 1.0 capped
at 2.0, vulnerable_class 3.0, relative_velocity 2.0 over divisor 600.0, velocity_change 1.5
over divisor 400.0, post_impact_displacement 1.0 over divisor 250.0, high 4.5, medium 2.0.
Maximum attainable score is 9.5. The divisors are pixel-space and were set from the
magnitudes observed on the 10-clip regression sample, not fitted to severity labels.

Peak relative velocity and the velocity drop are measured by the reasoner during scoring and
passed forward in the 'kinematics' field of its result; they are not recomputed here.
Post-impact displacement is the mean centroid travel of the involved tracks over the
post_impact_window_frames following the impact frame, in pixels.
"""

import math
from typing import Dict, Any, List, Optional

from rads.config.config_loader import ConfigLoader

def _post_impact_displacement(track_history, involved_ids: List[int], impact_frame: Optional[int],
                              window: int) -> float:
    """Mean centroid travel, in pixels, of the involved tracks after the impact frame."""
    if impact_frame is None or not involved_ids:
        return 0.0

    displacements = []
    for obj_id in involved_ids:
        traj = track_history.get_trajectory(obj_id)
        pts = [p for p in traj if impact_frame <= p['frame_index'] <= impact_frame + window]
        if len(pts) < 2:
            continue
        dx = pts[-1]['cx'] - pts[0]['cx']
        dy = pts[-1]['cy'] - pts[0]['cy']
        displacements.append(math.hypot(dx, dy))

    if not displacements:
        return 0.0
    return sum(displacements) / len(displacements)

def estimate_severity(accident_result: Dict[str, Any], track_history,
                      config: Optional[ConfigLoader] = None) -> Dict[str, Any]:
    """Returns {severity, score, evidence: [{factor, value, contribution}, ...]}.

    severity is None when no accident was reported.
    """
    if config is None:
        config = ConfigLoader.defaults()

    if not accident_result.get('accident'):
        return {"severity": None, "score": 0.0, "evidence": []}

    weights = config.severity_weights
    vulnerable_classes = set(config.severity_vulnerable_classes)
    involved_ids = accident_result.get('involved_object_ids', [])
    kinematics = accident_result.get('kinematics', {}) or {}

    evidence = []
    score = 0.0

    num_objects = len(involved_ids)
    count_value = max(0, num_objects - 1)
    count_contribution = min(count_value * weights['object_count'], weights['object_count_cap'])
    score += count_contribution
    evidence.append({"factor": "num_objects_involved", "value": num_objects,
                     "contribution": round(count_contribution, 4)})

    involved_classes = []
    for obj_id in involved_ids:
        traj = track_history.get_trajectory(obj_id)
        if traj:
            involved_classes.append(traj[0].get('class_name', 'unknown'))
    has_vulnerable = any(c in vulnerable_classes for c in involved_classes)
    vulnerable_contribution = weights['vulnerable_class'] if has_vulnerable else 0.0
    score += vulnerable_contribution
    evidence.append({"factor": "vulnerable_class_involved",
                     "value": sorted({c for c in involved_classes if c in vulnerable_classes}),
                     "contribution": round(vulnerable_contribution, 4)})

    peak_rel_vel = float(kinematics.get('peak_relative_velocity', 0.0))
    relvel_contribution = min(peak_rel_vel / weights['relative_velocity_divisor'], 1.0) * weights['relative_velocity']
    score += relvel_contribution
    evidence.append({"factor": "peak_relative_velocity", "value": round(peak_rel_vel, 2),
                     "contribution": round(relvel_contribution, 4)})

    velocity_drop = float(kinematics.get('max_velocity_drop', 0.0))
    dvel_contribution = min(max(velocity_drop, 0.0) / weights['velocity_change_divisor'], 1.0) * weights['velocity_change']
    score += dvel_contribution
    evidence.append({"factor": "velocity_change", "value": round(velocity_drop, 2),
                     "contribution": round(dvel_contribution, 4)})

    displacement = _post_impact_displacement(track_history, involved_ids,
                                             kinematics.get('impact_frame'),
                                             config.severity_post_impact_window_frames)
    disp_contribution = min(displacement / weights['post_impact_displacement_divisor'], 1.0) * weights['post_impact_displacement']
    score += disp_contribution
    evidence.append({"factor": "post_impact_displacement", "value": round(displacement, 2),
                     "contribution": round(disp_contribution, 4)})

    if score >= config.severity_threshold_high:
        severity = 'HIGH'
    elif score >= config.severity_threshold_medium:
        severity = 'MEDIUM'
    else:
        severity = 'LOW'

    return {"severity": severity, "score": round(score, 4), "evidence": evidence}
