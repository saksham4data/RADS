from rads.motion.motion_features import compute_motion_features
from rads.motion.trajectory import TrackHistory


def compute_motion_features_incremental(track_history, track_ids):
    scoped = TrackHistory()
    for track_id in track_ids:
        scoped.history[track_id] = track_history.get_trajectory(track_id)
    return compute_motion_features(scoped)
