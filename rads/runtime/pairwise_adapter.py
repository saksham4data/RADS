from rads.interaction.pairwise import compute_pairwise_metrics, enrich_with_motion
from rads.motion.trajectory import TrackHistory


def compute_pairwise_metrics_windowed(track_history, active_track_ids, window_frames):
    scoped = TrackHistory()
    for track_id in active_track_ids:
        traj = list(track_history.get_trajectory(track_id))
        if window_frames is not None and traj:
            cutoff = traj[-1]['frame_index'] - window_frames + 1
            traj = [pt for pt in traj if pt['frame_index'] >= cutoff]
        scoped.history[track_id] = traj
    pairwise = compute_pairwise_metrics(scoped)
    enrich_with_motion(pairwise)
    return pairwise
