from rads.core import (
    TrackHistory,
    Tracker,
    build_event_result,
    detect_interactions,
    estimate_severity,
    evaluate_accident,
)
from rads.runtime.motion_adapter import compute_motion_features_incremental
from rads.runtime.pairwise_adapter import compute_pairwise_metrics_windowed


class FrameProcessor:
    def __init__(self, config):
        self.config = config
        self.tracker = Tracker(
            model_path=config.device_model_path,
            tracker_type=config.tracker_type,
            confidence=config.detector_confidence,
            iou=config.detector_iou,
            classes=config.detector_classes,
        )
        self.track_history = TrackHistory()
        self._pairwise = {}
        self._interaction_candidates = []
        self._frame_count = 0
        self._last_timestamp = None
        self._last_reasoning_result = None
        self._motion_features = {}
        self._prev_timestamp = None
        self._prev_frame_idx = None
        self._frame_dt = None
        self._sliding_window_s = config.runtime_sliding_window_s

    def process_frame(self, frame, frame_idx, timestamp_s):
        tracked_objects = self.tracker.track(frame, frame_idx, timestamp_s)
        self.track_history.update(tracked_objects, frame_idx, timestamp_s)
        self._prune(timestamp_s)
        active_ids = self.track_history.get_all_track_ids()
        window_frames = self._window_frames(frame_idx, timestamp_s)
        self._pairwise = compute_pairwise_metrics_windowed(
            self.track_history, active_ids, window_frames)
        self._interaction_candidates = detect_interactions(self._pairwise, self.config)
        self._motion_features = compute_motion_features_incremental(
            self.track_history, active_ids)
        self._frame_count += 1
        self._last_timestamp = timestamp_s
        if not self._interaction_candidates:
            return None
        accident_result = evaluate_accident(
            self._interaction_candidates, self.track_history, self.config)
        self._last_reasoning_result = accident_result
        if not accident_result.get('accident'):
            return None
        severity_result = estimate_severity(
            accident_result, self.track_history, self.config)
        return build_event_result(
            self.config.runtime_source_uri,
            tracks_summary=self._tracks_summary(active_ids),
            accident_result=accident_result,
            severity_result=severity_result,
            interaction_candidates=self._interaction_candidates,
        )

    def reset(self):
        self.tracker.reset()
        self.track_history = TrackHistory()
        self._pairwise = {}
        self._interaction_candidates = []
        self._frame_count = 0
        self._last_timestamp = None
        self._last_reasoning_result = None
        self._motion_features = {}
        self._prev_timestamp = None
        self._prev_frame_idx = None
        self._frame_dt = None

    def get_state(self):
        return {
            'active_track_count': len(self.track_history.get_all_track_ids()),
            'frame_count': self._frame_count,
            'last_reasoning_result': self._last_reasoning_result,
        }

    def _prune(self, timestamp_s):
        cutoff = timestamp_s - self._sliding_window_s
        for track_id in list(self.track_history.history):
            kept = [
                pt for pt in self.track_history.history[track_id]
                if pt['timestamp'] >= cutoff
            ]
            if kept:
                self.track_history.history[track_id] = kept
            else:
                del self.track_history.history[track_id]

    def _window_frames(self, frame_idx, timestamp_s):
        if self._prev_timestamp is not None and timestamp_s > self._prev_timestamp:
            dt = timestamp_s - self._prev_timestamp
            df = frame_idx - self._prev_frame_idx
            if dt > 0 and df > 0:
                self._frame_dt = dt / df
        self._prev_timestamp = timestamp_s
        self._prev_frame_idx = frame_idx
        if not self._frame_dt:
            return None
        return max(1, int(round(self._sliding_window_s / self._frame_dt)))

    def _tracks_summary(self, track_ids):
        summary = {}
        for track_id in track_ids:
            traj = self.track_history.get_trajectory(track_id)
            if not traj:
                continue
            entry = {
                'class_name': traj[-1].get('class_name', 'unknown'),
                'first_frame': traj[0]['frame_index'],
                'last_frame': traj[-1]['frame_index'],
                'frame_count': len(traj),
            }
            if track_id in self._motion_features:
                entry['motion_features'] = self._motion_features[track_id]
            summary[track_id] = entry
        return summary
