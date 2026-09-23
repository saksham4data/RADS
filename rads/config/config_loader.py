import yaml
import os
from typing import Dict, Any, List

class ConfigLoader:
    """Loads and provides typed access to the pipeline configuration."""

    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self._load_yaml(config_path)

    @classmethod
    def defaults(cls) -> 'ConfigLoader':
        """Loader backed by an empty document, so every property falls back to its default."""
        loader = cls.__new__(cls)
        loader.config_path = None
        loader.config = {}
        return loader

    def _load_yaml(self, path: str) -> Dict[str, Any]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Configuration file not found: {path}")
        with open(path, 'r') as f:
            try:
                return yaml.safe_load(f)
            except yaml.YAMLError as exc:
                raise ValueError(f"Error parsing YAML config: {exc}")

    @property
    def detector_model(self) -> str:
        return self.config.get('detector', {}).get('model', 'yolo11n.pt')

    @property
    def detector_confidence(self) -> float:
        return float(self.config.get('detector', {}).get('confidence', 0.25))

    @property
    def detector_iou(self) -> float:
        return float(self.config.get('detector', {}).get('iou', 0.45))

    @property
    def detector_classes(self) -> List[int]:
        return self.config.get('detector', {}).get('classes', [0, 1, 2, 3, 5, 7])

    @property
    def tracker_type(self) -> str:
        return self.config.get('tracker', {}).get('type', 'bytetrack')

    @property
    def frame_skip(self) -> int:
        return int(self.config.get('pipeline', {}).get('frame_skip', 1))

    @property
    def interaction_normalized_proximity(self) -> float:
        return float(self.config.get('interaction', {}).get('normalized_proximity_threshold', 1.5))

    @property
    def interaction_min_persistence_frames(self) -> int:
        return int(self.config.get('interaction', {}).get('min_persistence_frames', 5))

    @property
    def interaction_overlap_iou(self) -> float:
        return float(self.config.get('interaction', {}).get('overlap_iou_threshold', 0.1))

    @property
    def interaction_relative_velocity(self) -> float:
        return float(self.config.get('interaction', {}).get('relative_velocity_threshold', 5.0))

    def _nested(self, section: str, keys, default):
        node = self.config.get(section) or {}
        for key in keys:
            if not isinstance(node, dict) or key not in node:
                return default
            node = node[key]
        return node

    def _reasoning(self, *keys, default=None):
        return self._nested('reasoning', keys, default)

    @property
    def reasoning_accident_threshold(self) -> float:
        return float(self._reasoning('accident_score_threshold', default=0.5))

    @property
    def reasoning_velocity_window_frames(self) -> int:
        return int(self._reasoning('velocity_window_frames', default=15))

    @property
    def reasoning_clustering_window_seconds(self) -> float:
        return float(self._reasoning('clustering_window_seconds', default=3.0))

    @property
    def reasoning_weight_relative_velocity(self) -> float:
        return float(self._reasoning('score_weights', 'relative_velocity', default=0.4))

    @property
    def reasoning_relative_velocity_divisor(self) -> float:
        return float(self._reasoning('score_weights', 'relative_velocity_divisor', default=50.0))

    @property
    def reasoning_weight_iou(self) -> float:
        return float(self._reasoning('score_weights', 'iou', default=0.3))

    @property
    def reasoning_iou_gate(self) -> float:
        return float(self._reasoning('score_weights', 'iou_gate', default=0.05))

    @property
    def reasoning_weight_overlap_evidence(self) -> float:
        return float(self._reasoning('score_weights', 'overlap_evidence', default=0.2))

    @property
    def reasoning_track_loss_boost(self) -> float:
        return float(self._reasoning('track_loss', 'boost', default=0.6))

    @property
    def reasoning_track_loss_unguarded_boost(self) -> float:
        return float(self._reasoning('track_loss', 'unguarded_boost', default=0.1))

    @property
    def reasoning_track_loss_guard_pre_velocity(self) -> float:
        return float(self._reasoning('track_loss', 'guard_pre_velocity', default=15.0))

    @property
    def reasoning_track_loss_guard_post_ratio(self) -> float:
        return float(self._reasoning('track_loss', 'guard_post_velocity_ratio', default=0.5))

    @property
    def reasoning_deceleration_boost(self) -> float:
        return float(self._reasoning('deceleration', 'boost', default=0.3))

    @property
    def reasoning_deceleration_pre_velocity(self) -> float:
        return float(self._reasoning('deceleration', 'pre_velocity', default=15.0))

    @property
    def reasoning_deceleration_post_ratio(self) -> float:
        return float(self._reasoning('deceleration', 'post_velocity_ratio', default=0.3))

    @property
    def reasoning_duplicate_track_iou(self) -> float:
        return float(self._reasoning('filters', 'duplicate_track_iou', default=0.85))

    @property
    def reasoning_duplicate_track_rel_velocity(self) -> float:
        return float(self._reasoning('filters', 'duplicate_track_rel_velocity', default=30.0))

    @property
    def reasoning_duplicate_track_scale(self) -> float:
        return float(self._reasoning('filters', 'duplicate_track_scale', default=0.1))

    @property
    def reasoning_passing_velocity_retention(self) -> float:
        return float(self._reasoning('filters', 'passing_velocity_retention', default=0.5))

    @property
    def reasoning_passing_both_survive_scale(self) -> float:
        return float(self._reasoning('filters', 'passing_both_survive_scale', default=0.1))

    @property
    def reasoning_passing_one_survives_scale(self) -> float:
        return float(self._reasoning('filters', 'passing_one_survives_scale', default=0.3))

    @property
    def reasoning_clustering_require_own_evidence(self) -> bool:
        return bool(self._reasoning('clustering', 'require_own_evidence', default=True))

    @property
    def reasoning_clustering_max_involved(self) -> int:
        return int(self._reasoning('clustering', 'max_involved_objects', default=4))

    @property
    def reasoning_clustering_qualifying_evidence(self) -> List[str]:
        value = self._reasoning('clustering', 'qualifying_evidence',
                                default=['overlap', 'proximity_and_convergence'])
        return list(value)

    @property
    def reasoning_confidence_bounded_transform(self) -> bool:
        return bool(self._reasoning('confidence', 'bounded_transform', default=True))

    @property
    def reasoning_confidence_transform_scale(self) -> float:
        return float(self._reasoning('confidence', 'transform_scale', default=0.5))

    def _severity(self, *keys, default=None):
        return self._nested('severity', keys, default)

    @property
    def severity_weights(self) -> Dict[str, float]:
        defaults = {
            'object_count': 1.0,
            'object_count_cap': 2.0,
            'vulnerable_class': 3.0,
            'relative_velocity': 2.0,
            'relative_velocity_divisor': 60.0,
            'velocity_change': 1.5,
            'velocity_change_divisor': 40.0,
            'post_impact_displacement': 1.0,
            'post_impact_displacement_divisor': 80.0,
        }
        configured = self._severity('weights', default={}) or {}
        defaults.update({k: float(v) for k, v in configured.items()})
        return defaults

    @property
    def severity_vulnerable_classes(self) -> List[str]:
        return list(self._severity('vulnerable_classes',
                                   default=['person', 'bicycle', 'motorcycle']))

    @property
    def severity_post_impact_window_frames(self) -> int:
        return int(self._severity('post_impact_window_frames', default=15))

    @property
    def severity_threshold_high(self) -> float:
        return float(self._severity('thresholds', 'high', default=4.5))

    @property
    def severity_threshold_medium(self) -> float:
        return float(self._severity('thresholds', 'medium', default=2.0))

    @property
    def visualization_summary_frame(self) -> bool:
        return bool(self.config.get('visualization', {}).get('summary_frame', True))

    @property
    def visualization_summary_frame_seconds(self) -> float:
        return float(self.config.get('visualization', {}).get('summary_frame_seconds', 2.0))

    @property
    def runtime_source(self) -> str:
        return self.config.get('runtime', {}).get('source', 'file')

    @property
    def runtime_source_uri(self) -> str:
        return self.config.get('runtime', {}).get('source_uri', '')

    @property
    def runtime_reconnect_interval_s(self) -> int:
        return int(self.config.get('runtime', {}).get('reconnect_interval_s', 5))

    @property
    def runtime_max_reconnect_attempts(self) -> int:
        return int(self.config.get('runtime', {}).get('max_reconnect_attempts', -1))

    @property
    def runtime_sliding_window_s(self) -> int:
        return int(self.config.get('runtime', {}).get('sliding_window_s', 30))

    @property
    def runtime_event_confirmation_window_s(self) -> float:
        return float(self.config.get('runtime', {}).get('event_confirmation_window_s', 2.0))

    @property
    def runtime_event_resolution_timeout_s(self) -> float:
        return float(self.config.get('runtime', {}).get('event_resolution_timeout_s', 10.0))

    @property
    def device_compute(self) -> str:
        return self.config.get('device', {}).get('compute', 'auto')

    @property
    def device_model_path(self) -> str:
        return self.config.get('device', {}).get('model_path', 'models/yolo11n.pt')

    @property
    def api_enabled(self) -> bool:
        return bool(self.config.get('api', {}).get('enabled', False))

    @property
    def api_host(self) -> str:
        return self.config.get('api', {}).get('host', '0.0.0.0')

    @property
    def api_port(self) -> int:
        return int(self.config.get('api', {}).get('port', 8100))

    @property
    def api_event_buffer_size(self) -> int:
        return int(self.config.get('api', {}).get('event_buffer_size', 100))

