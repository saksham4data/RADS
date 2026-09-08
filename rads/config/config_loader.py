import yaml
import os
from typing import Dict, Any, List

class ConfigLoader:
    """Loads and provides typed access to the pipeline configuration."""

    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self._load_yaml(config_path)

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

    @property
    def interaction_convergence_angle(self) -> float:
        return float(self.config.get('interaction', {}).get('convergence_angle_threshold', 45.0))

