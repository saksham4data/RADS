"""
Scene classifier for the Metadata Generation Pipeline.

Classifies day/night and weather conditions from sampled video/image frames
using purely pixel-level OpenCV heuristics. No external models or networks.

Design (user-approved hybrid approach):
    - Each prediction is accompanied by a confidence score in [0.0, 1.0].
    - Predictions with confidence < config.confidence_threshold are set to NULL
      rather than emitting a low-confidence guess that could corrupt training.
    - Confidence scores are always written to the CSV regardless of threshold,
      so downstream users can apply their own filtering.

Heuristics used:
    day_time  — mean grayscale brightness across sampled frames.
                Bright → "day", dark → "night".
    weather   — mean HSV saturation + grayscale contrast (std dev).
                Low contrast → "fog", low saturation (non-fog) → "rain",
                otherwise → "clear".

Limitations (intentionally documented):
    - These are NOT classifier model outputs — they are thresholded pixel stats.
    - Nighttime urban scenes with streetlights may be misclassified as "day".
    - Heavy rain at night is indistinguishable from dry night by saturation alone.
    - Short clips with strong lighting changes may produce averaged-out results.
    - Treat all outputs as best-effort estimates, not ground truth.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from metadata_gen.adapters.base_adapter import MediaFile
from metadata_gen.config import MetadataGenConfig


logger = logging.getLogger(__name__)


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class SceneClassification:
    """
    Result of a single scene classification pass.

    Fields are None when:
      - No frames were available (empty input).
      - Confidence fell below config.confidence_threshold.
      - Scene classification is disabled in config.

    These map directly to the MediaFile fields that will be written to CSV.
    """
    day_time: Optional[str]             # "day" | "night" | None
    day_time_confidence: Optional[float]
    weather: Optional[str]              # "clear" | "rain" | "fog" | None
    weather_confidence: Optional[float]

    # Raw pixel features (for logging / debugging; not written to CSV)
    mean_brightness: Optional[float] = None
    mean_saturation: Optional[float] = None
    mean_contrast: Optional[float] = None

    def __str__(self) -> str:
        return (
            f"SceneClassification("
            f"day_time={self.day_time!r}({self.day_time_confidence:.2f}), "
            f"weather={self.weather!r}({self.weather_confidence:.2f})"
            f")"
            if self.day_time_confidence is not None and self.weather_confidence is not None
            else "SceneClassification(no_result)"
        )


# ── Classifier ────────────────────────────────────────────────────────────────

class SceneClassifier:
    """
    Pixel-heuristic scene classifier.

    Usage:
        classifier = SceneClassifier(config)
        result = classifier.classify(frames)
        classifier.apply(media_file, frames)   # convenience: classify + write to MediaFile
    """

    def __init__(self, config: MetadataGenConfig) -> None:
        """
        Args:
            config: Pipeline 0 configuration. Used thresholds:
                - night_brightness_threshold   (default 60.0)
                - rain_saturation_threshold    (default 50.0)
                - fog_contrast_threshold       (default 30.0)
                - confidence_threshold         (default 0.40)
        """
        self.config = config

    # ── Public interface ──────────────────────────────────────────────────

    def classify(self, frames: List[np.ndarray]) -> SceneClassification:
        """
        Classify scene conditions from a list of BGR frames.

        Averages pixel features across all frames before classifying,
        which suppresses noise from any single outlier frame.

        Args:
            frames: List of BGR np.ndarray frames (H×W×3), as returned
                    by VideoAnalyzer or ImageAnalyzer. May be empty.

        Returns:
            SceneClassification with predictions and confidence scores.
            Fields are None when confidence < threshold or input is empty.
        """
        null_result = SceneClassification(
            day_time=None, day_time_confidence=None,
            weather=None, weather_confidence=None,
        )

        if not self.config.enable_scene_classification:
            return null_result

        if not frames:
            logger.debug("SceneClassifier: No frames provided — returning NULL result.")
            return null_result

        # ── 1. Extract per-frame pixel features ───────────────────────
        brightness_vals: List[float] = []
        saturation_vals: List[float] = []
        contrast_vals:   List[float] = []

        for i, frame in enumerate(frames):
            if frame is None or frame.size == 0:
                logger.warning("SceneClassifier: Frame %d is None/empty — skipping.", i)
                continue

            b, s, c = _extract_frame_features(frame)
            brightness_vals.append(b)
            saturation_vals.append(s)
            contrast_vals.append(c)

        if not brightness_vals:
            logger.warning(
                "SceneClassifier: All frames were invalid — returning NULL result."
            )
            return null_result

        # ── 2. Average features across frames ─────────────────────────
        mean_brightness = float(np.mean(brightness_vals))
        mean_saturation = float(np.mean(saturation_vals))
        mean_contrast   = float(np.mean(contrast_vals))

        logger.debug(
            "SceneClassifier: features over %d frames — "
            "brightness=%.1f, saturation=%.1f, contrast=%.1f",
            len(brightness_vals), mean_brightness, mean_saturation, mean_contrast,
        )

        # ── 3. Classify day/night ──────────────────────────────────────
        day_time, day_time_confidence = _classify_day_night(
            mean_brightness=mean_brightness,
            threshold=self.config.night_brightness_threshold,
        )

        # ── 4. Classify weather ────────────────────────────────────────
        weather, weather_confidence = _classify_weather(
            mean_saturation=mean_saturation,
            mean_contrast=mean_contrast,
            rain_sat_threshold=self.config.rain_saturation_threshold,
            fog_contrast_threshold=self.config.fog_contrast_threshold,
        )

        # ── 5. Apply confidence threshold → NULL if too uncertain ──────
        min_conf = self.config.confidence_threshold

        if day_time_confidence < min_conf:
            logger.debug(
                "SceneClassifier: day_time confidence %.3f < threshold %.3f → NULL",
                day_time_confidence, min_conf,
            )
            day_time = None

        if weather_confidence < min_conf:
            logger.debug(
                "SceneClassifier: weather confidence %.3f < threshold %.3f → NULL",
                weather_confidence, min_conf,
            )
            weather = None

        result = SceneClassification(
            day_time=day_time,
            day_time_confidence=round(day_time_confidence, 4),
            weather=weather,
            weather_confidence=round(weather_confidence, 4),
            mean_brightness=round(mean_brightness, 2),
            mean_saturation=round(mean_saturation, 2),
            mean_contrast=round(mean_contrast, 2),
        )

        logger.debug("SceneClassifier: %s", result)
        return result

    def apply(self, media_file: MediaFile, frames: List[np.ndarray]) -> None:
        """
        Classify frames and write results directly into the MediaFile.

        Convenience method that combines classify() + field assignment.
        Always sets all four scene fields (day_time, day_time_confidence,
        weather, weather_confidence) — to a value or to None.

        Args:
            media_file: MediaFile to write classification results into.
            frames:     List of BGR frames from the appropriate Analyzer.
        """
        result = self.classify(frames)
        media_file.day_time             = result.day_time
        media_file.day_time_confidence  = result.day_time_confidence
        media_file.weather              = result.weather
        media_file.weather_confidence   = result.weather_confidence


# ── Heuristic functions ───────────────────────────────────────────────────────

def _extract_frame_features(frame: np.ndarray) -> Tuple[float, float, float]:
    """
    Extract the three pixel-level features from a single BGR frame.

    Args:
        frame: BGR np.ndarray of shape (H, W, 3).

    Returns:
        (mean_brightness, mean_saturation, mean_contrast) tuple.
        - mean_brightness: mean grayscale intensity [0, 255].
        - mean_saturation: mean HSV saturation channel value [0, 255].
        - mean_contrast:   std dev of grayscale intensity [0, ~128].
    """
    # Convert to grayscale for brightness and contrast
    if frame.ndim == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame  # Already greyscale

    mean_brightness = float(np.mean(gray))
    mean_contrast   = float(np.std(gray))

    # Convert to HSV for saturation
    if frame.ndim == 3:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # HSV: channel 0=Hue, 1=Saturation, 2=Value
        mean_saturation = float(np.mean(hsv[:, :, 1]))
    else:
        # Greyscale has no colour saturation — treat as 0
        mean_saturation = 0.0

    return mean_brightness, mean_saturation, mean_contrast


def _classify_day_night(
    mean_brightness: float,
    threshold: float,
) -> Tuple[str, float]:
    """
    Classify a frame set as "day" or "night" from mean grayscale brightness.

    Confidence is proportional to how far the brightness is from the
    threshold: 0.0 at the threshold, 1.0 at the extremes (0 or 255).

    Args:
        mean_brightness: Average grayscale intensity across sampled frames.
        threshold:       Brightness boundary between night and day.
                         Frames below this value → "night".

    Returns:
        (label, confidence) where label is "day" or "night".
    """
    if mean_brightness <= threshold:
        # Night: confidence rises as brightness falls toward 0
        # At brightness=0: confidence=1.0; at brightness=threshold: confidence=0.0
        if threshold > 0:
            confidence = float((threshold - mean_brightness) / threshold)
        else:
            confidence = 1.0
        return "night", min(1.0, max(0.0, confidence))
    else:
        # Day: confidence rises as brightness approaches 255
        # At brightness=255: confidence=1.0; at brightness=threshold: confidence=0.0
        remaining = 255.0 - threshold
        if remaining > 0:
            confidence = float((mean_brightness - threshold) / remaining)
        else:
            confidence = 1.0
        return "day", min(1.0, max(0.0, confidence))


def _classify_weather(
    mean_saturation: float,
    mean_contrast: float,
    rain_sat_threshold: float,
    fog_contrast_threshold: float,
) -> Tuple[str, float]:
    """
    Classify weather condition as "fog", "rain", or "clear".

    Priority order: fog is checked first because its signal (low contrast)
    is stronger and more distinct than rain's (low saturation).

    Heuristics:
        fog  — grayscale contrast (std dev) < fog_contrast_threshold.
                Fog creates a uniform grey veil that compresses dynamic range.
        rain — HSV saturation < rain_sat_threshold and not fog.
                Rain desaturates colours through the windshield.
        clear — neither fog nor rain threshold is triggered.

    Confidence:
        Proportional to how far the measured value is from its threshold.
        The confidence for "clear" is the minimum of both non-fog, non-rain
        distances: it's only high when both signals are strongly absent.

    Args:
        mean_saturation:      Average HSV saturation over sampled frames.
        mean_contrast:        Average grayscale std dev over sampled frames.
        rain_sat_threshold:   Saturation below this → potential rain.
        fog_contrast_threshold: Contrast below this → potential fog.

    Returns:
        (label, confidence) where label is "fog", "rain", or "clear".
    """
    # ── Fog: dominant signal is low contrast ──────────────────────────
    if mean_contrast < fog_contrast_threshold:
        if fog_contrast_threshold > 0:
            fog_conf = (fog_contrast_threshold - mean_contrast) / fog_contrast_threshold
        else:
            fog_conf = 1.0
        return "fog", min(1.0, max(0.0, float(fog_conf)))

    # ── Rain: low saturation (colour washout), not fog ────────────────
    if mean_saturation < rain_sat_threshold:
        if rain_sat_threshold > 0:
            rain_conf = (rain_sat_threshold - mean_saturation) / rain_sat_threshold
        else:
            rain_conf = 1.0
        return "rain", min(1.0, max(0.0, float(rain_conf)))

    # ── Clear: compute confidence from both distances ─────────────────
    # How far is contrast above the fog threshold?
    fog_margin = mean_contrast - fog_contrast_threshold
    fog_range  = 255.0 - fog_contrast_threshold  # plausible max contrast
    fog_clear_conf = fog_margin / fog_range if fog_range > 0 else 1.0

    # How far is saturation above the rain threshold?
    sat_margin  = mean_saturation - rain_sat_threshold
    sat_range   = 255.0 - rain_sat_threshold
    sat_clear_conf = sat_margin / sat_range if sat_range > 0 else 1.0

    # Confidence in "clear" = the weaker of the two margins
    clear_conf = min(fog_clear_conf, sat_clear_conf)
    return "clear", min(1.0, max(0.0, float(clear_conf)))
