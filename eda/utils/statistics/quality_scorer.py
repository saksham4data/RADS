# ─────────────────────────────────────────────────────────────
# Composite Dataset Quality Scorer
# ─────────────────────────────────────────────────────────────
"""
Computes a weighted composite Dataset Quality Score (0–100)
from component scores: missing values, duplicates, class
imbalance, invalid bounding boxes, and corruption.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


# ── Grade mapping ───────────────────────────────────────────

_GRADE_MAP = [
    (95, "A+"),
    (90, "A"),
    (85, "B+"),
    (80, "B"),
    (75, "C+"),
    (70, "C"),
    (60, "D"),
    (0, "F"),
]


class QualityScorer:
    """Computes composite Dataset Quality Score.

    Each component score is a float in [0, 1] where
    1 = perfect (no issues) and 0 = worst possible.

    Usage::

        scorer = QualityScorer(
            weights=config.quality_weights,
            missing_values=overview.compute_missing_values_score(),
            duplicates=overview.compute_duplicate_score(),
            class_imbalance=class_analyzer.compute_class_imbalance_score(),
            invalid_bboxes=bbox_analyzer.compute_invalid_bbox_score(),
            corruption=1.0,  # from validation status
        )
        result = scorer.compute()
        print(result["overall"])   # 87.3
        print(result["grade"])     # B+
    """

    def __init__(
        self,
        weights: Dict[str, float],
        *,
        missing_values: float = 1.0,
        duplicates: float = 1.0,
        class_imbalance: float = 1.0,
        invalid_bboxes: float = 1.0,
        corruption: float = 1.0,
    ) -> None:
        self.weights = weights
        self.components: Dict[str, float] = {
            "missing_values": max(0.0, min(1.0, missing_values)),
            "duplicates": max(0.0, min(1.0, duplicates)),
            "class_imbalance": max(0.0, min(1.0, class_imbalance)),
            "invalid_bboxes": max(0.0, min(1.0, invalid_bboxes)),
            "corruption": max(0.0, min(1.0, corruption)),
        }

    def compute(self) -> Dict[str, Any]:
        """Compute the weighted overall score and grade."""
        total_weight = sum(
            self.weights.get(k, 0) for k in self.components
        )
        if total_weight == 0:
            total_weight = 1.0

        weighted_sum = sum(
            self.weights.get(k, 0) * v
            for k, v in self.components.items()
        )
        overall = (weighted_sum / total_weight) * 100
        overall = round(max(0.0, min(100.0, overall)), 1)

        grade = self._get_grade(overall)

        return {
            "overall": overall,
            "grade": grade,
            "components": {
                k: {
                    "score": round(v * 100, 1),
                    "weight": self.weights.get(k, 0),
                    "raw": round(v, 4),
                }
                for k, v in self.components.items()
            },
            "weights": self.weights,
        }

    @staticmethod
    def _get_grade(score: float) -> str:
        """Map a 0–100 score to a letter grade."""
        for threshold, grade in _GRADE_MAP:
            if score >= threshold:
                return grade
        return "F"

    @classmethod
    def from_analyzers(
        cls,
        weights: Dict[str, float],
        *,
        overview_analyzer: Optional[Any] = None,
        class_analyzer: Optional[Any] = None,
        bbox_analyzer: Optional[Any] = None,
        corruption_score: float = 1.0,
    ) -> QualityScorer:
        """Factory that extracts component scores from analyzer instances."""
        missing = 1.0
        duplicates = 1.0
        imbalance = 1.0
        invalid_bbox = 1.0

        if overview_analyzer is not None:
            missing = overview_analyzer.compute_missing_values_score()
            duplicates = overview_analyzer.compute_duplicate_score()
        if class_analyzer is not None:
            imbalance = class_analyzer.compute_class_imbalance_score()
        if bbox_analyzer is not None:
            invalid_bbox = bbox_analyzer.compute_invalid_bbox_score()

        return cls(
            weights=weights,
            missing_values=missing,
            duplicates=duplicates,
            class_imbalance=imbalance,
            invalid_bboxes=invalid_bbox,
            corruption=corruption_score,
        )
