# ─────────────────────────────────────────────────────────────
# Statistics Sub-package
# ─────────────────────────────────────────────────────────────
"""
Analyzer class hierarchy for structured, extensible
statistical computation across all EDA notebooks.
"""

from eda.utils.statistics.base_analyzer import BaseAnalyzer
from eda.utils.statistics.overview_analyzer import OverviewAnalyzer
from eda.utils.statistics.class_analyzer import ClassAnalyzer
from eda.utils.statistics.video_analyzer import VideoAnalyzer
from eda.utils.statistics.bbox_analyzer import BBoxAnalyzer
from eda.utils.statistics.split_analyzer import SplitAnalyzer
from eda.utils.statistics.quality_scorer import QualityScorer
from eda.utils.statistics.metadata_quality_analyzer import MetadataQualityAnalyzer

__all__ = [
    "BaseAnalyzer",
    "OverviewAnalyzer",
    "ClassAnalyzer",
    "VideoAnalyzer",
    "BBoxAnalyzer",
    "SplitAnalyzer",
    "QualityScorer",
    "MetadataQualityAnalyzer",
]
