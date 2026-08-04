# Picek Dataset Preprocessing Pipeline v2
"""
Modular preprocessing pipeline for traffic accident video datasets.
Designed for extensibility — future processors (TU-DAT, Kaggle) 
implement the same BaseDatasetProcessor interface.
"""

from pipeline.config import PipelineConfig
from pipeline.picek_processor import PicekProcessor

__all__ = ["PipelineConfig", "PicekProcessor"]
