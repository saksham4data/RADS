# Metadata Generation Pipeline (Pipeline 0)
"""
Generates metadata CSVs for raw datasets that lack author-provided metadata.

This pipeline scans raw media files (videos/images), extracts technical
properties via OpenCV, applies scene classification heuristics, and outputs
a metadata.csv compatible with Pipeline 1's ingestion format.

Designed for TU-DAT, Kaggle, and future datasets.
Does NOT modify or depend on the existing Picek preprocessing pipeline.
"""
