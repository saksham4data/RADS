"""
Deterministic unique ID generator for video samples.

Uses UUID-5 (SHA-1 based, deterministic) so that re-running the pipeline
on the same dataset produces identical IDs. This enables stable references,
incremental updates, and reproducible results.
"""

import uuid


# Fixed namespace for all dataset video IDs.
# Chosen arbitrarily but must remain constant across pipeline versions.
_NAMESPACE = uuid.UUID("7a3d8f1e-2b4c-5d6e-8f9a-0b1c2d3e4f5a")


def generate_video_id(dataset_name: str, source_type: str, filename: str) -> str:
    """
    Generate a deterministic, human-readable video ID.

    Args:
        dataset_name: Name of the dataset (e.g. "picek").
        source_type: Source type (e.g. "real" or "synthetic").
        filename: Original video filename (e.g. "Z4kg2Ev3vhk_00.mp4").

    Returns:
        A string like "picek_real_a1b2c3d4e5f6" (dataset + source + 12-char hex).
    """
    # Build a canonical key for UUID generation
    key = f"{dataset_name}/{source_type}/{filename}"
    uid = uuid.uuid5(_NAMESPACE, key)

    # Shorten to first 12 hex chars for readability (collision risk negligible at <100k samples)
    short_hex = uid.hex[:12]

    prefix = "syn" if source_type == "synthetic" else source_type
    return f"{dataset_name}_{prefix}_{short_hex}"
