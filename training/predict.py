#!/usr/bin/env python
# ─────────────────────────────────────────────────────────────
# Production Inference / Prediction Entry Point
# ─────────────────────────────────────────────────────────────
"""
Run inference on a single video or directory of videos.

Usage::

    python training/predict.py --checkpoint best.pt --input video.mp4
    python training/predict.py --checkpoint best.pt --input videos/
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import torch
import torch.nn as nn
import sys

# Add project root to sys.path so 'training' module can be imported
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.datasets.video_sampling import (
    compute_sample_indices,
    probe_decodable_frame_count,
    read_frame_with_fallback,
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RADS — Prediction Script (Pipeline 4)",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to training config YAML.",
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to model checkpoint (.pt).",
    )
    parser.add_argument(
        "--input", type=str, required=True,
        help="Path to a video file or directory of videos.",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Path to save prediction results JSON.",
    )
    return parser.parse_args()


def predict_video(
    model: nn.Module,
    video_path: Path,
    transform,
    device: torch.device,
    class_names: List[str],
    frames_per_video: int = 5,
    sampling_strategy: str = "uniform",
) -> Dict:
    """Run inference on a single video.

    Samples frames uniformly, runs the model, and aggregates
    predictions via majority vote with average confidence.

    Returns
    -------
    dict
        Prediction result with class, confidence, per-frame details.
    """
    # Sample frames uniformly
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"video": str(video_path), "error": "Cannot open video"}

    reported_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_frames = probe_decodable_frame_count(video_path, reported_frames)
    if total_frames <= 0:
        cap.release()
        return {"video": str(video_path), "error": "No frames in video"}

    indices = compute_sample_indices(
        total_frames, frames_per_video, sampling_strategy,
    )

    frames = []
    for idx in indices:
        frame = read_frame_with_fallback(video_path, idx)
        if frame is not None:
            frames.append(frame)
    cap.release()

    if not frames:
        return {"video": str(video_path), "error": "Could not read frames"}

    # Transform and batch
    tensors = []
    for frame in frames:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = transform(frame_rgb)
        tensors.append(tensor)

    batch = torch.stack(tensors).unsqueeze(0).to(device)

    # Inference
    model.eval()
    with torch.no_grad():
        logits = model(batch)
        probs = torch.softmax(logits, dim=1)
        # Replicate the clip-level prediction for each frame so the rest of the script works
        probs = probs.repeat(len(frames), 1)

    # Aggregate: average confidence across frames
    avg_probs = probs.mean(dim=0).cpu().numpy()
    pred_class_idx = int(avg_probs.argmax())
    pred_class = class_names[pred_class_idx]
    confidence = float(avg_probs[pred_class_idx])

    # Per-frame predictions
    per_frame = []
    for i, (prob, frame_idx) in enumerate(zip(probs.cpu().numpy(), indices)):
        per_frame.append({
            "frame_index": frame_idx,
            "predicted_class": class_names[int(prob.argmax())],
            "confidence": float(prob.max()),
            "probabilities": {
                name: float(prob[j]) for j, name in enumerate(class_names)
            },
        })

    return {
        "video": str(video_path),
        "predicted_class": pred_class,
        "confidence": round(confidence, 4),
        "avg_probabilities": {
            name: round(float(avg_probs[j]), 4)
            for j, name in enumerate(class_names)
        },
        "num_frames_sampled": len(frames),
        "per_frame_predictions": per_frame,
    }


def main() -> None:
    args = parse_args()

    from training.utils.config import load_training_config
    from training.utils.seed import set_global_seed
    from training.utils.device import get_device
    from training.models.model_factory import create_model
    from training.callbacks.checkpoint import CheckpointManager
    from training.datasets.transforms import get_val_transforms

    config = load_training_config(args.config)
    set_global_seed(config.seed)
    device = get_device()

    # Load model
    model = create_model(config)
    state = CheckpointManager.load(Path(args.checkpoint))
    model.load_state_dict(state["model_state_dict"])
    model = model.to(device)
    model.eval()

    class_names = config.resolved_class_names

    transform = get_val_transforms(config.image_size)

    # Collect video files
    input_path = Path(args.input)
    if input_path.is_file():
        video_files = [input_path]
    elif input_path.is_dir():
        video_files = sorted(
            p for p in input_path.rglob("*")
            if p.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}
        )
    else:
        print(f"Input not found: {input_path}")
        return

    print(f"Running inference on {len(video_files)} video(s)...")

    # Predict
    results = []
    for vf in video_files:
        result = predict_video(
            model, vf, transform, device, class_names,
            frames_per_video=config.frames_per_video,
            sampling_strategy=config.sampling_strategy,
        )
        results.append(result)
        print(
            f"  {vf.name}: {result.get('predicted_class', 'ERROR')} "
            f"(conf={result.get('confidence', 0):.2%})"
        )

    # Save results
    output_path = Path(args.output) if args.output else Path("predictions.json")
    output_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
