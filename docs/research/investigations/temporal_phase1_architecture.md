# Phase 1 Architecture: Temporal Modeling Recommendation

## 1. Recommended Architecture

```
Video
  |
  v
Ordered Frames [T, 3, 224, 224]
  |
  v
SpatialEncoder (ResNet18, fc head removed)
  -> [T, 512]
  |
  v
TemporalEncoder (GRU)
  -> [D]  (D = hidden_dim = 256)
  |
  v
Dropout (0.3)
  |
  v
Linear Classifier (256 -> 2)
  -> logits [2]
```

Batched: `[B, T, 3, 224, 224]` -> `[B, 2]`

---

## 2. Temporal Insertion Point

**Selected: Feature level (Option A)**

The temporal encoder operates on the 512-dim spatial feature vectors extracted by ResNet18 (with its fc head removed), NOT on the 2-dim logits.

**Rationale**:

- The 512-dim feature vector retains rich spatial information about each frame
- Operating on 2-dim logits (Option B) would reduce the temporal model to learning a weighted average of frame predictions, offering negligible improvement over the current `probs.mean(dim=0)` at inference time
- Removing the fc head from ResNet18 is a single-line change (`backbone.fc = nn.Identity()`)
- This is the standard approach in video understanding literature

---

## 3. Temporal Encoder Selection

**Selected: GRU (Gated Recurrent Unit)**

**Rationale** (following TECH.md architecture selection criteria):

| Criterion | GRU Assessment |
|---|---|
| Controlled comparison | Single new component between backbone and classifier |
| Low unnecessary complexity | Fewer parameters than LSTM (no cell state), simpler than attention |
| Suitable for sequence length | T=8 frames is well within GRU capacity |
| Compatible with ResNet18 | Accepts 512-dim input, outputs fixed-dim hidden state |
| Reproducibility | Deterministic with fixed seed |
| Isolates temporal effect | Only change relative to E07 is the GRU + sequence grouping |

**Alternatives considered and documented**:

- **Temporal Pooling**: Too simple to learn temporal patterns; useful only as a control baseline
- **LSTM**: More parameters than GRU for comparable capacity at T=8; not justified as a first experiment
- **Conv1D**: Captures local temporal patterns but less natural for ordered sequences than RNN
- **Attention/Transformer**: More complex, more parameters, higher overfitting risk on 93 videos; not justified as a first experiment

All four alternatives are implemented and available via `temporal.architecture` config for future experiments.

---

## 4. Architectural Boundaries

### SpatialEncoder (training/models/temporal_model.py)

- Wraps any torchvision backbone
- Strips classification head, returns pooled features
- Input: `[B, T, C, H, W]` -> Output: `[B, T, F]`
- F = 512 for ResNet18

### TemporalEncoder (training/models/temporal_model.py)

- Abstract interface with `output_dim` attribute
- Concrete implementations: GRU, LSTM, Conv1D, Pool
- Input: `[B, T, F]` -> Output: `[B, D]`
- Factory function: `create_temporal_encoder(architecture, ...)`

### TemporalClassificationModel (training/models/temporal_model.py)

- Composes SpatialEncoder + TemporalEncoder + Linear Classifier
- Input: `[B, T, C, H, W]` -> Output: `[B, num_classes]`
- Same output interface as ClassificationModel, compatible with existing Trainer

---

## 5. Dataset Architecture

### VideoSequenceDataset (training/datasets/video_dataset.py)

- Replaces the flat `(vid_idx, frame_idx)` sample index with per-video entries
- `__getitem__` returns `(frames[T, C, H, W], label)` for one video
- `__len__` returns the number of videos (not frames)
- Frame indices computed using the same `compute_sample_indices()` as the frame-based dataset
- Frames decoded in sorted temporal order

---

## 6. Configuration

Temporal parameters are controlled via a `temporal` section in the YAML config:

```yaml
temporal:
  enabled: true        # false = frame-based (E07 behavior)
  architecture: "gru"  # gru | lstm | conv1d | pool
  hidden_dim: 256
  num_layers: 1
  dropout: 0.0
  bidirectional: false
  classifier_dropout: 0.3
```

When `temporal.enabled` is `false` (the default), the system behaves identically to E07.

---

## 7. T01 Experiment Specification

The first temporal experiment (T01) changes exactly one variable relative to E07:

| Parameter | E07 | T01 | Changed? |
|---|---|---|---|
| Dataset | TUDAT v2 | TUDAT v2 | No |
| Splits | Frozen | Frozen | No |
| Frames/video | 8 | 8 | No |
| Sampling | uniform, safe-range | uniform, safe-range | No |
| Backbone | ResNet18 (pretrained) | ResNet18 (pretrained) | No |
| Temporal model | None (frame-level) | GRU(512->256) | **Yes** |
| Classifier | Linear(512->2) | Linear(256->2) | Changed by architecture |
| Loss | CrossEntropy (weighted) | CrossEntropy (weighted) | No |
| Optimizer | Adam, lr=1e-3 | Adam, lr=1e-3 | No |
| Scheduler | StepLR(5, 0.1) | StepLR(5, 0.1) | No |
| Epochs | 50 | 50 | No |
| Batch size | 32 (frames) | 32 (videos) | Unit changed |
| Seed | 42 | 42 | No |

---

## 8. What Remains for Phase 2

1. Run the T01 experiment: `python training/train.py --config training/config/training_config_temporal.yaml`
2. Compare T01 metrics against E07 baseline
3. Update prediction pipeline to support temporal models
4. If T01 shows improvement, plan follow-up experiments (LSTM, bidirectional, different hidden dims)
5. If T01 shows no improvement, investigate whether the sequence length, sampling strategy, or temporal architecture needs adjustment before concluding that temporal modeling is not beneficial for this dataset
