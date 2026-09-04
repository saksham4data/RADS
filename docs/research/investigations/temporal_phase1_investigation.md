# Phase 1 Investigation: Frame-Based Representation Analysis

## Objective

Determine how the current TUDAT pipeline represents a video, where temporal information is preserved or discarded, and what representation limitation motivates temporal modeling.

---

## 1. Current Data Flow

### 1.1 Dataset Construction

**File**: `training/datasets/frame_dataset.py`
**Class**: `VideoFrameDataset`

The dataset reads `global_master_metadata_v2.csv`, filters by dataset name and split, then builds a flat sample index:

```python
# _build_sample_index() — line 288
for row_idx, row in self._df.iterrows():
    frame_indices = self._compute_frame_indices(total_frames, frames_per_video, strategy)
    vid_idx = len(self._video_paths)
    self._video_paths.append(video_path)
    self._video_labels.append(label_idx)
    for fi in frame_indices:
        self._samples.append((vid_idx, fi))
```

Each `(vid_idx, frame_idx)` pair becomes an independent sample. For 93 videos with 8 frames each, this produces 744 independent samples in the dataset.

**Consequence**: The dataset `__len__()` returns the total number of individual frames, not the number of videos. The DataLoader sees 744 independent images, not 93 videos.

### 1.2 Frame Sampling

**File**: `training/datasets/video_sampling.py`
**Function**: `compute_sample_indices()`

For the `uniform` strategy (used in E07), frame indices are computed as:

```python
lo, hi = safe_range(total_frames)  # [15%, 85%] of decodable range
return [lo + int(i * (safe_count - 1) / (n - 1)) for i in range(n)]
```

These indices are sorted by construction. For a video with 100 decodable frames and T=8, the indices would be approximately: [15, 25, 35, 45, 55, 65, 75, 85].

The frame order is deterministic and temporally ordered within a video's index list. However, this ordering is discarded when frames are flattened to independent samples.

### 1.3 Frame Decoding

**File**: `training/datasets/video_sampling.py`
**Function**: `read_frame_with_fallback()`

Each frame is decoded independently via OpenCV seek + read. The decoder does not maintain state between frames. Transforms (resize, augment, normalize) are applied per-frame with no cross-frame interaction.

### 1.4 Batching

**File**: `training/datasets/dataloader.py`
**Function**: `create_dataloaders()`

The DataLoader uses `WeightedRandomSampler(replacement=True)` for training:

```python
sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True,
)
```

This sampler draws individual frame samples with replacement to address class imbalance. Frames from the same video may end up in different batches, or multiple copies of the same frame may appear in one batch. The temporal relationship between frames from the same video is completely destroyed at this stage.

The collated batch has shape `[B, 3, 224, 224]` where B is the batch size (32 individual frames, not 32 videos).

### 1.5 Model Forward Pass

**File**: `training/models/classification_model.py`
**Class**: `ClassificationModel`

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.backbone(x)  # [B, 3, H, W] -> [B, num_classes]
```

The model receives `[B, 3, 224, 224]` and returns `[B, 2]`. Each frame is classified independently through the full ResNet18 (conv layers + avgpool + fc). There is no temporal dimension anywhere in the forward pass.

### 1.6 Inference Aggregation

**File**: `training/predict.py`
**Function**: `predict_video()`

Only at inference time are multiple frames from the same video grouped:

```python
batch = torch.stack(tensors).to(device)        # [T, 3, H, W]
logits = model(batch)                           # [T, 2]
probs = torch.softmax(logits, dim=1)            # [T, 2]
avg_probs = probs.mean(dim=0).cpu().numpy()     # [2]
pred_class_idx = int(avg_probs.argmax())
```

This is arithmetic mean of softmax probabilities across independently classified frames. It is not learned temporal modeling.

---

## 2. Where Temporal Information Is Discarded

| Stage | File | Line(s) | What happens |
|---|---|---|---|
| Sample index | frame_dataset.py | 350-356 | Frame indices flattened to independent `(vid_idx, frame_idx)` tuples |
| DataLoader sampler | dataloader.py | 71-76 | `WeightedRandomSampler` shuffles and resamples individual frames |
| Batch collation | dataloader.py | 93, 157-166 | `default_collate` stacks frames as `[B, C, H, W]` with no video grouping |
| Model input | classification_model.py | 142-155 | Expects `[B, 3, H, W]`, no temporal axis |

**Summary**: Temporal information is discarded at dataset construction time (step 1) and is never recoverable in the training pipeline. The model has no mechanism to learn from frame ordering, motion, or temporal transitions.

---

## 3. What the Current Representation Cannot Express

The current frame-level classification treats each frame as an independent image. The model cannot learn:

1. **Frame ordering**: Whether frame A comes before or after frame B in a video is unknown to the model during training.

2. **Motion**: The direction and velocity of objects between adjacent frames cannot be captured because the model never sees adjacent frames together.

3. **State transitions**: The progression from normal driving to an accident (or the absence of such a progression in non-accident videos) is invisible to the model.

4. **Temporal context**: A frame showing a car at an unusual angle might be an accident or might be a normal turn. The distinction often depends on what happened in preceding frames.

5. **Event progression**: The temporal signature of an accident (approach, impact, aftermath) cannot be modeled because these phases are never presented as a sequence.

---

## 4. What the Current Representation Can Express

The model can learn:

- Static visual cues associated with accidents (damage, unusual vehicle positions, debris)
- Static visual cues associated with normal driving (clear road, normal traffic)
- Individual frame-level appearance features

The inference-time probability averaging provides a weak form of aggregation but cannot learn temporal patterns because the averaging weights are uniform and fixed (not learned).

---

## 5. ResNet18 Feature Dimensions

From `classification_model.py`:

```python
_BACKBONE_FEATURES = {
    "resnet18": 512,
    "efficientnet_b0": 1280,
}
```

- ResNet18 avgpool output: `[B, 512]`
- ResNet18 fc layer: `Linear(512, num_classes)`
- For temporal modeling, the 512-dim feature vector (before the fc head) is the natural insertion point

---

## 6. Conclusion

The current system performs frame-level classification with post-hoc probability averaging. Temporal information is structurally absent from the training pipeline. To model temporal relationships, the system must:

1. Present frames from the same video as an ordered sequence
2. Extract per-frame features while preserving the temporal dimension
3. Apply a learned temporal model to the feature sequence
4. Classify based on the temporally-aware representation

This is addressed in the architecture recommendation document.
