# Investigation 1 — Metadata / Feature Leakage Audit
**Date:** 2026-08-15
**Status:** COMPLETE
**Investigator:** Antigravity diagnostic pass (read-only, no code modified)
**Pipeline version:** E06 / TUDAT binary classification / ResNet18

---

## 1. Complete Data Flow: metadata -> dataset -> tensor -> model

```
global_master_metadata.csv   (5338 rows, 51 columns)
        |
        |  _load_metadata()  [frame_dataset.py:103-111]
        |  pd.read_csv(path, low_memory=False)
        v
full DataFrame
        |
        |  _filter_dataset()  [frame_dataset.py:113-130]
        |  df["dataset_name"] == "tudat"
        v
tudat_df  (111 rows)
        |
        |  _assign_or_filter_split()  [frame_dataset.py:139-172]
        |  split_in_distribution col: ALL NaN for TUDAT
        |  -> falls through to _generate_splits()
        |  -> sklearn stratified train/val/test on "type" column, seed=42
        v
split_df  (train~77 | val~17 | test~17 rows)
        |
        |  _build_sample_index()  [frame_dataset.py:280-350]
        |  for each row:
        |    label_str = _normalize_label(row["type"])
        |                "challenging" -> "accident" if label_mode=="binary"
        |    label_idx = class_mapping[label_str]          <- integer 0 or 1
        |    video_path = raw_base_dir / "tudat" / "Final_videos" / original_path
        |    frame_indices = _compute_frame_indices(total_frames, 8, "uniform")
        |    -> stored as (video_path, label_idx, frame_indices)
        v
_samples: List[Tuple[int,int]]     (vid_idx, frame_idx)
_video_paths: List[Path]
_video_labels: List[int]           (integers only)
        |
        |  __getitem__(idx)  [frame_dataset.py:439-464]
        |    frame = _read_frame(video_path, frame_idx)
        |            -> BGR numpy ndarray [H,W,3]  or  None
        |    frame = self.transform(frame)
        |            -> Tensor[3,224,224]  (pixels only)
        v
(frame_tensor: torch.Tensor[3,224,224],  label: int)    <- ITEM RETURNED
        |
        |  _safe_collate()  [dataloader.py:157-166]
        |  filters out None items; calls default_collate
        v
(images: Tensor[B,3,224,224],  labels: Tensor[B])       <- BATCH
        |
        |  trainer._train_one_epoch()  [trainer.py:272-307]
        |    images = images.to(device)
        |    labels = labels.to(device)
        |    logits = model(images)        <- ONLY TENSOR ENTERS forward()
        |    loss   = CrossEntropyLoss(logits, labels)
        v
logits: Tensor[B, num_classes]
```

**VERDICT: The only thing that enters model.forward() is an image tensor of
shape [B,3,224,224] containing normalized pixel values. No metadata, no path,
no string of any kind is part of this tensor.**

---

## 2. Metadata Fields Actually Used by the Pipeline

Source: `Datasets/processed/global_master_metadata.csv` (51 columns, 5338 rows total)

| Field | File:Line | Purpose |
|---|---|---|
| `dataset_name` | frame_dataset.py:116 | Filter rows to TUDAT |
| `original_path` | frame_dataset.py:291-306 | Construct video_path on disk |
| `type` | frame_dataset.py:316, 286 | Ground-truth label (configured as label_column) |
| `split_in_distribution` | frame_dataset.py:150 | Attempted split column (all NaN for TUDAT) |
| `no_frames` | frame_dataset.py:327 | Frame count for index computation |

**All other 46 columns are never accessed by any pipeline file.**

---

## 3. Complete Metadata Field Disposition Table

| Field | Locate video | Split selection | Label | Model input | Bookkeeping | Not used | Notes |
|---|:---:|:---:|:---:|:---:|:---:|:---:|---|
| `original_path` | YES | | | | | | Path-only; never passed to model |
| `processed_path` | | | | | | NOT READ | Not accessed anywhere |
| filename part of original_path | | | | | | INDIRECT | See Section 6 |
| folder part of original_path | | | | | | INDIRECT | See Section 6 — critical |
| `dataset_name` | | YES | | | YES | | Filter + logging |
| `video_id` | | | | | | NOT READ | Column exists; never read |
| `event_id` | | | | | | NOT PRESENT | Not in TUDAT rows |
| `duplicate_group_id` | | | | | | NOT READ | Column exists; never read |
| `is_duplicate` | | | | | | NOT READ | Column exists; never read |
| `split_in_distribution` | | ATTEMPTED | | | | | All NaN for TUDAT -> fallback to auto-split |
| `proposed_split` | | | | | | NOT PRESENT | Column does not exist in CSV |
| `type` | | | YES | | | | Ground-truth label column |
| class name strings | | | YES | | | | Converted to int before any model use |
| `no_frames` | YES | | | | | | Frame index computation only |
| `duration` | | | | | YES | | In CSV; not read by pipeline |
| `fps` | | | | | YES | | In CSV; not read by pipeline |
| `weather`, `region`, `scene_layout` | | | | | YES | | In CSV; not read |
| All other 39 fields | | | | | | NOT READ | Ignored entirely |

---

## 4. Exact Location Where the Final Model Input Tensor is Created

**File:** `training/datasets/frame_dataset.py`
**Function:** `VideoFrameDataset.__getitem__`
**Lines 439-458**

```python
def __getitem__(self, idx: int) -> Optional[Tuple[torch.Tensor, int]]:
    vid_idx, frame_idx = self._samples[idx]
    video_path = self._video_paths[vid_idx]       # Path object only
    label = self._video_labels[vid_idx]            # integer only

    frame = self._read_frame(video_path, frame_idx)   # -> BGR numpy or None
    if frame is None:
        raise IOError("Frame extraction failed ...")

    if self.transform is not None:
        frame = self.transform(frame)             # -> Tensor[3,224,224]
    else:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0

    return frame, label    # (Tensor[3,224,224], int)
```

Transform chain for training (`transforms.py`):
```
ToPILImage() -> Resize((224,224)) -> RandomHorizontalFlip(p=0.5)
             -> RandomRotation(5) -> ColorJitter(brightness=0.15, contrast=0.15)
             -> ToTensor() -> Normalize(mean=ImageNet, std=ImageNet)
```

Transform chain for val/test:
```
ToPILImage() -> Resize((224,224)) -> CenterCrop((224,224))
             -> ToTensor() -> Normalize(mean=ImageNet, std=ImageNet)
```

**Only pixel values are encoded in the output tensor.**

---

## 5. Exact Location Where Labels Are Created

**File:** `training/datasets/frame_dataset.py`
**Function:** `VideoFrameDataset._build_sample_index`
**Lines 316-324**

```python
# Line 316
label_str = self._normalize_label(row[label_col])  # "type" -> "accident"|"non-accident"
# Line 317-321
if label_str not in self._class_mapping:
    logger.warning("Unknown label '%s' in row %d, skipping.", label_str, row_idx)
    continue
# Line 324
label_idx = self._class_mapping[label_str]          # -> 0 or 1
```

Label normalization (`_normalize_label`, lines 132-137):
```python
def _normalize_label(self, raw_label: Any) -> str:
    label = str(raw_label).strip()
    if self.config.label_mode == "binary" and label == "challenging":
        return "accident"
    return label
```

Active class mapping (binary mode, auto-detected from data):
```
"accident"     -> 0
"non-accident" -> 1
```

The label integer (0 or 1) is the only representation of the label passed to the
model via the loss function. The string "accident" never touches the model.

---

## 6. Critical Finding: Folder Name Encodes Label (Data Quality Concern)

### OBSERVED FACT (verified by code + data)

The TUDAT `original_path` values carry folder prefixes that perfectly predict
the ground-truth label. Cross-tabulation of 111 rows:

```
type                     accident  challenging  non-accident
folder
Negative_Videos                 0            0            50
Positive_Vidoes               44            0             0
challenging-environment         0           17             0
```

The folder-label correspondence is:
- `Positive_Vidoes/`  ->  type="accident"  ->  label=0 (accident)
- `Negative_Videos/`  ->  type="non-accident"  ->  label=1 (non-accident)
- `challenging-environment/`  ->  type="challenging"  ->  label=0 (accident, binary)

This is 100% deterministic. No exceptions.

The CSV column `metadata_source` value is `'folder_structure'` for TUDAT rows,
confirming the label was derived from the folder name during metadata generation.

### DOES THIS LEAK INTO THE MODEL? OBSERVED FACT: NO

The `original_path` is consumed at dataset construction time to resolve the
filesystem path to the video file. The path string is then discarded. Only the
pixel bytes decoded from the video file reach the model.

Specifically:
- `video_path` (a `pathlib.Path`) is passed to `cv2.VideoCapture()`.
- OpenCV reads raw pixel bytes from disk.
- Returns `numpy.ndarray` of shape [H,W,3].
- Path is never embedded, encoded, hashed, or otherwise converted to a numeric
  representation that could enter a tensor.

In inference (`predict.py`), the model receives:
```python
logits = model(batch)   # batch = Tensor[N, 3, 224, 224] - pixels only
```

### INFERENCE: What is the actual risk?

The risk is at the **data quality / label integrity layer**, not at runtime:

1. Labels are correct only if every video was placed in the correct folder.
2. If a video was misplaced, its label is silently wrong.
3. There is no independent ground-truth verification in the pipeline.
4. This should be validated independently against the original TUDAT paper annotations.

This is a data integrity concern, **not a feature leakage concern**.

---

## 7. Additional Code-Level Checks

### `_safe_collate` — `dataloader.py:157-166`
Filters out None items. Passes only `(frame_tensor, label_int)` tuples to the
training loop. No metadata.

### `WeightedRandomSampler` — `dataloader.py:62-75`
Builds per-sample weights from `class_weights` tensor (inverse-frequency).
Uses `_video_labels[vid_idx]` (integer). No metadata strings.

### `create_loss` — `classification_loss.py:22-60`
`nn.CrossEntropyLoss(weight=class_weight_tensor)` — pure tensor math.

### `MetricsTracker` — `classification_metrics.py`
Accumulates `preds.cpu().tolist()` and `targets.cpu().tolist()`.
Class name strings are used only for metric dictionary keys
(e.g., `"val/f1_accident"`), never fed to the model or loss.

### `predict.py` — inference path
`model(batch)` where `batch = torch.stack(tensors)`.
Class name strings are used only for labelling the output JSON.
Never fed to the model.

### `ClassificationModel.forward` — `classification_model.py:142-155`
```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.backbone(x)     # x is Tensor[B,3,H,W]
```
This is the ResNet18 backbone. The only input is the image tensor.

---

## 8. Verdict

### PRIMARY VERDICT: PASS — Model input is visual-only

The model `forward()` receives only image tensors of shape [B, 3, 224, 224]
containing normalized pixel values. No metadata field, no path, no folder name,
no filename is encoded in or concatenated to this tensor at any point.

Labels are integer indices (0, 1) derived from the `type` column. The string
form of the label is never passed to the model.

### SECONDARY FINDING (not leakage, but data quality):

- TUDAT labels are folder-derived; no independent annotation verification in pipeline.
- `split_in_distribution` is entirely NaN for all 111 TUDAT rows.
  Every training run auto-generates splits from scratch.
  Reproducibility depends entirely on `seed=42` being kept constant.
  **If the seed changes or data is added/removed, the splits change silently.**

### TERTIARY FINDING:

- `proposed_split` column mentioned in investigation brief does not exist in the CSV.

---

## 9. File / Function Reference Summary

| Finding | File | Function | Lines |
|---|---|---|---|
| Metadata loading | frame_dataset.py | _load_metadata | 103-111 |
| Dataset filtering | frame_dataset.py | _filter_dataset | 113-130 |
| Label normalization | frame_dataset.py | _normalize_label | 132-137 |
| Split assignment | frame_dataset.py | _assign_or_filter_split | 139-172 |
| Auto-split generation | frame_dataset.py | _generate_splits | 174-261 |
| Label integer creation | frame_dataset.py | _build_sample_index | 316-324 |
| Video path resolution | frame_dataset.py | _build_sample_index | 291-306 |
| Tensor creation | frame_dataset.py | __getitem__ | 439-458 |
| Transform pipeline (train) | transforms.py | get_train_transforms | 25-100 |
| Transform pipeline (val/test) | transforms.py | get_val_transforms | 103-127 |
| Model forward pass | classification_model.py | forward | 142-155 |
| Loss computation | trainer.py | _train_one_epoch | 284-291 |
| Validation forward | trainer.py | _validate_one_epoch | 347-351 |
| Evaluation forward | evaluator.py | evaluate | 113-114 |
| Inference forward | predict.py | predict_video | 106-108 |
| Collate function | dataloader.py | _safe_collate | 157-166 |
