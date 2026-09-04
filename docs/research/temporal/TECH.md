# Technical Reference: Temporal Modeling Phase 1

## 1. Current Dataset

The controlled dataset for temporal-modeling development is:

**TUDAT v2**

The dataset is the versioned canonical dataset created after the earlier metadata and duplicate investigation.

Important properties:

- binary classification
- accident vs non-accident
- frozen train/validation/test assignments
- versioned metadata
- existing E07 baseline
- raw videos remain unchanged

Do not modify the dataset definition as part of temporal architecture work unless a concrete compatibility problem is discovered and documented.

---

## 2. Existing Spatial Baseline

The established baseline uses:

**ResNet18**

The current system processes video frames through the spatial backbone.

The exact implementation of frame processing and aggregation MUST be verified from source code.

Do not assume:

- where aggregation occurs
- whether aggregation happens on features or logits
- whether probabilities are averaged
- whether frames are independently classified
- whether any hidden temporal operation already exists

Inspect the implementation.

---

## 3. Temporal Modeling Target

The intended conceptual architecture is:

ordered video frames
→ spatial representation
→ temporal representation
→ classification

The temporal component should have access to the ordered sequence rather than receiving an already-collapsed representation.

The important question is therefore not simply:

"Which temporal model should we use?"

The more important question is:

"What representation should the temporal model receive?"

---

## 4. Candidate Temporal Architectures

Possible candidates include:

### Temporal pooling

Example:

`frame features → temporal pooling → classifier`

Advantages:
- simple
- low computational cost
- useful as a control

Limitation:
- limited ability to model temporal relationships

---

### 1D Temporal Convolution

Example:

`frame features → temporal Conv1D → classifier`

Advantages:
- lightweight
- explicitly models local temporal patterns
- easy to control experimentally

---

### GRU / LSTM

Example:

`frame features → recurrent sequence model → classifier`

Advantages:
- explicitly models ordered sequences
- suitable for relatively short sequences
- conceptually straightforward

Potential disadvantages:
- recurrent optimization
- more parameters than simple pooling

---

### Temporal Attention / Transformer-style Encoder

Example:

`frame features + positional information → temporal encoder → classifier`

Advantages:
- flexible temporal relationships
- explicit attention over frames

Potential disadvantages:
- more complexity
- more opportunities for overfitting on a small dataset
- harder to interpret as a minimal architectural change

---

## 5. Architecture Selection Rule

Do not select an architecture because it is currently popular.

The first temporal experiment should prioritize:

1. controlled comparison
2. low unnecessary complexity
3. suitability for the available sequence length
4. compatibility with the existing ResNet18 representation
5. reproducibility
6. ability to isolate the effect of temporal modeling

If multiple approaches are reasonable, document the alternatives and choose the simplest defensible first experiment.

---

## 6. Sequence Representation

The temporal pipeline should explicitly represent:

`B × T × F`

where:

- `B` = batch size
- `T` = number of frames
- `F` = spatial feature dimension

The exact dimensions must be determined from the current implementation.

Do not hard-code feature dimensions until the backbone output has been inspected.

---

## 7. Frame Ordering

Temporal modeling requires ordered frames.

Verify:

- how frame indices are generated
- whether indices are sorted
- whether dataloader collation preserves sequence order
- whether transforms operate independently without reordering
- whether inference follows the same ordering

Frame order must be deterministic for the controlled experiment.

---

## 8. Decoder and Sampling

The previous frame-decoding investigation identified unreliable container-reported frame counts.

The decoder/sampling implementation was subsequently corrected to use the actually decodable range.

That behavior is part of the current baseline.

Do not replace it with a different decoder strategy during temporal modeling unless a new problem is demonstrated.

---

## 9. Evaluation

The established evaluation system includes:

- top-1 accuracy
- macro F1
- balanced accuracy
- accident precision
- accident recall
- accident F1
- non-accident precision
- non-accident recall
- non-accident F1
- AUROC
- validation loss

The first temporal experiment should preserve these metrics.

Checkpoint selection should remain explicitly configured and documented.

---

## 10. Configuration

Temporal experiments must be configuration-driven.

Do not hard-code:

- temporal model type
- sequence length
- hidden dimension
- dropout
- learning rate
- checkpoint metric
- dataset path

into training code when these are experiment parameters.

Use a dedicated configuration for the temporal experiment rather than altering historical configurations.

---

## 11. Existing Experiment Protection

E01-E07 are historical experimental states.

Temporal development must not silently alter them.

New experiments should receive new configuration/version identifiers.

Historical outputs and checkpoints should remain reproducible.

---

## 12. Engineering Preference

Prefer a clean separation:

Dataset
→ sequence sampling
→ spatial encoder
→ temporal encoder
→ classifier
→ training/evaluation

The spatial encoder and temporal encoder should ideally have explicit interfaces so that future experiments can change the temporal component without rewriting the entire training system.