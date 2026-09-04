# Phase 1 Scope and Change Control

## Purpose

This phase is allowed to modify the repository.

The goal is to prepare a clean, reusable foundation for temporal modeling.

Changes must be driven by the investigation and documented.

---

# In Scope

The agent may modify:

- dataset interfaces if required to expose ordered frame sequences
- dataloader/collation if required for sequence tensors
- model interfaces
- temporal-model infrastructure
- configuration system
- trainer interfaces if required
- evaluator interfaces if required
- prediction interfaces if required
- tests
- documentation

The agent may create new modules where this produces a cleaner architecture.

---

# Dataset Rules

TUDAT v2 is the controlled dataset.

Do not:

- delete videos
- modify raw videos
- change labels
- regenerate frozen splits
- remove records
- silently alter metadata

If the current metadata format creates a genuine technical blocker, document the blocker before changing it.

Prefer adapting the data-loading interface instead.

---

# Historical Experiment Protection

Do not modify the semantics of:

- E01
- E02
- E03
- E04
- E05
- E06
- E07

Do not overwrite their:

- configurations
- checkpoints
- outputs
- metrics
- dataset definitions

If shared code must be changed, verify that historical behavior remains compatible.

---

# New Experiment Isolation

Temporal work must use new experiment/config identifiers.

Do not repurpose the E07 configuration.

Create a dedicated configuration for the temporal experiment when the architecture is finalized.

---

# Architecture Changes

Architecture changes are allowed when justified.

Preferred structure:

```text
Video
  ↓
Ordered Frames
  ↓
Spatial Encoder
  ↓
Sequence of Frame Features
  ↓
Temporal Encoder
  ↓
Video Representation
  ↓
Classifier