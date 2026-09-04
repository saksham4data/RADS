# Phase 1 Execution Plan: Temporal Modeling

## Objective

Understand the existing TUDAT frame-based pipeline, identify the representation limitation that motivates temporal modeling, design the temporal architecture boundary, and implement only the infrastructure changes justified by the investigation.

---

# Phase 1 Workflow

## Step 1: Repository Reconnaissance

Inspect:

- dataset implementation
- dataloader
- frame sampling
- decoder utilities
- transforms
- model
- trainer
- evaluator
- prediction pipeline
- configuration system
- existing tests
- existing investigation documentation

### Record

- relevant file paths
- classes
- functions
- tensor shapes
- data interfaces
- configuration interfaces

### Status

`NOT STARTED`

---

## Step 2: Trace Video Data Flow

Trace one video from metadata to final prediction.

Document:

```text
video
↓
metadata
↓
frame indices
↓
frame decoding
↓
transforms
↓
batch
↓
spatial encoder
↓
frame representation
↓
aggregation
↓
classifier
↓
loss
↓
metrics

Questions
How many frames enter the model?
Are they ordered?
What tensor shape reaches the model?
What shape leaves the spatial encoder?
Where does aggregation happen?
Is aggregation feature-level or prediction-level?
Where is temporal information discarded?
Status

NOT STARTED

Step 3: Representation Investigation

Determine what the current representation can and cannot express.

Investigate:

frame ordering
temporal transitions
motion
pre-event/event progression
changes between adjacent frames
whether the current aggregation preserves any of these relationships

Do not make theoretical claims without connecting them to the implementation.

Status

NOT STARTED

Step 4: Identify the Temporal Insertion Point

Evaluate possible boundaries:

Option A
frames
→ spatial encoder
→ temporal model
→ classifier
Option B
frames
→ spatial encoder
→ frame logits
→ temporal model
→ classifier
Option C

Another repository-supported design.

Determine:

information retained
information lost
implementation complexity
compatibility with ResNet18
impact on controlled comparison
Status

NOT STARTED

Step 5: Determine Required Infrastructure Changes

Identify exactly what must change in:

dataset
dataloader
batch representation
model interfaces
trainer
evaluator
prediction
configuration
tests

Separate:

Required

Changes without which the temporal pipeline cannot work.

Recommended

Changes that improve maintainability or extensibility.

Unnecessary

Changes that should not be made during this phase.

Status

NOT STARTED

Step 6: Design the First Temporal Experiment

The experiment specification must define:

dataset
split
sequence length
frame sampling
frame ordering
spatial backbone
feature dimension
temporal architecture
temporal hidden dimension
classifier
loss
optimizer
scheduler
evaluation metrics
checkpoint metric
seed
experiment identifier

The experiment should change as few variables as possible relative to E07.

Status

NOT STARTED

Step 7: Implement Required Infrastructure

Only after Steps 1-6 are documented.

Implement the minimum required changes.

Requirements:

configuration-driven
reusable
testable
backward-compatible where practical
no historical experiment changes
Status

NOT STARTED

Step 8: Test the Infrastructure

Run focused tests for:

sequence shape
sequence ordering
dataloader behavior
temporal model forward pass
configuration loading
checkpoint compatibility where relevant
regression behavior for existing baseline components

Do not begin expensive training until these pass.

Status

NOT STARTED

Step 9: Produce Phase 1 Documentation

Create:

docs/research/investigations/temporal_phase1_investigation.md

and:

docs/research/investigations/temporal_phase1_architecture.md

The documentation must contain:

what was found
what was changed
why it was changed
what was tested
unresolved issues
proposed Phase 2 experiment
Status

NOT STARTED

Phase Completion Criteria

Phase 1 is complete when:

 Current frame pipeline is fully traced.
 Current aggregation is verified from code.
 Frame ordering is verified.
 Representation limitations are documented.
 Temporal insertion point is justified.
 Required infrastructure changes are identified.
 Necessary infrastructure changes are implemented.
 Focused tests pass.
 Existing E01-E07 behavior is protected.
 Phase 2 temporal experiment is explicitly specified.
 Investigation documentation is complete.

 Phase 2 Boundary

Do not consider Phase 1 complete merely because code compiles.

The next phase is:

Build + validate the temporal pipeline and run the first controlled temporal experiment on TUDAT v2.

Phase 2 begins only after the architecture and infrastructure produced here are validated.