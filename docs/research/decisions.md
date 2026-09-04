# Decisions log

## This file contains all the decisions made by the author during the development of the project.

- For each decision I will keep a record of 
    - date
    - decision made
    - reason
    - impact on the project
    

- ## Decisions

- ## D001

-- Date: 2026-08-08

-- Decision:Merge challenging into accident for binary experiment.

-- Reason:
Manual inspection showed every challenging video contains an accident.
Dataset documentation describes challenging videos by environmental conditions (night, snow, cloudy, dim light) rather than a different semantic event.
The deployment objective of RADS is accident detection, not challenging-condition classification.

-- Evidence:
- Manual inspection of all challenging videos.
- TUDAT documentation.
- Baseline confusion matrix.
- Codex analysis.

-- Status: Executed


- ## D002

-- Date: 2026-08-08

-- Decision: Centralize experiment identity inside `training_config.yaml`. All scripts, notebooks, utility managers, logs, output directories, checkpoints, manifests, and W&B runs inherit the experiment name automatically through the shared configuration.


--Reason: Experiment names were previously generated independently in multiple entry points, creating maintenance overhead and inconsistent W&B runs.

-- Status: executed

- ## D003

-- Date: 2026-08-09

-- Decision: Seeing the val of binary baseline training, decided to edit epoch to 45 and enabling early stopping

-- Reason: 
- To prevent overfitting on training data. The val accuracy drops sharply after 10 epochs.
- To allow the model to train for longer if it is still learning.

-- Impact:
- Training time decreased.
- Model didnt generalize better to unseen data.
- Current val accuracy is low, and it didnt improved

-- Status: Executed


## D004

--Date: 2026-08-10

-- Decision: Increase frames per video from 5 to 8

-- Reason:

- Only 5 frames are currently sampled from each video, which may leave important accident-related visual information out.
- Accident detection depends on visual changes throughout the video, so sampling more frames may provide better coverage of the accident event.
- To test whether increasing frame coverage improves the model's ability to generalize to unseen videos.

-- Impact:

- Increased the number of sampled frames per video from 5 to 8.
- Training dataset size and training time will increase.
- Model architecture, batch size, optimizer, learning rate, dataset split, and other training settings remain unchanged to isolate the effect of frame sampling.
- The experiment will determine whether additional frame coverage improves validation and test performance.

-- Status: Executed

## D005 

-- Date: 2026-08-10

-- Decision: Fix checkpoint isolation and misleading test metadata

-- Reason:

- Evaluation scripts (`test.py`, `validate.py`, `predict.py`) must never modify training checkpoints such as `best.pt`, `last.pt`, or files inside `latest/checkpoints/`.
- Test metadata incorrectly reports `total_epochs: 1`, which is misleading because testing does not perform training.

-- Impact:

- Only `train.py` will be permitted to create or update checkpoints.
- Test/validation/prediction will load checkpoints in read-only mode and write only their own evaluation outputs.
- Test metadata will record the actual checkpoint used and test execution information instead of fake training epoch information.

-- Status: Executed


## D006

-- Date: 2026-08-26

-- Decision: Freeze T01 Temporal Proof-of-Concept and Transition to Small PICEK Temporal Subset Phase.

-- Reason:
- T01 validated that temporal sequence modeling (ResNet18 + GRU) substantially outperforms the 2D spatial baseline (E07), achieving 100% test accident recall (0 false negatives) and doubling validation F1 (0.3119 -> 0.6316).
- T01 line of inquiry on TUDAT v2 is complete and verified with full artifact lineage.
- Continuing progress requires scaling beyond the small 93-video TUDAT dataset to a larger, multi-class collision dataset (PICEK).
- To maintain rapid iteration and avoid premature optimization on 2000+ videos, a deterministic Small PICEK Temporal Subset will be designed and validated first.

-- Impact:
- T01 configuration, checkpoints, and artifacts are frozen and immutable (`training_config_t01_frozen.yaml`).
- TUDAT v2 benchmark remains unchanged.
- A dedicated, versioned metadata subset (`picek_v1_small`) will be extracted for the next temporal modeling phase.

-- Status: Executed


## D007

-- Date: 2026-09-03

-- Decision: Transition RADS from ResNet18 + GRU video-level classification to YOLO-based object-centric temporal event reasoning architecture.

-- Reason:

- All five specification documents (MASTER_SPEC.md, AI.md, SYSTEM.md, TECH_STACK.md, MVP.md) converge on the same target: an object-centric, temporal, event-understanding pipeline.
- The existing ResNet18 + GRU baseline (P02) achieved only 45.9% accuracy with an 83.8% normal false-positive rate, demonstrating that video-level classification is fundamentally insufficient for accident detection.
- The specifications explicitly describe the current architecture as a "baseline to be surpassed" (MASTER_SPEC §15, AI §27), not the target system.
- A full repository audit (see `docs/architecture/IMPLEMENTATION_AUDIT_AND_ORDER.md`) confirmed that **none of the 14 specified pipeline stages exist in code**. The entire MVP pipeline must be built from scratch.
- The new architecture uses: Pretrained YOLO → Multi-Object Tracking (ByteTrack or equivalent) → Persistent Object IDs → Trajectories → Motion Features → Object Interaction Analysis → Temporal Event Reasoning → Accident Detection → Event Localization → Severity Estimation → Visualization → Structured Output.

-- Evidence:
- Repository audit against all 5 specification documents.
- P02 baseline results: 45.9% accuracy, 0.535 AUROC, 83.8% FPR on normal class.
- T01 temporal POC showed temporal reasoning improves accident recall to 100% but the architecture remains fundamentally a video classifier, not an event detector.
- No YOLO, tracking, trajectory, interaction, severity, or structured output code exists in the repository.

-- Impact:
- The ResNet18 + GRU baseline code is preserved as a frozen reference (not deleted).
- A new `rads/` module directory will be created with the full pipeline structure.
- Dependencies `ultralytics` (YOLO) and a tracking library will be added.
- Implementation will proceed in 14 phases as documented in `IMPLEMENTATION_AUDIT_AND_ORDER.md`.
- Progress will be tracked in `docs/architecture/IMPLEMENTATION_TRACKER.md`.

-- Status: Approved — Implementation Not Yet Started