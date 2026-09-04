# RADS — MVP & Evaluation Specification

**Project:** RADS
**Document:** MVP & Evaluation Specification
**Version:** 1.0
**Status:** Active
**Date:** 2026-09-03

---

# 1. Purpose

This document defines the scope, priorities, evaluation strategy, and success criteria for the RADS MVP.

The purpose is to ensure that development remains focused on producing a **coherent, demonstrable, and measurable accident-understanding system** rather than expanding indefinitely into production-level requirements.

The MVP should demonstrate the complete RADS concept:

```text
Video
  ↓
Detection
  ↓
Tracking
  ↓
Persistent IDs
  ↓
Trajectories
  ↓
Motion
  ↓
Interactions
  ↓
Accident Event
  ↓
Localization
  ↓
Severity
  ↓
Output
```

---

# 2. MVP Objective

The MVP should prove that RADS can move beyond simple video-level classification and reason about accident events using objects and their temporal behavior.

The MVP does not need to be perfect.

It needs to demonstrate that:

1. Objects can be detected.
2. Objects can be tracked.
3. Persistent identities can be maintained.
4. Object trajectories can be constructed.
5. Motion characteristics can be extracted.
6. Object interactions can be analyzed.
7. Accident-like events can be identified.
8. The event can be localized in time.
9. Objects involved in the event can be identified.
10. A basic severity estimate can be produced.
11. The complete pipeline can be demonstrated visually.
12. The system can produce structured results.

---

# 3. MVP Definition

A RADS MVP is considered a working system when a traffic video can be processed end-to-end:

```text
INPUT VIDEO
     ↓
OBJECT DETECTION
     ↓
OBJECT TRACKING
     ↓
PERSISTENT IDs
     ↓
TRACK HISTORIES
     ↓
TRAJECTORIES
     ↓
MOTION FEATURES
     ↓
OBJECT INTERACTIONS
     ↓
ACCIDENT EVENT
     ↓
EVENT TIMESTAMP
     ↓
INVOLVED OBJECTS
     ↓
SEVERITY
     ↓
VISUALIZATION
     ↓
STRUCTURED RESULT
```

The system should work on representative examples rather than only on a single hand-picked video.

---

# 4. MVP Priorities

Development priority should be:

```text
P0 — Required
P1 — Important
P2 — Useful
P3 — Future
```

---

# 5. P0 — Mandatory MVP Components

The following are mandatory.

## 5.1 Video Input

The system must accept traffic video files.

Minimum support:

```text
.mp4
```

Additional formats may be supported where convenient.

---

## 5.2 Object Detection

The MVP must contain a working pretrained object detector.

Current preferred direction:

```text
YOLO
```

The detector must produce:

```text
Class
Confidence
Bounding Box
Frame
Timestamp
```

---

## 5.3 Object Tracking

The MVP must associate detections across frames.

A tracker such as:

```text
ByteTrack
```

or another suitable tracker may be used.

The tracker must produce persistent object IDs.

Example:

```text
Vehicle → ID 3
Vehicle → ID 7
Motorcycle → ID 12
```

---

## 5.4 Track Histories

The system must retain the history of tracked objects.

Example:

```text
ID 3

Frame 100 → position
Frame 101 → position
Frame 102 → position
Frame 103 → position
...
```

---

## 5.5 Trajectory Extraction

The MVP must be able to construct trajectories from track histories.

At minimum:

```text
Timestamp
X position
Y position
Object ID
```

---

## 5.6 Motion Features

The MVP should derive useful motion information.

Initial candidates:

```text
Displacement
Direction
Velocity
Acceleration
Direction change
Relative velocity
Relative distance
```

The system must clearly distinguish image-space measurements from real-world physical measurements.

---

## 5.7 Interaction Analysis

The MVP must identify potentially meaningful interactions between objects.

Possible signals:

```text
Distance reduction
Trajectory convergence
Bounding-box overlap
Relative motion change
Abrupt direction change
Abrupt deceleration
```

An interaction candidate should not automatically be classified as an accident.

---

## 5.8 Accident Event Detection

The system must determine whether an interaction represents an accident.

The initial implementation may use:

```text
Rule-based reasoning
+
Engineered features
+
Lightweight ML / temporal model
```

The exact approach should be determined through experimentation.

---

## 5.9 Event Localization

The MVP should estimate:

```text
Event start
Impact / event time
Event end
```

Exact localization accuracy will depend on annotation quality and model performance.

---

## 5.10 Involved Object Identification

The system should identify which tracked objects are most likely involved in the accident.

Example:

```text
Accident detected

Objects:
ID 3
ID 7
```

---

## 5.11 Basic Severity Estimation

The MVP must provide a basic severity estimate.

Initial classes:

```text
LOW
MEDIUM
HIGH
```

A transparent scoring/rule-based system is acceptable.

A sophisticated learned severity model is not required for the MVP.

---

## 5.12 Visualization

The system should render:

```text
Bounding boxes
Object IDs
Trajectories
Accident marker
Objects involved
Severity
```

The goal is to make the system's reasoning visible.

---

# 6. P1 — Important Components

These should be implemented if time and reliability allow.

```text
Accident type classification
Event confidence
Improved temporal localization
Explainable evidence display
Dashboard integration
Telegram alert integration
Hard-negative evaluation
Near-collision handling
```

---

# 7. P2 — Useful but Not Required

Potential additions:

```text
Real-time processing
Advanced visualization
More accident categories
Advanced severity indicators
Camera-specific calibration
Performance optimization
Additional tracking comparisons
```

These should not delay the core MVP.

---

# 8. P3 — Future Work

The following are explicitly outside the MVP scope:

```text
Production-grade reliability
Nationwide deployment
Edge-device optimization
Large-scale distributed inference
Fully learned severity prediction
Advanced multimodal models
Large graph neural networks
Large temporal Transformers
Automatic emergency-service integration
Full autonomous response
```

These may become future development directions.

---

# 9. Dataset Strategy

The dataset is expected to expand beyond the existing P02 dataset.

The goal is not simply to increase the number of videos.

The expanded dataset should improve **behavioral diversity**.

Important categories include:

```text
Accidents
Normal traffic
Hard negatives
Near collisions
Different accident types
Different camera perspectives
Different traffic densities
Different weather
Day
Night
Different road environments
```

---

# 10. Positive Data

Positive examples should include genuine accident events.

Where possible, positive examples should contain:

```text
Pre-event context
Interaction
Impact
Post-impact behavior
```

This allows the temporal reasoning system to learn the event rather than only the appearance of the final frame.

---

# 11. Negative Data

Normal traffic should contain diverse behavior.

Examples:

```text
Normal driving
Lane changes
Overtaking
Merging
Traffic congestion
Stopping
Turning
Close following
Dense traffic
```

The goal is to prevent the system from treating normal traffic behavior as an accident.

---

# 12. Hard Negatives

Hard negatives should be treated as a dedicated category.

Important examples:

```text
Near collisions
Emergency braking
Sudden steering
Abrupt lane changes
Objects passing very close
Traffic conflicts without collision
Sudden stops
Camera motion
Occlusions
```

Hard negatives are particularly important because the current baseline demonstrates a serious false-positive problem.

---

# 13. Dataset Splitting

Dataset splitting must occur at the appropriate source level.

If multiple clips originate from the same source video, those clips should not create leakage between:

```text
Train
Validation
Test
```

Source-video separation must remain enforced.

---

# 14. Existing Baseline

P02 is the current baseline against which the new architecture should be compared where appropriate.

P02:

```text
Total clips:
500

Accident:
250

Normal:
250

Train:
350

Validation:
76

Test:
74
```

Architecture:

```text
ResNet18 + GRU
```

GRU hidden size:

```text
128
```

Optimizer:

```text
Adam
```

Learning rate:

```text
1e-4
```

Weight decay:

```text
1e-4
```

Scheduler:

```text
Cosine Annealing
```

---

# 15. P02 Baseline Results

Current P02 test results:

```text
Accuracy:
45.9%

Balanced Accuracy:
45.9%

AUROC:
0.535

Macro F1:
0.407

Accident F1:
0.583

Normal F1:
0.231
```

Confusion matrix:

```text
                 Predicted
              Normal  Accident

Actual Normal     6       31
Actual Accident   9       28
```

Normal false-positive rate:

```text
31 / 37 = 83.8%
```

This baseline demonstrates that the previous architecture is not sufficiently discriminative for the desired accident-detection system.

It should therefore be retained as a baseline rather than treated as the final architecture.

---

# 16. P01 Historical Context

Earlier P01 experiments used:

```text
200 videos
100 accident
100 normal
```

with a ResNet18 + GRU architecture.

The original six-configuration hyperparameter study was later determined to be invalid for model-selection conclusions because of an early-stopping configuration error.

The issue involved monitoring accident F1 with the wrong optimization direction.

Therefore:

> The original six-config results must not be interpreted as reliable evidence that one optimizer or hyperparameter configuration was superior to another.

The corrected baseline remains useful as historical evidence.

---

# 17. Evaluation Philosophy

RADS should not be evaluated using accuracy alone.

A model can achieve seemingly reasonable accuracy while failing catastrophically on one class.

Evaluation should therefore include:

```text
Accuracy
Balanced Accuracy
Precision
Recall
F1
Macro F1
AUROC
Confusion Matrix
False Positive Rate
False Negative Rate
```

Per-class results should be reported.

---

# 18. Accident Detection Metrics

For accident detection, the most important metrics include:

### Recall

How many real accidents were detected?

```text
TP / (TP + FN)
```

### Precision

How many predicted accidents were actually accidents?

```text
TP / (TP + FP)
```

### F1

Balance between precision and recall.

### AUROC

Measure of discrimination between accident and normal examples.

### False Positive Rate

Particularly important because excessive false alarms reduce the usefulness of an accident alerting system.

---

# 19. Event Localization Metrics

Once event localization exists, evaluation should include:

```text
Temporal localization error
Impact-time error
Event overlap
Detection delay
False alarms per unit time
```

Where suitable annotations exist, temporal IoU may be used.

---

# 20. Tracking Evaluation

Tracking should not be evaluated solely by visual inspection.

Where appropriate, metrics may include:

```text
IDF1
MOTA
HOTA
ID switches
Track fragmentation
```

The exact metrics depend on available ground truth annotations.

The MVP may initially use qualitative and limited quantitative tracking validation if annotated tracking ground truth is unavailable.

---

# 21. Severity Evaluation

Severity should eventually be evaluated using:

```text
Confusion Matrix
Macro F1
Per-class Recall
Per-class Precision
```

If the MVP severity engine is rule-based and ground-truth severity labels are limited, evaluation should explicitly state this limitation.

The system must not present an engineered severity score as a scientifically validated severity prediction model.

---

# 22. Detection Evaluation

Object detection should be evaluated independently from accident detection where possible.

Useful metrics include:

```text
Precision
Recall
mAP
```

Detection failures should be distinguishable from reasoning failures.

For example:

```text
Detector failed to detect vehicle
```

should not be confused with:

```text
Accident reasoner incorrectly classified the event
```

---

# 23. End-to-End Evaluation

The complete system should eventually be evaluated as:

```text
Video
 ↓
Detection
 ↓
Tracking
 ↓
Reasoning
 ↓
Accident
 ↓
Severity
```

The evaluation should measure:

```text
Did RADS detect the accident?
Did it detect it at approximately the correct time?
Did it identify the correct objects?
Did it assign a reasonable severity?
```

---

# 24. Baseline Comparison

The new architecture should be compared with P02 where meaningful.

The comparison should include:

```text
P02
vs.
YOLO + Tracking + Temporal Reasoning
```

Potential comparison metrics:

```text
Accuracy
Balanced Accuracy
AUROC
Macro F1
Accident Recall
Accident Precision
Normal Recall
False Positive Rate
```

For event-based systems, additionally:

```text
Event localization
Detection delay
Involved-object accuracy
```

---

# 25. Success Criteria

The MVP should not be judged solely by reaching one arbitrary accuracy number.

Success means demonstrating measurable improvement in the dimensions that matter for the intended architecture.

The new system should aim to:

```text
1. Produce stable object identities.
2. Produce meaningful trajectories.
3. Identify meaningful interactions.
4. Distinguish at least some hard negatives from accidents.
5. Localize accident events.
6. Identify involved objects.
7. Produce a defensible severity estimate.
8. Demonstrate the complete pipeline visually.
9. Improve meaningfully over the P02 baseline on relevant metrics.
```

---

# 26. What Would Constitute a Bad Result?

A model should not be considered successful merely because it produces a high accuracy number.

Warning signs include:

```text
Very high false-positive rate
All predictions concentrated in one class
No meaningful separation between accident and normal
Identical predictions across unrelated videos
Unstable tracking IDs
Large numbers of ID switches
Events triggered by ordinary traffic behavior
Severity predictions unsupported by evidence
Strong results caused by source-video leakage
```

These should trigger investigation rather than being hidden.

---

# 27. Experiment Discipline

Every meaningful experiment should record:

```text
Experiment ID
Dataset version
Train/validation/test split
Detector
Tracker
Feature configuration
Temporal model
Hyperparameters
Checkpoint
Evaluation metrics
Observations
Failure cases
Conclusion
```

Results should be traceable to the configuration that produced them.

---

# 28. Failure Analysis

When the system performs poorly, analysis should identify the stage responsible.

Potential failure categories:

```text
Detection Failure
Tracking Failure
Trajectory Failure
Motion Feature Failure
Interaction Failure
Temporal Reasoning Failure
Accident Classification Failure
Severity Failure
```

Example:

```text
Vehicle not detected
      ↓
No track created
      ↓
No trajectory
      ↓
No interaction
      ↓
Accident missed
```

This is fundamentally different from:

```text
Vehicle detected
      ↓
Track correct
      ↓
Interaction correct
      ↓
Accident reasoner says NORMAL
```

The two failures require different solutions.

---

# 29. Visualization as an Evaluation Tool

Visualization should not only be used for presentation.

Annotated videos should be used to inspect:

```text
Detection quality
ID consistency
Trajectory quality
Interaction candidates
Accident event timing
Severity reasoning
```

For difficult examples, the system should make it possible to see what the model saw.

---

# 30. MVP Development Order

Implementation should proceed in stages.

```text
PHASE 1
Video pipeline
     ↓

PHASE 2
YOLO detection
     ↓

PHASE 3
Multi-object tracking
     ↓

PHASE 4
Persistent IDs + track histories
     ↓

PHASE 5
Trajectory and motion features
     ↓

PHASE 6
Object interaction engine
     ↓

PHASE 7
Accident event logic
     ↓

PHASE 8
Temporal reasoning
     ↓

PHASE 9
Event localization
     ↓

PHASE 10
Severity engine
     ↓

PHASE 11
Visualization
     ↓

PHASE 12
Dashboard / Telegram integration
     ↓

PHASE 13
Evaluation and failure analysis
```

The system should be tested after each major phase.

---

# 31. Development Strategy

The project should follow:

```text
Build
 ↓
Test
 ↓
Inspect
 ↓
Measure
 ↓
Improve
```

rather than attempting to build the entire architecture before testing any component.

A working detector + tracker should exist before sophisticated event reasoning is implemented.

A working trajectory system should exist before training a temporal model.

This allows failures to be isolated.

---

# 32. Dataset Expansion Strategy

Dataset expansion should occur alongside system development.

The priority should be:

```text
Existing Dataset
      ↓
Baseline Pipeline
      ↓
Identify Failure Modes
      ↓
Collect / Add Relevant Examples
      ↓
Hard Negatives
      ↓
Near Collisions
      ↓
Diverse Accidents
      ↓
Re-evaluate
```

New data should be added based on observed weaknesses rather than simply maximizing video count.

---

# 33. MVP vs Final Product

The MVP is not the final RADS system.

## MVP

```text
Pretrained detector
+
Tracker
+
Trajectory extraction
+
Engineered motion features
+
Temporal accident reasoning
+
Basic severity
+
Visualization
```

## Future Product

```text
Advanced detector
+
Robust tracking
+
Learned trajectory representations
+
Object interaction modeling
+
Temporal event model
+
Accident classification
+
Event localization
+
Learned severity
+
Real-time processing
+
Production monitoring
+
Alert infrastructure
```

The distinction must remain explicit.

---

# 34. Things That Should Not Delay the MVP

The following should not block the MVP:

```text
Perfect accuracy
Perfect severity prediction
Perfect accident-type classification
Real-world velocity estimation
Full camera calibration
Real-time optimization
Massive dataset
Large Transformer architecture
Production deployment
Cloud infrastructure
Complete dashboard redesign
```

If the core pipeline works, these can be developed later.

---

# 35. MVP Demonstration

The final demonstration should ideally show a complete video flowing through RADS.

Example:

```text
VIDEO
  ↓
YOLO detects vehicles
  ↓
Tracker assigns IDs
  ↓
IDs persist across frames
  ↓
Trajectories are displayed
  ↓
Vehicles approach each other
  ↓
Interaction detected
  ↓
Motion changes abruptly
  ↓
Accident event identified
  ↓
Impact timestamp displayed
  ↓
Objects involved identified
  ↓
Severity estimated
  ↓
Result displayed
```

The demonstration should make the architecture visually understandable to a viewer who has not read the code.

---

# 36. Definition of Done

The RADS MVP is considered complete when:

### Pipeline

* [ ] Video input works.
* [ ] Frames are processed correctly.
* [ ] Object detection works.
* [ ] Tracking works.
* [ ] Persistent IDs are visible.
* [ ] Track histories are stored.
* [ ] Trajectories are generated.
* [ ] Motion features are calculated.
* [ ] Object interactions are identified.
* [ ] Accident events can be detected.
* [ ] Event timing is estimated.
* [ ] Involved objects are identified.
* [ ] Severity is estimated.
* [ ] Results are visualized.
* [ ] Structured output is produced.

### Evaluation

* [ ] Dataset split is verified.
* [ ] No source-video leakage exists.
* [ ] Baseline results are recorded.
* [ ] New system results are recorded.
* [ ] Confusion matrix is available.
* [ ] Precision/recall/F1 are available.
* [ ] AUROC is available where applicable.
* [ ] False-positive behavior is analyzed.
* [ ] Representative failure cases are documented.

### Engineering

* [ ] Configuration is reproducible.
* [ ] Model/checkpoint provenance is recorded.
* [ ] Major components are modular.
* [ ] Pipeline can be run from a clean configuration.
* [ ] Existing dashboard/alert integration is not broken.

---

# 37. Final MVP Principle

The objective of the MVP is not:

> Build the most sophisticated accident-detection AI possible.

The objective is:

> **Demonstrate a coherent object-centric accident-understanding pipeline that can detect, localize, and characterize accident events from video.**

The MVP should establish the foundation for the future RADS system.

---

# 38. Final Evaluation Principle

RADS should always prefer:

```text
Honest measurement
      over
Impressive numbers
```

A model with modest performance but clear failure analysis is more valuable than a model with suspiciously high accuracy caused by leakage, dataset shortcuts, or misleading metrics.

The final system should make it possible to answer:

```text
What did RADS detect?

Which objects were involved?

What happened over time?

Why did it classify the event as an accident?

When did it happen?

How severe does it appear?

How reliable is that conclusion?
```

Those questions define the practical success of the RADS MVP.

---

# 39. Final System Goal

The long-term goal is:

```text
                    RADS
                     │
                     ↓
                UNDERSTAND
                     │
          ┌──────────┴──────────┐
          ↓                     ↓
       OBJECTS               MOTION
          │                     │
          └──────────┬──────────┘
                     ↓
                 INTERACTION
                     ↓
                   EVENT
                     ↓
                 ACCIDENT
                     ↓
                 SEVERITY
                     ↓
              ACTIONABLE OUTPUT
```

The MVP should prove the feasibility of this direction.

The system should then evolve through evidence-driven experimentation rather than uncontrolled architectural expansion.
