## RADS -- Master Specification

- Document: 00_MASTER_SPEC.md
- Project: RADS
- Version: 2.0
- Status: Active -- Deployable Runtime v1
- Target: Production Runtime
- Last Updated: September 2026
- History: Version 1.0 covered the research MVP phase (completed). Version 2.0 covers the transition to a deployable runtime.

# 1. Project Definition
---

RADS is an AI-based road accident detection and severity analysis system designed to identify accident events from traffic video, determine when and how an accident occurred, and provide an interpretable estimate of accident severity.

The research MVP phase is complete. The core intelligence pipeline (detection, tracking, motion, interaction, reasoning, severity) has been demonstrated end-to-end on pre-recorded clips.

The current objective is to build RADS Deployable Runtime v1: a system that can process live and recorded video from real sources (files, webcams, RTSP/IP cameras), emit structured accident events over a network API, and run portably on any supported machine via Docker.

The system must:

Process continuous video streams, not only pre-recorded clips.
Maintain object identities across frames within a sliding temporal window.
Detect accident events incrementally as frames arrive.
Emit structured events with a defined lifecycle (candidate, detected, confirmed, resolved).
Expose results over REST and WebSocket APIs.
Run on CPU or CUDA without hardcoded local paths.
Be deployable via Docker with configuration-only setup.
Preserve the existing research and evaluation system unchanged.
---
# 2. Core Design Philosophy

RADS should reason about events and interactions, rather than treating an accident as a visual property of a single frame.

The system should move from:

"What does this frame look like?"

toward:

"What are the objects doing?"
        ↓
"How are they moving?"
        ↓
"How are they interacting?"
        ↓
"Did something abnormal happen?"
        ↓
"Did a collision/event occur?"
        ↓
"When did it occur?"
        ↓
"How severe was it?"

The fundamental principle is:

Accidents are temporal events involving interacting objects, not merely accident-looking frames.

This principle applies equally to batch processing of recorded clips and to live stream processing.
---
# 3. Target End-to-End System

The system operates in two modes sharing the same core intelligence modules:

Batch mode (research/evaluation):

    Complete Video File
         |
         v
    Full Processing
         |
         v
    Result JSON

Runtime mode (deployment):

    Source (file / webcam / RTSP)
         |
         v
    Frame Acquisition
         |
         v
    Object Detection (YOLO)
         |
         v
    Object Tracking (Persistent IDs)
         |
         v
    Track History (Sliding Window)
         |
         v
    Motion / Interaction Features
         |
         v
    Event Reasoning
         |
         v
    Accident Detection
         |
         v
    Event Lifecycle (candidate / detected / confirmed / resolved)
         |
         v
    Severity Estimation
         |
         v
    API / WebSocket / Alerts

The architecture is intentionally modular.

Each stage must produce structured information that can be inspected independently.

Both modes use the same detection, tracking, motion, interaction, reasoning, and severity modules.
---
# 4. Primary Objective

The deployable runtime must process video from any supported source through the complete reasoning chain and deliver structured accident events over a network API.

Input (file / webcam / RTSP)
  |
  v
Detect vehicles / relevant road objects
  |
  v
Assign persistent IDs
  |
  v
Track objects within a sliding temporal window
  |
  v
Calculate movement and interaction information
  |
  v
Identify collision-like temporal events
  |
  v
Classify accident vs non-accident
  |
  v
Assign event lifecycle status
  |
  v
Estimate severity
  |
  v
Emit structured event via API

The runtime requires:

Processing of all three source types (file, webcam, RTSP).
Automatic reconnection on RTSP disconnection.
Bounded memory usage via sliding window.
Structured event output with lifecycle tracking.
REST and WebSocket API for event access.
Graceful shutdown on signals.
Docker-based deployment without hardcoded paths.
Full backward compatibility with the batch pipeline.
---
# 5. Object-Centric Intelligence

RADS should use an object-centric representation as a core part of its reasoning.

Instead of relying only on raw RGB frames, the system should maintain information about detected objects.

For each tracked object, the system should aim to maintain:

Object ID
Object class
Bounding box
Detection confidence
Center position
Width / height
Frame timestamp
Track duration
Trajectory
Velocity estimate
Acceleration estimate

Additional features may be introduced as the system develops.

Potential object classes include:

Car
Truck
Bus
Motorcycle
Bicycle
Pedestrian
Other relevant road users

The exact classes depend on the selected detection model and dataset.
---
# 6. Persistent Object Identity

Tracking is a foundational component of RADS.

A detected object must not be treated as an entirely new object on every frame.

The system should instead maintain persistent identities:

Frame 001:
Vehicle → ID 07

Frame 002:
Vehicle → ID 07

Frame 003:
Vehicle → ID 07

...

Frame 120:
Vehicle → ID 07

This enables RADS to reconstruct trajectories and reason about changes in object behavior.

Tracking is therefore not merely a visualization feature.

It is a prerequisite for temporal reasoning.
---
# 7. Temporal Representation

RADS must represent events over time.

For each object:

Position(t)
Velocity(t)
Acceleration(t)
Direction(t)

For interacting objects:

Distance(A,B,t)
RelativeVelocity(A,B,t)
RelativeAcceleration(A,B,t)
TrajectoryRelationship(A,B,t)
BoundingBoxOverlap(A,B,t)

The system should use temporal changes in these quantities to identify unusual interactions.

The exact feature set and temporal model remain open to experimentation.
---
# 8. Accident Reasoning

The system should not classify an accident solely from a single frame.

Accident reasoning should consider evidence across a temporal window.

Potential evidence includes:

Rapid decrease in distance between objects
High relative velocity
Sudden deceleration
Abrupt trajectory change
Trajectory intersection
Bounding-box interaction / overlap
Collision-like contact
Sudden change in object direction
Post-impact displacement
Vehicle stopping unexpectedly
Secondary interactions

These features are candidate evidence, not individually sufficient definitions of an accident.

The reasoning system should combine multiple signals wherever possible.
---
# 9. Event-Based Accident Detection

RADS should ultimately produce an event rather than only a clip-level classification.

Conceptually:

Normal traffic
      ↓
Objects approach
      ↓
Interaction begins
      ↓
Motion becomes abnormal
      ↓
Collision-like event
      ↓
Post-event behavior
      ↓
Accident confirmed

The system should aim to identify:

Accident: YES / NO

Event start: approximately T1
Impact/event time: approximately T2
Event end: approximately T3

Objects involved:
ID X
ID Y
...

For the MVP, temporal localization may be approximate.

Exact frame-level annotation is a future refinement if the available data supports it.
---
# 10. Severity Estimation

Accident detection and severity estimation are separate tasks.

The MVP should provide a transparent severity estimation mechanism based on measurable event characteristics.

Potential severity evidence includes:

Number of objects involved
Vehicle/object types
Relative velocity
Estimated impact intensity
Magnitude of trajectory change
Post-impact displacement
Sudden deceleration
Multiple-object involvement
Pedestrian involvement
Secondary collisions
Rollover-like behavior
Road obstruction

The initial severity system may be:

LOW
MEDIUM
HIGH

A rule-based or weighted scoring approach is acceptable for the MVP if it is clearly documented and based on measurable features.

A learned severity model can be introduced later once sufficient annotated data exists.
---
# 11. Explainability

RADS should provide evidence for its decisions.

The system should avoid producing only:

ACCIDENT
Confidence: 94%

Instead, where possible, it should provide information such as:

ACCIDENT DETECTED

Event time: 00:07.4

Objects involved:
Vehicle #03
Vehicle #07

Observed evidence:
• Rapid relative approach
• Significant trajectory change
• Collision-like overlap
• Abrupt deceleration
• Post-event displacement

Estimated severity:
HIGH

The exact explanation format may change during implementation.

The principle should remain:

The system should be able to show why it believes an accident occurred.
---
# 12. Model Strategy

The system should use pretrained computer-vision components where appropriate rather than attempting to train every component from scratch.

The initial model strategy is:

Object Detection
        ↓
Pretrained YOLO

Object Tracking
        ↓
Dedicated multi-object tracker

Temporal Reasoning
        ↓
Temporal ML model and/or engineered temporal reasoning

Accident Detection
        ↓
Event-level classifier / reasoning system

Severity
        ↓
Transparent MVP scoring system
        ↓
Future learned model

Specific model variants should be selected based on:

Accuracy
Speed
Hardware constraints
Dataset compatibility
Ease of training/fine-tuning
Research value
MVP development time

The architecture should remain modular enough that individual models can be replaced.
---
# 13. Data Strategy

RADS should prioritize diversity and generalization.

The dataset should contain, where possible:

Accidents
Normal traffic
Hard negatives
Near-collisions
Sudden braking
Abrupt lane changes
Dense traffic
Sparse traffic
Day
Night
Rain
Snow
Different camera viewpoints
Different road environments
Different accident types

The dataset should avoid unnecessary shortcuts that allow the model to associate environmental characteristics with the accident label.
---
# 14. Data Leakage Prevention

Source-video separation is mandatory.

If multiple clips originate from the same source video, those clips must remain within the same dataset split.

The following must never occur:

Source Video A
    ├── Train
    └── Test

Instead:

Source Video A
    └── Train

Source Video B
    └── Validation

Source Video C
    └── Test

This is required to prevent overly optimistic evaluation caused by visual similarity between related clips.
---
# 15. Existing Baseline

RADS has an existing ResNet18 + GRU baseline.

The baseline should be preserved as a reference point rather than treated as the final architecture.

The corrected P01 baseline and P02 scaling experiment demonstrated that the existing architecture is weak for the current objective.

P02 used:

500 clips
250 accident
250 normal

350 train
76 validation
74 test

ResNet18 + GRU
Hidden dimension: 128

Adam
Learning rate: 1e-4
Weight decay: 1e-4

Cosine annealing

P02 results:

Accuracy:            45.9%
Balanced Accuracy:   45.9%
AUROC:               0.535
Macro F1:             0.407
Accident F1:         0.583
Normal F1:            0.231

The test confusion matrix showed:

Normal:
6 correctly classified
31 incorrectly classified as accident

Accident:
28 correctly classified
9 incorrectly classified as normal

Therefore the normal-class false-positive rate was approximately:

31 / 37 = 83.8%

P02 remains the baseline that the new system should be evaluated against.

The baseline should not be deleted.
---
# 16. Previous Experiment Integrity

The original six-configuration P01 hyperparameter experiment contained a training-control issue.

Early stopping monitored accident F1 using the wrong optimization direction:

mode = min

when it should have been:

mode = max

Therefore the original six-configuration comparison must not be treated as reliable evidence for optimizer superiority.

The corrected baseline should be used for meaningful comparison.

Historical experiment outputs may be archived, but their conclusions must not be silently presented as valid model comparisons.
---
# 17. Evaluation Philosophy

RADS must not optimize for accuracy alone.

The evaluation system should report:

Accident classification
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
Event detection

Where annotations permit:

Event localization accuracy
Temporal IoU
Detection delay
False alarms
Detection
Precision
Recall
mAP
Tracking

Where applicable:

ID consistency
ID switches
Tracking accuracy metrics
Severity
Per-class precision
Per-class recall
Macro F1
Confusion matrix

The system must report class-specific performance, particularly accident recall and normal false-positive rate.
---
# 18. Runtime v1 Success Criteria

The deployable runtime is considered successful if it can demonstrate:

Perception
Detect relevant road objects from file, webcam, and RTSP sources.
Maintain object identities within a sliding temporal window.
Tracking
Produce stable trajectories for relevant objects.
Preserve IDs sufficiently for downstream reasoning.
Reset state cleanly on stream reconnection.
Temporal reasoning
Calculate meaningful motion and interaction features incrementally.
Identify abnormal temporal behavior within the sliding window.
Accident detection
Distinguish accident events from normal traffic.
Produce an interpretable accident decision with lifecycle status.
Event localization
Identify approximately when the accident occurred.
Emit events with start_time, impact_time, and end_time.
Severity
Produce LOW / MEDIUM / HIGH severity with numeric score and evidence list.
API
Expose health, events, and WebSocket endpoints.
Function correctly when API is disabled.
Deployment
Run via Docker on a clean machine.
Support CPU and CUDA configurations.
Operate without hardcoded local paths.
---
# 19. Runtime v1 vs Future System

Runtime v1
YOLO
  |
  v
Tracking (ByteTrack)
  |
  v
Trajectories (Sliding Window)
  |
  v
Motion / Interaction Features
  |
  v
Temporal Event Reasoning (Rule-Based)
  |
  v
Accident Detection
  |
  v
Event Lifecycle
  |
  v
Severity Scoring (Heuristic)
  |
  v
REST API / WebSocket
  |
  v
Docker Deployment

Future RADS may include:

Advanced object detection
+
Robust multi-object tracking
+
Appearance embeddings
+
Trajectory representation
+
Object interaction graphs
+
Advanced temporal models
+
Accident type classification
+
Learned severity estimation
+
Multimodal reasoning
+
Multi-stream support
+
Edge deployment
+
Production monitoring and alerting
+
Camera calibration and world-space coordinates

Future features must not be introduced prematurely.
---
# 20. Development Priority

The core intelligence pipeline (stages 1-11 below) is implemented. Development priority is now the runtime and deployment layers.

Implemented (core intelligence):

1. Detection (YOLO)
2. Tracking / Persistent IDs (ByteTrack)
3. Track Data Representation
4. Motion Feature Extraction
5. Object Interaction Analysis
6. Temporal Event Detection
7. Accident Reasoning
8. Event Localization
9. Severity Estimation

Current priority (runtime v1):

10. Source Abstraction (file / webcam / RTSP)
11. Streaming Frame Processor
12. Event Lifecycle Manager
13. Runtime Engine and Main Loop
14. REST API and WebSocket
15. Docker Deployment
16. Integration Testing

See IMPLEMENTATION_PLAN.md for the detailed phase breakdown.

A later component must not be built on undocumented assumptions about an earlier component.

Each stage should have:

Input
Output
Contract
Tests
Visualization / debugging method

where practical.
---
# 21. Modularity Requirement

Every major stage should be independently replaceable.

For example:

YOLO
   ↓
Tracker

The tracker should not depend on the internal implementation details of YOLO beyond a defined detection interface.

Likewise:

Tracker
   ↓
Temporal Reasoning

should communicate through a structured track representation.

This allows future replacement of:

YOLO model
Tracker
Temporal model
Accident classifier
Severity model

without rewriting the entire system.
---
# 22. Engineering Principles

The implementation should prioritize:

Reproducibility
Modularity
Observability
Clear interfaces
Testability
Explainability
Dataset integrity
Simple solutions before unnecessary complexity

Experimental code must not silently modify production or evaluation behavior.

Important parameters should be configurable rather than hard-coded.

Model checkpoints, configurations, metrics and experiment metadata should be traceable.
---
# 23. Research vs MVP Boundary

RADS currently has two simultaneous objectives.

Research objective

Determine whether an object-centric, temporal architecture can provide better accident detection and event understanding than the existing RGB video-classification baseline.

MVP objective

Demonstrate a functioning end-to-end accident detection and severity pipeline.

These objectives must not be confused.

A component does not need to be scientifically final to be included in the MVP.

Similarly, an MVP component should not be described as production-ready without evidence.
---
# 24. Definition of Done

The MVP pipeline is complete when a user can provide a traffic video and RADS can produce:

VIDEO
  ↓
OBJECTS DETECTED
  ↓
OBJECTS TRACKED
  ↓
TRAJECTORIES GENERATED
  ↓
TEMPORAL INTERACTIONS ANALYZED
  ↓
ACCIDENT EVENT IDENTIFIED
  ↓
EVENT TIMESTAMP PROVIDED
  ↓
SEVERITY ESTIMATED
  ↓
RESULT VISUALIZED

The system should also retain enough intermediate information to inspect how the final decision was reached.
---
# 25. Non-Goals for Runtime v1

The following are explicitly not required for runtime v1:

Perfect real-time performance guarantees
Perfect accident classification
Large-scale distributed training
Fully learned severity model
Perfect frame-level localization
Full autonomous emergency response
Cloud-scale infrastructure
Universal performance across every camera/environment
Multi-stream processing (architecture supports it, not implemented)
Accident type classification
Camera calibration or world-space coordinates
Frontend dashboard UI

These may become future objectives.
---
# 26. Source of Truth

This document defines the highest-level direction of RADS.

Detailed specifications belong in the corresponding documents:

01_PROJECT_VISION.md
02_AI_BRAIN.md
03_DATA_SPEC.md
04_PIPELINE_DESIGN.md
05_MODEL_ARCHITECTURE.md
06_DETECTION_TRACKING.md
07_EVENT_REASONING.md
08_SEVERITY_ENGINE.md
09_EVALUATION_PLAN.md
10_MVP_SCOPE.md
DECISIONS.md

If an implementation conflicts with this specification, the conflict must be identified rather than silently ignored.

If a new technical discovery requires changing the architecture, the change should be documented in DECISIONS.md.
---
# 27. Current Strategic Direction

As of September 2026, RADS has completed the transition from:

Raw video
    |
    v
ResNet18
    |
    v
GRU
    |
    v
Video classification

to:

Raw video
    |
    v
YOLO
    |
    v
Persistent object tracking
    |
    v
Object trajectories
    |
    v
Motion + interaction reasoning
    |
    v
Temporal event detection
    |
    v
Accident detection
    |
    v
Severity estimation

The object-centric architecture is now the established system. The ResNet18 + GRU system remains as the experimental baseline for comparison.

The current phase is making this architecture deployable: processing live sources, exposing events over APIs, and packaging for portable execution via Docker.
---
# 28. Guiding Principle

The ultimate goal of RADS is not:

"Find frames that look like accidents."

It is:

"Understand what happened on the road, identify whether an accident occurred, determine when it occurred, identify the objects involved, and estimate its severity from observable evidence."

The deployable runtime makes this intelligence accessible to real video sources and downstream systems.