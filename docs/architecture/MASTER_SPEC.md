## RADS — Master Specification

- Document: 00_MASTER_SPEC.md
- Project: RADS
- Version: 1.0
- Status: Active — MVP Development
- Target: MVP / Demonstration
- Last Updated: September 2026

# 1. Project Definition
---

RADS is an AI-based road accident detection and severity analysis system designed to identify accident events from traffic video, determine when and how an accident occurred, and provide an interpretable estimate of accident severity.

The system is being developed as a research-oriented MVP.

The immediate objective is not to create a production-ready autonomous safety system.

The immediate objective is to demonstrate a technically coherent pipeline that can:

Understand objects in a traffic scene.
Maintain identities of those objects across frames.
Extract their movement and interaction over time.
Detect anomalous or collision-like events.
Determine whether an accident occurred.
Localize the event temporally.
Estimate accident severity using measurable evidence.
Present the result in an interpretable form.
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
---
# 3. Target End-to-End System

The intended architecture is:

                    INPUT VIDEO
                         │
                         ▼
                 Frame Extraction
                         │
                         ▼
                Object Detection
                      (YOLO)
                         │
                         ▼
                Object Tracking
             (Persistent Object IDs)
                         │
                         ▼
              Track Reconstruction
                         │
                         ▼
          Motion / Interaction Features
                         │
                         ▼
               Event Reasoning
                         │
                         ▼
             Accident Detection
                         │
                         ▼
             Event Localization
                         │
                         ▼
             Severity Estimation
                         │
                         ▼
                    OUTPUT

The architecture is intentionally modular.

Each stage must produce structured information that can be inspected independently.
---
# 4. Primary MVP Objective

The MVP must demonstrate the following complete flow:

Video
  ↓
Detect vehicles / relevant road objects
  ↓
Assign persistent IDs
  ↓
Track objects across time
  ↓
Calculate movement and interaction information
  ↓
Identify collision-like temporal events
  ↓
Classify accident vs non-accident
  ↓
Identify approximate event timestamp
  ↓
Estimate severity
  ↓
Display interpretable result

A successful MVP does not require perfect accuracy.

It requires:

A functioning end-to-end pipeline.
Consistent object tracking.
Meaningful temporal reasoning.
Demonstrable accident detection.
Event localization.
A defensible severity mechanism.
Clear outputs.
Reproducible evaluation.
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
# 18. MVP Success Criteria

The MVP is considered successful if it can demonstrate:

Perception
Detect relevant road objects.
Maintain object identities across multiple frames.
Tracking
Produce stable trajectories for relevant objects.
Preserve IDs sufficiently for downstream reasoning.
Temporal reasoning
Calculate meaningful motion and interaction features.
Identify abnormal temporal behavior.
Accident detection
Distinguish accident events from normal traffic better than the existing baseline.
Produce an interpretable accident decision.
Event localization
Identify approximately when the accident occurred.
Severity
Produce LOW / MEDIUM / HIGH severity.
Provide measurable evidence supporting the severity estimate.
Demonstration
Process a complete video.
Visualize detections and tracking.
Display the detected event.
Display accident confidence/result.
Display severity.
Show relevant reasoning evidence.
---
# 19. MVP vs Final System

The MVP should deliberately be smaller than the eventual RADS system.

MVP
YOLO
  ↓
Tracking
  ↓
Trajectories
  ↓
Motion / Interaction Features
  ↓
Temporal Event Reasoning
  ↓
Accident Detection
  ↓
Event Timestamp
  ↓
Prototype Severity Scoring
  ↓
Visualization / Dashboard
Future RADS

The long-term system may include:

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
Event localization
+
Accident type classification
+
Learned severity estimation
+
Multimodal reasoning
+
Real-time optimization
+
Edge deployment
+
Production monitoring

The future architecture must not dictate unnecessary MVP complexity.
---
# 20. Development Priority

Development should proceed in dependency order.

The current priority is:

1. Detection
        ↓
2. Tracking / Persistent IDs
        ↓
3. Track Data Representation
        ↓
4. Motion Feature Extraction
        ↓
5. Object Interaction Analysis
        ↓
6. Temporal Event Detection
        ↓
7. Accident Classification
        ↓
8. Event Localization
        ↓
9. Severity Estimation
        ↓
10. Visualization / Integration
        ↓
11. Evaluation

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
# 25. Non-Goals for the Current MVP

The following are explicitly not required for the first MVP:

Production-grade deployment
Perfect real-time performance
Perfect accident classification
Large-scale distributed training
Fully learned severity model
Perfect frame-level localization
Full autonomous emergency response
Cloud-scale infrastructure
Universal performance across every camera/environment

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

As of September 2026, RADS is transitioning from:

Raw video
    ↓
ResNet18
    ↓
GRU
    ↓
Video classification

toward:

Raw video
    ↓
YOLO
    ↓
Persistent object tracking
    ↓
Object trajectories
    ↓
Motion + interaction reasoning
    ↓
Temporal event detection
    ↓
Accident detection
    ↓
Severity estimation

The ResNet18 + GRU system remains the experimental baseline.

The new architecture is the primary development direction.

The purpose of this transition is not simply to replace one neural network with another.

The purpose is to change RADS from a visual classification system into an event understanding system.
---
# 28. Guiding Principle

The ultimate goal of RADS is not:

"Find frames that look like accidents."

It is:

"Understand what happened on the road, identify whether an accident occurred, determine when it occurred, identify the objects involved, and estimate its severity from observable evidence."

The MVP should be the smallest credible implementation of that idea.