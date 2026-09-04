# RADS — Technology Stack Specification

**Project:** RADS
**Document:** Technology Stack Specification
**Version:** 1.0
**Status:** Active
**Date:** 2026-09-03

---

# 1. Purpose

This document defines the technology stack and implementation technologies intended for RADS.

The purpose is to establish:

* Which technologies should be used initially
* Which technologies are preferred but replaceable
* Which components are still open for experimentation
* What each technology is responsible for
* Which technologies are appropriate for the MVP
* Which technologies should be avoided unless justified

This document defines the **technical implementation direction**, while the AI reasoning principles are defined in:

```text
02_AI_BRAIN.md
```

The complete system architecture is defined in:

```text
04_SYSTEM_ARCHITECTURE.md
```

---

# 2. Technology Philosophy

RADS should use technologies that allow rapid experimentation while keeping the architecture modular.

The MVP should prioritize:

```text
Reliability
+
Speed of development
+
Interpretability
+
Modularity
+
Reproducibility
```

over unnecessary complexity.

The project should prefer established, well-supported tools over custom implementations unless there is a clear research reason to build something from scratch.

---

# 3. Core Technology Stack

The initial RADS stack is conceptually:

```text
Python
   ↓
OpenCV
   ↓
YOLO-based Object Detection
   ↓
Multi-Object Tracking
   ↓
Trajectory / Motion Processing
   ↓
Temporal Accident Reasoning
   ↓
Severity Engine
   ↓
Output / Dashboard / Alerts
```

The exact versions of individual packages should be pinned in the project environment once implementation begins.

---

# 4. Programming Language

## Primary Language

**Python**

Python should be the primary language for:

* Model inference
* Model training
* Dataset processing
* Video processing
* Tracking
* Feature extraction
* Accident reasoning
* Evaluation
* Experiment scripts
* Backend integration

Python is preferred because the project is primarily an ML/computer-vision system and requires rapid experimentation.

---

# 5. Computer Vision

## OpenCV

OpenCV should be used for general video and image processing.

Potential responsibilities:

```text
Video loading
Frame extraction
Frame resizing
Frame writing
FPS handling
Timestamp handling
Image transformations
Visualization
Bounding-box rendering
Trajectory visualization
Video annotation
```

OpenCV should handle general video-processing tasks rather than being used as the primary accident-detection intelligence.

---

# 6. Object Detection

## YOLO

YOLO is the preferred initial object-detection family for RADS.

The detector should be used to identify relevant road users in individual frames.

Conceptually:

```text
Frame
  ↓
YOLO
  ↓
Objects
```

Each detection should provide at minimum:

```text
Class
Confidence
Bounding Box
Frame
Timestamp
```

The exact YOLO model/version should be selected during implementation based on:

* Detection accuracy
* Inference speed
* Available pretrained weights
* Hardware limitations
* Supported object classes
* Ease of integration with tracking

The project should prefer a pretrained detector for the MVP.

Training a detector from scratch is not an MVP requirement.

---

# 7. Object Classes

The detector should support the road-user classes required for accident reasoning.

Potential classes include:

```text
Car
Motorcycle
Truck
Bus
Van
Bicycle
Pedestrian
```

The exact class set depends on the selected pretrained model and the requirements of the dataset.

Additional classes should only be introduced when they provide useful information for RADS.

---

# 8. Multi-Object Tracking

Tracking is a core component of RADS.

The tracker should associate detections across consecutive frames and assign persistent IDs.

Conceptually:

```text
YOLO Detection
      ↓
Tracker
      ↓
Object ID
      ↓
Track History
```

The initial tracker should preferably be a modern, lightweight multi-object tracker.

**ByteTrack** is a primary candidate.

Other trackers may be evaluated if required by:

* Occlusion behavior
* ID stability
* Camera conditions
* Detection quality
* Computational cost

The tracker must remain replaceable.

---

# 9. Persistent Object IDs

Every tracked object should have an identifier.

Example:

```text
Vehicle → ID 3
Vehicle → ID 7
Motorcycle → ID 12
```

The ID should be propagated through subsequent frames whenever the tracker determines that the same physical object is present.

The tracking output should form the basis for all downstream temporal reasoning.

---

# 10. Track Data Representation

A track should contain information similar to:

```text
object_id
class
frame_index
timestamp
bounding_box
center_x
center_y
width
height
detection_confidence
```

Derived information may include:

```text
velocity
acceleration
direction
heading
displacement
trajectory
relative_velocity
relative_distance
```

The representation should remain simple and structured.

A standard tabular or serialized representation may be used initially.

---

# 11. Trajectory Processing

Trajectory processing should operate on the persistent tracks produced by the tracking layer.

Example:

```text
Object #3

t0 → (x0, y0)
t1 → (x1, y1)
t2 → (x2, y2)
t3 → (x3, y3)
...
```

The trajectory system should support calculation or estimation of:

```text
Position
Displacement
Direction
Velocity
Acceleration
Direction change
Trajectory curvature
```

These features may later become inputs to the temporal accident reasoning system.

---

# 12. Temporal Reasoning Technology

The temporal reasoning component should remain modular.

Potential implementations include:

```text
Rule-based logic
Machine-learning classifier
GRU
Temporal CNN
Transformer
Hybrid rule + ML system
```

The MVP should begin with the **simplest architecture capable of demonstrating meaningful temporal reasoning**.

The project should not automatically use a large Transformer merely because Transformers exist.

Model complexity should be justified by:

```text
Dataset size
Performance
Latency
Interpretability
Training requirements
```

---

# 13. Accident Reasoning

The accident reasoning layer will consume information generated by:

```text
Detection
Tracking
Trajectory Processing
Motion Features
Object Interactions
```

Conceptually:

```text
Tracks
   +
Motion
   +
Interactions
   +
Temporal Context
        ↓
Accident Reasoning
```

The reasoning layer may combine engineered features with a learned temporal model.

The exact implementation should be determined experimentally.

---

# 14. Severity Technology

Severity estimation should initially prioritize interpretability.

For the MVP, a **rule-based or scoring-based severity engine** is acceptable.

Potential inputs:

```text
Relative velocity
Motion change
Number of objects
Object classes
Pedestrian involvement
Post-impact displacement
Secondary collisions
Trajectory disruption
```

Example conceptual flow:

```text
Event Features
      ↓
Severity Score
      ↓
Low / Medium / High
```

A learned severity model may be introduced later when sufficient labeled severity data exists.

The MVP does not require a sophisticated neural severity model.

---

# 15. Video Processing

The video pipeline should support:

```text
Video input
      ↓
Frame decoding
      ↓
Frame processing
      ↓
Detection
      ↓
Tracking
      ↓
Event analysis
      ↓
Output video / results
```

The system should preserve:

* Original FPS where practical
* Frame numbering
* Timestamps
* Video duration
* Input metadata

Temporal alignment is important because event localization depends on accurate timing.

---

# 16. Visualization

The MVP should support visualizing the system's reasoning.

Useful overlays include:

```text
Object bounding boxes
Object IDs
Class labels
Detection confidence
Trajectories
Accident event marker
Objects involved
Severity
```

Example:

```text
┌────────────────────────────────────┐
│                                    │
│   CAR #3 ───────────────→          │
│                    ↓               │
│               CAR #7               │
│                    ✕               │
│                                    │
│      ACCIDENT DETECTED             │
│      Time: 00:07.4                 │
│      Severity: HIGH                │
│                                    │
└────────────────────────────────────┘
```

Visualization is important for both debugging and demonstration.

---

# 17. Dashboard and Alert Integration

RADS already has a dashboard and Telegram alert integration from an earlier project phase.

These components should remain separate from the core computer-vision reasoning pipeline.

The conceptual architecture is:

```text
RADS AI Pipeline
       ↓
Structured Event
       ↓
┌──────┴──────┐
↓             ↓
Dashboard   Telegram
```

The AI system should produce a structured event rather than directly coupling model logic to a specific user interface.

Example structured result:

```json
{
  "accident": true,
  "timestamp": 7.4,
  "severity": "high",
  "objects_involved": [3, 7],
  "type": "rear_end",
  "confidence": 0.91
}
```

The exact schema may evolve.

---

# 18. Experiment Tracking

Experiments should be reproducible.

The project may use an experiment-tracking system such as:

```text
Weights & Biases
```

for recording:

* Configuration
* Hyperparameters
* Training metrics
* Validation metrics
* Test metrics
* Model checkpoints
* Experiment metadata

Existing RADS experiments have already used W&B.

Future experiments should maintain clear provenance between:

```text
Experiment
→ Configuration
→ Checkpoint
→ Evaluation
→ Result
```

---

# 19. Configuration Management

Model and pipeline parameters should be configurable rather than hardcoded.

Examples:

```text
Detector model
Detection confidence
IoU threshold
Tracker parameters
Input resolution
FPS sampling
Temporal window
Model hyperparameters
Severity thresholds
```

Configuration files should be preferred over modifying source code for every experiment.

---

# 20. Model Checkpoints

Every trained model should have clear provenance.

The project should be able to determine:

```text
Which experiment produced this checkpoint?
Which configuration was used?
Which dataset version was used?
Which epoch was selected?
Which metric selected the checkpoint?
```

A checkpoint should never be treated as a valid experimental result if its provenance cannot be established.

---

# 21. Hardware Awareness

The system should be designed to operate within available hardware constraints.

The implementation should avoid unnecessary memory and compute requirements.

Where possible:

```text
Batch processing
Frame sampling
Model-size selection
Inference optimization
Caching
```

should be configurable.

A model that cannot run reliably on the available hardware is not a useful MVP, regardless of how impressive its architecture diagram looks.

---

# 22. Technology Selection Rules

When choosing between technologies, prioritize:

### 1. Proven reliability

Prefer mature tools where possible.

### 2. Integration

The technology should integrate cleanly with the rest of RADS.

### 3. Performance

The system should be fast enough for the intended MVP demonstration.

### 4. Reproducibility

Experiments must be repeatable.

### 5. Replaceability

Components should not create unnecessary architectural lock-in.

### 6. Simplicity

Do not introduce a technology simply because it is fashionable.

---

# 23. Technologies That Are Not Automatically Required

The following should not be introduced unless there is a clear reason:

```text
Large multimodal LLM
Complex agent framework
Distributed training
Kubernetes
Microservice architecture
Large database infrastructure
Cloud inference
Custom detector training
Large Transformer
Complex graph neural network
```

These may become useful later.

They are not requirements for the MVP.

---

# 24. MVP Technology Stack

The initial MVP should aim for something close to:

```text
Language:
Python

Video:
OpenCV

Detection:
Pretrained YOLO

Tracking:
ByteTrack or equivalent

Data Processing:
NumPy / Pandas as appropriate

Temporal Reasoning:
Lightweight temporal model and/or engineered reasoning

Severity:
Interpretable scoring/rule engine initially

Experiment Tracking:
Weights & Biases

Visualization:
OpenCV / existing dashboard

Alerts:
Existing Telegram integration

Configuration:
Existing project configuration system
```

The exact package versions and model variants should be finalized during implementation.

---

# 25. Technology Abstraction

The architecture should avoid tightly coupling the entire system to one model.

For example, downstream code should conceptually consume:

```text
DetectionResult
```

rather than directly depending on YOLO-specific internals.

Similarly, the reasoning system should consume:

```text
Track
Trajectory
Interaction
Event
```

rather than directly depending on the tracking library.

This allows components to be replaced without rewriting the entire project.

---

# 26. Replaceability Requirements

The following components should be replaceable:

```text
Detector
Tracker
Temporal Model
Severity Engine
Visualization Layer
```

For example:

```text
YOLO
  ↓
Tracker A
```

should eventually be replaceable with:

```text
Detector B
  ↓
Tracker B
```

without redesigning the entire accident reasoning system.

---

# 27. Recommended Initial Implementation Order

The technology should be implemented in the following order:

```text
1. Video Input / Frame Pipeline
             ↓
2. YOLO Detection
             ↓
3. Multi-Object Tracking
             ↓
4. Persistent Object IDs
             ↓
5. Track History Storage
             ↓
6. Trajectory / Motion Features
             ↓
7. Object Interaction Features
             ↓
8. Accident Event Logic
             ↓
9. Temporal Accident Model
             ↓
10. Event Localization
             ↓
11. Severity Engine
             ↓
12. Visualization
             ↓
13. Dashboard / Alert Integration
```

This order is intentional.

Tracking and persistent identity should be established before attempting sophisticated accident reasoning.

---

# 28. Technology Decision Status

The following decisions are currently established:

| Component           | Current Direction           | Status      |
| ------------------- | --------------------------- | ----------- |
| Programming         | Python                      | Established |
| Video Processing    | OpenCV                      | Established |
| Detection           | YOLO family                 | Preferred   |
| Tracking            | ByteTrack / equivalent      | Candidate   |
| Persistent IDs      | Tracker-generated IDs       | Required    |
| Trajectories        | Track-based                 | Required    |
| Motion Features     | Engineered initially        | Preferred   |
| Temporal Reasoning  | Lightweight / experimental  | Open        |
| Severity            | Rule/scoring initially      | Preferred   |
| Experiment Tracking | W&B                         | Existing    |
| Visualization       | OpenCV + existing dashboard | Existing    |
| Alerts              | Telegram integration        | Existing    |

---

# 29. Technology Decisions That Must Remain Open

The following should not be permanently locked before experimentation:

```text
Exact YOLO version
Exact YOLO model size
Exact tracker
Exact temporal model
Exact feature set
Exact severity formulation
Exact inference optimization
```

These should be selected based on actual experiments and constraints.

---

# 30. Final Technology Principle

RADS should follow this rule:

> **Use the simplest technology that allows the system to demonstrate the intended intelligence reliably.**

The objective is not to assemble the largest possible collection of AI technologies.

The objective is to create a coherent pipeline in which:

```text
Detection
    ↓
Tracking
    ↓
Motion
    ↓
Interaction
    ↓
Temporal Reasoning
    ↓
Accident Detection
    ↓
Severity
```

works as one system.

The technology stack exists to support that intelligence, not to become the intelligence itself.
