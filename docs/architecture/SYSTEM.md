# RADS — System Architecture Specification

**Project:** RADS
**Document:** System Architecture Specification
**Version:** 1.0
**Status:** Active
**Date:** 2026-09-03

---

# 1. Purpose

This document defines the technical architecture of RADS.

It describes:

* How video enters the system
* How frames are processed
* How objects are detected
* How persistent IDs are assigned
* How tracks are constructed
* How motion is derived
* How object interactions are identified
* How accident events are reasoned about
* How severity is estimated
* How results are visualized and delivered

The architecture should remain modular so that individual components can be replaced or improved without redesigning the entire system.

---

# 2. Core Architecture

The primary RADS architecture is:

```text
                         INPUT VIDEO
                              │
                              ↓
                    ┌──────────────────┐
                    │ VIDEO PROCESSING │
                    └────────┬─────────┘
                             │
                             ↓
                    ┌──────────────────┐
                    │ OBJECT DETECTION │
                    │      YOLO        │
                    └────────┬─────────┘
                             │
                             ↓
                    ┌──────────────────┐
                    │ OBJECT TRACKING  │
                    │  ID ASSIGNMENT   │
                    └────────┬─────────┘
                             │
                             ↓
                    ┌──────────────────┐
                    │ TRACK HISTORIES  │
                    │   TRAJECTORIES   │
                    └────────┬─────────┘
                             │
                             ↓
                    ┌──────────────────┐
                    │ MOTION FEATURES  │
                    └────────┬─────────┘
                             │
                             ↓
                    ┌──────────────────┐
                    │   INTERACTION    │
                    │     ANALYSIS     │
                    └────────┬─────────┘
                             │
                             ↓
                    ┌──────────────────┐
                    │ TEMPORAL EVENT   │
                    │    REASONING     │
                    └────────┬─────────┘
                             │
                             ↓
                  ┌──────────────────────┐
                  │ ACCIDENT DETECTION   │
                  │ + EVENT LOCALIZATION │
                  └──────────┬───────────┘
                             │
                             ↓
                    ┌──────────────────┐
                    │    SEVERITY      │
                    │    ESTIMATION    │
                    └────────┬─────────┘
                             │
                             ↓
                    ┌──────────────────┐
                    │ STRUCTURED EVENT │
                    │      OUTPUT      │
                    └───────┬──────────┘
                            │
                  ┌─────────┴─────────┐
                  ↓                   ↓
             DASHBOARD             ALERT
```

---

# 3. Architectural Principle

The system should be treated as a sequence of transformations.

```text
Video
  ↓
Frames
  ↓
Detections
  ↓
Tracks
  ↓
Trajectories
  ↓
Motion / Interactions
  ↓
Events
  ↓
Accident
  ↓
Severity
  ↓
Structured Result
```

Each stage should have a clear input and output.

A downstream component should not need to understand the internal implementation of the component before it.

---

# 4. Stage 1 — Video Input

The pipeline begins with a video.

Input may come from:

* Dataset video
* Uploaded video
* Camera stream
* Recorded traffic footage

For the MVP, file-based video input is sufficient.

The video-processing layer should extract or expose:

```text
Frame
Frame Index
Timestamp
FPS
Width
Height
Video Duration
```

Example:

```text
Frame 120
Timestamp: 4.00 s
FPS: 30
Resolution: 1280 × 720
```

---

# 5. Stage 2 — Frame Processing

Frames are passed sequentially or in controlled batches through the perception pipeline.

Conceptually:

```text
Video
  ↓
Frame t
  ↓
Detection
  ↓
Tracking
  ↓
Frame t+1
  ↓
Detection
  ↓
Tracking
```

The pipeline should preserve frame ordering.

Temporal information must not be lost between stages.

---

# 6. Stage 3 — Object Detection

The detector receives a frame and produces detections.

```text
Frame
  ↓
YOLO
  ↓
Detections
```

A detection should contain at least:

```text
class
confidence
bounding_box
frame_index
timestamp
```

Example:

```text
Frame 120

Detection 1:
Class: car
Confidence: 0.94
BBox: [x1, y1, x2, y2]

Detection 2:
Class: motorcycle
Confidence: 0.89
BBox: [x1, y1, x2, y2]
```

At this stage, the system does not yet know whether two detections belong to the same physical object across frames.

---

# 7. Stage 4 — Object Tracking and ID Assignment

The tracking layer receives detections from consecutive frames.

Its primary responsibility is:

> Determine which detection belongs to which existing object.

Conceptually:

```text
Frame t detections
       ↓
     Tracker
       ↓
Frame t+1 detections
       ↓
    Association
       ↓
 Persistent IDs
```

Example:

```text
Frame 100:

Car → ID 3
Car → ID 7
Motorcycle → ID 12


Frame 101:

Car → ID 3
Car → ID 7
Motorcycle → ID 12
```

The tracker is responsible for maintaining identity continuity.

---

# 8. Tracking State

Each active object should have an internal track state.

Conceptually:

```text
Track {
    object_id
    class
    bbox
    confidence
    timestamp
    frame_index
    center
    history
}
```

The track state should be updated whenever a new detection is associated with the object.

Example:

```text
ID 3

Frame 100 → (421, 310)
Frame 101 → (429, 311)
Frame 102 → (438, 313)
Frame 103 → (447, 315)
```

This history becomes the basis for trajectory and motion analysis.

---

# 9. Track Lifecycle

Objects may:

```text
ENTER FRAME
     ↓
BECOME TRACKED
     ↓
REMAIN ACTIVE
     ↓
TEMPORARILY DISAPPEAR
     ↓
REAPPEAR
     ↓
CONTINUE TRACK
```

or:

```text
ENTER
 ↓
TRACK
 ↓
LEAVE FRAME
 ↓
TRACK TERMINATED
```

The tracker should handle track creation and termination according to its configured behavior.

Short-term occlusion should not immediately result in unnecessary new IDs where the tracker can reliably maintain identity.

---

# 10. Stage 5 — Trajectory Construction

The tracking layer provides persistent identities.

The trajectory system converts the sequence of positions into an object trajectory.

Example:

```text
Object #3

t0 → (100, 400)
t1 → (110, 398)
t2 → (121, 396)
t3 → (134, 393)
t4 → (150, 390)
```

This can be represented conceptually as:

```text
Trajectory_3 =
[
    (t0, x0, y0),
    (t1, x1, y1),
    ...
]
```

The system should retain enough temporal information to calculate motion-related features.

---

# 11. Coordinate Representation

At minimum, trajectories may initially use image-space coordinates.

Useful representations include:

```text
Center X
Center Y
Bounding Box
Width
Height
```

Image-space velocity can then be approximated using changes in center position over time.

Actual physical speed should not be assumed unless the system has sufficient information for camera calibration or another valid scale estimation method.

The MVP should clearly distinguish:

```text
Pixel / image-space motion
```

from:

```text
Real-world physical speed
```

unless physical calibration is implemented.

---

# 12. Stage 6 — Motion Feature Extraction

Motion features should be derived from track histories.

Potential features include:

```text
Position
Displacement
Direction
Velocity
Acceleration
Direction change
Trajectory curvature
Motion consistency
```

For object A:

```text
Position(t)
      ↓
Position(t+1)
      ↓
Displacement
      ↓
Velocity
      ↓
Velocity change
      ↓
Acceleration
```

The implementation should account for video FPS when calculating temporal differences.

---

# 13. Stage 7 — Pairwise Object Relationships

RADS should construct relationships between relevant tracked objects.

For objects A and B:

```text
Distance(A,B)
Relative Position(A,B)
Relative Velocity(A,B)
Trajectory Relationship
Bounding-Box Relationship
```

Example:

```text
Vehicle A
      ↓
      ↓
      ↓

          Vehicle B
```

As the distance changes:

```text
d(t)
d(t+1)
d(t+2)
...
```

the system can identify whether objects are:

```text
Approaching
Separating
Parallel
Crossing
Converging
```

These relationships are inputs to accident reasoning.

---

# 14. Stage 8 — Interaction Detection

The interaction layer determines whether tracked objects are behaving in a way that suggests meaningful interaction.

Potential signals:

```text
Rapid distance reduction
Trajectory convergence
Bounding-box overlap
Sudden relative motion change
Abrupt direction changes
Simultaneous motion changes
Post-interaction displacement
```

Conceptually:

```text
Object A ───────→
                  \
                   \
                    X
                   /
                  /
Object B ───────→
```

This should generate an interaction candidate.

An interaction candidate is **not automatically an accident**.

---

# 15. Interaction Candidate

A possible event should be represented independently from the final accident decision.

Example:

```text
Interaction Candidate

Object A: ID 3
Object B: ID 7

Start:
6.8 s

Peak Interaction:
7.4 s

End:
8.2 s

Evidence:
- Rapid distance decrease
- Trajectory convergence
- Motion change
```

The event-reasoning system then evaluates this candidate.

---

# 16. Stage 9 — Temporal Event Window

Accident reasoning should operate on a temporal window.

Example:

```text
         EVENT WINDOW

   BEFORE      IMPACT       AFTER
     │           │            │
     ↓           ↓            ↓
   Motion → Interaction → Post-impact
```

The exact window length should be configurable.

The system should preserve enough context to understand:

```text
What happened before?
What happened during?
What happened immediately after?
```

---

# 17. Stage 10 — Accident Reasoning

The accident reasoning layer consumes:

```text
Track Data
Trajectory Data
Motion Features
Pairwise Relationships
Interaction Candidates
Temporal Context
```

Conceptually:

```text
Tracks
  +
Motion
  +
Interactions
  +
Temporal Window
       ↓
Accident Reasoning
       ↓
Accident Probability / Decision
```

The reasoning system may initially use engineered rules/features and later incorporate a learned temporal model.

The architecture should support both.

---

# 18. Stage 11 — Accident Event Representation

A detected accident should become a structured event.

Example:

```json
{
  "event_type": "accident",
  "confidence": 0.91,
  "start_time": 6.8,
  "impact_time": 7.4,
  "end_time": 8.2,
  "objects_involved": [3, 7]
}
```

Additional fields may be added later.

This event object becomes the common interface between AI reasoning and downstream systems.

---

# 19. Stage 12 — Accident Type

If sufficient evidence exists, the event may be classified as:

```text
Rear-End
Head-On
T-Bone
Sideswipe
Single-Vehicle
Other / Unknown
```

The accident-type classifier should consume event-level information rather than raw frame classification alone where practical.

Example:

```text
Event
  ↓
Object trajectories
  +
Relative motion
  +
Interaction geometry
  ↓
Accident Type
```

---

# 20. Stage 13 — Severity Estimation

Severity estimation receives the detected event and its characteristics.

```text
Accident Event
      ↓
Event Features
      ↓
Severity Engine
      ↓
Low / Medium / High
```

Potential inputs:

```text
Number of objects
Object classes
Relative motion
Motion change
Post-impact displacement
Pedestrian involvement
Secondary interactions
Trajectory disruption
```

For the MVP, the severity engine may be deterministic or rule-based.

The interface should allow a future learned model to replace it.

---

# 21. Stage 14 — Final Structured Result

The entire pipeline should eventually produce one structured event/result.

Example:

```json
{
  "accident": true,
  "confidence": 0.91,
  "event": {
    "start_time": 6.8,
    "impact_time": 7.4,
    "end_time": 8.2
  },
  "objects_involved": [
    {
      "id": 3,
      "class": "car"
    },
    {
      "id": 7,
      "class": "car"
    }
  ],
  "accident_type": "rear_end",
  "severity": "high"
}
```

The schema is illustrative and may evolve.

---

# 22. Visualization Pipeline

The system should be capable of rendering the reasoning output back onto the original video.

Conceptually:

```text
Original Frame
      +
Detections
      +
IDs
      +
Trajectories
      +
Event Marker
      +
Severity
      ↓
Annotated Video
```

Example overlay:

```text
CAR #3
──────────────→

              CAR #7
                 ↓
                 X

ACCIDENT DETECTED
TIME: 00:07.4
SEVERITY: HIGH
```

This is useful for:

* Debugging
* Research analysis
* Demonstrations
* Presentations
* Human verification

---

# 23. Dashboard Integration

The existing dashboard should consume structured RADS events.

It should not contain the core accident reasoning logic.

Architecture:

```text
AI Pipeline
     ↓
Structured Event
     ↓
Dashboard
```

This allows the AI system to be tested independently of the interface.

---

# 24. Telegram Alert Integration

The existing Telegram integration should also consume the structured event.

Conceptually:

```text
Accident Event
      ↓
Alert Formatter
      ↓
Telegram
```

Example alert:

```text
RADS ALERT

Accident detected.

Time: 00:07.4
Type: Rear-End
Severity: High
Objects involved: 2
Confidence: 91%
```

The exact alert format may change.

---

# 25. Separation of Concerns

The following responsibilities should remain separate:

```text
Video Processing
       │
       ↓
Detection
       │
       ↓
Tracking
       │
       ↓
Trajectory Processing
       │
       ↓
Interaction Analysis
       │
       ↓
Accident Reasoning
       │
       ↓
Severity
       │
       ↓
Output
```

No single component should contain the entire pipeline.

This is especially important for debugging.

If accident detection performs poorly, the system should allow investigation of:

```text
Detector
Tracker
Trajectory
Motion Features
Interaction Logic
Temporal Model
Severity
```

independently.

---

# 26. Module Boundaries

The implementation should conceptually expose modules similar to:

```text
video/
    video_reader
    frame_processor

detection/
    detector

tracking/
    tracker
    track_manager

motion/
    trajectory
    motion_features

interaction/
    interaction_engine

reasoning/
    event_detector
    accident_classifier

severity/
    severity_engine

output/
    event_schema
    visualizer
    dashboard_adapter
    telegram_adapter
```

The exact repository structure may differ.

The important requirement is functional separation.

---

# 27. Data Flow Contract

Each stage should produce a defined output.

```text
Video Reader
    ↓
Frame

Detector
    ↓
Detections

Tracker
    ↓
Tracks

Track Manager
    ↓
Track Histories

Motion Engine
    ↓
Motion Features

Interaction Engine
    ↓
Interaction Candidates

Event Reasoner
    ↓
Accident Event

Severity Engine
    ↓
Severity

Output Layer
    ↓
Structured Result
```

Downstream modules should consume these outputs rather than accessing hidden internal state.

---

# 28. Offline and Real-Time Modes

The architecture should support two conceptual modes.

## Offline Mode

```text
Complete Video
      ↓
Process
      ↓
Analyze
      ↓
Generate Result
```

This should be the initial MVP priority.

## Streaming / Real-Time Mode

```text
Incoming Frames
      ↓
Detection
      ↓
Tracking
      ↓
Rolling Temporal Window
      ↓
Event Reasoning
      ↓
Alert
```

Real-time optimization is a future concern unless required for the MVP demonstration.

---

# 29. Failure Isolation

Each stage should be testable independently.

Examples:

### Detection test

```text
Video → Detector → Detection Quality
```

### Tracking test

```text
Known Detections → Tracker → ID Stability
```

### Motion test

```text
Tracks → Motion Engine → Feature Correctness
```

### Event test

```text
Synthetic / Annotated Tracks → Event Reasoner
```

### Severity test

```text
Known Events → Severity Engine
```

This makes debugging significantly easier.

---

# 30. Logging and Debugging

The pipeline should provide enough logging to determine where a failure occurred.

Useful information:

```text
Video ID
Frame count
Detection count
Active track count
Track creation
Track termination
Interaction candidates
Event candidates
Final accident decision
Severity decision
Processing time
```

Debug logs should be configurable.

---

# 31. Configuration

The following should be configurable:

```text
Detector model
Detection confidence
Detection IoU
Tracker parameters
Frame sampling rate
Temporal window
Minimum track length
Interaction thresholds
Event thresholds
Severity thresholds
Visualization settings
```

Values should not be scattered throughout source code.

---

# 32. MVP Architecture

The MVP should implement the smallest coherent version of the architecture:

```text
Video
  ↓
YOLO
  ↓
Tracker
  ↓
Persistent IDs
  ↓
Track History
  ↓
Trajectory / Motion Features
  ↓
Interaction Logic
  ↓
Accident Event Logic
  ↓
Event Timestamp
  ↓
Objects Involved
  ↓
Severity Scoring
  ↓
Visualization
  ↓
Structured Output
```

The MVP should demonstrate the complete flow even if individual components remain imperfect.

---

# 33. Future Architecture

The architecture should be capable of evolving toward:

```text
Video
  ↓
Advanced Detection
  ↓
Advanced Tracking
  ↓
Trajectory Encoder
  ↓
Object Interaction Graph
  ↓
Temporal Representation
  ↓
Event Reasoning Model
  ↓
Accident Classification
  ↓
Event Localization
  ↓
Severity Model
  ↓
Explainable Output
```

Possible future additions include:

* Appearance embeddings
* Graph neural networks
* Temporal Transformers
* Learned trajectory encoders
* Multimodal reasoning
* Camera calibration
* Real-world velocity estimation
* Advanced severity prediction

These are not MVP requirements.

---

# 34. Architecture Constraints

The following constraints apply:

### Constraint 1

Tracking must occur before trajectory-based reasoning.

### Constraint 2

Persistent IDs must be maintained before object histories are constructed.

### Constraint 3

Accident reasoning must have access to temporal information.

### Constraint 4

Severity estimation must operate on an identified event rather than independently declaring an accident.

### Constraint 5

Dashboard and alert systems must consume structured outputs rather than implement AI logic themselves.

### Constraint 6

Detection and tracking components must remain replaceable.

### Constraint 7

Real-world physical quantities must not be claimed unless the system has sufficient calibration or measurement support.

### Constraint 8

The system should remain capable of processing an expanded dataset without fundamental architectural redesign.

---

# 35. Architectural Success Criteria

The architecture will be considered successful if:

1. A video can pass through the entire pipeline.
2. Objects can be detected.
3. Objects can receive persistent IDs.
4. Tracks can be constructed.
5. Motion features can be derived.
6. Object interactions can be identified.
7. Accident events can be detected.
8. Events can be localized in time.
9. Involved objects can be identified.
10. Severity can be estimated.
11. Results can be visualized.
12. Results can be consumed by the dashboard/alert system.
13. Individual components can be tested independently.
14. Components can be replaced without rewriting the entire system.

---

# 36. Final Architecture Principle

The RADS architecture should follow one central rule:

> **First understand the objects, then understand their behavior, then understand their interactions, then determine whether an accident occurred.**

The intended transformation is:

```text
PIXELS
  ↓
OBJECTS
  ↓
IDENTITIES
  ↓
TRAJECTORIES
  ↓
MOTION
  ↓
INTERACTIONS
  ↓
EVENTS
  ↓
ACCIDENT
  ↓
SEVERITY
  ↓
ACTIONABLE OUTPUT
```

RADS is therefore not intended to be merely a video classifier.

It is intended to become a **temporal, object-centric accident understanding system**.
