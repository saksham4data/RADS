# RADS — AI Brain Specification

**Project:** RADS
**Document:** AI Brain Specification
**Version:** 1.0
**Status:** Active
**Date:** 2026-09-03

---

# 1. Purpose

This document defines the intended intelligence and reasoning philosophy of RADS.

It describes **how RADS should reason about accidents**, rather than prescribing one fixed implementation.

The system should progressively transform raw visual information into an understanding of:

```text
Objects
    ↓
Identity
    ↓
Motion
    ↓
Interaction
    ↓
Event
    ↓
Accident
    ↓
Severity
```

The central principle is:

> RADS should reason about what is happening in the scene, not merely what the scene looks like.

---

# 2. Core Intelligence Model

RADS should operate as a hierarchical reasoning system.

```text
                    VIDEO
                      │
                      ↓
                PERCEPTION
                      │
                "What exists?"
                      │
                      ↓
                  IDENTITY
                      │
              "Which object is which?"
                      │
                      ↓
                   MOTION
                      │
              "What is it doing?"
                      │
                      ↓
                INTERACTION
                      │
            "What is happening between
                 the objects?"
                      │
                      ↓
                    EVENT
                      │
             "Did something unusual
                  happen?"
                      │
                      ↓
                  ACCIDENT
                      │
            "Does this constitute an
                   accident?"
                      │
                      ↓
                  SEVERITY
                      │
             "How serious is it?"
```

Each level should build upon information produced by the previous level.

---

# 3. Perception

The first stage is scene perception.

RADS should identify relevant road users and other objects that may participate in an event.

Potential object categories include:

* Cars
* Motorcycles
* Trucks
* Buses
* Vans
* Pedestrians
* Bicycles
* Other relevant road users

The purpose of perception is not to determine whether an accident occurred.

Its purpose is to establish:

> **What objects are present in the scene?**

Example:

```text
Frame 100

Car
Motorcycle
Car
Pedestrian
```

The perception layer should remain independent from accident classification where practical.

---

# 4. Identity

After detecting objects, RADS must determine which detections correspond to the same physical object across frames.

Example:

```text
Frame 1:
Car → ID 3

Frame 2:
Car → ID 3

Frame 3:
Car → ID 3

Frame 4:
Car → ID 3
```

The system should build persistent object histories.

This allows RADS to reason about:

```text
Where was object 3?
Where is object 3 now?
How fast is it moving?
Did its direction change?
Did it interact with another object?
```

Identity is therefore a prerequisite for meaningful temporal reasoning.

---

# 5. Motion Understanding

Once object identities are established, RADS should reason about movement.

For each tracked object, the system should be able to derive or estimate information such as:

```text
Position
Direction
Velocity
Acceleration
Displacement
Trajectory
Heading
```

The objective is to understand:

> **How is each object behaving over time?**

Example:

```text
Normal:

Vehicle #3
→
→
→
→
→


Potential abnormal event:

Vehicle #3
→
→
↘
↓
STOP
```

A sudden change in motion can provide evidence of an event, but should not automatically be interpreted as an accident.

---

# 6. Relative Motion

Absolute motion alone is insufficient.

RADS should also reason about the relationship between objects.

For two objects A and B:

```text
Relative Position
Relative Velocity
Relative Acceleration
Distance
Distance Change
Trajectory Relationship
```

Example:

```text
Vehicle A →→→

             ← Vehicle B

Distance rapidly decreasing
          ↓
Potential interaction
```

The system should distinguish between:

```text
Objects moving close together
```

and:

```text
Objects exhibiting collision-like interaction
```

This distinction is critical.

---

# 7. Interaction Understanding

RADS should treat object interactions as a central source of accident evidence.

Potential interaction signals include:

* Rapidly decreasing distance
* Trajectory convergence
* Bounding-box overlap
* Sudden relative velocity change
* Abrupt direction changes
* Sudden braking
* Simultaneous motion changes
* Post-interaction displacement

Conceptually:

```text
Object A
     ↓
approaches
     ↓
Object B
     ↓
distance decreases
     ↓
trajectories converge
     ↓
motion changes abruptly
     ↓
possible collision
```

However:

> Proximity does not equal collision.

Traffic naturally contains close-following vehicles, overtaking, merging, and other interactions that are not accidents.

---

# 8. Temporal Reasoning

RADS should reason across a temporal window.

An accident should generally be understood as a sequence:

```text
BEFORE
  ↓
Normal motion
  ↓
APPROACH
  ↓
Objects move toward interaction
  ↓
INTERACTION
  ↓
Potential collision
  ↓
IMPACT / ABRUPT CHANGE
  ↓
POST-IMPACT
  ↓
Motion changes / displacement / stopping
```

The model should therefore have access to information before and after a suspected event whenever possible.

This is preferable to making an accident decision from a single frame.

---

# 9. Event-Centric Thinking

RADS should ultimately reason in terms of **events**.

An event is a meaningful change in the state or behavior of objects in the scene.

Examples:

```text
Vehicle suddenly changes direction
Vehicle abruptly stops
Two vehicles converge rapidly
Vehicle and pedestrian interact
Vehicle trajectory becomes abnormal
Multiple objects experience simultaneous motion changes
```

The system must then determine whether the event represents:

```text
Normal traffic behavior
       OR
Abnormal but non-accident behavior
       OR
Accident event
```

---

# 10. Accident Reasoning

An accident should be considered the result of multiple pieces of temporal and physical evidence.

Conceptually:

```text
Object Behavior
      +
Relative Motion
      +
Interaction
      +
Abrupt Change
      +
Collision Evidence
      +
Post-Impact Behavior
      ↓
Accident Evidence
      ↓
Accident Decision
```

The exact combination and weighting of these signals should be determined experimentally.

RADS should avoid relying on a single arbitrary rule such as:

```text
Bounding boxes overlap → Accident
```

or:

```text
Vehicle stops suddenly → Accident
```

Such signals may be useful evidence, but they are not sufficient on their own.

---

# 11. Before / During / After Reasoning

A strong accident detector should consider three temporal regions.

## Before

Understand:

* Object positions
* Velocities
* Directions
* Relative distances
* Existing trajectories

## During

Identify:

* Rapid convergence
* Collision/contact evidence
* Abrupt motion changes
* Trajectory disruption
* Simultaneous changes between objects

## After

Identify:

* Sudden stopping
* Changed trajectories
* Displacement
* Secondary movement
* Objects remaining stationary
* Continued movement after impact
* Additional interactions

Conceptually:

```text
       BEFORE
          │
          ↓
   What was happening?
          │
          ↓
       DURING
          │
          ↓
    What changed?
          │
          ↓
        AFTER
          │
          ↓
   What happened as
      a consequence?
```

This temporal structure should be central to RADS.

---

# 12. Accident Localization

RADS should not merely answer:

```text
"This video contains an accident."
```

It should eventually answer:

```text
"An accident occurred at approximately 7.4 seconds."
```

Where possible, the system should identify:

```text
Event Start
Impact / Event Time
Event End
```

The temporal resolution will depend on:

* Video frame rate
* Annotation quality
* Detector/tracker performance
* Temporal model design

---

# 13. Identifying Involved Objects

Once an accident event is detected, RADS should identify the objects most likely involved.

Example:

```text
Accident:
YES

Objects involved:
Vehicle #3
Vehicle #7
```

Potential reasoning:

```text
Vehicle #3
    ↕
Strong interaction
    ↕
Vehicle #7

Other objects:
No significant interaction
```

The system should distinguish involved objects from nearby objects that were merely present in the scene.

---

# 14. Accident Type Reasoning

RADS may eventually classify accident type.

Possible categories:

```text
Rear-End
Head-On
T-Bone
Sideswipe
Single-Vehicle
Other / Unknown
```

Accident type should be inferred from object trajectories and interaction patterns where possible.

Example:

```text
Vehicle A
→→→→→
       Vehicle B
       ↓

Strong lateral interaction
        ↓
Potential T-Bone
```

The accident-type classifier should not be treated as mandatory for every event in the initial MVP.

If evidence is insufficient:

```text
Other / Unknown
```

is preferable to an unsupported confident classification.

---

# 15. Severity Reasoning

Severity should be treated as a downstream reasoning task.

The system should first establish:

```text
Did an accident occur?
```

Then:

```text
How severe does the event appear to be?
```

Potential evidence includes:

```text
Number of objects involved
Object types
Relative velocity
Magnitude of motion change
Trajectory disruption
Post-impact displacement
Pedestrian involvement
Secondary collisions
Rollover-like behavior
Road obstruction
```

The MVP may use an interpretable scoring system.

A future system may learn severity directly from data.

---

# 16. Evidence-Based Decision Making

RADS should ideally maintain an internal representation of accident evidence.

Conceptually:

```text
Evidence:

Object interaction       → Strong
Relative velocity        → Strong
Trajectory convergence   → Strong
Abrupt deceleration      → Moderate
Post-impact displacement → Strong

                    ↓

             Accident Score
                    ↓
             Final Decision
```

The exact implementation may be:

* Rule-based
* Statistical
* Machine-learning based
* Neural
* Hybrid

The architecture should remain open to experimentation.

---

# 17. Confidence

Every major prediction should ideally have an associated confidence or evidence strength.

Examples:

```text
Accident:
YES

Confidence:
0.91
```

or:

```text
Accident:
UNCERTAIN

Confidence:
0.54
```

Confidence should not be presented as a guarantee of correctness.

It should represent the model's estimated certainty according to its learned or engineered decision process.

---

# 18. Hard Negative Awareness

RADS must explicitly account for events that look abnormal but are not accidents.

Examples:

```text
Hard braking
Near collision
Sudden lane change
Sharp turn
Overtaking
Traffic congestion
Vehicle stopping normally
Camera movement
Object occlusion
Crowded intersections
```

These examples are particularly important because a system that simply reacts to abrupt motion may produce excessive false positives.

The system should therefore be designed to distinguish:

```text
ABNORMAL MOTION
        ≠
ACCIDENT
```

---

# 19. Near-Collision Reasoning

Near collisions are an important category of hard negatives.

A near collision may contain:

```text
Rapid approach
Small inter-object distance
Trajectory convergence
Hard braking
Abrupt steering
```

but ultimately:

```text
No collision
No meaningful post-impact behavior
```

The system should learn or reason about this distinction.

Conceptually:

```text
Near Collision:

Approach
   ↓
Very close interaction
   ↓
Abrupt braking / steering
   ↓
Objects separate
   ↓
NO ACCIDENT


Actual Collision:

Approach
   ↓
Interaction
   ↓
Collision evidence
   ↓
Motion disruption
   ↓
Post-impact behavior
   ↓
ACCIDENT
```

This distinction should become an important part of training and evaluation.

---

# 20. Environmental Robustness

RADS should avoid learning shortcuts based on environmental appearance.

The system should not conclude:

```text
Dark scene → Accident
Rain → Accident
Busy road → Accident
Specific camera → Accident
Specific background → Accident
```

Instead, accident reasoning should rely increasingly on:

```text
Object behavior
+
Temporal relationships
+
Physical interaction
```

Dataset diversity should support this goal.

---

# 21. Separation of Perception and Reasoning

The system should maintain a conceptual separation between:

```text
PERCEPTION
```

and:

```text
REASONING
```

Perception answers:

> What objects are present?

Reasoning answers:

> What are those objects doing?

And eventually:

> Did an accident occur?

This separation allows the detector/tracker to be improved independently from the accident reasoning system.

---

# 22. Hierarchical Decision Process

The intended reasoning flow is:

```text
1. Detect objects
        ↓
2. Assign persistent IDs
        ↓
3. Build object histories
        ↓
4. Estimate motion
        ↓
5. Compare objects
        ↓
6. Identify interactions
        ↓
7. Detect abnormal temporal events
        ↓
8. Determine whether event is an accident
        ↓
9. Localize event
        ↓
10. Identify involved objects
        ↓
11. Determine accident type
        ↓
12. Estimate severity
```

Not every stage must be a separate neural network.

Some stages may be deterministic or engineered.

The separation is conceptual and functional.

---

# 23. MVP Intelligence

The MVP should prioritize the following capabilities:

```text
Object Detection
        ↓
Object Tracking
        ↓
Persistent IDs
        ↓
Trajectory Extraction
        ↓
Basic Motion Features
        ↓
Object Interaction Detection
        ↓
Temporal Accident Reasoning
        ↓
Accident Localization
        ↓
Basic Severity Estimation
```

The MVP does NOT require:

* Perfect accident classification
* Production-level reliability
* Perfect severity prediction
* Full autonomous emergency response
* Large-scale edge deployment
* A massive end-to-end neural network
* Every possible accident category

The objective is to demonstrate that the proposed intelligence architecture works as a coherent system.

---

# 24. Long-Term Intelligence

The eventual RADS intelligence system may evolve toward:

```text
Visual Perception
       +
Object Tracking
       +
Trajectory Encoding
       +
Object Interaction Modeling
       +
Temporal Reasoning
       +
Contextual Understanding
       +
Accident Classification
       +
Severity Estimation
```

Potential future approaches may include:

* Temporal Transformers
* Graph-based interaction models
* Learned trajectory representations
* Appearance + trajectory fusion
* Multimodal reasoning
* Learned severity prediction
* Event-aware representation learning

These are future directions, not mandatory MVP requirements.

---

# 25. Explainable Accident Reasoning

Where possible, RADS should produce a compact explanation.

Example:

```text
ACCIDENT DETECTED

Time:
00:07.4

Objects:
Vehicle #3
Vehicle #7

Evidence:
- Rapid reduction in distance
- Strong trajectory convergence
- Abrupt velocity change
- Collision interaction detected
- Significant post-impact displacement

Classification:
Rear-End Collision

Severity:
High
```

This output should make it possible for a human reviewer to understand the basis of the decision.

---

# 26. Design Principle: Do Not Overfit the Architecture to Current Data

The current dataset and experiments are not assumed to represent the final dataset.

The dataset is expected to expand.

Therefore:

* Components should remain modular.
* Thresholds should be configurable.
* Model choices should remain replaceable.
* Dataset paths should not be hardcoded.
* Training and inference should be reproducible.
* New annotation fields should be supportable.
* Detection and tracking should remain independently testable.

The architecture should be able to grow without requiring a complete rewrite.

---

# 27. Design Principle: Baselines Must Be Preserved

The existing ResNet18 + GRU experiments should not be discarded.

They provide evidence about the limitations of the previous approach.

P02 remains the current baseline.

The new object-centric architecture should be evaluated against this baseline where comparison is meaningful.

The goal is not to hide weak historical results.

The goal is to demonstrate why the architecture evolved.

---

# 28. Research Philosophy

RADS should follow an iterative research process:

```text
Hypothesis
    ↓
Implementation
    ↓
Experiment
    ↓
Evaluation
    ↓
Failure Analysis
    ↓
Revision
```

A poor result is not automatically treated as a failure of the entire project.

Instead, the system should determine:

```text
What failed?
Why did it fail?
What evidence supports that conclusion?
What should change?
```

Architecture changes should be based on evidence whenever possible.

---

# 29. Non-Negotiable Principles

The following principles should guide all future RADS development:

### 1. Temporal reasoning is fundamental.

Accidents are events, not static images.

### 2. Object identity matters.

The system must know which detections correspond to which physical objects.

### 3. Proximity is not collision.

Close objects are common in normal traffic.

### 4. Abnormal motion is not automatically an accident.

Hard braking and evasive maneuvers can be normal non-accident events.

### 5. Accident decisions should use multiple signals.

No single arbitrary feature should define an accident.

### 6. Source-video leakage must be avoided.

Evaluation must reflect genuine generalization.

### 7. Explainability is valuable.

The system should expose useful evidence behind predictions.

### 8. MVP simplicity is preferable to unnecessary complexity.

A smaller system that works coherently is preferable to a sophisticated architecture that cannot be evaluated.

### 9. Components should remain modular.

Detection, tracking, reasoning, and severity should be replaceable independently.

### 10. The system should evolve with the dataset.

Current assumptions should not unnecessarily constrain future development.

---

# 30. Final Mental Model

The intended RADS intelligence can be summarized as:

```text
                 RAW VIDEO
                     │
                     ↓
              "WHAT EXISTS?"
                     │
                Detection
                     │
                     ↓
              "WHO IS WHO?"
                     │
                 Tracking
                     │
                     ↓
             "WHAT ARE THEY DOING?"
                     │
            Motion / Trajectories
                     │
                     ↓
          "HOW ARE THEY INTERACTING?"
                     │
           Interaction Analysis
                     │
                     ↓
             "WHAT CHANGED?"
                     │
            Temporal Reasoning
                     │
                     ↓
             "IS THIS AN ACCIDENT?"
                     │
             Accident Decision
                     │
                     ↓
              "WHEN / WHO?"
                     │
        Event + Object Localization
                     │
                     ↓
               "HOW BAD?"
                     │
             Severity Estimation
                     │
                     ↓
                  OUTPUT
```

The fundamental idea behind RADS is therefore:

> **Do not classify the scene. Understand the event.**

---
