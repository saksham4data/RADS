# AI Instructions: Temporal Modeling Phase 1

## Role

You are the primary coding and research agent for Phase 1 of the temporal-modeling work in the TUDAT project.

You are not being asked to immediately build a temporal model.

Your first responsibility is to understand the existing system thoroughly, identify the actual limitation of the current frame-based approach, determine what must change to support temporal modeling, and then implement only the changes that are justified by that investigation.

Think like a research engineer, not a code generator.

---

## Project Direction

The broader research roadmap is:

TUDAT
↓
Understand current failure
↓
Build + validate temporal pipeline
↓
Prove temporal modeling works
↓
Small PICEK subset
↓
Validate scalability + annotation handling
↓
Full PICEK training
↓
Final experiments / comparison

The earlier TUDAT investigation and E07 work have already established the current frame-based baseline.

We are now entering the temporal-modeling stage.

Phase 1 is the bridge between the existing frame-based system and the future temporal pipeline.

---

## Primary Objective

Determine:

1. How the current frame-based pipeline represents a video.
2. Whether frame ordering is preserved.
3. Where temporal information is currently retained.
4. Where temporal information is lost or collapsed.
5. How the current frame representations are aggregated.
6. Whether the current representation can adequately model temporal changes.
7. What architectural boundary is best suited for a temporal module.
8. What changes are required in the dataset, dataloader, model, trainer, evaluator, and configuration systems.
9. What should remain unchanged for a scientifically controlled comparison.
10. What the first temporal experiment should look like.

After establishing these facts, implement the necessary infrastructure changes required for the next temporal-modeling phase.

---

## Required Working Method

Follow this order.

### Step 1: Read before modifying

Read:

- project documentation
- relevant research/investigation reports
- TUDAT v2 documentation
- E07 documentation
- training configuration
- dataset implementation
- dataloader
- frame sampling/decoder code
- model implementation
- trainer
- evaluator
- prediction/inference code
- experiment configuration

Do not begin editing files before understanding the relevant execution path.

---

### Step 2: Trace the real data flow

Trace one complete video through the system:

video
→ metadata
→ selected frame indices
→ frame decoding
→ transforms
→ batch
→ spatial model
→ frame representation
→ aggregation
→ classifier
→ loss
→ metrics
→ checkpoint

Record the actual tensor shapes and interfaces.

Do not infer these from filenames or documentation if the source code can verify them.

---

### Step 3: Identify the representation problem

The key research question is:

> What information about temporal change is unavailable to the current model?

Investigate whether the current system can represent:

- frame ordering
- motion
- direction of change
- transitions between visual states
- pre-event → event progression
- temporal relationships between frames

Do not automatically assume that the current system has no temporal information.

Show exactly where temporal structure is preserved or discarded.

---

### Step 4: Determine the architectural boundary

Find the cleanest location for a future structure such as:

`ordered frames`
→ `spatial feature extractor`
→ `temporal module`
→ `classifier`

Evaluate the existing code and determine whether temporal modeling should occur at:

- frame feature level
- frame logit level
- another representation level

The decision must be based on the actual implementation and controlled-experiment requirements.

---

### Step 5: Plan before implementation

Before making substantive code changes:

1. update `plan.md`
2. document the discovered architecture
3. document the proposed changes
4. identify affected files
5. identify files that must remain untouched
6. identify tests required to validate the changes

Do not implement speculative improvements.

---

## Implementation Principle

Changes ARE allowed in this phase.

The goal is not to freeze the codebase.

The goal is to make the minimum justified changes necessary to establish a clean, reusable temporal-modeling pipeline while preserving the existing frame-based baseline.

Prefer:

- reusable components
- clean interfaces
- explicit sequence dimensions
- deterministic behavior
- configuration-driven experimentation
- backward compatibility where practical
- tests for changed behavior

Avoid:

- unrelated refactoring
- changing existing experiment semantics
- silently changing dataset definitions
- mixing temporal architecture changes with unrelated optimization
- hard-coded experiment-specific behavior

---

## Scientific Control

The first temporal experiment must be comparable with the established TUDAT baseline.

Where practical, preserve:

- TUDAT v2
- existing frozen splits
- labels
- frame sampling policy
- spatial backbone
- optimizer
- training infrastructure
- evaluation metrics

The temporal component should be the principal experimental change.

If any of these must change, document why.

---

## Evidence Requirements

Important conclusions must be supported by repository evidence.

For each important finding, record:

- file path
- class/function
- relevant implementation behavior
- consequence for temporal modeling

Do not write conclusions such as "the model averages frames" unless the implementation confirms it.

Do not write conclusions such as "frame order is preserved" unless the data path confirms it.

---

## Testing Requirements

Any implementation change must have an appropriate validation strategy.

At minimum, consider:

- unit tests for new temporal utilities
- tensor shape tests
- sequence ordering tests
- dataset/dataloader smoke tests
- model forward-pass tests
- configuration loading tests
- regression tests for the existing baseline where affected

Do not run a full expensive experiment merely to prove that an interface works if a focused test can establish the same fact.

---

## Restrictions

Do not:

- delete TUDAT v2
- modify raw videos
- change frozen dataset splits without explicit justification
- rewrite historical experiments
- overwrite E07 results
- start PICEK training
- claim temporal modeling has improved performance before an experiment demonstrates it
- fabricate metrics or experimental conclusions

---

## Final Deliverable

At the end of Phase 1, the repository should contain:

1. A clear investigation of the current frame-based representation.
2. A documented temporal architecture recommendation.
3. A documented implementation plan for the next temporal experiment.
4. Necessary reusable infrastructure changes, if justified.
5. Tests validating those changes.
6. No unexplained changes to the existing experimental baseline.

The final report must clearly distinguish:

- what was discovered
- what was changed
- why it was changed
- what was tested
- what remains for Phase 2