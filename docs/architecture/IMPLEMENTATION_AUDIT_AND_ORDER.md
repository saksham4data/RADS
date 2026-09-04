# RADS — Implementation Audit & Execution Plan

**Project:** RADS
**Document:** Repository vs. Specification Audit — Revision 2
**Revision:** 2
**Date:** 2026-09-03
**Status:** Active

---

## Preamble

This document audits the RADS repository against five specification documents and defines the execution plan for the MVP.

### Specification Documents

- [MASTER_SPEC.md](file:///e:/Rads/docs/architecture/MASTER_SPEC.md)
- [AI.md](file:///e:/Rads/docs/architecture/AI.md)
- [SYSTEM.md](file:///e:/Rads/docs/architecture/SYSTEM.md)
- [TECH_STACK.md](file:///e:/Rads/docs/architecture/TECH_STACK.md)
- [MVP.md](file:///e:/Rads/docs/architecture/MVP.md)

### Target Architecture

All five specifications converge on one pipeline:

```text
Video → YOLO Detection → Tracking (Persistent IDs)
→ Track Histories → Trajectories → Motion Features
→ Object Interactions → Accident Event Reasoning
→ Event Localization → Severity Estimation
→ Visualization → Structured Output
```

### Revision 2 — What Changed From Revision 1

1. **Vertical-slice strategy.** Phases are restructured so a thin end-to-end pipeline runs as early as Phase 3, then each subsequent phase deepens it.
2. **Pipeline orchestrator moved to Phase 1** instead of Phase 12. Integration happens from the start.
3. **Clear P0/P1 scope boundary.** Dashboard, Telegram, accident-type classification, and per-component evaluation are explicitly P1 / post-MVP.
4. **Severity is explicitly rule-based.** Not an ML model. Not validated against ground truth (none exists).
5. **Baseline comparison protocol** defined: reduce new system output to clip-level labels for fair P02 comparison.
6. **Performance/profiling checkpoint** added at Phase 3 (earliest point where YOLO + tracker run together).
7. **Phase 0 scoped down** to only safe, non-code-breaking scaffolding.
8. **Preservation mandate explicit.** Nothing in the existing repository is deleted or overwritten.
9. **IMPLEMENTATION_TRACKER.md** format specified with concrete columns.
10. **Verification criteria strengthened** for every phase — each has a testable, binary pass/fail condition.

---

# PART 1 — AUDIT FINDINGS

---

## 1. Conflicts Between Specifications and Repository

### 1.1 Architecture Gap (CRITICAL)

The five specs mandate an object-centric, temporal, event-understanding pipeline. The repository contains only the older ResNet18 + GRU video-level classification pipeline, which the specs designate as the baseline to surpass (MASTER_SPEC §15, AI §27).

| Spec Requirement | Repository Reality |
|---|---|
| Pretrained YOLO detection | No YOLO code or dependency |
| Multi-object tracking / persistent IDs | No tracker code or dependency |
| Trajectory and motion features | No trajectory or motion code |
| Object interaction analysis | No interaction code |
| Temporal event-level accident reasoning | Clip-level binary classification only |
| Event localization (start / impact / end) | Not implemented |
| Severity estimation (LOW / MEDIUM / HIGH) | Not implemented |
| Structured JSON event output | Not implemented |
| Annotated visualization pipeline | Not implemented |

**The entire specified MVP architecture does not yet exist in code.** The transition has not begun.

### 1.2 Module Structure Gap

SYSTEM.md §26 specifies functional modules: `video/`, `detection/`, `tracking/`, `motion/`, `interaction/`, `reasoning/`, `severity/`, `output/`.

The repository has: `training/` (old baseline), `eda/`, `Datasets/`, `scripts/`, `scratch/`.

SYSTEM.md §26 also says *"The exact repository structure may differ. The important requirement is functional separation."* — so the new code must be functionally separated but need not replicate the exact directory names verbatim.

### 1.3 Dependency Gap

Original audit finding: no YOLO library (`ultralytics`) and no tracking library existed in `requirements.txt` or `requirements-training.txt`.

Post-Phase-0 repository cleanup: dependencies were consolidated so `requirements.txt` is the single source of truth for EDA, baseline training, and Phase 1+ pipeline work. `ultralytics` is now listed there. `requirements-training.txt` remains as a backward-compatible wrapper that includes `requirements.txt`.

### 1.4 Dashboard / Telegram Clarification

TECH_STACK.md §17 and SYSTEM.md §23–24 reference "existing" dashboard and Telegram integrations. **No such code exists in the repository.** However, both specs explicitly mark these as consuming a structured event output (i.e., downstream of the core pipeline). MVP.md §6 classifies both as P1. This is not a blocker for the core MVP.

---

## 2. Missing Components — P0 vs P1

### P0 — Required for Core MVP

These form the end-to-end pipeline described in MASTER_SPEC §24 and MVP §36.

| # | Component | Spec Reference |
|---|---|---|
| 1 | Video input / frame pipeline | SYSTEM §4–5, MVP §5.1 |
| 2 | YOLO object detection | MASTER_SPEC §3, SYSTEM §6, TECH_STACK §6, MVP §5.2 |
| 3 | Multi-object tracking + persistent IDs | MASTER_SPEC §6, SYSTEM §7–8, TECH_STACK §8, MVP §5.3 |
| 4 | Track history storage | SYSTEM §8, MVP §5.4 |
| 5 | Trajectory extraction | SYSTEM §10, AI §5, MVP §5.5 |
| 6 | Motion feature extraction | SYSTEM §12, AI §5–6, MVP §5.6 |
| 7 | Pairwise object relationships | SYSTEM §13, AI §6 |
| 8 | Interaction detection | SYSTEM §14–15, AI §7, MVP §5.7 |
| 9 | Accident event reasoning (rule-based initially) | SYSTEM §17, AI §10, MVP §5.8 |
| 10 | Event localization + involved objects | SYSTEM §18, AI §12–13, MVP §5.9–5.10 |
| 11 | Severity estimation (rule-based heuristic) | SYSTEM §20, AI §15, MVP §5.11 |
| 12 | Structured event output (JSON) | SYSTEM §21, TECH_STACK §17 |
| 13 | Visualization (annotated video) | SYSTEM §22, TECH_STACK §16, MVP §5.12 |
| 14 | Pipeline orchestrator + entry point | MASTER_SPEC §3, SYSTEM §2 |
| 15 | Pipeline configuration (YAML) | SYSTEM §31, TECH_STACK §19 |
| 16 | Clip-level evaluation + P02 comparison | MVP §17–24 |

### P1 — Post-MVP / Important but Deferred

| # | Component | Spec Reference |
|---|---|---|
| 17 | Accident type classification | SYSTEM §19, AI §14, MVP §6 |
| 18 | Dashboard integration | SYSTEM §23, TECH_STACK §17, MVP §6 |
| 19 | Telegram alert integration | SYSTEM §24, TECH_STACK §17, MVP §6 |
| 20 | Hard-negative evaluation pipeline | AI §18–19, MVP §6 |
| 21 | Detection-specific evaluation (mAP) | MVP §22 |
| 22 | Tracking-specific evaluation (IDF1, MOTA) | MVP §20 |
| 23 | Severity evaluation (requires ground truth) | MVP §21 |
| 24 | Event confidence calibration | MVP §6 |

---

## 3. Existing Components — Preservation Mandate

**Nothing in the existing repository is deleted, renamed, or overwritten.** The new pipeline lives in a new `rads/` directory alongside the existing code.

### 3.1 PRESERVE — Required

| Component | Path | Reason |
|---|---|---|
| ResNet18 + GRU temporal model | [temporal_model.py](file:///e:/Rads/training/models/temporal_model.py) | Baseline reference (MASTER_SPEC §15) |
| ResNet18 classification model | [classification_model.py](file:///e:/Rads/training/models/classification_model.py) | Baseline reference |
| Training engine + evaluator | [training/engine/](file:///e:/Rads/training/engine/) | Baseline reference |
| Classification metrics | [classification_metrics.py](file:///e:/Rads/training/metrics/classification_metrics.py) | Reusable for new system evaluation |
| W&B manager | [wandb_manager.py](file:///e:/Rads/training/utils/wandb_manager.py) | Reusable |
| Logger | [logger.py](file:///e:/Rads/training/utils/logger.py) | Reusable |
| Config system + all YAML configs | [training/configs/](file:///e:/Rads/training/configs/) and [training/config/](file:///e:/Rads/training/config/) | Historical reference |
| Dataset pipeline | [Datasets/pipeline/](file:///e:/Rads/Datasets/pipeline/) | Active — needed for data |
| Processed metadata + manifests | [Datasets/processed/](file:///e:/Rads/Datasets/processed/) | Active — needed for data |
| EDA infrastructure | [eda/](file:///e:/Rads/eda/) | Active analysis tooling |
| All experiment outputs | [training/outputs/](file:///e:/Rads/training/outputs/), [training/p02_prediction.json](file:///e:/Rads/training/p02_prediction.json) | P02 benchmark evidence |
| Research docs + decisions | [docs/research/](file:///e:/Rads/docs/research/) | Historical record |
| Specification documents | [docs/architecture/](file:///e:/Rads/docs/architecture/) | Source of truth |
| Scripts | [scripts/](file:///e:/Rads/scripts/) | Dataset building scripts |
| Scratch debug artifacts | [scratch/](file:///e:/Rads/scratch/) | Diagnostic history |

### 3.2 Components Irrelevant to New Architecture (But Not Deleted)

| Component | Path | Note |
|---|---|---|
| Frame-level dataset | `training/datasets/frame_dataset.py` | Old architecture — frame classification |
| Video-level dataset | `training/datasets/video_dataset.py` | Old architecture — clip classification |
| Video sampling | `training/datasets/video_sampling.py` | Old architecture |
| Hyperparameter study | `training/run_hyperparameter_study.py` | Old architecture study |

These remain in place as historical artifacts. They do not interfere with the new `rads/` module.

---

## 4. Implementation Risks

### HIGH

| Risk | Mitigation |
|---|---|
| **Scope creep** — attempting all 13 spec stages before anything works end-to-end | Vertical-slice approach: thin end-to-end path first, deepen each stage iteratively |
| **YOLO + tracker integration** — occlusion handling, ID stability, detection-tracking coupling | Use `ultralytics` built-in YOLO+tracker first (BoT-SORT/ByteTrack). Validate on 1 video before dataset-scale runs. |
| **No per-object ground truth** — datasets have clip-level accident labels only, no bounding-box or tracking annotations | Visual/qualitative tracking validation for MVP (MVP §20). Clip-level labels still allow accident/normal classification evaluation. |
| **No severity ground truth** — no severity labels in any dataset | Severity engine is a transparent rule-based heuristic, not a validated prediction model. This is clearly stated. |
| **Hardware constraints** — YOLO + tracking per-frame is heavier than old frame-sampling | Profile at Phase 3. Use YOLO-nano/small. Allow configurable frame-skip. |

### MEDIUM

| Risk | Mitigation |
|---|---|
| **Temporal reasoning model selection** — specs leave open (rule-based / GRU / CNN / Transformer) | Start rule-based. Add ML only if rule-based is demonstrably insufficient. |
| **Image-space vs real-world motion** | All velocities labeled as image-space (px/frame). No physical speed claims without calibration. (SYSTEM §11) |
| **Baseline comparison validity** — old system = clip classification, new system = event detection | Define comparison protocol: reduce new system's per-video output to binary clip label, then compute same metrics against same test split. |

### LOW

| Risk | Mitigation |
|---|---|
| **Dashboard / Telegram don't exist** | P1. Structured event JSON comes first; delivery adapters are downstream consumers. |
| **Config proliferation** | New pipeline gets its own YAML under `rads/config/`. Old configs are untouched. |

---

## 5. Baseline Comparison Protocol

The P02 ResNet18 + GRU baseline produced clip-level binary predictions (accident / normal). The new YOLO-based system produces event-level structured output.

To compare fairly:

1. **Reduce** the new system's output to a clip-level binary label per video:
   - If the pipeline detects an accident event with confidence ≥ threshold → `accident`
   - Otherwise → `normal`
2. **Use the same test split** from the existing dataset (source-video leakage rules preserved).
3. **Report identical metrics**: Accuracy, Balanced Accuracy, Precision, Recall, F1, Macro F1, AUROC, Confusion Matrix, FPR.
4. **Additionally report** (new system only): event localization quality, involved-object identification, severity distribution — these have no P02 counterpart and represent architectural advantages.

P02 baseline reference values:

```text
Accuracy:          45.9%
Balanced Accuracy: 45.9%
AUROC:             0.535
Macro F1:          0.407
Accident F1:       0.583
Normal F1:         0.231
Normal FPR:        83.8%
```

---

# PART 2 — EXECUTION PLAN

---

## Strategy: Vertical Slice First

The plan uses a **vertical-slice-first** approach: reach a thin but complete end-to-end path as soon as possible, then deepen each stage.

```text
Phase 0    Scaffolding (safe, no code logic)
Phase 1    Video → YOLO → Tracker → JSON stub → pipeline glue    ← FIRST VERTICAL SLICE
Phase 2    Profiling checkpoint, visual verification
Phase 3    Track histories + trajectories + motion features
Phase 4    Pairwise relationships + interaction detection
Phase 5    Accident reasoning + event localization + involved objects
Phase 6    Severity heuristic
Phase 7    Visualization (annotated video)
Phase 8    Structured output finalization + evaluation + P02 comparison
            ← MVP COMPLETE
Phase 9+   P1 work (dashboard, telegram, accident types, etc.)
```

After Phase 1, the system can process a video and produce a (stub) JSON result. Each subsequent phase replaces a stub with real logic while the pipeline keeps running end-to-end.

---

### Phase 0 — Scaffolding

**Goal:** Create the directory structure, add dependencies, create `IMPLEMENTATION_TRACKER.md`, and set up the pipeline config. No functional code. No changes to any existing file.

**Tasks:**

- [ ] Create directory tree under `rads/`:
  ```text
  rads/
  ├── __init__.py
  ├── config/
  │   ├── __init__.py
  │   └── pipeline_config.yaml
  ├── video/
  │   └── __init__.py
  ├── detection/
  │   └── __init__.py
  ├── tracking/
  │   └── __init__.py
  ├── motion/
  │   └── __init__.py
  ├── interaction/
  │   └── __init__.py
  ├── reasoning/
  │   └── __init__.py
  ├── severity/
  │   └── __init__.py
  ├── output/
  │   └── __init__.py
  └── pipeline/
      └── __init__.py
  ```
- [x] Consolidate requirements:
  - `requirements.txt` is the single dependency source of truth.
  - `requirements-training.txt` remains as a compatibility wrapper.
  - The separate `requirements-pipeline.txt` scaffold file was removed because its dependencies are now covered by `requirements.txt`.
- [ ] Create minimal `rads/config/pipeline_config.yaml` with placeholder sections:
  ```yaml
  detector:
    model: "yolo11n.pt"        # smallest model for initial testing
    confidence: 0.25
    iou: 0.45
    classes: [0, 1, 2, 3, 5, 7]  # COCO: person, bicycle, car, motorcycle, bus, truck
  tracker:
    type: "bytetrack"
  pipeline:
    frame_skip: 1              # process every frame (1 = no skip)
  ```
- [ ] Create `docs/architecture/IMPLEMENTATION_TRACKER.md` (template — see Part 3)

**Does NOT touch:** Any existing file in `training/`, `eda/`, `Datasets/`, `scripts/`, `scratch/`, `docs/research/`, or any `requirements*.txt`.

**Verification (pass/fail):**
- `rads/` directory tree exists with all `__init__.py` files
- `requirements.txt` exists and includes the EDA, baseline training, and Phase 1+ pipeline dependencies
- `requirements-training.txt` remains usable for existing training documentation by including `requirements.txt`
- `pipeline_config.yaml` is valid YAML and loads without error
- `IMPLEMENTATION_TRACKER.md` exists
- No existing file was modified (verify via `git status` — only new files)

---

### Phase 1 — First Vertical Slice: Video → Detection → Tracking → Stub Output

**Goal:** Run YOLO + tracker on a video and produce a minimal JSON result. This is the first end-to-end path through the system, even though downstream reasoning is stubbed.

**Tasks:**

- [ ] `rads/config/config_loader.py` — loads `pipeline_config.yaml`, provides typed access
- [ ] `rads/video/video_reader.py` — opens video via OpenCV, yields `(frame_index, timestamp, frame)` tuples; exposes FPS, resolution, duration
- [ ] `rads/detection/detector.py` — wraps `ultralytics.YOLO`, runs inference on a frame, returns list of `Detection` dicts: `{class_id, class_name, confidence, bbox_xyxy, frame_index, timestamp}`
- [ ] `rads/tracking/tracker.py` — uses `ultralytics` model `.track()` or standalone ByteTrack to associate detections and assign persistent `track_id`; returns list of `TrackedObject` dicts per frame: `{track_id, class_name, confidence, bbox_xyxy, center_xy, frame_index, timestamp}`
- [ ] `rads/output/event_schema.py` — defines the structured result dict; for now returns a stub: `{"accident": null, "confidence": null, "event": null, "objects_involved": [], "severity": null, "tracks_summary": {...}}`
- [ ] `rads/pipeline/pipeline.py` — orchestrates: load config → open video → per-frame detection+tracking → collect all tracks → call (stub) reasoning → produce result JSON
- [ ] `run_pipeline.py` (project root) — CLI entry point: `python run_pipeline.py --video <path> --config rads/config/pipeline_config.yaml --output results.json`

**Depends on:** Phase 0

**Verification (pass/fail):**
- `python run_pipeline.py --video <test_video.mp4> --config rads/config/pipeline_config.yaml --output results.json` completes without error
- `results.json` is valid JSON containing `tracks_summary` with at least one tracked object
- Tracked objects have consistent `track_id` values (same object = same ID across frames, confirmed by manual inspection of console log or saved frame samples)
- No existing file was modified

---

### Phase 2 — Visual Verification & Performance Profiling

**Goal:** Validate detection and tracking quality visually. Establish performance baseline. This is a gating checkpoint — if detection or tracking is fundamentally broken, fix it before proceeding.

**Tasks:**

- [ ] `rads/output/visualizer.py` — renders bounding boxes + track IDs + class labels onto frames; writes annotated video via OpenCV `VideoWriter`
- [ ] Add `--visualize` flag to `run_pipeline.py` that produces an annotated `.mp4`
- [ ] Add timing instrumentation to `pipeline.py`: log per-frame processing time, total time, average FPS
- [ ] Run on at least 3 videos: 1 accident, 1 normal, 1 with dense traffic
- [ ] Document results: detection quality (are vehicles detected?), tracking quality (do IDs persist?), processing speed (frames/sec), GPU/CPU memory

**Depends on:** Phase 1

**Verification (pass/fail):**
- Annotated video shows correct bounding boxes with stable IDs on moving vehicles
- No ID assigned to a non-vehicle object (e.g., background) in the sampled videos
- Per-frame processing time is logged and total pipeline time is recorded
- If YOLO-nano is too slow for the hardware, switch to a viable model size and document the change
- Visual verification documented in `IMPLEMENTATION_TRACKER.md` with screenshot or timestamp references

---

### Phase 3 — Track Histories, Trajectories, and Motion Features

**Goal:** Store per-object position histories, construct trajectories, and derive motion features.

**Tasks:**

- [ ] `rads/motion/trajectory.py`:
  - `TrackHistory` class: stores `{track_id: [(frame_index, timestamp, cx, cy, w, h), ...]}` for all tracked objects
  - Built incrementally during per-frame tracking loop in `pipeline.py`
  - Queryable: get trajectory for any `track_id`, get all active tracks at any frame
- [ ] `rads/motion/motion_features.py`:
  - Per-object: displacement, direction, velocity (px/frame), acceleration, direction change
  - Computed from `TrackHistory` after video processing
  - All quantities clearly documented as image-space
- [ ] Integrate into `pipeline.py`: after tracking completes, compute motion features for all tracks
- [ ] Add trajectory visualization to `visualizer.py`: draw trailing path behind each tracked object

**Depends on:** Phase 2 (verified detection + tracking)

**Verification (pass/fail):**
- For a vehicle moving left-to-right, velocity x-component is positive; for right-to-left, negative
- For a vehicle that stops (visible in video), velocity drops near zero in later frames
- Trajectory trails rendered on annotated video match visually observed paths
- Unit test with synthetic trajectory data: known positions → expected velocity/acceleration

---

### Phase 4 — Pairwise Relationships & Interaction Detection

**Goal:** Identify when two tracked objects are interacting in a collision-like manner.

**Tasks:**

- [ ] `rads/interaction/pairwise.py`:
  - For each pair of co-existing tracks: compute distance(t), relative velocity(t), bbox overlap(t)
  - Efficient filtering: only consider pairs where minimum distance < configurable threshold
- [ ] `rads/interaction/interaction_engine.py`:
  - `InteractionCandidate`: `{object_a_id, object_b_id, start_time, peak_time, end_time, min_distance, max_relative_velocity, evidence_list}`
  - Detection logic: rapid distance reduction + trajectory convergence + bbox overlap or proximity
  - Configurable thresholds in `pipeline_config.yaml`
- [ ] Integrate into `pipeline.py`: after motion features, compute pairwise relationships and detect interaction candidates

**Depends on:** Phase 3

**Verification (pass/fail):**
- On a known accident video: at least one `InteractionCandidate` is generated with `peak_time` near the visible collision moment
- On a normal traffic video (no accident): either no candidates, or candidates have weak evidence scores
- Unit test with two synthetic tracks on a collision course → interaction detected; two parallel tracks → no interaction

---

### Phase 5 — Accident Reasoning, Event Localization, Involved Objects

**Goal:** Determine whether an interaction candidate is an accident. Identify when it happened and who was involved.

**Tasks:**

- [ ] `rads/reasoning/accident_reasoner.py`:
  - Consumes `InteractionCandidate` list + `TrackHistory` + motion features
  - Rule-based logic evaluating multiple signals: interaction strength, abrupt motion change at/after interaction, post-interaction displacement, simultaneous velocity changes
  - Produces: `{accident: bool, confidence: float, event_start, impact_time, event_end, involved_object_ids, evidence_list}`
  - Confidence is a normalized weighted sum of evidence signals (not a probability model)
  - Configurable thresholds and weights in `pipeline_config.yaml`
- [ ] Integrate into `pipeline.py`: after interaction detection, run accident reasoner; populate result JSON event fields (replacing stubs from Phase 1)

**Depends on:** Phase 4

**Verification (pass/fail):**
- On 3+ known accident videos: `accident: true` with `impact_time` within ±2 seconds of visually observed collision
- On 3+ normal videos: `accident: false` or `confidence` below threshold
- `involved_object_ids` match the vehicles visually participating in the collision
- `evidence_list` contains at least 2 distinct evidence types

---

### Phase 6 — Severity Heuristic

**Goal:** Produce a transparent LOW / MEDIUM / HIGH severity estimate. This is explicitly a **rule-based heuristic**, not a validated ML model, because no severity ground truth exists.

**Tasks:**

- [ ] `rads/severity/severity_engine.py`:
  - Input: accident event + involved tracks + motion features
  - Rule-based scoring on documented factors:
    - Number of objects involved
    - Object types (pedestrian involvement → higher severity)
    - Relative velocity at interaction peak (image-space)
    - Magnitude of velocity change (deceleration)
    - Post-impact displacement
  - Each factor contributes a configurable weighted score
  - Total score mapped to: `LOW` / `MEDIUM` / `HIGH` via configurable thresholds
  - Output: `{severity: str, score: float, evidence: [{factor, value, contribution}, ...]}`
  - **Documentation requirement:** The scoring formula, weights, and thresholds must be fully documented in comments and in `IMPLEMENTATION_TRACKER.md`. This is a heuristic, not a learned model.
- [ ] Integrate into `pipeline.py`: after accident reasoning, compute severity for detected events; populate result JSON

**Depends on:** Phase 5

**Verification (pass/fail):**
- Severity engine produces a result for every detected accident event
- Evidence list is non-empty and factors are traceable
- Manual review: a visually severe collision (high speed, multiple vehicles) scores higher than a minor one
- The scoring formula and thresholds are documented

---

### Phase 7 — Visualization Finalization

**Goal:** Produce a complete annotated video showing the full pipeline output.

**Tasks:**

- [ ] Enhance `rads/output/visualizer.py`:
  - Bounding boxes with track IDs and class labels (already from Phase 2)
  - Trajectory trails (already from Phase 3)
  - Accident event marker: highlight the frame range of the detected event
  - Involved-object highlighting: change bbox color for involved objects
  - On-screen overlay text: "ACCIDENT DETECTED", event time, severity, involved objects
  - Post-event summary frame (optional): display structured result as text overlay
- [ ] The `--visualize` flag on `run_pipeline.py` produces the fully annotated video

**Depends on:** Phase 6

**Verification (pass/fail):**
- Annotated video shows all overlays: boxes, IDs, trajectories, event marker, severity text
- On an accident video, the event marker appears at approximately the correct time
- On a normal video, no event marker appears
- The video is playable and visually demonstrates the RADS concept to a non-technical viewer

---

### Phase 8 — Evaluation, Baseline Comparison, MVP Completion

**Goal:** Run the pipeline on the full test split. Compute evaluation metrics. Compare with P02 baseline. Document results and failure cases. This phase marks **MVP completion**.

**Tasks:**

- [ ] `rads/evaluation/evaluator.py`:
  - Runs pipeline on all videos in the test split
  - Collects per-video: `{video_id, predicted_label, ground_truth_label, confidence, event_time (if any), severity (if any)}`
  - Computes: Accuracy, Balanced Accuracy, Precision, Recall, F1, Macro F1, Confusion Matrix, FPR, FNR
  - AUROC where a continuous confidence score is available
- [ ] `rads/evaluation/baseline_comparison.py`:
  - Loads P02 results from [training/p02_prediction.json](file:///e:/Rads/training/p02_prediction.json)
  - Produces side-by-side comparison table: P02 vs. YOLO pipeline on identical metrics
- [ ] Verify dataset split integrity: confirm source-video separation (reuse existing split from `global_master_metadata.csv`)
- [ ] Document at least 5 failure cases: what failed, which pipeline stage caused it (detection failure vs. tracking failure vs. reasoning failure)
- [ ] Record all results in `IMPLEMENTATION_TRACKER.md`
- [ ] Verify against MVP Definition of Done (MVP §36)

**Depends on:** Phase 7

**Verification (pass/fail):**
- Pipeline runs on every test-split video without crashing
- Evaluation report is produced with all specified metrics
- P02 comparison table is available
- Source-video leakage check passes
- Failure cases are documented with identified failure stage
- All Pipeline items in MVP §36 Definition of Done are checked

---

### Phase 9+ — Post-MVP / P1 Work

These are explicitly deferred until the core MVP (Phases 0–8) is complete.

- [ ] Accident type classification (SYSTEM §19, AI §14)
- [ ] Dashboard integration (SYSTEM §23)
- [ ] Telegram alert integration (SYSTEM §24)
- [ ] Hard-negative evaluation pipeline (AI §18–19)
- [ ] Detection-specific evaluation: mAP, precision, recall (MVP §22)
- [ ] Tracking-specific evaluation: IDF1, MOTA, ID switches (MVP §20)
- [ ] Severity evaluation against ground truth (requires annotations — MVP §21)
- [ ] Event confidence calibration
- [ ] ML-based temporal model replacing or augmenting rule-based reasoning
- [ ] Real-time / streaming mode (SYSTEM §28)

---

# PART 3 — IMPLEMENTATION TRACKER

---

Create `docs/architecture/IMPLEMENTATION_TRACKER.md` with this template:

```markdown
# RADS — Implementation Tracker

Companion to [IMPLEMENTATION_AUDIT_AND_ORDER.md](file:///e:/Rads/docs/architecture/IMPLEMENTATION_AUDIT_AND_ORDER.md).

Updated after every implementation phase.

---

## Progress

| Phase | Name | Status | Date Started | Date Completed | Notes |
|---|---|---|---|---|---|
| 0 | Scaffolding | Not Started | — | — | |
| 1 | First Vertical Slice | Not Started | — | — | |
| 2 | Visual Verification & Profiling | Not Started | — | — | |
| 3 | Track Histories & Motion Features | Not Started | — | — | |
| 4 | Pairwise & Interaction Detection | Not Started | — | — | |
| 5 | Accident Reasoning & Localization | Not Started | — | — | |
| 6 | Severity Heuristic | Not Started | — | — | |
| 7 | Visualization Finalization | Not Started | — | — | |
| 8 | Evaluation & MVP Completion | Not Started | — | — | |

---

## Phase Completion Records

### Phase N — [Name]

**Date:** YYYY-MM-DD
**Files created/modified:**
- `rads/...`

**Verification outcome:**
- [ ] Criterion 1: PASS / FAIL — notes
- [ ] Criterion 2: PASS / FAIL — notes

**Blockers / Deviations:**
- None / description

**MVP Definition of Done status:**
- Reference MVP.md §36 items that are now satisfied

---

## Configuration Snapshot

Record key configuration decisions here:
- YOLO model: (to be decided at Phase 1)
- Tracker: (to be decided at Phase 1)
- Severity thresholds: (to be decided at Phase 6)

---

## Performance Log

| Phase | Video | Resolution | FPS | Frames | Pipeline Time (s) | Per-Frame (ms) | GPU Mem (MB) |
|---|---|---|---|---|---|---|---|
| 2 | (first test) | | | | | | |
```

---

# PART 4 — SUMMARY: CURRENT STATE VS. TARGET

---

```text
SPECIFICATION TARGET               REPO STATUS        MVP PHASE
─────────────────────               ───────────        ─────────
Video Input Pipeline                ✗ Missing          Phase 1
YOLO Object Detection               ✗ Missing          Phase 1
Multi-Object Tracking                ✗ Missing          Phase 1
Persistent Object IDs                ✗ Missing          Phase 1
Visual Verification                  ✗ Missing          Phase 2
Performance Profiling                ✗ Missing          Phase 2
Track Histories                      ✗ Missing          Phase 3
Trajectory Extraction                ✗ Missing          Phase 3
Motion Feature Engine                ✗ Missing          Phase 3
Pairwise Object Relationships        ✗ Missing          Phase 4
Interaction Detection                ✗ Missing          Phase 4
Accident Event Reasoning             ✗ Missing          Phase 5
Event Localization                   ✗ Missing          Phase 5
Involved Object Identification       ✗ Missing          Phase 5
Severity Estimation                  ✗ Missing          Phase 6
Visualization Pipeline               ✗ Partial (Ph2)   Phase 7
Structured Event Output              ✗ Stub (Ph1)      Phase 8
Evaluation + P02 Comparison          ✗ Missing          Phase 8
Pipeline Orchestrator                ✗ Missing          Phase 1
Configuration System                 ✗ Missing          Phase 0–1

ResNet18 + GRU Baseline              ✓ PRESERVED
Dataset Pipeline                     ✓ PRESERVED
EDA Infrastructure                   ✓ PRESERVED
W&B Experiment Tracking              ✓ PRESERVED
Historical Experiments + Decisions   ✓ PRESERVED
All Specification Documents          ✓ PRESERVED

Dashboard Integration                ✗ P1 (post-MVP)
Telegram Integration                 ✗ P1 (post-MVP)
Accident Type Classification         ✗ P1 (post-MVP)
```

---

# PART 5 — FINAL CONSISTENCY CHECK

---

## Spec Compliance Matrix

| Spec Section | Requirement | Plan Coverage |
|---|---|---|
| MASTER_SPEC §3 | End-to-end pipeline | Phase 1 (vertical slice) through Phase 8 |
| MASTER_SPEC §15 | Preserve P02 baseline | §3.1 Preservation Mandate — no deletions |
| MASTER_SPEC §14 | Source-video split integrity | Phase 8 evaluation — reuse existing splits |
| MASTER_SPEC §20 | Dependency-order development | Phases follow detection → tracking → motion → interaction → reasoning → severity → visualization |
| MASTER_SPEC §21 | Modular, replaceable components | `rads/` subdirectories with clean interfaces |
| AI §4–7 | Perception → Identity → Motion → Interaction | Phases 1–4 |
| AI §8–13 | Temporal reasoning → Accident → Localization | Phase 5 |
| AI §15 | Severity as downstream task | Phase 6 |
| AI §18–19 | Hard-negative awareness | Phase 5 reasoning rules + P1 evaluation |
| AI §27 | Baselines preserved | §3.1 — nothing deleted |
| SYSTEM §4–22 | 14-stage pipeline | Phases 1–8 cover all stages |
| SYSTEM §26 | Module boundaries | `rads/` directory structure |
| SYSTEM §31 | Configurable parameters | `pipeline_config.yaml` |
| TECH_STACK §6 | YOLO detection | Phase 1 |
| TECH_STACK §8 | ByteTrack or equivalent | Phase 1 |
| TECH_STACK §14 | Rule-based severity for MVP | Phase 6 — explicitly heuristic |
| TECH_STACK §17 | Dashboard + Telegram | P1 — deferred |
| TECH_STACK §27 | Implementation order | Phases follow specified order |
| MVP §5.1–5.12 | P0 mandatory components | All covered by Phases 1–8 |
| MVP §6 | P1 components | Deferred to Phase 9+ |
| MVP §13 | Source-video leakage prevention | Phase 8 evaluation |
| MVP §17–24 | Evaluation metrics | Phase 8 |
| MVP §30 | Development order | Phases match MVP §30 phases |
| MVP §36 | Definition of Done | Phase 8 verification checklist |

## Remaining Spec Contradictions

| Item | Contradiction | Resolution |
|---|---|---|
| SYSTEM §23, TECH_STACK §17 refer to "existing dashboard" | No dashboard code exists in the repository | Treat as P1 new development, not integration with existing code. Not a blocker. |
| SYSTEM §24 refers to "existing Telegram integration" | No Telegram code exists | Same — P1 new development. |
| MVP §36 says "Existing dashboard/alert integration is not broken" | There is nothing to break | This DoD item is satisfied vacuously. Note it in tracker. |
| SYSTEM §26 prescribes exact module names | SYSTEM §26 also says "exact structure may differ" | Use the prescribed names where they match naturally; deviate only where justified. |

---
