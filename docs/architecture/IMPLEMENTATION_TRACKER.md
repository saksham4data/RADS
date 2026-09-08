# RADS — Implementation Tracker

Companion to [IMPLEMENTATION_AUDIT_AND_ORDER.md](file:///e:/Rads/docs/architecture/IMPLEMENTATION_AUDIT_AND_ORDER.md).

Updated after every implementation phase.

---

## Progress

| Phase | Name | Status | Date Started | Date Completed | Notes |
|---|---|---|---|---|---|
| 0 | Scaffolding | Completed | 2026-09-05 | 2026-09-05 | Dependencies and skeleton added |
| 1 | First Vertical Slice | Completed | 2026-09-06 | 2026-09-06 | YOLO detection + tracker integration |
| 2 | Visual Verification & Profiling | Completed | 2026-09-06 | 2026-09-06 | Visualizer built, performance tested |
| 3 | Track Histories & Motion Features | Completed | 2026-09-06 | 2026-09-06 | Trajectory memory and kinematics derived |
| 4 | Pairwise & Interaction Detection | Completed | 2026-09-08 | 2026-09-08 | Detected interactions using normalized proximity, relative velocity, and bbox overlap |
| 5 | Accident Reasoning & Localization | Completed | 2026-09-08 | 2026-09-08 | Implemented rule-based reasoning engine |
| 6 | Severity Heuristic | Completed | 2026-09-08 | 2026-09-08 | Rule-based heuristic based on objects and class |
| 7 | Visualization Finalization | Completed | 2026-09-08 | 2026-09-08 | Two-pass rendering and clip extraction |
| 8 | Evaluation & MVP Completion | Not Started | — | — | |

---

## Phase Completion Records

### Phase 0 — Scaffolding

**Date:** 2026-09-05
**Files created/modified:**
- `rads/` module structure and `__init__.py` files
- `rads/config/pipeline_config.yaml`
- `requirements.txt`
- `requirements-training.txt`
- `docs/architecture/IMPLEMENTATION_TRACKER.md`

**Verification outcome:**
- [x] `rads/` directory tree exists with all `__init__.py` files: PASS
- [x] Dependency requirements are present: PASS - `requirements.txt` is now the single source of truth; `requirements-training.txt` remains as a compatibility wrapper.
- [x] `pipeline_config.yaml` is valid YAML and loads without error: PASS
- [x] `IMPLEMENTATION_TRACKER.md` exists: PASS
- [x] No existing file was modified: PASS

**Blockers / Deviations:**
- Dependency layout changed after the initial Phase 0 scaffold. The separate `requirements-pipeline.txt` file was removed to avoid three competing dependency files. `ultralytics` was moved into `requirements.txt`; existing `pip install -r requirements-training.txt` workflows still work because that file now includes `requirements.txt`.

**MVP Definition of Done status:**
- Scaffolding complete.

---

### Phase 1 — First Vertical Slice

**Date:** 2026-09-06
**Files created/modified:**
- `rads/config/config_loader.py`
- `rads/video/video_reader.py`
- `rads/detection/detector.py`
- `rads/tracking/tracker.py`
- `rads/output/event_schema.py`
- `rads/pipeline/pipeline.py`
- `run_pipeline.py`

**Verification outcome:**
- [x] `run_pipeline.py` executes without error on a test video: PASS
- [x] `results.json` contains tracked objects with consistent `track_id` values: PASS
- [x] No existing file modified: PASS

**Blockers / Deviations:**
- None.

**MVP Definition of Done status:**
- Video input pipeline (MVP §5.1) — foundational pieces implemented.
- YOLO object detection (MVP §5.2) — implemented via ultralytics.
- Multi-object tracking (MVP §5.3) — implemented via ultralytics track().

---

### Phase 2 — Visual Verification & Profiling

**Date:** 2026-09-06
**Files created/modified:**
- `rads/output/visualizer.py`
- `rads/pipeline/pipeline.py`
- `run_pipeline.py`

**Verification outcome:**
- [x] Annotated video shows correct bounding boxes with stable IDs: PASS
- [x] Per-frame processing time is logged: PASS
- [x] FPS is sufficient for MVP logic testing: PASS (approx. 7 FPS on CPU/standard hardware with YOLO11n)

**Blockers / Deviations:**
- None. YOLO11n provides adequate performance and tracking stability for this phase.

**MVP Definition of Done status:**
- Progressed tracking (MVP §5.3) and visual verification capabilities.

### Phase 3 — Track Histories & Motion Features

**Date:** 2026-09-06
**Files created/modified:**
- `rads/motion/trajectory.py`
- `rads/motion/motion_features.py`
- `rads/pipeline/pipeline.py`
- `rads/output/visualizer.py`

**Verification outcome:**
- [x] Kinematics (velocity, acceleration, direction) are successfully computed in image-space (pixels/sec): PASS
- [x] Light smoothing reduces detection jitter in velocity gradients: PASS
- [x] Visualizer renders historic trailing trajectories correctly behind moving vehicles: PASS

**Blockers / Deviations:**
- Note: Computations are strictly in image-space coordinates. No world-coordinate scaling (pixels to meters) is applied per the MVP scope constraint.

**MVP Definition of Done status:**
- Implemented Track history storage (MVP §5.4).
- Implemented Trajectory extraction (MVP §5.5).
- Implemented Motion feature extraction (MVP §5.6).

---

### Phase 4 — Pairwise Relationships & Interaction Detection

**Date:** 2026-09-08
**Files created/modified:**
- `rads/interaction/__init__.py`
- `rads/interaction/pairwise.py`
- `rads/interaction/interaction_engine.py`
- `rads/interaction/test_interactions.py`
- `rads/pipeline/pipeline.py`
- `rads/output/event_schema.py`
- `rads/config/pipeline_config.yaml`

**Verification outcome:**
- [x] Tested against 3 positive and 3 negative clips from `picek_sorted`: PASS
- [x] Identified interaction candidates correctly based on normalized proximity, relative velocity, trajectory convergence, and bbox overlap: PASS
- [x] Extracted evidence such as distance, normalized proximity, and relative velocity per pair per frame: PASS

**Blockers / Deviations:**
- No arbitrary fixed pixel thresholds used for proximity; utilized normalized metrics (relative to object size bounding box).
- Convergence angle was simplified to `relative_velocity` (rate of change of distance) as a sufficient indicator of trajectory convergence for the MVP.

**MVP Definition of Done status:**
- Implemented Pairwise object relationships (MVP §5.7 partial).
- Implemented Interaction detection (MVP §5.7).

---

### Phases 5, 6, and 7 — Reasoning, Severity, and Visualization

**Date:** 2026-09-08
**Files created/modified:**
- `rads/reasoning/__init__.py`
- `rads/reasoning/accident_reasoner.py`
- `rads/severity/__init__.py`
- `rads/severity/severity_engine.py`
- `rads/output/visualizer.py`
- `rads/output/event_schema.py`
- `rads/motion/trajectory.py`
- `rads/pipeline/pipeline.py`

**Verification outcome:**
- [x] Logic evaluates candidates, assigns accident confidence, and localizes event bounds (impact time).
- [x] Severity assigned based on number and class type of involved objects.
- [x] Two-pass pipeline architecture implemented successfully for rendering downstream reasoning info onto the video frames.
- [x] Event-evidence clip automatically sliced and exported.
- [x] Visually verified: The annotated video overlays accident status, confidence, severity, and visually highlights involved objects during the event window.

**Blockers / Deviations:**
- Visualization moved to a post-processing second pass since reasoning relies on full-video temporal context.
- Used OpenCV for evidence clip slicing rather than subprocess ffmpeg to minimize external dependencies.

**MVP Definition of Done status:**
- Implemented Accident Event detection (MVP §5.8).
- Implemented Confidence & Severity output (MVP §5.8).
- Visually verifiable output pipeline completed.

---

### Phase 8 — Evaluation & MVP Completion
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
- YOLO model: `yolo11n.pt` (Phase 1)
- Tracker: ByteTrack (Phase 1)
- Severity thresholds: (to be decided at Phase 6)

---

## Performance Log

| Phase | Video | Resolution | FPS | Frames | Pipeline Time (s) | Per-Frame (ms) | GPU Mem (MB) |
|---|---|---|---|---|---|---|---|
| 2 | `-2UPLUV7JLg_00.mp4` | Unknown | 7.20 | 299 | 41.54 | ~138 | CPU |
| 2 | `-6SQSDj8cYU_00.mp4` | Unknown | 6.60 | 450 | 68.23 | ~151 | CPU |
| 2 | `022uvRkRJ8E_00.mp4` | Unknown | 7.15 | 596 | 83.38 | ~140 | CPU |
