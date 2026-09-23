# RADS -- Implementation Tracker

Companion to [IMPLEMENTATION_PLAN.md](file:///e:/Rads/docs/architecture/IMPLEMENTATION_PLAN.md).

Updated after every implementation phase.

This document contains two progress tables: the historical MVP phases (0-8, completed) and the current Runtime v1 phases. All historical records from the MVP phase are preserved unmodified below.

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
| 6 | Severity Heuristic | Claimed Completed 2026-09-08 — CORRECTED 2026-09-20 to Incomplete, re-verification pending | 2026-09-08 | 2026-09-08 (claim withdrawn 2026-09-20) | Rule-based heuristic based on objects and class. See Correction Record C1. |
| 7 | Visualization Finalization | Claimed Completed 2026-09-08 — CORRECTED 2026-09-20 to Incomplete, re-verification pending | 2026-09-08 | 2026-09-08 (claim withdrawn 2026-09-20) | Two-pass rendering and clip extraction. See Correction Record C2. |
| 8 | Evaluation & MVP Completion | Claimed Completed 2026-09-13 — CORRECTED 2026-09-20 to Incomplete, re-verification pending | 2026-09-13 | 2026-09-13 (claim withdrawn 2026-09-20) | 500-video evaluation harness built. See Correction Record C3 and C4. |

The Status cells for Phases 6, 7 and 8 were amended on 2026-09-20. The original claim is preserved verbatim in each cell and the Phase Completion Records below are unmodified, per the preservation mandate in `IMPLEMENTATION_AUDIT_AND_ORDER.md` section 3, which applies to the project's own records as well as to its code.

---

## Runtime v1 Progress

Phases defined in [IMPLEMENTATION_PLAN.md](file:///e:/Rads/docs/architecture/IMPLEMENTATION_PLAN.md). Detailed requirements in [PROD.md](file:///e:/Rads/docs/architecture/PROD.md).

| Phase | Name | Status | Date Started | Date Completed | Notes |
|---|---|---|---|---|---|
| 0 | Repository Stabilization | Not Started | -- | -- | Restore deleted files, verify tests, move model weights |
| 1A | Core Library Namespace | Not Started | -- | -- | `rads/core/__init__.py` re-exports |
| 1B | Configuration Extension | Not Started | -- | -- | Add runtime/device/api sections to config |
| 1C | Environment Variable Resolver | Not Started | -- | -- | `rads/config/env_resolver.py` |
| 2A | Source Abstraction | Not Started | -- | -- | `rads/runtime/source.py` (file/webcam/RTSP) |
| 2B | Frame Processor | Not Started | -- | -- | `rads/runtime/frame_processor.py` (streaming per-frame pipeline) |
| 2C | Runtime Engine | Not Started | -- | -- | `rads/runtime/engine.py` (main loop) |
| 2D | Health and Signal Handling | Not Started | -- | -- | `rads/runtime/health.py` |
| 2E | Event Lifecycle Manager | Not Started | -- | -- | `rads/runtime/event_lifecycle.py` |
| 2F | Runtime Entry Point | Not Started | -- | -- | `rads_runtime.py` CLI |
| 3A | API Server and Health Endpoint | Not Started | -- | -- | FastAPI server with `/health` |
| 3B | Event Endpoints | Not Started | -- | -- | `/events`, `/events/latest`, `/events/{id}` |
| 3C | WebSocket Event Stream | Not Started | -- | -- | `/ws/events` |
| 4A | Dockerfile and Build | Not Started | -- | -- | `Dockerfile`, `.dockerignore` |
| 4B | Docker Compose and Deployment | Not Started | -- | -- | `docker-compose.yml`, GPU override |
| 5 | Integration Testing | Not Started | -- | -- | End-to-end verification |

---

## Historical MVP Phase Completion Records

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
**Date:** 2026-09-13
**Files created/modified:**
- `rads/evaluation/evaluator.py`
- `rads/evaluation/baseline_comparison.py`
- `rads/config/create_splits.py`
- `rads/config/picek_500_split.csv`
- `rads/config/p02_test_split.csv`

**Verification outcome:**
- [x] Evaluator supports checkpoint recovery via JSONL: PASS
- [x] Baseline comparison calculates classification metrics using scikit-learn: PASS
- [x] P02 test set and held-out 500-video set generated strictly: PASS

**Blockers / Deviations:**
- The MVP pipeline is completely frozen. Evaluation proceeds on the pre-existing 74-video P02 test split (for apples-to-apples baseline comparison) and a newly isolated 500-video held-out test split (for final evaluation).

**MVP Definition of Done status:**
- Implemented robust progress tracking and evaluation harness (satisfies MVP evaluation readiness).

---

## Correction Record — 2026-09-20

Added 2026-09-20. This section amends status claims made above. Nothing above is deleted. Where a correction contradicts an earlier record, the earlier record stands as the historical claim and this section states what was verified against it.

Authority: `docs/architecture/MASTER_SPEC.md` section 26 requires that a conflict between an implementation and the specification be identified rather than silently ignored. `docs/architecture/IMPLEMENTATION_AUDIT_AND_ORDER.md` section 3 forbids deletion or rewriting of existing records.

### C1 — Phase 6 (Severity Heuristic)

Claimed Completed 2026-09-08. Corrected to Incomplete, re-verification pending.

`rads/severity/severity_engine.py` scored only object count and vulnerable-class presence and returned a bare severity string. It produced no numeric score, no evidence list, no relative velocity, no deceleration and no post-impact displacement, all of which the Phase 6 contract requires.

Internal contradiction within this document: the Configuration Snapshot below still reads "Severity thresholds: (to be decided at Phase 6)". A phase cannot be Completed while its own configuration snapshot records its central parameters as undecided.

A rewrite of the severity engine is in progress at the time of writing. This record does not assert that the rewrite is correct or complete. Status is corrected to Incomplete and the phase requires re-verification against the Phase 6 contract once the rewrite lands.

### C2 — Phase 7 (Visualization Finalization)

Claimed Completed 2026-09-08. Corrected to Incomplete, re-verification pending.

The annotated output lacked the impact timestamp overlay. `docs/architecture/TECH_STACK.md` section 16 lists the accident event marker among the required overlays and its worked example renders `Time: 00:07.4` alongside `ACCIDENT DETECTED` and `Severity: HIGH`. The optional end-of-video summary frame was also absent.

### C3 — Phase 8 (Evaluation and MVP Completion)

Claimed Completed 2026-09-13. Corrected to Incomplete, re-verification pending.

Phase 8's own stated verification criteria, as recorded in `IMPLEMENTATION_AUDIT_AND_ORDER.md` Phase 8, fail as follows.

| Criterion | Outcome | Basis |
|---|---|---|
| Record all results in `IMPLEMENTATION_TRACKER.md` | FAIL | No metrics appeared anywhere in this document before the Pre-Fix Metrics section added below on 2026-09-20 |
| Side-by-side P02 comparison table | FAIL | `rads/evaluation/baseline_comparison.py` printed the literal string `TBD` for every baseline value, so no comparison table existed |
| Document at least 5 failure cases with the failing stage named | FAIL | No such document existed. Created 2026-09-20 as `rads/evaluation/failure_analysis.md` |
| Verify dataset split integrity, source-video separation | FAIL | `rads/config/create_splits.py` extracted source ids incorrectly, so the separation check could not have passed. See C6 |

### C4 — Documented conflict: Phase 8 was placed on hold and then implemented

`docs/NEW.md` line 2 states in capital letters `PHASE 8 MUST REMAIN ON HOLD`. The same document states at line 8 `DO NOT implement Phase 8.`, at line 10 `DO NOT create the Phase 8 evaluator.`, at line 161 `Do not modify the Phase 8 plan or Phase 8 files.`, at line 237 `Do NOT mark Phase 8 as complete.` and at line 241 lists `Phase 8 evaluator` among the excluded items.

Phase 8 was nonetheless implemented and recorded above as Completed on 2026-09-13, with `rads/evaluation/evaluator.py` among its created files.

This is recorded as a factual conflict between two project records, per MASTER_SPEC section 26. No attribution of cause or responsibility is made here. The conflict is not resolved by this entry; it is identified. Whichever of the two records is to be superseded should be superseded explicitly rather than by silence.

### C5 — Version control integrity, verified 2026-09-20

The `.gitignore` rules `models/` and `output/` were unanchored. A git ignore pattern containing no leading or embedded slash matches at any directory depth, so `models/` matched `training/models/` and `output/` matched `rads/output/`.

Effect established:

- `models/` was actively hiding `training/models/temporal_model.py`, `classification_model.py`, `model_factory.py` and `__init__.py`. `IMPLEMENTATION_AUDIT_AND_ORDER.md` section 3.1 lists the ResNet18+GRU temporal model and the ResNet18 classification model as PRESERVE — Required, as the MASTER_SPEC section 15 baseline reference. They were untracked.
- `output/` was a latent hazard rather than an active loss. `rads/output/__init__.py`, `event_schema.py` and `visualizer.py` were already tracked, and git does not retroactively ignore tracked files, so nothing was lost. Any newly added file under `rads/output/` would have been silently ignored.

All four affected rules are now root-anchored. Verification run 2026-09-20, real output:

```
### git check-ignore -v --no-index (preserved files)
exit=1 (1 = no path ignored)

### git status --porcelain -- rads/output training/models
 M rads/output/event_schema.py
 M rads/output/visualizer.py
?? training/models/

### gitignore rules
135: /Outputs/
149: /models/
221: /output/
222: /output_results/
```

The `check-ignore` invocation was `git check-ignore -v --no-index rads/output/visualizer.py rads/output/event_schema.py training/models/temporal_model.py training/models/classification_model.py training/models/model_factory.py training/models/__init__.py`. Exit status 1 with no output means git matched no ignore rule against any of the six paths. `git ls-files rads/output` returns `rads/output/__init__.py`, `rads/output/event_schema.py`, `rads/output/visualizer.py`, confirming those three were already tracked and that the ` M` status is a content modification, not a new addition. `training/models/` is reported as untracked-new, which is the state that makes the four preserved files addable.

`training/models/` on disk contains `classification_model.py`, `model_factory.py`, `temporal_model.py`, `__init__.py` and a `__pycache__` directory.

### C6 — Rebuilt evaluation split `rads/config/picek_heldout_split_v2.csv`

Written by `rads/config/create_splits.py` under seed 42. Figures below are from `rads/evaluation/split_integrity_report.md` and from direct re-reading of the CSV on 2026-09-20.

Re-verified directly from the CSV:

| Check | Value |
|---|---|
| Rows | 500 |
| Columns | `video_id`, `source_video_id`, `binary_label`, `clip_role`, `processed_path`, `split_in_distribution` |
| Class balance | 250 label 1, 250 label 0 |
| Clip roles | 250 `positive_accident`, 250 `negative_pre_accident` |
| Distinct `video_id` | 500, zero duplicates |
| Distinct `source_video_id` | 250 |
| `processed_path` entries missing on disk | 0 |

Pool counts as recorded in `split_integrity_report.md` section 5: 2,027 positive clips and 1,530 negative clips; 1,789 unique sources in the positive directory and 1,428 in the negative directory; 1,428 sources present in both classes and therefore pairable; 30 pairable sources removed by the exclusion rules; 1,398 eligible pairs after exclusions; 250 pairs selected. Two consecutive runs produced byte-identical files, MD5 `C6EA958D93F6F857CAF8E7281D2977CB`.

Pairing rationale, reproduced from `split_integrity_report.md` section 3. Each selected source contributes exactly one pair, its positive clip and the negative clip sharing the same basename. This was chosen because MVP.md section 13 and MASTER_SPEC.md section 14 require all clips from one source to stay within a single split, which pairing satisfies, and because it makes every negative a hard negative from the same camera, scene and lighting as its positive, which is the hard-negative category MVP.md section 12 calls for. The rejected alternative was disjoint sources per class, which would reintroduce appearance-based separability between accident and normal and let scene identity rather than motion evidence drive the decision.

Leakage claim, stated exactly as the report states it. The claim the split supports is **"held out from threshold tuning"**, not "held out from P02". Four of the 250 selected sources also appear in `rads/config/p02_test_split.csv`: `0puo8kJOlmU`, `CnsDyfXdf8s`, `QB42gU5X634`, `aF2Hbp8C23A`. They were deliberately retained and reported rather than excluded, because no P02 source was used to tune any threshold. The consequence, which the report states and which is repeated here rather than buried: the two result sets are not statistically independent at the source level, and because each pair shares a source video, any metric assuming 500 independent samples overstates the effective sample size — the effective unit is the 250 pairs. A further 81 of the 250 selected sources also appear in the previous `picek_500_split.csv`, which was never used for tuning either.

Defects in the superseded split `rads/config/picek_500_split.csv`, which is retained unmodified as evidence: 500 rows, 454 distinct sources under the corrected extraction rule, 38 sources contributing more than one clip, and 29 duplicate `video_id` values spanning 58 rows. Re-verified 2026-09-20 by direct read: 500 rows, 471 distinct `video_id`, 29 duplicated ids over 58 rows. Because `rads/evaluation/evaluator.py` keys checkpoint recovery on `video_id`, a resumed run would have silently skipped the second occurrence of each duplicated id.

### C7 — Spec correction: wrong P02 baseline artifact cited in the audit document

`docs/architecture/IMPLEMENTATION_AUDIT_AND_ORDER.md` line 504 instructs, under Phase 8, that `rads/evaluation/baseline_comparison.py` "Loads P02 results from `training/p02_prediction.json`".

That reference is wrong. Verified 2026-09-20 by direct read: `training/p02_prediction.json` is a JSON list of length 1. Its single entry is for `E:\Rads\Datasets\processed\picek_sorted\trimmed\positive\real\jD8ybdMZOU8_00.mp4` with `predicted_class: accident`, `confidence: 0.5972` and 8 sampled frames. It is a single-video demo dump, not test-set predictions, and cannot support a 74-row comparison.

The correct artifacts, both verified present and read on 2026-09-20:

| Artifact | Contents verified |
|---|---|
| `training/outputs/2026-08-31_19-55-16/predictions/test_predictions.json` | Keys `predictions`, `targets`, `confidences`, `class_names`, `label_mode`. `predictions`, `targets` and `confidences` each have length 74. `class_names` has length 2, `label_mode` is `binary` |
| `training/outputs/2026-08-31_19-55-16/metrics/test_metrics.json` | `test/top1_accuracy` 0.4594594594594595, `test/macro_f1` 0.40705128205128205, `test/balanced_accuracy` 0.4594594594594595, `test/f1_accident` 0.5833333333333334, `test/f1_normal` 0.23076923076923078, `test/precision_accident` 0.4745762711864407, `test/recall_accident` 0.7567567567567568, `test/auroc` 0.5346968590211834, `test/confusion_matrix` [[6, 31], [9, 28]] |

Both paths sit under `training/outputs/`, which is gitignored, so they are not present on a clean clone. `baseline_comparison.py` already falls back to the values recorded in MASTER_SPEC section 15 and MVP section 15 in that case and states which source it used.

Per MASTER_SPEC section 26 this conflict is identified here and **not** patched in the audit document. `IMPLEMENTATION_AUDIT_AND_ORDER.md` is unmodified.

---

## Phase 8 Reporting — Pre-Fix Metrics

Added 2026-09-20. **These are PRE-FIX numbers.** They were produced by evaluation runs that predate the fixes to the tracker-state bug B1, where one `Tracker` instance with `persist=True` was reused across every video in a run so track IDs and track state leaked from one video into the next, and to the split-integrity bug B3, where `extract_source_id` returned wrong source ids and duplicate `video_id` values were emitted. They are recorded as the baseline against which the post-fix re-run is compared. They are not the MVP result.

All figures in this section were recomputed from the JSONL files on 2026-09-20 using `compute_metrics` from `rads/evaluation/baseline_comparison.py`, with the confusion matrix independently recounted by direct iteration as a cross-check. No figure here is carried over from an earlier report.

### Result files

| File | Rows | Ground truth balance | Predictions |
|---|---|---|---|
| `rads/evaluation/picek_500_results.jsonl` | 500 | 250 accident, 250 normal | 178 accident, 322 normal |
| `rads/evaluation/p02_test_results.jsonl` | 74 | 37 accident, 37 normal | 27 accident, 47 normal |

Neither file contains a `status` field on any row, so error rows are indistinguishable from completed rows. This is bug B5 and it means the row counts above cannot be read as "500 videos processed successfully", only as "500 rows written".

### Confusion matrices, PRE-FIX

| File | TN | FP | FN | TP | Total |
|---|---|---|---|---|---|
| `picek_500_results.jsonl` | 187 | 63 | 135 | 115 | 500 |
| `p02_test_results.jsonl` | 26 | 11 | 21 | 16 | 74 |

### Metrics, PRE-FIX

| Metric | `picek_500_results.jsonl` | `p02_test_results.jsonl` |
|---|---|---|
| Accuracy | 0.6040 | 0.5676 |
| Balanced accuracy | 0.6040 | 0.5676 |
| Precision, accident | 0.6461 | 0.5926 |
| Recall, accident | 0.4600 | 0.4324 |
| F1, accident | 0.5374 | 0.5000 |
| Precision, normal | 0.5807 | 0.5532 |
| Recall, normal | 0.7480 | 0.7027 |
| F1, normal | 0.6538 | 0.6190 |
| Macro F1 | 0.5956 | 0.5595 |
| FPR | 0.2520 | 0.2973 |
| FNR | 0.5400 | 0.5676 |
| AUROC over `confidence` | 0.6076 | 0.5749 |

Balanced accuracy equals accuracy in both files because both splits are exactly class-balanced.

### AUROC caveat

AUROC over the pre-fix `confidence` field is heavily tie-broken and should not be read as a ranking quality measure.

| File | Distinct confidence values | Rows at 0.0 | Rows at 1.0 | Share of rows at 0.0 or 1.0 |
|---|---|---|---|---|
| `picek_500_results.jsonl` | 48 | 172 | 135 | 307 of 500, 61.4 percent |
| `p02_test_results.jsonl` | 18 | 23 | 23 | 46 of 74, 62.2 percent |

Full frequency table, `p02_test_results.jsonl`: 0.0 x23, 0.02 x1, 0.04 x5, 0.06 x1, 0.09 x1, 0.1 x1, 0.11 x1, 0.13 x1, 0.15 x3, 0.17 x2, 0.27 x1, 0.3 x1, 0.4 x6, 0.5 x1, 0.52 x1, 0.74 x1, 0.9 x1, 1.0 x23.

Full frequency table, `picek_500_results.jsonl`: 0.0 x172, 0.01 x1, 0.02 x1, 0.03 x2, 0.04 x17, 0.06 x2, 0.08 x2, 0.09 x4, 0.11 x1, 0.12 x2, 0.13 x3, 0.15 x30, 0.17 x2, 0.19 x3, 0.21 x5, 0.23 x1, 0.24 x5, 0.25 x3, 0.26 x2, 0.27 x1, 0.28 x3, 0.29 x2, 0.3 x9, 0.31 x1, 0.35 x1, 0.39 x2, 0.4 x42, 0.44 x1, 0.46 x1, 0.49 x1, 0.5 x7, 0.56 x1, 0.58 x1, 0.59 x1, 0.6 x1, 0.66 x1, 0.69 x1, 0.7 x18, 0.74 x1, 0.77 x1, 0.8 x1, 0.83 x1, 0.86 x1, 0.9 x4, 0.95 x1, 0.96 x1, 0.97 x1, 1.0 x135.

Correction to an earlier planning note, recorded because it affects how this caveat is read: the claim that the pre-fix confidence field takes only the three values 0.04, 0.74 and 1.0 is not supported by the files. It takes 18 distinct values in the P02 file and 48 in the held-out file. The concentration at the two endpoints is real; the three-value claim is not.

### Severity distribution over predicted accidents, PRE-FIX

No severity ground truth exists. Per MVP.md section 21 these counts describe the engine's output and carry no correctness signal.

| File | Predicted accidents | HIGH | MEDIUM | LOW |
|---|---|---|---|---|
| `picek_500_results.jsonl` | 178 | 20 | 131 | 27 |
| `p02_test_results.jsonl` | 27 | 2 | 23 | 2 |

Every row predicted normal carries `severity: null`; no predicted accident carries a null severity.

### Event localization availability, PRE-FIX

| File | Rows with non-null `event.impact_time` | Rows predicted accident |
|---|---|---|
| `picek_500_results.jsonl` | 178 | 178 |
| `p02_test_results.jsonl` | 27 | 27 |

The two columns are equal in both files: an impact time is emitted for exactly the rows predicted accident and for no others. Impact-time error against the `accident_time` column of `p02_test_split.csv` has not been computed and is PENDING.

### Data integrity of the pre-fix result files

| Check | `picek_500_results.jsonl` | `p02_test_results.jsonl` |
|---|---|---|
| Distinct `video_id` | 471 of 500 rows | 74 of 74 rows |
| Duplicate `video_id` values | 29 ids spanning 58 rows | 0 |
| Rows with `num_tracks: 0` | 500 of 500 | 74 of 74 |
| Rows with a `status` field | 0 | 0 |

The 29 duplicated `video_id` values in `picek_500_results.jsonl` match the 29 duplicates independently counted in its source `rads/config/picek_500_split.csv`, which also has 500 rows and 471 distinct ids. The duplication originates in the split, not in the evaluator.

`num_tracks` is zero on every row of both files. This is bug B2: the evaluator read `track_summaries` while `rads/output/event_schema.py` emitted `tracks_summary`. No track count from either pre-fix file is usable, and per-clip stage attribution cannot be recovered from the JSONL alone.

### Failure cases, PRE-FIX

Eight stage-attributed failure cases plus the B9 limitation are documented in `rads/evaluation/failure_analysis.md`, created 2026-09-20 from the forensic dumps under `rads/evaluation/experiments/`. That document satisfies the MVP.md section 28 requirement to name the responsible stage and to distinguish clips that produced no interaction candidate from clips where candidates existed and were scored below threshold. All eight cases are marked for re-confirmation after the post-fix batch.

---

## Phase 8 Reporting — Post-Fix Re-Run

Added 2026-09-20 as a scaffold. Every cell below is PENDING. No cell is blank; a blank cell must not be read as a zero or as a pass.

### Provenance

| Item | Value |
|---|---|
| Commit SHA at launch | PENDING |
| Launch command, P02 | PENDING |
| Launch command, held-out v2 | PENDING |
| Start time | PENDING |
| Effective config snapshot path | PENDING |
| P02 result file | PENDING (planned `rads/evaluation/p02_test_results_v2.jsonl`) |
| Held-out result file | PENDING (planned `rads/evaluation/picek_heldout_results_v2.jsonl`) |
| Rows written vs rows expected, P02 | PENDING |
| Rows written vs rows expected, held-out | PENDING |
| Rows with `status: error` | PENDING |

### Confusion matrices, POST-FIX

| File | TN | FP | FN | TP | Total |
|---|---|---|---|---|---|
| `p02_test_results_v2.jsonl` | PENDING | PENDING | PENDING | PENDING | PENDING |
| `picek_heldout_results_v2.jsonl` | PENDING | PENDING | PENDING | PENDING | PENDING |

### Metrics, POST-FIX

| Metric | `p02_test_results_v2.jsonl` | `picek_heldout_results_v2.jsonl` |
|---|---|---|
| Accuracy | PENDING | PENDING |
| Balanced accuracy | PENDING | PENDING |
| Precision, accident | PENDING | PENDING |
| Recall, accident | PENDING | PENDING |
| F1, accident | PENDING | PENDING |
| Precision, normal | PENDING | PENDING |
| Recall, normal | PENDING | PENDING |
| F1, normal | PENDING | PENDING |
| Macro F1 | PENDING | PENDING |
| FPR | PENDING | PENDING |
| FNR | PENDING | PENDING |
| AUROC | PENDING | PENDING |
| Distinct confidence values | PENDING | PENDING |

### P02 side-by-side comparison, POST-FIX

Baseline column is read from `training/outputs/2026-08-31_19-55-16/metrics/test_metrics.json`, with the MASTER_SPEC section 15 / MVP section 15 values as the clean-clone fallback. See C7.

| Metric | RADS POST-FIX | ResNet18+GRU baseline |
|---|---|---|
| Accuracy | PENDING | PENDING |
| Balanced accuracy | PENDING | PENDING |
| Precision, accident | PENDING | PENDING |
| Recall, accident | PENDING | PENDING |
| F1, accident | PENDING | PENDING |
| F1, normal | PENDING | PENDING |
| Macro F1 | PENDING | PENDING |
| FPR | PENDING | PENDING |
| FNR | PENDING | PENDING |
| AUROC | PENDING | PENDING |
| Confusion matrix | PENDING | PENDING |
| Baseline source actually used | PENDING | PENDING |
| Index-alignment check result | PENDING | PENDING |

### Severity distribution, POST-FIX

| File | Predicted accidents | HIGH | MEDIUM | LOW |
|---|---|---|---|---|
| `p02_test_results_v2.jsonl` | PENDING | PENDING | PENDING | PENDING |
| `picek_heldout_results_v2.jsonl` | PENDING | PENDING | PENDING | PENDING |

### Event localization, POST-FIX

| Quantity | Value |
|---|---|
| P02 rows with non-null `impact_time` | PENDING |
| P02 rows with a ground-truth `accident_time` | PENDING |
| Mean absolute impact-time error, descriptive only | PENDING |
| Median absolute impact-time error, descriptive only | PENDING |

### Failure cases, POST-FIX re-confirmation

| Case in `failure_analysis.md` | Re-confirmed | Post-fix outcome |
|---|---|---|
| 1 `-2UPLUV7JLg_00` | PENDING | PENDING |
| 2 `-9oifpjUxxM_00` | PENDING | PENDING |
| 3 `-FQxK6HdxNU_00` | PENDING | PENDING |
| 4 `-NgnSm_oEB4_00` | PENDING | PENDING |
| 5 `-SNFUobKjoM_00` | PENDING | PENDING |
| 6 `-PpBteU0p3Q_00` | PENDING | PENDING |
| 7 `-dmYsQc-odI_00` | PENDING | PENDING |
| 8 single-vehicle, B9 | PENDING | PENDING |

---

## MVP Definition of Done — MVP.md section 36

Added 2026-09-20. Ticked only where evidence exists in the working tree today. Everything depending on the post-fix re-run is PENDING and is marked so explicitly. No item is left blank.

### Pipeline

| Item | Status | Evidence |
|---|---|---|
| Video input works | PENDING | Not re-verified by this pass. Phase 1 record above claims PASS; no run was performed to confirm |
| Frames are processed correctly | PENDING | Bug B15, `processed_frames` computed after the `with` block exits, is listed as open in the completion plan |
| Object detection works | PENDING | Not re-verified by this pass |
| Tracking works | PENDING | Not re-verified by this pass. Bug B1, shared tracker state across videos, invalidates the pre-fix runs |
| Persistent IDs are visible | PENDING | Not re-verified by this pass |
| Track histories are stored | PARTIAL | Track summaries with `first_frame`, `last_frame`, `frame_count`, `lifespan_frames` are present in every forensic dump, for example `frame_skip_comparison/forensics/FN_-2UPLUV7JLg_00.json` with 8 tracks. However `num_tracks` is 0 on all 500 and all 74 recorded result rows because of bug B2, so the JSONL does not evidence this |
| Trajectories are generated | YES | `motion_features` with `total_displacement` and `net_displacement` per track id in all four `frame_skip_comparison` forensic dumps |
| Motion features are calculated | YES | `mean_velocity`, `max_velocity`, `mean_acceleration`, `max_acceleration`, `mean_direction` per track in the same dumps |
| Object interactions are identified | YES | `interaction_candidates` with `min_distance`, `peak_iou`, `max_relative_velocity` and `evidence_list`; 46 candidates in `FN_-FQxK6HdxNU_00.json`, 38 in `FP_-NgnSm_oEB4_00.json`, 36 in `FP_-SNFUobKjoM_00.json` |
| Accident events can be detected | YES | 178 of 500 rows in `picek_500_results.jsonl` and 27 of 74 in `p02_test_results.jsonl` predicted accident, with 115 and 16 of those correct respectively |
| Event timing is estimated | YES | Non-null `event.impact_time` on all 178 and all 27 predicted-accident rows; `FP_-dmYsQc-odI_00.json` records `impact_time` 3.4856 s |
| Involved objects are identified | PARTIAL | `involved_object_ids` is populated on positive decisions, for example `[49, 25, 7]` in `FP_-dmYsQc-odI_00.json`. In the pre-fix artifacts it is over-inclusive: `FP_-NgnSm_oEB4_00.json` names 10 of 13 tracks. That was bug B11. Clustering was tightened in code on 2026-09-20, behind config flags; the statement above scopes to the recorded artifacts, and the effect on real output is unmeasured until the re-run |
| Severity is estimated | PARTIAL | Severity is emitted on every predicted accident: 20 HIGH, 131 MEDIUM, 27 LOW over the 178 held-out positives. In those recorded artifacts it is a bare label with no score and no evidence list, which is C1 above. `estimate_severity` was rewritten in code on 2026-09-20 to return `{severity, score, evidence}`; no evaluation artifact in the tree carries that shape yet. Re-verification PENDING |
| Results are visualized | PENDING | Phase 7 corrected to Incomplete, see C2. Impact timestamp overlay missing |
| Structured output is produced | PARTIAL | The recorded JSONL rows carry `video_id`, `ground_truth`, `prediction`, `confidence`, `severity`, `event.{start_time,impact_time,end_time}`, `num_tracks`. In those rows `num_tracks` is always 0 (B2), there is no `status` field (B5), and `evidence_list` is not carried into the saved result (B7). All three were fixed in code on 2026-09-20; the statements above scope to the pre-fix artifacts, which are the only evaluation output in the tree. Re-verification PENDING |

### Evaluation

| Item | Status | Evidence |
|---|---|---|
| Dataset split is verified | YES, for `picek_heldout_split_v2.csv` only | 500 rows, 250/250 class balance, 250/250 clip roles, 500 distinct `video_id`, 250 distinct sources, 0 missing `processed_path`, all re-read directly on 2026-09-20. See C6. The superseded `picek_500_split.csv` fails this check with 29 duplicate ids over 58 rows |
| No source-video leakage exists | QUALIFIED | Source-video separation as defined in MVP.md section 13 holds in v2: each of the 250 sources appears in exactly one split with both its clips. The published claim is "held out from threshold tuning", not "held out from P02". Four sources overlap P02 and are named in C6. The 250 pairs, not the 500 rows, are the effective sample unit |
| Baseline results are recorded | YES | `training/outputs/2026-08-31_19-55-16/metrics/test_metrics.json`, read 2026-09-20, values quoted in C7. Also recorded in MASTER_SPEC section 15 and MVP section 15 for clean clones |
| New system results are recorded | PRE-FIX YES, POST-FIX PENDING | Pre-Fix Metrics section above |
| Confusion matrix is available | PRE-FIX YES, POST-FIX PENDING | 187/63/135/115 for the held-out set and 26/11/21/16 for P02, recomputed 2026-09-20 |
| Precision/recall/F1 are available | PRE-FIX YES, POST-FIX PENDING | Metrics table above, both classes |
| AUROC is available where applicable | PRE-FIX YES WITH CAVEAT, POST-FIX PENDING | 0.6076 and 0.5749. 61.4 percent and 62.2 percent of rows sit at confidence 0.0 or 1.0, so the ranking is largely tie-broken. See the AUROC caveat above |
| False-positive behavior is analyzed | PRE-FIX YES, POST-FIX PENDING | FPR 0.2520 held-out and 0.2973 P02. Four false-positive cases attributed to a stage with numeric evidence in `rads/evaluation/failure_analysis.md`, cases 4 to 7 |
| Representative failure cases are documented | PRE-FIX YES, POST-FIX PENDING | `rads/evaluation/failure_analysis.md`, eight cases plus the B9 limitation, each naming the responsible stage, with category A / B / C separating no-candidate from scored-and-rejected per MVP.md section 28 |

### Engineering

| Item | Status | Evidence |
|---|---|---|
| Configuration is reproducible | PARTIAL | `rads/config/create_splits.py` reproduces `picek_heldout_split_v2.csv` byte-identically under seed 42, MD5 `C6EA958D93F6F857CAF8E7281D2977CB`. The pipeline-side config snapshot for the post-fix run is PENDING |
| Model/checkpoint provenance is recorded | PARTIAL | YOLO model `yolo11n.pt` and ByteTrack recorded in the Configuration Snapshot below. Baseline checkpoint provenance is the `training/outputs/2026-08-31_19-55-16/` run directory. Commit SHA for the post-fix run is PENDING |
| Major components are modular | YES | Separate packages under `rads/`: `config`, `detection`, `tracking`, `motion`, `interaction`, `reasoning`, `severity`, `output`, `pipeline`, `evaluation`, `tests` |
| Pipeline can be run from a clean configuration | PENDING | Clean-clone smoke test not performed by this pass |
| Existing dashboard/alert integration is not broken | VACUOUSLY SATISFIED | No dashboard and no Telegram code exists in the repository. `IMPLEMENTATION_AUDIT_AND_ORDER.md` PART 5 - FINAL CONSISTENCY CHECK, subsection "Remaining Spec Contradictions", resolves this: "MVP section 36 says 'Existing dashboard/alert integration is not broken' / There is nothing to break / This DoD item is satisfied vacuously. Note it in tracker." Noted here as instructed. Section 1.4 of the same document states that no such code exists and classifies both as P1 new development, not integration with existing code |

---

## Configuration Snapshot

Record key configuration decisions here:
- YOLO model: `yolo11n.pt` (Phase 1)
- Tracker: ByteTrack (Phase 1)
- Severity thresholds: (to be decided at Phase 6)
  - Amendment 2026-09-20: the line above is retained as the historical record. It was never filled in, which contradicts the Phase 6 Completed claim. See Correction Record C1. Actual severity weights and LOW/MEDIUM/HIGH cut-offs: PENDING, to be recorded from `rads/config/pipeline_config.yaml` once the severity rewrite lands.
  - Amendment 2026-09-21: the severity rewrite has landed, so the PENDING above is superseded. Read from `rads/config/pipeline_config.yaml` on 2026-09-21: cut-offs `severity.threshold_medium` 2.0 and `severity.threshold_high` 4.5, so a score below 2.0 is LOW, 2.0 up to 4.5 is MEDIUM, and 4.5 or above is HIGH. Weights: `object_count` 1.0 capped at 2.0, `vulnerable_class` 3.0 for `person`, `bicycle` and `motorcycle`, plus a relative-velocity term. `estimate_severity` returns `{severity, score, evidence}`. These values are engineered choices, not fitted ones: no labelled severity ground truth exists in this project and none was used to place them, consistent with MVP.md section 21 and TECH_STACK.md section 14, which specify a heuristic rather than a learned severity model. They are unvalidated and should be treated as such.
- Reasoning accident score threshold: 0.5, read from `rads/config/config_loader.py` on 2026-09-20, which returns `reasoning.accident_score_threshold` with default `0.5`.

---

## Performance Log

| Phase | Video | Resolution | FPS | Frames | Pipeline Time (s) | Per-Frame (ms) | GPU Mem (MB) |
|---|---|---|---|---|---|---|---|
| 2 | `-2UPLUV7JLg_00.mp4` | Unknown | 7.20 | 299 | 41.54 | ~138 | CPU |
| 2 | `-6SQSDj8cYU_00.mp4` | Unknown | 6.60 | 450 | 68.23 | ~151 | CPU |
| 2 | `022uvRkRJ8E_00.mp4` | Unknown | 7.15 | 596 | 83.38 | ~140 | CPU |

---

## Runtime v1 Phase Completion Records

Records will be added here as each runtime phase is completed.

(No phases completed yet.)
