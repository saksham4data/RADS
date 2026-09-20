# RADS — Session Handoff

Written 2026-09-21. Companion to [IMPLEMENTATION_TRACKER.md](file:///e:/Rads/docs/architecture/IMPLEMENTATION_TRACKER.md).

This document exists so the working tree can be committed now and picked up later by someone returning cold. It records what was verified, what was deliberately left undone, and what in the repository is now stale or misleading.

---

## 1. Why the session stopped

The user halted work on 2026-09-21 in order to reconsider the system architecture and the MVP scope before any further implementation. The halt was deliberate and came before the MVP evaluation batch was launched, so the corrected 74-clip P02 re-run and the 500-clip held-out re-run were never executed. Every code fix from this session is therefore in the tree and verified to import, parse and pass its unit tests, but no evaluation numbers were produced from the fixed code. The pre-fix result files remain the only evaluation artifacts in the repository and they are superseded, as recorded in section 5.

---

## 2. State of the working tree

Everything from this session is uncommitted. `HEAD` is unchanged at `b53706b`, dated 2026-09-14. No agent in this session was permitted to run any git write command; no `add`, `commit`, `restore`, `checkout`, `stash`, `clean` or `reset` was executed. `git stash list` is empty. The user commits.

### git status --porcelain -uall, 2026-09-21

```
 M .gitignore
 M docs/architecture/IMPLEMENTATION_TRACKER.md
 M rads/config/config_loader.py
 M rads/config/create_splits.py
 M rads/config/pipeline_config.yaml
 M rads/evaluation/baseline_comparison.py
 M rads/evaluation/dashboard.md
 M rads/evaluation/dashboard_updater.py
 M rads/evaluation/evaluator.py
 M rads/output/event_schema.py
 M rads/output/visualizer.py
 M rads/pipeline/pipeline.py
 M rads/reasoning/accident_reasoner.py
 M rads/severity/severity_engine.py
 M rads/tracking/tracker.py
 M run_pipeline.py
 M scripts/forensic_diagnostic.py
?? docs/architecture/MVP_COMPLETION_PLAN.md
?? docs/architecture/SESSION_HANDOFF.md
?? docs/frontend/DASHBOARD_DESIGN_OUTLINE.md
?? rads/config/picek_heldout_split_v2.csv
?? rads/evaluation/__init__.py
?? rads/evaluation/experiments/mvp_freeze_regression/gate_results.json
?? rads/evaluation/experiments/mvp_freeze_regression/regression_summary.md
?? rads/evaluation/experiments/mvp_freeze_regression/run_gate.py
?? rads/evaluation/failure_analysis.md
?? rads/evaluation/p02_baseline_comparison.md
?? rads/evaluation/split_integrity_report.md
?? rads/tests/__init__.py
?? rads/tests/test_interaction_unit.py
?? rads/tests/test_motion_features.py
?? rads/tests/test_pipeline_isolation.py
?? rads/tests/test_severity.py
?? rads/tests/test_split_integrity.py
?? training/models/__init__.py
?? training/models/classification_model.py
?? training/models/model_factory.py
?? training/models/temporal_model.py
```

Status captured 2026-09-21 after the documentation pass described in sections 4.1 and 5. Three of those entries are from that pass rather than from the implementation work: `docs/architecture/MVP_COMPLETION_PLAN.md` is the phase plan relocated out of the ignored `.cursor/plans/`, `docs/architecture/SESSION_HANDOFF.md` is this file, and `rads/evaluation/dashboard.md` carries only a one-line superseded marker. `docs/architecture/IMPLEMENTATION_TRACKER.md` carries two factual corrections on top of its implementation-session edits.

Git emits `LF will be replaced by CRLF` warnings on the modified text files; this is the repository's existing line-ending configuration on Windows and is not a defect introduced by this session.

---

## 3. What is verified working

All checks below were run on 2026-09-21 against the tree as it stands. No check was inferred.

### 3.1 No process left running

`Get-Process python*` returned nothing and `Get-CimInstance Win32_Process -Filter "Name like '%python%'"` returned an empty result set. No evaluation, no dashboard updater and no pipeline run was left alive. Nothing needed to be terminated.

### 3.2 Unit tests

`python -m unittest discover -s rads/tests -t . -v`

```
Ran 23 tests in 37.982s
OK
```

23 tests, zero failures, zero errors. The five test files are `test_interaction_unit.py` (2 tests), `test_motion_features.py` (5), `test_pipeline_isolation.py` (2), `test_severity.py` (4) and `test_split_integrity.py` (10).

Note for a future reader: `test_pipeline_isolation.py` is not a pure unit test. It runs the real pipeline over two clips, `-PpBteU0p3Q_00.mp4` and `-Qt5bDJNT84_00.mp4`, twice each, in order to prove that a video's result does not depend on the video processed before it. That is why the suite takes roughly 38 seconds and prints pipeline logs. It requires those two clips to be present under `Datasets/processed/picek_sorted/trimmed/positive/real/`.

### 3.3 Module imports

Every package and module changed this session imports cleanly.

| Module | Result |
|---|---|
| `rads.config.config_loader` | OK |
| `rads.video.video_reader` | OK |
| `rads.detection.detector` | OK |
| `rads.tracking.tracker` | OK |
| `rads.motion.trajectory` | OK |
| `rads.motion.motion_features` | OK |
| `rads.interaction.pairwise` | OK |
| `rads.interaction.interaction_engine` | OK |
| `rads.reasoning.accident_reasoner` | OK |
| `rads.severity.severity_engine` | OK |
| `rads.output.event_schema` | OK |
| `rads.output.visualizer` | OK |
| `rads.pipeline.pipeline` | OK |
| `rads.evaluation.evaluator` | OK |
| `rads.evaluation.baseline_comparison` | OK |
| `rads.evaluation.dashboard_updater` | OK |
| `rads.config.create_splits` | OK |

Reported failures: 0.

### 3.4 CLI entry points

All three parse their arguments and exit 0 on `--help`.

| Entry point | Accepted flags |
|---|---|
| `run_pipeline.py` | `--video`, `--config`, `--output`, `--visualize`, `--output-video` |
| `rads/evaluation/evaluator.py` | `--csv`, `--output`, `--config`, `--dump-dir` |
| `rads/evaluation/baseline_comparison.py` | `--jsonl`, `--baseline-name`, `--baseline-metrics`, `--baseline-predictions`, `--split-csv`, `--rads-label`, `--note`, `--out` |

The `--dump-dir` flag on the evaluator is the new addition and is optional.

### 3.5 scripts/ compile and import resolution

`py_compile` on all 12 files under `scripts/`: 12 OK, 0 failures.

```
scripts\build_p01_small_binary_dataset.py   OK
scripts\build_p02_binary_dataset.py         OK
scripts\experiment_frame_skip.py            OK
scripts\experiment_regression.py            OK
scripts\forensic_check_tracks.py            OK
scripts\forensic_diagnostic.py              OK
scripts\forensic_fn2_check.py               OK
scripts\picek_negative_extractor.py         OK
scripts\picekl_dataset_maker.py             OK
scripts\replay_e06_to_wandb.py              OK
scripts\smoke_test_p01_pipeline.py          OK
scripts\test_full_pipeline.py               OK
```

Every top-level import across those 12 files was extracted by AST and resolved: 43 distinct modules, 0 unresolvable.

Three scripts have no `if __name__ == "__main__"` guard and their module bodies execute on import: `forensic_check_tracks.py`, `forensic_diagnostic.py`, `forensic_fn2_check.py`. They were verified by compilation and by import resolution of their dependencies rather than by import, because importing `forensic_diagnostic.py` would run the pipeline over four videos. A future reader must not `import` these three; run them as `python scripts/<name>.py`.

### 3.6 scripts/forensic_diagnostic.py against the new severity contract

`estimate_severity` now has signature `(accident_result, track_history, config=None) -> Dict[str, Any]` returning `{severity, score, evidence}`. `forensic_diagnostic.py` calls it correctly and unpacks it correctly:

```
 95:    severity_result = estimate_severity(accident_result, track_history, config)
118:        "severity": severity_result["severity"],
120:            "score": severity_result["score"],
121:            "evidence": severity_result["evidence"]
```

It takes `severity_result["severity"]` where a label is expected. It does not write the whole dict into the `severity` key. It additionally emits a new `severity_detail` object carrying `score` and `evidence`. This is a richer output shape than the pre-contract version produced, and it is intentional: the script was adapted to the new contract rather than broken by it. No change was made. Anyone reading old forensic JSON dumps under `rads/evaluation/experiments/` should expect them to lack `severity_detail`.

`forensic_diagnostic.py` is the only file under `scripts/` that references `estimate_severity`, `get_stub_event_result` or `build_event_result`.

### 3.7 Configuration

`python -c "import yaml; yaml.safe_load(open('rads/config/pipeline_config.yaml'))"` succeeds.

`ConfigLoader('rads/config/pipeline_config.yaml')` exposes 43 public properties. Every one resolves against the YAML without raising. Property failures: 0. Selected values as they stand:

| Property | Value |
|---|---|
| `detector_model` | `yolo11n.pt` |
| `detector_confidence` | 0.25 |
| `tracker_type` | `bytetrack` |
| `frame_skip` | 1 |
| `reasoning_accident_threshold` | 0.5 |
| `reasoning_weight_relative_velocity` | 0.4 |
| `reasoning_weight_iou` | 0.3 |
| `reasoning_weight_overlap_evidence` | 0.2 |
| `reasoning_confidence_bounded_transform` | True |
| `reasoning_confidence_transform_scale` | 0.5 |
| `reasoning_clustering_window_seconds` | 3.0 |
| `reasoning_clustering_max_involved` | 4 |
| `reasoning_relative_velocity_divisor` | 50.0 |
| `severity_threshold_medium` | 2.0 |
| `severity_threshold_high` | 4.5 |
| `severity_vulnerable_classes` | `['person', 'bicycle', 'motorcycle']` |

A repository-wide AST sweep of attribute accesses on objects named `config` or `cfg` across `rads/`, `scripts/` and `run_pipeline.py` found 57 distinct attribute names. All 41 that belong to `ConfigLoader` resolve. The 16 that do not (`dataset_name`, `experiment_id`, `experiment_name`, `frames_per_video`, `image_size`, `label_mode`, `metadata_path`, `monitor_metric`, `num_classes`, `project_name`, `temporal_architecture`, `temporal_enabled`, `temporal_hidden_dim`, `to_dict`, `training_version`, `wandb`) belong to `training.configs.config.TrainingConfig` and `training.utils.config`, used only by `scripts/smoke_test_p01_pipeline.py` and `scripts/replay_e06_to_wandb.py`. They are a different config object, not a stale RADS key.

### 3.8 The gitignore anchoring fix

`git check-ignore -v --no-index rads/output/visualizer.py training/models/temporal_model.py` produces no output and exits 1, meaning neither path is ignored. The rules still exist and still fire at the repository root, confirmed by a positive control:

```
.gitignore:221:/output/    output/foo.txt
.gitignore:149:/models/    models/foo.pt
```

The diff anchors `Outputs/`, `models/`, `output/` and `output_results/` to the root and adds `.cursor/plans/`. That last rule is kept, for ephemeral Cursor files, and the phase plan was moved out from under it; see section 5.

---

## 4. What was deliberately not done

| Item | State | Consequence |
|---|---|---|
| 74-clip P02 evaluation re-run | Not launched | `rads/evaluation/p02_test_results_v2.jsonl` does not exist |
| 500-clip held-out evaluation re-run on `picek_heldout_split_v2.csv` | Not launched | `rads/evaluation/picek_heldout_results_v2.jsonl` does not exist |
| MVP.md section 36 Evaluation items | Open | Every POST-FIX cell in the tracker remains PENDING; no post-fix metric, confusion matrix, AUROC or failure-case re-confirmation exists |
| Frontend | Design only | `docs/frontend/DASHBOARD_DESIGN_OUTLINE.md` exists. There is no frontend code anywhere in the repository, no HTML, no JS, no Streamlit app |
| Bug B9, single-vehicle accident detection | Accepted limitation | The reasoner operates on pairs of tracks only, so a single-vehicle event has no counterpart to pair against and cannot produce an interaction candidate. This is documented, not fixed, and is recorded as case 8 in `rads/evaluation/failure_analysis.md` |

The Phase 3 regression gate did run and its output is preserved under `rads/evaluation/experiments/mvp_freeze_regression/`.

### 4.1 The Phase 4 pre-gate did run, and passed

One thing in the Phase 4 sequence was completed before the halt: step 0, the mandatory 3-clip pre-gate. It ran to completion and passed. The batch itself was never launched.

The run was a 3-clip slice of `rads/config/picek_heldout_split_v2.csv`, containing at least one positive and one negative clip, at `frame_skip` 1, with the new `--dump-dir` option enabled.

| Check | Result |
|---|---|
| Completeness | 3 successful rows against 3 expected from the CSV |
| Rows with `status: error` | 0 |
| `num_tracks` per clip | 5, 4 and 42 |
| Positive clip evidence | The one clip predicted accident carried `evidence_list` of `proximity_and_convergence` and `track_loss` |
| Positive clip severity | MEDIUM at score 3.0485 |
| Per-clip dump shape | All three dumps contained the thirteen top-level keys, including `objects_involved`, `severity_detail`, `kinematics`, `tracks_summary` and `interaction_candidates` |
| Dump sizes | 1.8 KB and 2.2 KB for the two clips with no interaction candidates, 44 KB for the clip with 48 candidates |

The non-zero `num_tracks` is the point of the gate. Every one of the 574 pre-fix records reported `num_tracks` as zero because the evaluator read `track_summaries` while the schema emitted `tracks_summary`. These three rows are empirical proof that the B2 fix landed and that it landed on the rebuilt split, which had never been consumed by the evaluator before.

Extrapolating the dump sizes, a full 500-clip run with `--dump-dir` enabled projects to roughly 10 to 20 MB of per-clip JSON. Plan disk and gitignore accordingly before launching.

The temporary pre-gate output was deleted during cleanup, so the figures above come from the run report rather than from a file still on disk. Nothing in the tree reproduces them.

Consequence for the next session: Phase 4 step 0 can be skipped and the batch launched directly.

---

## 5. Known stale or misleading artifacts

| Artifact | Why it is stale or misleading |
|---|---|
| `rads/evaluation/picek_500_results.jsonl`, 500 rows | Produced before the tracker-state fix (B1), where one `Tracker` with `persist=True` was reused across every video, and before the split-integrity fix (B3). Its source split `picek_500_split.csv` has 29 duplicate `video_id` values over 58 rows. Every row is superseded and must not be compared against a future run |
| `rads/evaluation/p02_test_results.jsonl`, 74 rows | Same pre-B1 provenance. Superseded for the same reason |
| `num_tracks` in both files above | Zero on all 574 rows because of bug B2, where the evaluator read `track_summaries` while the schema emitted `tracks_summary`. No track count in either file is usable |
| `rads/evaluation/dashboard.md` | Generated 2026-09-14 by `dashboard_updater.py` during the pre-fix 500-clip run. It states `Status: Completed` and `500 / 500 videos processed (100.0%)`. That is true of the superseded run and false of the current state of the project. It is also structurally out of date: the current `dashboard_updater.py` writes `Source` and `Error rows` fields that this file does not contain, which is itself the tell that it predates the rewrite. A one-line SUPERSEDED marker was added at the top on 2026-09-21. The file is regenerated wholesale by `dashboard_updater.py`, so that marker disappears the next time the script runs |
| The phase plan, formerly `.cursor/plans/MVP_COMPLETION_PLAN.md` | RESOLVED 2026-09-21. The `.gitignore` change this session added `.cursor/plans/`, so the plan would not have committed and would not have existed in a clean clone, even though section 7 points to it as the resume document. The plan was moved to `docs/architecture/MVP_COMPLETION_PLAN.md`, where `.gitignore` line 200 `!docs/**` explicitly un-ignores it, and `git status` now lists it as untracked. The `.cursor/plans/` ignore rule was deliberately kept for ephemeral Cursor files. Only one copy of the plan exists. Its own Phase 0 steps 1 and 2 still read "Create `.cursor/plans/` and keep this plan there" and "add a Cursor section ignoring `.cursor/plans/`"; that text is left as the historical record of what Phase 0 did and no longer describes where the plan lives |
| `docs/architecture/IMPLEMENTATION_TRACKER.md`, Configuration Snapshot | CORRECTED 2026-09-21. It stated "Actual severity weights and LOW/MEDIUM/HIGH cut-offs: PENDING, to be recorded from `rads/config/pipeline_config.yaml` once the severity rewrite lands." The rewrite has landed. A dated Amendment 2026-09-21 was appended recording the real cut-offs, 2.0 and 4.5, the weights, and the fact that they are engineered choices with no ground-truth validation. The original PENDING line is preserved above it as the historical record |
| `docs/architecture/IMPLEMENTATION_TRACKER.md`, MVP Definition of Done, Pipeline table | CORRECTED 2026-09-21. Three rows described defects fixed in code this session but were written in the present tense about the code rather than about the superseded artifacts: "Severity is estimated", "Structured output is produced" and "Involved objects are identified". Each was reworded to scope its claim to the recorded pre-fix artifacts and to note the code fix separately, without asserting that the pending re-run has happened. All three remain PARTIAL with re-verification pending |
| `docs/architecture/IMPLEMENTATION_TRACKER.md`, POST-FIX sections | Every cell in "Phase 8 Reporting — Post-Fix Re-Run" reads `PENDING`. This is correct and intentional, and the section header says so explicitly. Flagged only so a future reader does not mistake the scaffold for a result set |
| `rads/evaluation/experiments/**/forensics/*.json` | Produced by pre-B1 runs. Track ids, track lifespans and every candidate in them can change after the fix. `rads/evaluation/failure_analysis.md` already states this at the top of the document and marks all eight cases for re-confirmation |

The staleness of the two result JSONL files is stated clearly in four places already: `docs/architecture/IMPLEMENTATION_TRACKER.md` in the "Phase 8 Reporting — Pre-Fix Metrics" preamble, `rads/evaluation/p02_baseline_comparison.md` lines 4 and 11, `rads/evaluation/failure_analysis.md` line 5, and `docs/frontend/DASHBOARD_DESIGN_OUTLINE.md` line 41. It is not stated in `rads/evaluation/dashboard.md`, nor in the JSONL files themselves, which carry no provenance field.

### 5.1 Config key `interaction.convergence_angle_threshold`

Removed from both `rads/config/pipeline_config.yaml` and `rads/config/config_loader.py`. Confirmed that nothing reads it: a repository-wide search for the literal string returns only two hits, both in the phase plan, now `docs/architecture/MVP_COMPLETION_PLAN.md`, where it appears in the historical bug description for B8 and in the instruction to either wire it up or delete it. No code path references it.

`rads/interaction/pairwise.py` still computes a `convergence_angle` field as per-frame evidence data, and `scripts/forensic_diagnostic.py` reads that computed field at line 154. Both are intentional and unrelated to the removed config key.

### 5.2 Tooling blind spot: the Grep tool cannot see rads/output/

This caused real confusion during the session and will cause it again. The agent-facing Grep tool returns no matches for symbols that demonstrably exist in `rads/output/event_schema.py` and `rads/output/visualizer.py`. Verified 2026-09-21:

| Query | Grep tool | Shell `rg` |
|---|---|---|
| `def build_event_result` in `rads/output` | No matches found | `rads/output/event_schema.py:3` |
| `def build_event_result` in `rads` | No matches found | `rads/output/event_schema.py:3` |
| `build_event_result` repository-wide | 5 hits, in `docs/frontend/`, `rads/pipeline/pipeline.py` and the phase plan. The definition site in `rads/output/event_schema.py` is absent | 5 hits including `rads/output/event_schema.py:3` |
| `def ` in `rads/output/visualizer.py` | No matches found | 9 hits, lines 8 through 153 |

Both files are tracked (`git ls-files rads/output` lists `__init__.py`, `event_schema.py`, `visualizer.py`) and both are modified in the current diff. They are not ignored. The files are real; the indexer does not see them. A future reader who greps for a symbol in `rads/output/` will conclude the module is missing that symbol. Use shell `rg` for anything under `rads/output/`.

### 5.3 Residual, low priority

`.gitignore` still carries two unanchored directory rules, `checkpoints/` at line 147 and `weights/` at line 148, which are the same class of defect as the `models/` and `output/` rules that were anchored this session. They cause no current harm: the only `checkpoints` directories in the tree are nested under `training/outputs/`, which is ignored on its own account, and no `weights` directory exists. Left alone.

---

## 6. Suggested commit grouping

Six coherent commits rather than one blob. File lists are taken from the real `git status -uall` above; every modified and untracked path is assigned to exactly one group.

### Group 1 — Version control recovery

Root-anchors the `Outputs/`, `models/`, `output/` and `output_results/` rules so they stop matching nested directories, and restores the four `training/models/*.py` files that the unanchored `models/` rule had been hiding from git since they were written.

```
.gitignore
training/models/__init__.py
training/models/classification_model.py
training/models/model_factory.py
training/models/temporal_model.py
```

Commit this first and alone. It is the only change that alters which files git can see, and reviewing it separately makes the recovered files obvious. The same diff adds `.cursor/plans/` to the ignore list; that rule is intentional and the phase plan was moved out from under it into group 6.

### Group 2 — Pipeline correctness fixes, with their tests

Tracker state isolation, the severity contract rewrite, the event schema rewrite, involved-object clustering and the bounded confidence transform, plus the five test files that pin them.

```
rads/tracking/tracker.py
rads/pipeline/pipeline.py
rads/severity/severity_engine.py
rads/reasoning/accident_reasoner.py
rads/output/event_schema.py
rads/output/visualizer.py
run_pipeline.py
rads/tests/__init__.py
rads/tests/test_interaction_unit.py
rads/tests/test_motion_features.py
rads/tests/test_pipeline_isolation.py
rads/tests/test_severity.py
rads/tests/test_split_integrity.py
```

`rads/tests/test_split_integrity.py` reads `picek_heldout_split_v2.csv` from group 4, so commit group 4 before or with this one if the test suite must pass at every commit.

### Group 3 — Configuration extraction

Roughly 30 hardcoded reasoning and severity constants moved into YAML, with the loader properties that expose them and the removal of the unread `convergence_angle_threshold` key.

```
rads/config/pipeline_config.yaml
rads/config/config_loader.py
```

### Group 4 — Evaluation split rebuild

The corrected source-id extraction and the rebuilt 250-pair held-out split.

```
rads/config/create_splits.py
rads/config/picek_heldout_split_v2.csv
rads/evaluation/split_integrity_report.md
```

`create_splits.py` reproduces the CSV byte-identically under seed 42, MD5 `C6EA958D93F6F857CAF8E7281D2977CB`.

### Group 5 — Evaluation and reporting tooling

The missing package marker, the `tracks_summary` key fix, error rows, the completeness check, the `--dump-dir` option, the real baseline comparison, the dashboard updater rewrite, the Phase 3 regression gate, and the adaptation of the forensic script to the new severity contract.

```
rads/evaluation/__init__.py
rads/evaluation/evaluator.py
rads/evaluation/baseline_comparison.py
rads/evaluation/dashboard_updater.py
rads/evaluation/p02_baseline_comparison.md
rads/evaluation/failure_analysis.md
rads/evaluation/experiments/mvp_freeze_regression/run_gate.py
rads/evaluation/experiments/mvp_freeze_regression/gate_results.json
rads/evaluation/experiments/mvp_freeze_regression/regression_summary.md
scripts/forensic_diagnostic.py
```

### Group 6 — Documentation

The dated correction records, the phase plan relocated out of the ignored `.cursor/plans/`, the frontend design outline, and this handoff.

```
docs/architecture/IMPLEMENTATION_TRACKER.md
docs/architecture/MVP_COMPLETION_PLAN.md
docs/architecture/SESSION_HANDOFF.md
docs/frontend/DASHBOARD_DESIGN_OUTLINE.md
rads/evaluation/dashboard.md
```

---

## 7. Where to resume

The phase plan is `docs/architecture/MVP_COMPLETION_PLAN.md`, moved there on 2026-09-21 from `.cursor/plans/` so that it commits. Phases 0 through 3 are complete: repo integrity, the bugs that invalidate any evaluation, split integrity and portability, and the contract completion and freeze including its regression gate. Phase 4 onward is untouched.

| Plan phase | State |
|---|---|
| Phase 0 — Repo integrity and version control | Done, see group 1 |
| Phase 1 — Bugs that invalidate any evaluation | Done, see group 2 |
| Phase 2 — Split integrity and portability | Done, see group 4 |
| Phase 3 — Complete the contracts, then freeze | Done, gate output under `rads/evaluation/experiments/mvp_freeze_regression/` |
| Phase 4 — Launch the overnight batch | Not started |
| Phase 5 — Work parallel to the batch | Partially pre-empted: B4 baseline comparison and the visualizer changes landed, demo artifacts did not |
| Phase 6 — Metrics, comparison, DoD closure | Not started |
| Phase 7 — Freeze and verify | Not started |

Phase 4 step 0, the mandatory 3-clip pre-gate over `picek_heldout_split_v2.csv`, already ran and passed. It can be skipped and the batch launched directly. See section 4.1 for its figures.

---

## 8. Open questions carried forward

### 8.1 Frontend, from docs/frontend/DASHBOARD_DESIGN_OUTLINE.md section 9

Eight questions, each answerable in one line. Q6 is the one to answer first.

| # | Question |
|---|---|
| Q1 | Does the dashboard read local files directly from disk, or is a small read-only serving layer acceptable? |
| Q2 | Is in-browser playback of the annotated mp4 and the evidence clip required for the demo, or is opening them in a media player acceptable? |
| Q3 | If playback is required, is re-encoding the visualizer output from `mp4v` to H.264 acceptable? |
| Q4 | Which run is the demo run: P02 at 74 clips, or the held-out split at 500? |
| Q5 | Will bug B7 be fixed before the frontend is built, so negatives carry `evidence_list` and involved ids? |
| Q6 | Should the pipeline write a per-clip result JSON during batch evaluation, not just the JSONL row? Without it, the proposed clip-result page is reachable only for manually re-run clips |
| Q7 | Is the demo audience technical or non-technical? |
| Q8 | Do the summary pages render the markdown artifacts live, or is a pre-render step acceptable? |

Q5 is partly overtaken by events: the severity and event-schema rewrites in group 2 changed what a negative decision emits. Re-read the current `rads/output/event_schema.py` before answering it.

Q6 is partly addressed by the new `--dump-dir` option on the evaluator, which writes one full-result JSON per video when supplied. It defaults to `None`, so the question of whether batch runs should use it by default is still open.

### 8.2 Unvalidated numeric choices

These were engineering judgements made without ground truth. Nothing in the repository validates them and no sensitivity analysis was run.

| Choice | Current value | Why it is unvalidated |
|---|---|---|
| `reasoning.relative_velocity_divisor` | 50.0 | Normalizes relative velocity into the score. The divisor sets how quickly the velocity term saturates. Chosen, not fitted |
| `severity` score divisors and weights | `object_count` 1.0 capped at 2.0, `vulnerable_class` 3.0, plus the relative-velocity term | The weights set the relative importance of occupant vulnerability against impact energy. No labelled severity data exists to fit them against |
| `severity.threshold_medium` / `threshold_high` | 2.0 / 4.5 | The LOW/MEDIUM/HIGH cut-offs on the severity score. Placed by inspection of the pre-fix distribution, which is itself superseded |
| `reasoning.confidence_transform_scale` | 0.5 | Scale of the bounded confidence transform, active because `reasoning_confidence_bounded_transform` is True. It changes the shape of the confidence distribution and therefore AUROC, without changing any thresholded decision. Chosen to spread the pre-fix pile-up at 0.0 and 1.0, but the spreading was never measured on a post-fix run |

The pre-fix AUROC caveat in the tracker, that 61.4 percent of held-out rows and 62.2 percent of P02 rows sat at confidence exactly 0.0 or 1.0, is the reason the transform was added. Whether it helped cannot be known until a post-fix run exists.

### 8.3 Architecture, raised by the halt itself

The user stopped to reconsider architecture and MVP scope. Nothing in this document presumes an answer. The two constraints most likely to bear on that reconsideration are recorded here because they are structural rather than incidental: reasoning is pairwise over track pairs, which is what makes B9 single-vehicle detection unreachable without a design change; and every decision threshold in the system is a hand-placed constant, which is what makes section 8.2 a list rather than a footnote.
