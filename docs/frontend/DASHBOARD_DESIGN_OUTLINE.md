# RADS Frontend / Dashboard Design Outline

Written 2026-09-20. This is a design outline only. No frontend code exists and none is created by this document.

---

# 1. Status and Scope

## 1.1 What exists

No frontend exists in this repository. Verified by:

- Globbing the working tree for `*.html`, `*.htm`, `*.css`, `*.scss`, `*.js`, `*.jsx`, `*.ts`, `*.tsx`, `*.vue`, `*.svelte`: zero files.
- Globbing for `package.json`, `package-lock.json`, `vite.config.*`, `next.config.*`, `tailwind.config.*`, `index.html`: zero files.
- Searching all tracked and untracked content for `streamlit`, `gradio`, `flask`, `fastapi`, `dash.`, `plotly`, `django`: zero matches.
- Searching for directories named `frontend`, `dashboard`, `web`, `ui`, `static`, `templates`: none.
- `requirements.txt` lists no web framework, no templating engine and no server dependency.
- `docs/` contains 33 markdown files and no design file, mockup, wireframe, page inventory, layout specification or component list.

Two files match the string "dashboard" and neither is a frontend:

| File | What it actually is |
|---|---|
| `rads/evaluation/dashboard_updater.py` | A CLI loop that counts rows in an evaluation JSONL and rewrites a markdown progress file every 15 seconds. |
| `rads/evaluation/dashboard.md` | The markdown file that script writes. Its entire content is a progress bar reading 500/500. |

## 1.2 What the specs say

`docs/architecture/IMPLEMENTATION_AUDIT_AND_ORDER.md` section 1.4 records the discrepancy directly: TECH_STACK.md section 17 and SYSTEM.md sections 23 and 24 reference "existing" dashboard and Telegram integrations, and the audit states "No such code exists in the repository." PART 5 of the same audit resolves TECH_STACK section 17 as "P1 — deferred" and MVP section 6 P1 components as "Deferred to Phase 9+". The phase table places `Phase 9+ P1 work (dashboard, telegram, accident types, etc.)` after Phase 8, which is marked MVP COMPLETE.

MVP.md section 6 lists "Dashboard integration" and "Telegram alert integration" under P1, "These should be implemented if time and reliability allow." MVP.md section 34 lists "Complete dashboard redesign" among the things that should not block the MVP.

MVP.md section 36 Definition of Done contains exactly one dashboard item, under Engineering: "Existing dashboard/alert integration is not broken." Since no dashboard or alert integration exists, nothing can be broken by the MVP work, so the item is vacuously satisfied. That is the audit's resolution and this document does not reopen it.

## 1.3 Consequence for this document

There is no prior design to verify against. There is no specified page count, no specified layout, no specified widget set and no specified frontend technology. Every page, layout, component and technology named below is **PROPOSED** by this document. Each proposal is traced either to a spec section that constrains it or to a data field that actually exists in the pipeline output. Where neither exists, the item is marked as an open question rather than presented as a decision.

## 1.4 Timing

The core MVP evaluation is still in progress. The evaluation numbers available today are labelled PRE-FIX: they were produced before the tracker-state bug (B1) and the split-integrity bug (B3) were fixed, and a corrected re-run is pending. `rads/evaluation/failure_analysis.md` carries a re-confirmation checklist in which all eight failure cases are PENDING. No frontend work should start before the corrected re-run lands, because the numbers a results view would display are known to be about to change.

---

# 2. The Hard Architectural Constraint

SYSTEM.md section 23 states that the dashboard should consume structured RADS events and "should not contain the core accident reasoning logic", with the architecture given as pipeline to structured event to dashboard, so "the AI system to be tested independently of the interface."

SYSTEM.md section 34 Constraint 5 states: "Dashboard and alert systems must consume structured outputs rather than implement AI logic themselves."

TECH_STACK.md section 17 repeats the separation: the components "should remain separate from the core computer-vision reasoning pipeline" and the AI system "should produce a structured event rather than directly coupling model logic to a specific user interface."

In practice this forbids the following in the frontend:

| Forbidden in the frontend | Why | Where it must stay |
|---|---|---|
| Applying any accident decision threshold to `score` or `confidence` | The accept/reject decision is `reasoning.accident_score_threshold`, default 0.5, read in `rads/config/config_loader.py` | `rads/reasoning/accident_reasoner.py` |
| Computing or recomputing a severity band from `severity_detail.score` | Banding is `HIGH` at score >= 4.5, `MEDIUM` at >= 2.0, else `LOW`, from config | `rads/severity/severity_engine.py` |
| Deriving which objects were involved from `tracks_summary` or `interaction_candidates` | Involvement is the result of candidate clustering with the B11 own-evidence rule | `rads/reasoning/accident_reasoner.py` |
| Re-deriving event start, impact or end times from candidate times | Event timing is selected from the best candidate and its cluster | `rads/reasoning/accident_reasoner.py` |
| Computing evidence tokens from kinematics | The four tokens are emitted by the interaction and reasoning stages | `rads/interaction/interaction_engine.py`, `rads/reasoning/accident_reasoner.py` |
| Recomputing accuracy, precision, recall, F1, AUROC or the confusion matrix from raw JSONL rows | These are produced by the comparison script and written to a markdown artifact | `rads/evaluation/baseline_comparison.py` |
| Inferring an accident type | Not implemented anywhere; the field is a literal placeholder | Nowhere yet, see section 3.3 |

The frontend is allowed to: read the emitted fields, format them for display (timestamp formatting, rounding, sorting, filtering), lay them out, and play back video files that the pipeline wrote. Sorting and filtering a table of already-computed results is presentation, not reasoning, and is permitted.

A useful test for any proposed frontend feature: if the pipeline output changed and the frontend's displayed value would not change accordingly, the frontend is computing something it should not.

---

# 3. The Real Data Contract

## 3.1 Per-video result, from `rads/output/event_schema.py`

`build_event_result` is the authoritative per-video contract. It returns a dictionary with exactly thirteen top-level keys. This is the complete list; nothing else is emitted.

| Field | Type | Present when no accident | Notes |
|---|---|---|---|
| `video` | string | Always | The `video_path` argument passed in. `rads/pipeline/pipeline.py` passes the path it was invoked with. |
| `accident` | boolean | Always, `false` | From `accident_result.accident`. Defaults to `None` if the reasoner dictionary is absent. |
| `confidence` | float | Always | Bounded confidence derived from `score` by `_confidence` in the reasoner. Present and non-null on negatives; observed values in the evaluation JSONL include `0.0`, `0.04`, `0.11` and `0.3`. |
| `score` | float | Always | Raw aggregate candidate score, rounded to 4 places. Not the same quantity as `confidence`. Observed `0.4` on a negative and `1.3` on a positive in `gate_results.json`. |
| `event.start_time` | float or null | `null` | Seconds. |
| `event.impact_time` | float or null | `null` | Seconds. Taken from the best candidate's `peak_time`. |
| `event.end_time` | float or null | `null` | Seconds. |
| `objects_involved` | list of `{id: int, class: string}` | `[]` | Built by `_objects_involved`, which looks each involved id up in `tracks_summary` and falls back to `class` = `"unknown"` when the id is missing. |
| `accident_type` | string | Always, `"unknown"` | Hardcoded literal. See section 3.3. |
| `evidence_list` | list of strings | `[]` | See section 3.4 for the closed token set. |
| `kinematics` | object | `{}` | See section 3.5. |
| `severity` | string or null | `null` | `"LOW"`, `"MEDIUM"` or `"HIGH"`. `event_schema.py` documents that this stays a plain string because the evaluator, the visualizer and the JSONL consumers read it directly. |
| `severity_detail.score` | float | `0.0` | Rounded to 4 places. Maximum attainable is 9.5 per the `severity_engine.py` docstring. |
| `severity_detail.evidence` | list of `{factor, value, contribution}` | `[]` | Always five entries when an accident is reported. See section 3.6. |
| `interaction_candidates` | list of objects | Possibly non-empty | Critically, this is **not** empty on negatives. `gate_results.json` records 46 candidates on a clip predicted normal. |
| `tracks_summary` | object keyed by track id | Possibly non-empty | See section 3.7. |

The nesting is: `event` is an object with three keys; `severity_detail` is an object with two keys. Expanding those two, the contract is thirteen top-level keys giving sixteen addressable entries, which is why the table above has sixteen rows.

## 3.2 Null-versus-present summary

The distinction a frontend must handle correctly:

```text
accident = false
  ├── event.start_time / impact_time / end_time  →  null
  ├── objects_involved                           →  []
  ├── evidence_list                              →  []      (bug B7: evidence is computed then discarded)
  ├── kinematics                                 →  {}
  ├── severity                                   →  null
  ├── severity_detail.score                      →  0.0     (not null)
  ├── severity_detail.evidence                   →  []
  ├── confidence / score                         →  present, numeric, possibly 0.0
  ├── accident_type                              →  "unknown" (same as on positives)
  ├── interaction_candidates                     →  may be a long list
  └── tracks_summary                             →  may be a long object
```

`severity_detail.score` being `0.0` rather than `null` on negatives matters: a view that tests for presence rather than for `accident` will render a severity score of zero on every normal clip.

`rads/evaluation/failure_analysis.md` records bug B7: on a negative decision the reasoner returns an empty `evidence_list` and empty `involved_object_ids` even when candidates were scored, so "the evidence is computed and then discarded when the decision is negative, which ... makes negative cases unauditable without the forensic dump." B7 is listed as being fixed in Phase 3. If B7 is fixed before the frontend is built, negatives may carry evidence and the table above changes. This is an open question, see section 9.

## 3.3 `accident_type`

`event_schema.py` sets `"accident_type": "unknown"` as a hardcoded literal with the inline comment `post-MVP: classification is not implemented`. MVP.md section 6 lists "Accident type classification" as P1. The value is `"unknown"` on every clip, accident or not. SYSTEM.md section 21 shows `"rear_end"` and AI.md section 25 shows "Rear-End Collision" in their illustrative examples; those are illustrations of a future schema, not of current output. The frontend must not present a type. See section 8.

## 3.4 `evidence_list` token set

The tokens are a closed set of four, emitted by two stages:

| Token | Emitted by | Condition |
|---|---|---|
| `overlap` | `rads/interaction/interaction_engine.py` | `peak_iou` above the IoU threshold |
| `proximity_and_convergence` | `rads/interaction/interaction_engine.py` | `min_prox` below the proximity threshold and `max_rel_vel` above the relative-velocity threshold |
| `track_loss` | `rads/reasoning/accident_reasoner.py` | Track termination condition |
| `sudden_deceleration` | `rads/reasoning/accident_reasoner.py` | Velocity-drop condition, appended at two separate branches |

All four appear in real output: `gate_results.json` shows `["proximity_and_convergence", "track_loss", "sudden_deceleration"]` on clip `-NgnSm_oEB4_00.mp4`, and `failure_analysis.md` Case 3 shows `overlap` on candidate-level evidence. A frontend can map these four tokens to human-readable strings, since that is presentation. It must not add a fifth token or infer one.

AI.md section 25 and MASTER_SPEC.md section 11 both require a compact evidence explanation. AI.md section 25 gives the shape: time, objects, an evidence bullet list, classification, severity. MASTER_SPEC.md section 11 states the principle: "The system should be able to show why it believes an accident occurred", and that the system "should avoid producing only ACCIDENT / Confidence: 94%". The evidence list plus the severity factor breakdown are the fields that satisfy this.

## 3.5 `kinematics`

Empty object on negatives. On a positive, `_event_kinematics` in the reasoner returns:

| Key | Type | Source |
|---|---|---|
| `peak_relative_velocity` | float, pixels per second | Max `max_relative_velocity` over the candidate cluster |
| `peak_iou` | float | Max `peak_iou` over the cluster |
| `max_velocity_drop` | float, pixels per second | Max pre-to-post velocity drop over the cluster |
| `impact_frame` | int | Frame index of impact |
| `best_candidate.object_a_id` | int | |
| `best_candidate.object_b_id` | int | |
| `best_candidate.evidence_list` | list of strings | |

Real values from `gate_results.json` clip `-NgnSm_oEB4_00.mp4`: `peak_relative_velocity` 668.7972700358924, `peak_iou` 0.020699440916606363, `max_velocity_drop` 496.58149321246873, `impact_frame` 13, `best_candidate` `{object_a_id: 6, object_b_id: 9, evidence_list: [proximity_and_convergence, track_loss, sudden_deceleration]}`.

These are image-space pixel quantities. `severity_engine.py` states the divisors "are pixel-space and were set from the magnitudes observed on the 10-clip regression sample". `failure_analysis.md` records "No world-coordinate scaling" as a standing limitation and notes velocities of 366 and 497 px/s arising from four-frame tracks. SYSTEM.md section 34 Constraint 7 forbids claiming real-world physical quantities without calibration support. Any frontend that displays these must label them as pixels per second in image space, never as km/h or mph.

## 3.6 `severity_detail.evidence` factors

`estimate_severity` in `rads/severity/severity_engine.py` appends exactly five entries, always in this order, each `{factor, value, contribution}`:

| `factor` | `value` type | Real example from `gate_results.json` |
|---|---|---|
| `num_objects_involved` | int | value 4, contribution 2.0 |
| `vulnerable_class_involved` | sorted list of matching class names, often `[]` | value `[]`, contribution 0.0 |
| `peak_relative_velocity` | float, px/s, rounded to 2 | value 668.8, contribution 2.0 |
| `velocity_change` | float, px/s, rounded to 2 | value 496.58, contribution 1.5 |
| `post_impact_displacement` | float, pixels, rounded to 2 | value 66.6, contribution 0.2664 |

Contributions sum to `severity_detail.score`; the example sums to 5.7664, which the file records as the clip's `severity_score` with severity `HIGH`.

The `severity_engine.py` module docstring states plainly: "This is an engineered heuristic. It has no ground-truth validation: the datasets in use carry no severity labels ... Per MVP.md section 21 and TECH_STACK.md section 14, the output must not be presented as a validated severity prediction." Any frontend surface that shows severity must carry that caveat as visible text, not as a footnote. `failure_analysis.md` Case 7 shows a `MEDIUM` severity emitted on a clip whose ground truth is normal, which is the concrete reason the caveat is needed.

## 3.7 `tracks_summary`

Keyed by integer track id. `rads/pipeline/pipeline.py` populates each entry with `class_name`, `first_frame`, `last_frame`, `frame_count`, and then attaches `motion_features` for every track that has them. The visualizer's summary frame renders `len(tracks_summary)` as the track count.

## 3.8 `interaction_candidates`

The list is passed straight through from `detect_interactions`. `rads/interaction/interaction_engine.py` constructs each candidate with eleven keys: `object_a_id`, `object_b_id`, `start_frame`, `end_frame`, `start_time`, `peak_time`, `end_time`, `min_distance` (normalised proximity, smaller is closer), `max_relative_velocity`, `peak_iou`, `evidence_list`.

`rads/reasoning/accident_reasoner.py` then mutates these same dictionaries in place during scoring, setting `evidence_list`, `score`, and a candidate-level `kinematics` containing `pre_velocity_a`, `post_velocity_a`, `pre_velocity_b`, `post_velocity_b`. Because the pipeline passes the same list object into `build_event_result`, scored candidates carry those extra keys in the emitted JSON.

Caveat: I could not confirm the serialised candidate shape against a real artifact. `gate_results.json` stores only `interaction_candidate_count`, not the candidate list, and the two evaluation JSONL files store only the count as well. The key list above is read from source, not from a sample file. Resolving this requires one saved full-result JSON from `run_pipeline.py`.

## 3.9 Evaluation record shape

Two different shapes exist and a frontend must not assume they are the same.

The current `rads/evaluation/evaluator.py` writes, per successful video: `video_id`, `ground_truth`, `status` (`"ok"`), `prediction` (0 or 1), `confidence`, `score`, `severity`, `severity_score`, `event` (the three-key object), `evidence_list`, `interaction_candidate_count`, `num_tracks`. Error rows carry only `video_id`, `ground_truth`, `status` (`"error"`) and `error`.

The two JSONL files currently on disk predate that. Both `rads/evaluation/p02_test_results.jsonl` and `rads/evaluation/picek_500_results.jsonl` carry the reduced shape: `video_id`, `ground_truth`, `prediction`, `confidence`, `severity`, `event`, `num_tracks`. They have no `status`, no `score`, no `severity_score`, no `evidence_list` and no `interaction_candidate_count`. `failure_analysis.md` records that `num_tracks` is 0 on all 74 P02 rows and all 500 held-out rows because of bug B2, where the evaluator read `track_summaries` while the schema emitted `tracks_summary`.

A frontend built against today's files would show a track count of zero everywhere. It should be built against the post-fix shape, which is another reason to wait for the corrected re-run.

## 3.10 Video artifacts

From `run_pipeline.py` and `rads/pipeline/pipeline.py`:

| Artifact | Path | Condition |
|---|---|---|
| Result JSON | `--output`, required argument, written with `indent=4` | Always |
| Annotated video | `--output-video`, default `output_videos/output.mp4` | Only with `--visualize` |
| Evidence clip | Same path with `_evidence` inserted before the extension, for example `output_videos/output_evidence.mp4` | Only with `--visualize` **and** `result["accident"]` true |

Both videos are written with the `mp4v` fourcc. The evidence clip is cut from the already-annotated video, so it carries the overlays, and spans the event window padded by 2.0 seconds on each side; if start and end times are null it falls back to the impact time, and if that is null too it writes nothing.

The `Visualizer` draws, per `rads/output/visualizer.py`: trajectory trails over the last 30 points, bounding boxes with `class_name` and track id labels, involved objects highlighted in red during the event window and orange-versus-red banner text, an `ACCIDENT DETECTED | Conf | Sev` banner or a `NORMAL TRAFFIC` banner, and a `TIME:` line formatted `MM:SS.s`. Optionally it appends a held summary frame listing time, window, severity with score, type, involved objects, evidence, confidence with raw score, and track count. This covers everything MVP.md section 5.12 asks the system to render (bounding boxes, object IDs, trajectories, accident marker, objects involved, severity) and everything TECH_STACK.md section 16 lists as useful overlays except detection confidence, which the visualizer does not draw per box.

I found no generated `.mp4` or result `.json` at `output_videos/` in the working tree, so the artifacts are producible but not currently present.

## 3.11 Evaluation content a results view would surface

| Source file | Content |
|---|---|
| `rads/evaluation/p02_baseline_comparison.md` | A metric table comparing RADS PRE-FIX against ResNet18+GRU on 74 P02 rows, plus confusion matrices and a provenance section. |
| `rads/evaluation/split_integrity_report.md` | Nine sections on source-id extraction, defects in the previous split, pairing policy, exclusions, counts, schema, P02 source overlap, what the leakage claim covers and does not cover, and reproduction commands. |
| `rads/evaluation/failure_analysis.md` | Eight failure cases in three categories, a limitations section including bug B9 on single-vehicle accidents, and a pending re-confirmation checklist. |
| `docs/architecture/IMPLEMENTATION_TRACKER.md` | Phase and bug tracking. Concurrently edited by other agents; treat as a live file. |
| `rads/evaluation/experiments/mvp_freeze_regression/gate_results.json` | Ten clips with full per-clip result fields, plus a recorded baseline and acceptance criteria. The best available realistic sample data. |

Verified numbers that a results view would display, all PRE-FIX and all subject to change after the corrected re-run:

| Run | Scope | Accuracy | TN | FP | FN | TP |
|---|---|---|---|---|---|---|
| Held-out split | 500 clips | 0.6040 | 187 | 63 | 135 | 115 |
| P02 | 74 clips | 0.5676 | 26 | 11 | 21 | 16 |
| ResNet18+GRU baseline | 74 clips | 0.4595 | 6 | 31 | 9 | 28 |

The baseline's false-positive rate is 0.8378 against RADS PRE-FIX at 0.2973. Severity distribution on the held-out set was 20 HIGH, 131 MEDIUM, 27 LOW.

---

# 4. Page Inventory

**Proposed page count: 4.** Two are P0 for the demonstration, two are P1.

The count is driven by the artifacts that exist. There is one per-video result object, one batch of result rows, one set of evaluation markdown reports, and one failure-analysis report. Each maps to one page. Nothing else has a data source, so nothing else is proposed.

| # | Page | Priority | Purpose | Reads | Primary question it answers |
|---|---|---|---|---|---|
| 1 | Clip Result | P0 | Show one video's full reasoning output alongside its annotated video. | One result JSON from `run_pipeline.py --output`; `output_videos/<name>.mp4`; `output_videos/<name>_evidence.mp4` | "What did RADS conclude about this clip, and on what evidence?" |
| 2 | Run Index | P0 | List every clip in an evaluation run, filterable, linking to page 1. | One evaluation JSONL (`p02_test_results.jsonl` or `picek_500_results.jsonl`, post-fix) | "Which clips did RADS get right and wrong, and which one should I open?" |
| 3 | Evaluation Summary | P1 | Present the headline metrics, confusion matrices and the baseline comparison. | `p02_baseline_comparison.md`; `split_integrity_report.md` | "How good is RADS overall, and is the measurement trustworthy?" |
| 4 | Failure Analysis | P1 | Browse documented failure cases by stage and category. | `failure_analysis.md`; optionally the forensic JSON dumps it cites | "When RADS is wrong, which stage is responsible?" |

Justification per page:

- **Page 1** is the only page MVP.md section 35 actually requires. That section asks the demonstration to show a complete video flowing through RADS, ending "Result displayed", and to be "visually understandable to a viewer who has not read the code." MASTER_SPEC.md section 24 gives the same chain ending in RESULT VISUALIZED, and adds that "the system should also retain enough intermediate information to inspect how the final decision was reached." AI.md section 25 and MASTER_SPEC.md section 11 specify that the explanation be compact and evidence-bearing. Page 1 is where that lands.
- **Page 2** is justified by MVP.md section 29, which requires visualization to be usable as an evaluation tool for inspecting detection quality, ID consistency, trajectory quality, interaction candidates, event timing and severity reasoning "for difficult examples". Finding the difficult examples across 500 clips requires an index. Without page 2, page 1 is only reachable by knowing a filename in advance.
- **Page 3** is P1 rather than P0 because `p02_baseline_comparison.md` is already a readable markdown table, generated by `baseline_comparison.py`. Rendering it in a browser adds convenience, not capability.
- **Page 4** is P1 for the same reason: `failure_analysis.md` is already readable, and its eight cases are all marked PENDING re-confirmation, so a browsing surface over provisional content has low value today.

Rejected pages, with reasons, so the count is not later padded by drift:

| Rejected | Reason |
|---|---|
| Live monitoring / map view | No live data source. SYSTEM.md section 28 makes streaming a future concern. No geospatial field exists in any output. |
| Alert inbox | SYSTEM.md section 24 puts Telegram on a separate consumer branch; no alert records are persisted anywhere. |
| Accident type breakdown | `accident_type` is the constant `"unknown"`. A breakdown of one constant value is not a page. |
| Configuration editor | Would require write access to `pipeline_config.yaml` and would make displayed results irreproducible. TECH_STACK.md section 22 rule 4 requires reproducibility. |
| Upload-and-process page | Requires the frontend to invoke the pipeline. That is a job-runner, not a dashboard, and is outside the consumer role SYSTEM.md section 23 defines. |

---

# 5. Per-Page Layout Outline

Proposed. No pixel values, no colour system, no component-library assumption. The order of regions is the reading order; how that maps to columns is an implementation choice.

## 5.1 Page 1, Clip Result

The layout follows the AI.md section 25 explanation shape, extended with the fields that actually exist.

```text
+---------------------------------------------------------------+
| Verdict bar                                                     |
|   accident (true/false)   confidence   score                    |
|   PRE-FIX / provisional run banner                              |
+---------------------------------------------------------------+
| Video region                | Event facts                       |
|   annotated mp4             |   start_time                      |
|   evidence clip (if any)    |   impact_time                     |
|   (see open question Q2)    |   end_time                        |
|                             |   objects_involved [{id, class}]  |
+-----------------------------+-----------------------------------+
| Evidence region                                                 |
|   evidence_list, one row per token, plain-language label        |
|   empty state: "no evidence recorded" (see bug B7)              |
+---------------------------------------------------------------+
| Severity region                                                 |
|   severity band + severity_detail.score                         |
|   five-row factor table: factor | value | contribution          |
|   unvalidated-heuristic caveat, always visible                  |
+---------------------------------------------------------------+
| Kinematics region                                               |
|   peak_relative_velocity, peak_iou, max_velocity_drop,          |
|   impact_frame, best_candidate pair                             |
|   units labelled as image-space px/s                            |
+---------------------------------------------------------------+
| Detail region, collapsed by default                             |
|   tracks_summary table: id | class | first | last | frames      |
|   interaction_candidates table, sortable by score               |
|   raw result JSON                                               |
+---------------------------------------------------------------+
```

Notes on behaviour, all presentation-only:

- On `accident: false`, the event, evidence, severity and kinematics regions collapse to a single explicit empty state rather than rendering nulls or a severity score of 0.0 as if it were meaningful.
- `interaction_candidates` may be long and non-empty on negatives, up to 46 on one recorded clip. That is the most diagnostically useful thing on a false negative and should stay reachable, but collapsed.
- The verdict bar shows `confidence` and `score` side by side because they are different quantities and the failure analysis argues about both.

## 5.2 Page 2, Run Index

```text
+---------------------------------------------------------------+
| Run header                                                      |
|   source JSONL path, row count, PRE-FIX / POST-FIX label        |
+---------------------------------------------------------------+
| Filters (client-side, presentation only)                        |
|   outcome: TP | TN | FP | FN     severity: HIGH/MEDIUM/LOW/none |
|   has evidence token: overlap | proximity | track_loss | decel  |
+---------------------------------------------------------------+
| Result table, one row per video_id                              |
|   video_id | ground_truth | prediction | outcome | confidence   |
|   | score | severity | impact_time | num_tracks | candidates    |
|   row click -> page 1 for that clip                             |
+---------------------------------------------------------------+
```

The `outcome` column is TP/TN/FP/FN derived from `ground_truth` and `prediction`, both of which are already in the record. That is a two-field lookup, not reasoning. Rows with `status: "error"` render as an error row showing the `error` string, not as a prediction.

## 5.3 Page 3, Evaluation Summary

```text
+---------------------------------------------------------------+
| Run provenance                                                  |
|   run label, results file, baseline source, alignment check     |
+---------------------------------------------------------------+
| Metric table, RADS vs baseline (rendered from the md artifact)  |
+---------------------------------------------------------------+
| Confusion matrices, one per system                              |
+---------------------------------------------------------------+
| Caveats, rendered as body text not as a footnote                |
|   single-vehicle recall ceiling (B9)                            |
|   pair-level effective sample size from split_integrity_report  |
|   P02 source overlap (4 of 250)                                 |
+---------------------------------------------------------------+
```

The caveat region is not optional. `failure_analysis.md` states a reporting obligation: "any recall figure quoted for RADS on a split containing `single` clips must state the proportion of those clips, because the achievable recall is bounded below 100 percent by construction", citing MVP.md section 38. `split_integrity_report.md` section 8 adds that "Any metric that assumes independent samples across the 500 rows is overstating its effective sample size; the effective unit is the 250 pairs." A page showing accuracy without these is a spec violation, not a styling choice.

## 5.4 Page 4, Failure Analysis

```text
+---------------------------------------------------------------+
| Category legend: A no candidate | B rejected | C wrongly accepted|
+---------------------------------------------------------------+
| Case list, filterable by responsible stage                      |
|   case id | clip | FP/FN | stage | category | status            |
+---------------------------------------------------------------+
| Case detail                                                     |
|   the case's field table and candidate table, as authored       |
|   link to that clip on page 1, if a result exists for it        |
+---------------------------------------------------------------+
```

This page renders authored markdown content. It derives nothing.

---

# 6. Shared Components

Each component maps to a specific field. Nothing here computes.

| Component | Renders | Source field |
|---|---|---|
| VerdictBadge | Accident yes/no | `accident` |
| ConfidenceReadout | Two numbers, labelled distinctly | `confidence`, `score` |
| EventTimeline | Three marks on one time axis | `event.start_time`, `event.impact_time`, `event.end_time` |
| TimestampLabel | `MM:SS.s` formatting, matching `format_timestamp` in `visualizer.py` | any time field |
| ObjectChipList | One chip per involved object, `CLASS #id` | `objects_involved[].class`, `objects_involved[].id` |
| EvidenceList | One row per token, mapped to a plain-language label | `evidence_list` |
| SeverityBadge | Band label with the unvalidated-heuristic caveat attached | `severity` |
| SeverityFactorTable | Five fixed rows | `severity_detail.evidence[].factor`, `.value`, `.contribution`, totalling `severity_detail.score` |
| KinematicsPanel | Labelled image-space quantities | `kinematics.peak_relative_velocity`, `.peak_iou`, `.max_velocity_drop`, `.impact_frame`, `.best_candidate` |
| TrackTable | One row per track | `tracks_summary[id].class_name`, `.first_frame`, `.last_frame`, `.frame_count` |
| CandidateTable | One row per candidate | `interaction_candidates[].object_a_id`, `.object_b_id`, `.start_time`, `.peak_time`, `.end_time`, `.min_distance`, `.max_relative_velocity`, `.peak_iou`, `.evidence_list`, `.score` |
| VideoPane | Playback of a written artifact | annotated mp4 path, evidence clip path |
| ResultRow | One evaluation record | `video_id`, `ground_truth`, `prediction`, `confidence`, `score`, `severity`, `event.impact_time`, `num_tracks`, `interaction_candidate_count` |
| ProvisionalRunBanner | Static text stating PRE-FIX or POST-FIX | run label, taken from the artifact header |
| EmptyState | Explicit "not applicable, no accident detected" | driven by `accident` |
| RawJsonView | The unmodified result object | whole result |

`AccidentTypeBadge` is deliberately absent. See section 8.

---

# 7. Tech Stack Options

The specs do not name a frontend technology. TECH_STACK.md section 24 names Python as the MVP language for the pipeline; it does not address a UI. TECH_STACK.md section 23 warns against introducing technologies "unless there is a clear reason". Three realistic options follow, assessed against the six selection rules in TECH_STACK.md section 22 and against the requirement that the demonstration run offline from local artifacts.

## Option A: Static HTML page reading JSON, served by the Python standard-library HTTP server

A hand-written page, no build step, no npm, no bundler. Served by `python -m http.server` from the repository root so that result JSON and mp4 files load over `http://localhost`.

| Rule | Assessment |
|---|---|
| Proven reliability | Highest. The serving layer is the Python standard library, already installed. |
| Integration | Reads the JSON the pipeline already writes. No change to any pipeline module. |
| Performance | 500 rows is trivial. Video playback is browser-native for `mp4v`-encoded H.264-container files, subject to the codec caveat below. |
| Reproducibility | Highest. No lockfile, no dependency resolution, no version drift. |
| Replaceability | Highest. Nothing depends on it, and it can be deleted without touching the pipeline. |
| Simplicity | Highest. |

Honest tradeoffs: no component model, so the four pages will share code by hand or duplicate it. Opening the page directly as a `file://` URL will fail to fetch sibling JSON because browsers block cross-origin requests for local files, so the static-server step is not optional and must be documented. Markdown artifacts (pages 3 and 4) either need a client-side markdown renderer or need to be converted to HTML ahead of time.

## Option B: Streamlit, or another Python-first UI framework

Keeps everything in Python, consistent with the existing stack.

| Rule | Assessment |
|---|---|
| Proven reliability | Good. Mature and widely used for exactly this kind of artifact viewer. |
| Integration | Best on paper. Could import `rads` modules directly, which is precisely the risk: SYSTEM.md section 34 Constraint 5 is easiest to violate when reasoning code is one import away. |
| Performance | Adequate. Re-runs the script on every interaction, which is noticeable but not disqualifying at this data size. |
| Reproducibility | Weaker. Adds a dependency with a large transitive tree to `requirements.txt`, which currently has no web dependency at all. |
| Replaceability | Moderate. Layout is expressed in framework-specific calls and does not port. |
| Simplicity | Moderate. Zero HTML to write, but a new runtime and a new failure surface. |

Honest tradeoffs: the strongest argument for it, that the team already writes Python, is also the strongest argument against it, because the architectural constraint exists specifically to keep the UI unable to reason. A rule that says "do not import `rads.reasoning`" is weaker than an architecture in which importing it is impossible. Video playback and precise layout control are also weaker than in a plain page.

## Option C: React with Vite, or an equivalent modern JS framework

| Rule | Assessment |
|---|---|
| Proven reliability | Good, in general. Less good here, because nobody in this repository currently maintains a JS toolchain. |
| Integration | Weakest. Introduces Node, npm, a bundler and a second language to a single-language project. |
| Performance | Excellent, and irrelevant at this scale. |
| Reproducibility | Weakest for a research repo. A `node_modules` tree and a lockfile must be reproduced alongside the Python environment, and a build step must run before the demo. |
| Replaceability | Good at the component level, poor at the toolchain level. |
| Simplicity | Weakest. |

Honest tradeoffs: it is the right answer if the dashboard is expected to grow into a real product surface with many views, authentication and a live feed. None of that is in scope, and MVP.md section 34 names "Complete dashboard redesign" as something that must not delay the MVP. Choosing C now is paying the cost of a future that the specs defer.

## Recommendation

**Option A.** It is the only option that adds zero dependencies to a repository whose reproducibility is already under scrutiny, it wins on all six TECH_STACK section 22 rules, it runs fully offline from local artifacts with a command that is already available in the environment, and its structural inability to import `rads.reasoning` enforces SYSTEM.md section 34 Constraint 5 by construction rather than by discipline.

Codec caveat: `rads/output/visualizer.py` writes with the `mp4v` fourcc, which is MPEG-4 Part 2, not H.264. Browser support for MPEG-4 Part 2 in an mp4 container is inconsistent. Whether the written videos play in the demo browser is unverified and I did not test it, because no annotated video currently exists in the tree and generating one would require running the pipeline, which is out of scope for this task. This is open question Q3 and it affects Option A and Option C equally; Option B would hit the same limitation.

---

# 8. Explicitly Out of Scope

| Out of scope | Reason |
|---|---|
| Telegram alerting | SYSTEM.md section 24 places Telegram on a branch parallel to the dashboard, both consuming the structured event. TECH_STACK.md section 17 draws the same fork. It is a separate P1 consumer with its own formatter and its own delivery concerns, and no alert code or alert record exists. It should be specified separately. |
| Accident-type display beyond the placeholder | `accident_type` is the hardcoded string `"unknown"`. Rendering "Rear-End Collision" as SYSTEM.md section 21 and AI.md section 25 illustrate would be fabricating an output the system does not produce. Deferred until MVP.md section 6 accident-type classification is implemented. |
| Live streaming or real-time view | SYSTEM.md section 28 marks offline mode as "the initial MVP priority" and states "Real-time optimization is a future concern unless required for the MVP demonstration." MVP.md section 7 lists real-time processing as P2. There is no streaming source to render. |
| Authentication, users, roles | Nothing in any spec requires it. The demonstration is local and offline. Adding it creates a security surface with no corresponding requirement. |
| Any persistence layer | TECH_STACK.md section 23 names "Large database infrastructure" among technologies not to introduce without a clear reason. The artifacts are files on disk; the frontend reads them. A database would add a second source of truth and break reproducibility. |
| Any frontend recomputation of AI outputs | SYSTEM.md section 34 Constraint 5 and section 23. The specific forbidden operations are enumerated in section 2 of this document. |
| Editing configuration or triggering pipeline runs | Would make displayed results irreproducible, against TECH_STACK.md section 22 rule 4, and exceeds the consumer role. |
| Complete dashboard redesign | MVP.md section 34 names this explicitly among things that must not block the MVP. |

---

# 9. Open Questions Requiring a Human Decision

Each is phrased to be answerable in one line.

| # | Question | Why it blocks | Default if unanswered |
|---|---|---|---|
| Q1 | Does the dashboard read local files directly from disk, or is a small read-only serving layer acceptable? | Browsers block `fetch` of sibling files under `file://`, so a page opened by double-click cannot load result JSON. A one-line static server solves it. | Assume a read-only static server (`python -m http.server`), documented as a demo step. |
| Q2 | Is in-browser playback of the annotated mp4 and the evidence clip required for the demo, or is opening them in a media player acceptable? | Determines whether page 1 needs a video pane at all, and whether Q3 matters. | Assume playback is required, since MVP.md section 35 asks the demo to be understandable to a non-technical viewer. |
| Q3 | If playback is required, is re-encoding the visualizer output to H.264 acceptable? | `visualizer.py` writes `mp4v` (MPEG-4 Part 2); browser support is inconsistent and untested here. Changing the fourcc means editing pipeline code, which this outline does not propose. | Unknown. Test one generated clip in the demo browser before committing to a video pane. |
| Q4 | Which run is the demo run: P02 (74 clips) or the held-out split (500 clips)? | Page 2 needs one JSONL as its source, and the two have different provenance and different caveats. | Assume P02, because only P02 has a baseline comparison artifact. |
| Q5 | Will bug B7 be fixed before the frontend is built, so that negatives carry `evidence_list` and involved ids? | Changes the empty-state design of page 1 and makes false negatives explainable in the UI. | Assume B7 is fixed and design the empty state to degrade gracefully either way. |
| Q6 | Should the pipeline write a per-clip result JSON during batch evaluation, not just the JSONL row? | Page 2 links to page 1, but the JSONL row lacks `objects_involved`, `severity_detail`, `kinematics`, `tracks_summary` and `interaction_candidates`. Without per-clip JSON, page 1 only works for clips re-run individually. | Unknown. This is the single largest determinant of whether pages 1 and 2 connect. |
| Q7 | Is the demo audience technical or non-technical? | Decides whether the kinematics and candidate regions on page 1 are collapsed by default or absent. | Assume non-technical, per MVP.md section 35's "a viewer who has not read the code". |
| Q8 | Do pages 3 and 4 render the markdown artifacts live, or is a pre-render step at documentation time acceptable? | Live rendering needs a markdown library; pre-rendering needs a build step. | Assume live client-side rendering of the two files, since both are already committed markdown. |

Q6 is the one to answer first. If per-clip result JSON is not written during batch runs, the proposed page 1 is reachable only for manually re-run clips and the page inventory should be reconsidered.

---

# 10. Proposed Implementation Order

Smallest demonstrable slice first. Effort is a rough estimate in working days for one developer and assumes the corrected evaluation re-run has landed and Q1 through Q3 are answered. These are estimates, not measurements.

| Step | Deliverable | Depends on | Rough effort |
|---|---|---|---|
| 0 | Generate one annotated video and one full result JSON for a known-positive clip, as the fixture everything else is built against. Answers Q3. | Corrected re-run complete | 0.5 day |
| 1 | Page 1 for a single hardcoded result JSON: verdict, event times, objects involved, evidence list, severity band and factor table. No video, no index. This is the smallest thing that satisfies AI.md section 25. | Step 0 | 1 day |
| 2 | Add the video pane to page 1: annotated mp4 plus evidence clip when present. | Steps 0, 1; Q2, Q3 | 0.5 day |
| 3 | Add the collapsed detail region: tracks table, candidate table, raw JSON. | Step 1 | 0.5 day |
| 4 | Page 2, the run index, reading one evaluation JSONL with client-side filters and the TP/TN/FP/FN column. | Post-fix JSONL; Q4 | 1 day |
| 5 | Wire page 2 rows to page 1. | Steps 1, 4; **Q6** | 0.5 day if per-clip JSON exists; otherwise blocked pending a pipeline change that is outside this outline's scope |
| 6 | Page 3, evaluation summary, rendering `p02_baseline_comparison.md` and the required caveats. | Q8 | 0.5 day |
| 7 | Page 4, failure analysis browser. | Q8 | 0.5 day |
| 8 | Demo script: the exact commands and click path that reproduce the MVP.md section 35 flow end to end. | Steps 1 through 5 | 0.5 day |

Steps 0 through 3 deliver the P0 demonstration for a single clip and are the only steps required for MVP.md section 35. Steps 4, 5 and 8 complete the P0 set. Steps 6 and 7 are P1 and can be dropped without affecting the demonstration.

---

# 11. What This Document Does Not Establish

- It does not establish that a frontend should be built now. MVP.md section 6 and section 34, and the audit's Phase 9+ placement, all say it should not be, until the core MVP is complete.
- It does not establish a colour system, typography, spacing, iconography or any visual design. None of that is derivable from the specs and none is proposed.
- It does not verify that the `mp4v`-encoded outputs play in a browser. See Q3.
- It does not verify the serialised shape of `interaction_candidates`, because no committed artifact contains the list. See section 3.8.
- It does not reflect post-fix evaluation numbers, which do not exist yet. Every number quoted here is PRE-FIX and is labelled as such at each occurrence.
