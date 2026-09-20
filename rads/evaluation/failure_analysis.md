# RADS Failure Analysis (pre-fix)

Written 2026-09-20. Satisfies the MVP.md section 28 requirement that failure analysis name the stage responsible, and the Phase 8 task in `docs/architecture/IMPLEMENTATION_AUDIT_AND_ORDER.md` requiring at least five documented failure cases.

**Provenance and validity.** Every case below is derived from forensic JSON dumps produced by pre-fix runs, held under `rads/evaluation/experiments/frame_skip_comparison/forensics/` and `rads/evaluation/experiments/track_loss_reasoning_fix/forensics/`. Those runs predate the tracker-state fix (B1), where one `Tracker` instance was reused across videos with `persist=True`, so track IDs, track lifespans and therefore every candidate in these files can change after the fix. All eight cases must be re-confirmed against the post-fix batch before they are cited as current behaviour. Nothing here was re-run; no pipeline execution was performed to produce this document.

**Decision threshold.** `rads/config/config_loader.py` line 82 returns `reasoning.accident_score_threshold` with a default of 0.5. All "below threshold" statements below are against 0.5.

**Reading the numbers.** `norm_prox` and `min_distance` are centre distances normalised by object size; smaller is closer. Negative `rel_vel` means the pair is separating. Velocities are image-space pixels per second; MVP scope forbids world-coordinate scaling, so a short track on a fast-moving or jittering box can report several hundred px/s without physical meaning.

## Failure category distinction

MVP.md section 28 requires separating a clip that produced no interaction candidate at all from a clip where candidates existed and the reasoner rejected them. The two need different fixes.

| Category | Meaning | Cases below |
|---|---|---|
| A. No candidate produced | The interaction stage emitted zero candidates, so the reasoner had nothing to score. Fix lies upstream in detection, tracking or the interaction gates. | 1, 2, 8 |
| B. Candidates produced, rejected | Candidates existed; the aggregate score fell below 0.5. Fix lies in reasoning weights or evidence quality. | 3 |
| C. Candidates produced, wrongly accepted | Candidates existed and scored above 0.5 on evidence that does not indicate a collision. Fix lies in reasoning rules and clustering. | 4, 5, 6, 7 |

---

## Case 1 — `-2UPLUV7JLg_00`, false negative, interaction stage (category A)

Source: `frame_skip_comparison/forensics/FN_-2UPLUV7JLg_00.json`.

| Field | Value |
|---|---|
| Ground truth | accident |
| Prediction | normal, confidence 0.0 |
| Clip | 79 frames, 9.966 fps, 7.93 s |
| Unique tracks | 8 |
| Pairs formed | 10 |
| Interaction candidates | 0 |

Stage responsible: **interaction**, with **tracking** as the upstream contributor.

Evidence. Four of the eight tracks (IDs 1, 2, 3, 18) have `lifespan_frames: 0` and a single frame each, and their motion features are identically zero (`total_displacement 0.0`, `mean_velocity 0.0`). They cannot contribute kinematic evidence. Of the four surviving tracks, the only pair that ever reaches close proximity is `1_2` at frame 0, `norm_prox 0.007` with `iou 0.9566` over one co-existing frame — two boxes on the same object, not two vehicles interacting. The longest-lived pair, `4_12` with 24 co-existing frames, separates monotonically: `norm_prox` rises 3.174 to 5.694 and `rel_vel` is negative at every frame after the first, from -80.01 down to -345.75. Every remaining pair sits above `norm_prox 2.6` with `iou 0.0`. `accident_result.evidence_list` is empty and `severity` is null.

Interpretation: the accident objects were never tracked long enough or close enough in image space for a candidate to form. The reasoner was never consulted.

## Case 2 — `-9oifpjUxxM_00`, false negative, interaction stage (category A)

Source: `frame_skip_comparison/forensics/FN_-9oifpjUxxM_00.json`.

| Field | Value |
|---|---|
| Ground truth | accident |
| Prediction | normal, confidence 0.0 |
| Clip | 120 frames, 14.947 fps, 8.03 s |
| Unique tracks | 14 |
| Pairs formed | 26 |
| Interaction candidates | 0 |

Stage responsible: **interaction**.

Evidence. Tracking is healthier here than in Case 1 — tracks 3, 22, 29, 34 and 51 run for 52, 50, 55, 62 and 31 frames respectively — so the failure is not track fragmentation. The minimum normalised proximity across all 26 pairs is 1.171 (`34_60`, 3 co-existing frames) and the maximum IoU across all 26 pairs is 0.0. No pair ever produces bounding-box contact, and no pair crosses the proximity gate. `interaction_candidates` is an empty list; `evidence_list` is empty.

Interpretation: distinct from Case 1. Objects were tracked well, but never came close enough in image space under the current normalised-proximity gate. The gate, not the tracker, is the lever.

## Case 3 — `-FQxK6HdxNU_00`, false negative, reasoning stage (category B)

Source: `track_loss_reasoning_fix/forensics/FN_-FQxK6HdxNU_00.json`.

| Field | Value |
|---|---|
| Ground truth | accident |
| Prediction | normal, confidence 0.40 |
| Unique tracks | 27 |
| Interaction candidates | 46 |

Stage responsible: **reasoning**.

Evidence. This is the direct contrast to Cases 1 and 2: 46 candidates were produced and scored, and the best aggregate score was 0.40, below the 0.5 threshold. Candidates carrying the strongest evidence include:

| Pair | Frames | min_distance | peak_iou | max_rel_vel | Evidence |
|---|---|---|---|---|---|
| 4_27 | 53-60 | 0.4024 | 0.1824 | 74.15 | overlap, proximity_and_convergence, track_loss (obj 27 terminated, 88.99 to 41.06) |
| 1_5 | 1-36 | 0.3064 | 0.3129 | 274.01 | overlap, proximity_and_convergence |
| 47_63 | 63-68 | 0.3290 | 0.1183 | 106.01 | overlap, proximity_and_convergence |
| 1_2 | 34-43 | 0.7543 | 0.0 | 272.02 | proximity_and_convergence, track_loss (obj 1 terminated, 143.77 to 0.0) |

A second defect is visible in the same record: `accident_result.evidence_list` is `[]` and `involved_object_ids` is `[]` despite 46 scored candidates. The evidence is computed and then discarded when the decision is negative, which is bug B7 and which makes negative cases unauditable without the forensic dump.

## Case 4 — `-NgnSm_oEB4_00`, false positive, reasoning stage (category C)

Source: `track_loss_reasoning_fix/forensics/FP_-NgnSm_oEB4_00.json`.

| Field | Value |
|---|---|
| Ground truth | normal |
| Prediction | accident, confidence 1.00 |
| Clip span | frames 0-33 |
| Unique tracks | 13 |
| Interaction candidates | 38 |
| Evidence | proximity_and_convergence, track_loss, sudden_deceleration |
| Involved object IDs | 1, 2, 3, 5, 6, 9, 10, 12, 14, 23 (10 of 13 tracks) |

Stage responsible: **reasoning**, plus **involved-object clustering** (bug B11).

Evidence. The track-loss rule fires on tracks that are too short to have reliable velocity. Track 5 exists for frames 0-3 only and reports `pre_vel 366.01`, `post_vel 0.0`, `terminated: true`; track 12 exists for frames 6-9 and reports `pre_vel 496.58`, `post_vel 0.0`, `terminated: true`. Those velocities are image-space artefacts of four-frame tracks, and their disappearance is read as an impact. The deceleration rule then fires on the same artefacts: candidate `9_10` has `pre_vel 250.61` to `post_vel 66.78`, candidate `1_12` has `255.36` to `74.58`.

Separately, naming 10 of 13 tracks as involved in one event is not a plausible collision description. This is the over-inclusion recorded as B11 and it makes the severity object-count factor meaningless on this clip.

## Case 5 — `-SNFUobKjoM_00`, false positive, interaction and reasoning stages (category C)

Source: `track_loss_reasoning_fix/forensics/FP_-SNFUobKjoM_00.json`.

| Field | Value |
|---|---|
| Ground truth | normal |
| Prediction | accident, confidence 0.70 |
| Clip span | frames 0-149 |
| Unique tracks | 16 |
| Interaction candidates | 36 |
| Evidence | overlap, proximity_and_convergence, track_loss |
| Involved object IDs | 2, 3, 4, 20 |

Stage responsible: **interaction** gates, confirmed by **reasoning**.

Evidence. The two firing candidates involve near-stationary vehicles:

| Pair | Frames | min_distance | peak_iou | max_rel_vel | Velocity context |
|---|---|---|---|---|---|
| 4_20 | 14-20 | 0.2776 | 0.3137 | 8.84 | obj 4: 4.71 to 0.0, terminated |
| 3_20 | 21-27 | 0.2023 | 0.4378 | 12.86 | obj 20: 3.95 to 0.0, terminated |

Maximum relative velocity of 8.84 and 12.86 px/s is at the level of detection jitter, and track 2 reports velocities between 0.57 and 4.36 px/s across its 150-frame life. Track 20 exists for frames 14-27 only. Two parked or queued vehicles whose boxes overlap, one of which the tracker drops, satisfy `overlap` plus `track_loss` and reach 0.70.

Interpretation: the interaction gates admit overlap between stationary objects with no closing speed, and the reasoner has no minimum-velocity guard to reject them.

## Case 6 — `-PpBteU0p3Q_00`, false positive, reasoning stage (category C)

Source: `track_loss_reasoning_fix/forensics/FP_-PpBteU0p3Q_00.json`.

| Field | Value |
|---|---|
| Ground truth | normal |
| Prediction | accident, confidence 1.00 |
| Clip span | frames 0-33 |
| Unique tracks | 10 |
| Interaction candidates | 4 |
| Evidence | proximity_and_convergence, track_loss |
| Involved object IDs | 1, 2, 3 |

Stage responsible: **reasoning**.

Evidence. Both firing candidates have `peak_iou: 0.0`, that is no bounding-box contact at any frame, yet the result is accident at confidence 1.00.

| Pair | Frames | min_distance | peak_iou | max_rel_vel | Velocity context |
|---|---|---|---|---|---|
| 1_3 | 2-22 | 0.7780 | 0.0 | 283.43 | obj 1: 140.54 to 0.0, terminated at frame 22 of 33 |
| 2_3 | 1-13 | 0.9132 | 0.0 | 44.06 | obj 2: 67.50 to 0.0, terminated at frame 13 of 33 |

Track 1 runs frames 0-22 and track 2 runs frames 0-13 while the clip continues to frame 33. Both are ordinary mid-clip tracker drops, and both are scored as impacts. The track-loss rule alone, with zero overlap, is sufficient to reach the maximum confidence.

## Case 7 — `-dmYsQc-odI_00`, false positive, tracking then reasoning (category C)

Source: `frame_skip_comparison/forensics/FP_-dmYsQc-odI_00.json`.

| Field | Value |
|---|---|
| Ground truth | normal |
| Prediction | accident, confidence 1.00 |
| Clip | 38 frames, 8.033 fps, 4.73 s |
| Unique tracks | 12 |
| Interaction candidates | 5 |
| Reported impact_time | 3.4856 s |
| Severity | MEDIUM |
| Involved object IDs | 49, 25, 7 |

Stage responsible: **tracking** (ID churn) propagating into **reasoning**, with a **severity** label attached to a non-event.

Evidence. The firing candidate `25_49` spans frames 26-28 with `min_distance 0.9968`, `peak_iou 0.0` and `max_rel_vel 54.08`. Its velocity context shows both objects terminating at the same frame: track 25 `113.58` to `0.0`, track 49 `75.25` to `0.0`. Track 25 (truck) lives frames 12-28 and track 49 (truck) lives frames 24-28. Two trucks whose tracks end on the same frame is a tracker drop on a single physical vehicle, not a two-vehicle collision. The clip has 12 tracks in 38 frames with six of them shorter than seven frames (IDs 11, 20, 33, 49, 61 and 8), which is the signature of unstable IDs.

A severity of MEDIUM was emitted for this non-event. Per MVP.md section 21 there is no severity ground truth, so severity carries no correctness signal here; it only demonstrates that severity is produced unconditionally whenever the reasoner fires.

## Case 8 — single-vehicle accidents, structurally undetectable (bug B9, category A)

Stage responsible: **interaction**. Reasoning is pairwise only, so a clip whose accident involves one vehicle and no second tracked object can never produce an interaction candidate, and therefore can never be scored as an accident regardless of thresholds.

Affected clips were identified by reading `rads/config/p02_test_split.csv` and cross-referencing `rads/evaluation/p02_test_results.jsonl` on `video_id`. The `collision_type` distribution over the 74 rows is: `single` 16, `t-bone` 16, `head-on` 14, `rear-end` 14, `sideswipe` 14.

Sixteen rows carry `collision_type: single`. They form eight source pairs, each contributing one `positive_accident` row and one `negative_pre_accident` row.

| video_id | binary_label | clip_role | prediction | confidence | Outcome |
|---|---|---|---|---|---|
| picek_p02_pos_UcJKeuDYOzc_9_00 | 1 | positive_accident | 0 | 0.30 | FN |
| picek_p02_pos_SBIUNqe_XTk_00 | 1 | positive_accident | 0 | 0.00 | FN |
| picek_p02_pos_o5neAPqmNm8_00 | 1 | positive_accident | 0 | 0.00 | FN |
| picek_p02_pos_71QSBkIXKXI_00 | 1 | positive_accident | 0 | 0.00 | FN |
| picek_p02_pos_HXttLdePt0k_00 | 1 | positive_accident | 0 | 0.00 | FN |
| picek_p02_pos_0puo8kJOlmU_00 | 1 | positive_accident | 0 | 0.06 | FN |
| picek_p02_pos_7m77G8C7hiE_00 | 1 | positive_accident | 0 | 0.04 | FN |
| picek_p02_pos_-dVU8qW4ik8_00 | 1 | positive_accident | 1 | 1.00 | TP |
| picek_p02_neg_UcJKeuDYOzc_9_00 | 0 | negative_pre_accident | 0 | 0.11 | TN |
| picek_p02_neg_SBIUNqe_XTk_00 | 0 | negative_pre_accident | 0 | 0.00 | TN |
| picek_p02_neg_o5neAPqmNm8_00 | 0 | negative_pre_accident | 0 | 0.00 | TN |
| picek_p02_neg_71QSBkIXKXI_00 | 0 | negative_pre_accident | 0 | 0.00 | TN |
| picek_p02_neg_HXttLdePt0k_00 | 0 | negative_pre_accident | 0 | 0.00 | TN |
| picek_p02_neg_0puo8kJOlmU_00 | 0 | negative_pre_accident | 0 | 0.00 | TN |
| picek_p02_neg_7m77G8C7hiE_00 | 0 | negative_pre_accident | 0 | 0.04 | TN |
| picek_p02_neg_-dVU8qW4ik8_00 | 0 | negative_pre_accident | 1 | 1.00 | FP |

Counts: 16 `single` clips, of which 8 are positives. Seven of the 8 positives were missed. Named: `picek_p02_pos_UcJKeuDYOzc_9_00`, `picek_p02_pos_SBIUNqe_XTk_00`, `picek_p02_pos_o5neAPqmNm8_00`, `picek_p02_pos_71QSBkIXKXI_00`, `picek_p02_pos_HXttLdePt0k_00`, `picek_p02_pos_0puo8kJOlmU_00`, `picek_p02_pos_7m77G8C7hiE_00`.

Five of the seven missed positives were returned with confidence exactly 0.00, consistent with no candidate having been produced at all rather than a candidate scoring low.

The one detected single-vehicle positive, `picek_p02_pos_-dVU8qW4ik8_00`, cannot be counted as a success for single-vehicle reasoning: its `negative_pre_accident` counterpart from the same source was also predicted accident at confidence 1.00, so the pair was classified identically regardless of label. Whatever fired on that source fired on both clips.

Caveat, stated rather than hidden: the pre-fix result rows carry `num_tracks: 0` for all 74 P02 rows and all 500 held-out rows, because the evaluator read `track_summaries` while the schema emitted `tracks_summary` (bug B2). The per-clip candidate counts for these 16 clips are therefore not recoverable from the JSONL; the structural argument above rests on the pairwise design of the reasoner and on the confidence values, not on a recorded candidate count for these specific clips.

---

## Note on `FP_-6SQSDj8cYU_00.json`

That file is named as a false positive but its recorded `accident_result` is `accident: false, confidence: 0.15`, with `start_time`, `impact_time` and `end_time` all null and an empty `involved_object_ids`. Its `ground_truth` field is `normal`. On its own contents it is a correct rejection, not a false positive. `frame_skip_comparison/comparison_summary.md` confirms that the normal clip of `-6SQSDj8cYU_00` was predicted normal at both `frame_skip=1` and `frame_skip=3`; the false positive that the filename refers to is not present in this dump. The file is not used as a failure case here.

It is still informative as a negative control for Case 6: eight candidates were produced, five of them carrying `track_loss`, including `3_7` with `max_rel_vel 384.32` and object 3 terminating from `301.35` to `0.0`. The aggregate reached only 0.15. The same evidence type that drives Case 6 to 1.00 produces 0.15 here, which indicates the track-loss contribution is being amplified by co-occurring evidence rather than acting alone. This should be re-examined after the Phase 3 reasoning changes land.

---

## Limitations

### B9 — pairwise-only reasoning cannot detect single-vehicle accidents

Status: accepted and documented limitation of this MVP. Not a defect to be fixed in this window.

Statement. `rads/reasoning/accident_reasoner.py` scores interaction candidates, and an interaction candidate is by construction a pair of tracked objects. A single-vehicle accident — a rollover, a run-off-road, a collision with unmodelled static infrastructure — produces no second track and therefore no candidate. No threshold change can recover these clips; the recall ceiling on them is structural.

Measured impact on the P02 test split: 8 of the 37 positive clips (21.6 percent) are `collision_type: single`, and 7 of those 8 were predicted normal. The listing above names them.

Decision, per the MVP completion plan section 2: reasoning stays pairwise for this MVP. The rationale is that a single-track anomaly path is a new detection mechanism with its own false-positive profile, and introducing it inside the evaluation window would make the measured numbers uninterpretable.

P1 fix: a single-track anomaly path scoring one track in isolation on abrupt heading change, abrupt deceleration, post-event displacement collapse and aspect-ratio change indicating rollover, with its own threshold and its own evidence entry, so that single-vehicle detections remain separable from pairwise detections in the output and in the metrics.

Reporting obligation while the limitation stands: any recall figure quoted for RADS on a split containing `single` clips must state the proportion of those clips, because the achievable recall is bounded below 100 percent by construction. MVP.md section 38 requires honest measurement over impressive numbers; quoting recall without this note would violate it.

### Secondary limitations visible in the cases above

| Limitation | Evidence | Disposition |
|---|---|---|
| Severity has no ground truth | Case 7 emits `MEDIUM` on a non-event | Stated per MVP.md section 21; severity is an engineered heuristic, not a validated prediction |
| No world-coordinate scaling | Velocities of 366 and 497 px/s on four-frame tracks in Case 4 | In MVP scope by design; means velocity thresholds are resolution- and framerate-dependent |
| Negative decisions carry no evidence | Case 3 has 46 candidates and an empty `evidence_list` | Bug B7, being fixed in Phase 3 |
| `num_tracks` is 0 on every recorded row | Both JSONL result files | Bug B2, being fixed in Phase 1; blocks per-clip stage attribution from the JSONL alone |

---

## Re-confirmation checklist for the post-fix batch

| Case | What to re-check | Status |
|---|---|---|
| 1 | Whether track fragmentation persists once the tracker is reset per video | PENDING |
| 2 | Whether the minimum normalised proximity over all pairs is still above the gate | PENDING |
| 3 | Whether the best candidate still scores below 0.5, and whether `evidence_list` is now populated on negatives | PENDING |
| 4 | Whether `involved_object_ids` is still 10 of 13 tracks after the B11 clustering cap | PENDING |
| 5 | Whether a minimum-velocity guard now rejects the stationary-overlap candidates | PENDING |
| 6 | Whether track-loss alone, with `peak_iou 0.0`, still reaches confidence 1.00 | PENDING |
| 7 | Whether the same-frame termination of tracks 25 and 49 still fires | PENDING |
| 8 | Recount of `single`-clip outcomes on the post-fix P02 run | PENDING |
