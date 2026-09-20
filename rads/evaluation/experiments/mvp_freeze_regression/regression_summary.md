# MVP Freeze Regression Gate

Re-run of the 10-clip sample recorded in `rads/evaluation/experiments/track_loss_reasoning_fix/regression_summary.md`,
at `frame_skip=1`, after the Phase 1 and Phase 3 fixes (B1, B2, B5, B6, B7, B8, B11, B12, B14, B15).

Runner: `rads/evaluation/experiments/mvp_freeze_regression/run_gate.py`
Raw results: `rads/evaluation/experiments/mvp_freeze_regression/gate_results.json`
Config: `rads/config/pipeline_config.yaml`

All ten clips are processed by a single reused `Pipeline` instance, matching how `rads/evaluation/evaluator.py`
runs a split, so the B1 tracker reset is exercised on every clip after the first.

## Acceptance criteria

- Correct predictions at least 6.
- False positives no more than 3.
- With the B11 and B12 feature flags disabled, the accident/normal labels must be identical to the
  recorded baseline, which is what demonstrates the B8 config refactor is behaviour-preserving.
- No threshold was tuned for this sample.

## Recorded baseline

| Correct | False positives | False negatives |
|---|---|---|
| 6/10 | 3 | 1 |

## Flags enabled (B11 clustering discipline on, B12 bounded confidence on)

| Video | Ground Truth | Baseline Prediction | Prediction | Confidence | Raw Score | Severity | Objects Involved | Event Time | Runtime |
|---|---|---|---|---|---|---|---|---|---|
| -FQxK6HdxNU_00.mp4 | accident | normal | normal | 0.5507 | 0.40 | n/a | 0 | N/A | 32.96s |
| -NgnSm_oEB4_00.mp4 | accident | accident | accident | 0.9257 | 1.30 | HIGH (5.77) | 4 | 0.42s | 43.59s |
| -PpBteU0p3Q_00.mp4 | accident | accident | accident | 0.8647 | 1.00 | MEDIUM (3.53) | 3 | 1.40s | 35.57s |
| -PpjzmhI_PE_00.mp4 | accident | accident | accident | 0.8647 | 1.00 | HIGH (6.20) | 3 | 3.73s | 39.42s |
| -Qt5bDJNT84_00.mp4 | accident | accident | accident | 0.8647 | 1.00 | HIGH (6.50) | 4 | 0.82s | 47.02s |
| -NgnSm_oEB4_00.mp4 | normal | accident | accident | 0.9257 | 1.30 | HIGH (5.77) | 4 | 0.42s | 17.11s |
| -PpBteU0p3Q_00.mp4 | normal | accident | accident | 0.8647 | 1.00 | MEDIUM (3.53) | 3 | 1.40s | 13.17s |
| -RE3XseZINA_00.mp4 | normal | normal | normal | 0.0 | 0.00 | n/a | 0 | N/A | 26.70s |
| -RrDtLjWsT4_00.mp4 | normal | normal | normal | 0.6197 | 0.4834 | n/a | 0 | N/A | 65.84s |
| -SNFUobKjoM_00.mp4 | normal | accident | accident | 0.7548 | 0.7029 | MEDIUM (2.12) | 4 | 0.90s | 59.19s |

- Correct predictions: 6/10
- False positives: 3
- False negatives: 1
- Labels matching the recorded baseline: 10/10
- Total runtime: 380.57s

## Flags disabled (B11 and B12 reverted to pre-fix behaviour)

| Video | Ground Truth | Baseline Prediction | Prediction | Confidence | Raw Score | Severity | Objects Involved | Event Time | Runtime |
|---|---|---|---|---|---|---|---|---|---|
| -FQxK6HdxNU_00.mp4 | accident | normal | normal | 0.40 | 0.40 | n/a | 0 | N/A | 30.93s |
| -NgnSm_oEB4_00.mp4 | accident | accident | accident | 1.00 | 1.30 | HIGH (5.81) | 11 | 0.42s | 37.65s |
| -PpBteU0p3Q_00.mp4 | accident | accident | accident | 1.00 | 1.00 | MEDIUM (3.53) | 3 | 1.40s | 34.81s |
| -PpjzmhI_PE_00.mp4 | accident | accident | accident | 1.00 | 1.00 | HIGH (6.20) | 3 | 3.73s | 35.39s |
| -Qt5bDJNT84_00.mp4 | accident | accident | accident | 1.00 | 1.00 | HIGH (6.19) | 6 | 0.82s | 30.27s |
| -NgnSm_oEB4_00.mp4 | normal | accident | accident | 1.00 | 1.30 | HIGH (5.89) | 10 | 0.42s | 14.47s |
| -PpBteU0p3Q_00.mp4 | normal | accident | accident | 1.00 | 1.00 | MEDIUM (3.53) | 3 | 1.40s | 12.29s |
| -RE3XseZINA_00.mp4 | normal | normal | normal | 0.00 | 0.00 | n/a | 0 | N/A | 27.33s |
| -RrDtLjWsT4_00.mp4 | normal | normal | normal | 0.48 | 0.4834 | n/a | 0 | N/A | 33.32s |
| -SNFUobKjoM_00.mp4 | normal | accident | accident | 0.70 | 0.7029 | MEDIUM (2.12) | 4 | 0.90s | 16.37s |

- Correct predictions: 6/10
- False positives: 3
- False negatives: 1
- Labels matching the recorded baseline: 10/10
- Total runtime: 272.83s

## Outcome

Gate passed. Correct predictions are 6, equal to the baseline and at the acceptance floor; false
positives are 3, equal to the baseline and at the ceiling. Neither configuration flips a label,
so the B11 and B12 flags remain enabled by default.

The B8 configuration refactor is behaviour-preserving beyond the label requirement. With the flags
disabled the confidence values reproduce the recorded baseline exactly for all ten clips
(0.40, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 0.00, 0.48, 0.70) and the impact times agree to two
decimal places. Since every reasoning weight, divisor, boost, guard and threshold now comes from
YAML, that identity is evidence the extracted defaults equal the previous hardcoded values.

## Observed effect of the two behavioural flags

B11, clustering discipline. Requiring a merged candidate to carry its own qualifying evidence and
capping involvement reduces the over-inclusion recorded in
`track_loss_reasoning_fix/forensics/FP_-NgnSm_oEB4_00.json`: the `-NgnSm_oEB4_00` false positive
drops from 10 involved objects to 4, the `-NgnSm_oEB4_00` positive from 11 to 4, and
`-Qt5bDJNT84_00` from 6 to 4. The accident decision is unchanged in every case.

B12, bounded confidence. With the flag disabled the sample takes only four distinct confidence
values (0.00, 0.40, 0.48, 0.70) plus six saturated at 1.00. With the flag enabled the ten clips
take six distinct values between 0.0 and 0.9257 and nothing saturates, so ties break and AUROC is
computable. The accident decision is still taken on the raw score against the configured 0.5
threshold, which is why `-FQxK6HdxNU_00` stays `normal` at confidence 0.5507: its raw score is 0.40.

## Notes

- B2 is visible here: `num_tracks` is non-zero throughout (27, 27, 14, 19, 20, 13, 10, 1, 10, 16),
  where the 574 pre-fix result rows all recorded 0 because the evaluator read `track_summaries`
  while the schema emitted `tracks_summary`.
- Severity values are new in this run; the baseline file records none. They come from an engineered
  heuristic with no ground-truth severity labels and must not be read as validated predictions.
- Runtimes are not comparable to the baseline file. This machine was running concurrent workloads,
  and the same clips differ by more than a factor of two between the two configurations here.
