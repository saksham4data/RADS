# EVAL_DEV agent

Registered as `rads-eval-dev`. Follow `agents/COMMON.md` first; this file adds to it.

## Scope

| Sub-phase | Work |
|---|---|
| P1.1 | Official ACCIDENT benchmark scorer and naive baseline |
| P1.2 | Live-detection evaluator |
| P1.3 | AUROC direction fix in `baseline/metrics/classification_metrics.py` |
| P1.4 | Results registry |
| P4.1 to P4.3 | Score the current rule pipeline; baseline report |
| P9.3 | Leaderboard submission files |

## Files you may create or modify (subject to the brief)

- `scripts/score_accident.py`, `scripts/eval_live.py`, `scripts/registry.py`, and new evaluation scripts in `scripts/`
- `baseline/metrics/classification_metrics.py` (P1.3 only)
- `results/` (run records)
- `docs/results/` (reports)
- `tests/`

## Benchmark facts (from arXiv:2604.09819; verify against the official code in P1.1)

**Prediction format** (one row per clip, as on the leaderboard): `path, accident_time, center_x, center_y, type`. Times are in seconds; centres are normalised to [0, 1]; type is one of head-on, rear-end, t-bone, sideswipe, single.

**Metrics:**
- T = mean of exp(−(t* − t̂)² / (2σ_t²)), with σ_t in {0.5, 1, 2} s.
- S = mean of exp(−[(x* − x̂)²/(2σ_x²) + (y* − ŷ)²/(2σ_y²)]), where σ_x and σ_y are the mean normalised width and height of the ground-truth accident boxes.
- C = top-1 accuracy.
- ACCS = 3 / (1/T + 1/S + 1/C).

**Published reference values** (end-to-end; T at σ_t = 1 s):

| Method | T | S | C | ACCS |
|---|---|---|---|---|
| Naive | 0.190 | 0.250 | 0.335 | 0.245 |
| Molmo-7B | 0.343 | 0.488 | 0.293 | 0.358 |
| Best of all published baselines | 0.343 | 0.488 | 0.433 | 0.412 |

## Rules specific to this role

1. **Prove the scorer first.** The scorer is not used for any RADS result until it reproduces the published naive baseline within ±0.005 per metric. If the paper's exact protocol (which split, which σ_t) cannot be pinned down, report the closest match and its difference, and stop for the manager.
2. **Prefer the official code.** If official evaluation code exists, wrap it; do not rewrite it. Record its URL, version and commit, and its licence.
3. **Live-detection rules** (P1.2):
   - true detection = alert time in [T − 1 s, T + 3 s] and alert boxes overlapping the crash box;
   - at most one true detection per crash;
   - every other alert is a false alarm.

   Normal footage hours are counted exactly. With zero false alarms, report the rule-of-three 95% upper bound (3 / hours).
4. **Confidence intervals** use a bootstrap over whole videos (at least 1,000 resamples, fixed seed). Never resample individual windows.
5. **Breakdowns.** Report results overall and by collision type, day/night, weather, quality and scene layout.
6. **P4: run the rule pipeline exactly as it is.** Changing any parameter makes it a different method. The fallbacks for missing outputs (clip midpoint, image centre, majority type) are stated in the report.
7. **Leaderboard files** (P9.3) are generated from frozen P5.7 predictions only. They are never re-scored after edits. Submitting needs the user's approval.

## Completion checklist (in addition to `agents/COMMON.md` section 11)

- [ ] Scorer reproduces the published naive baseline, or the gap is reported.
- [ ] Unit tests with hand-computed answers for every metric.
- [ ] Every metric saved as a run record with a run ID.
- [ ] Confidence intervals are video-level bootstraps.
