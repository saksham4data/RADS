# MODEL_DEV agent

Registered as `rads-model-dev`. Follow `agents/COMMON.md` first; this file adds to it.

## Scope

| Sub-phase | Work |
|---|---|
| P3.3 | Tracker quality on synthetic ground truth |
| P5.1 | Feature extractor (shared by batch and live) |
| P5.2 | LightGBM event model |
| P5.3 | Impact time and impact point |
| P5.4 | Collision-type model |
| P5.5 | Single-track branch |
| P5.6 | Threshold and alert parameters |
| P5.7 | Single test run on IID, OOD and zero-shot |
| P6.1 to P6.4 | Severity formula, synthetic speed check, rubric and labelling tool, fitting and agreement |
| P9.1, P9.2 | ResNet18 + GRU retrain (optional), ablations |

## Files you may create or modify (subject to the brief)

- `rads/features.py`, `rads/event_model.py`, `rads/collision_type.py`, `rads/severity.py`
- `rads/config.yaml`: new sections only, plus severity weights
- `scripts/` (training and experiment scripts)
- `models/rads/<version>/`: trained model files, config and hashes
- `docs/severity/`
- `baseline/` (P9.1 only)
- `tests/`

## Method summary (from `docs/RADS_STRATEGY.md` sections 4 and 5)

**Input.** Cached tracks, resampled to 10 Hz.

**Features:**
- **Speed** in body lengths per second (displacement divided by the object's own box diagonal, per second).
- **Per track:** speed, acceleration, jerk, largest deceleration, heading change, aspect-ratio change, how the track ended (frame border, occlusion under another box, or mid-scene), stationary afterwards, class, confidence.
- **Per pair:** normalised distance and its minimum, time to collision, closing speed, relative heading, box overlap and its duration, speed change of both objects across the closest point, track endings near that point.
- **Scene:** number of tracks, mean speed.

**Event model.** LightGBM over pair and single-track candidates per window. Class weights; no duplicated samples.

**Impact.**
- Time = peak of the smoothed event score.
- Point = contact point of the pair's boxes (single-vehicle: box centre).

**Collision type.** A 5-class LightGBM on features at impact: relative heading, contact edges, following relation, number of involved tracks, rollover indicator.

**Severity:**
- Score = w_E·min(v_c/v_ref, 1.5)² + w_D·min(Δv/Δv_ref, 1.5) + w_V·V + w_H·H + w_R·R + w_S·S + w_N·min(N − 1, 2).
- Overrides: vulnerable road user at speed, or rollover, gives at least MEDIUM.
- Every term's value and contribution is returned.

## Rules specific to this role

1. **Train and tune only on train splits,** with the 5 grouped folds from P2.2. Folds are fixed by DATA_DEV; do not re-split.
2. **Track augmentations** (position jitter, dropped detections, ±20% time stretch, horizontal flip) apply only to training folds, never to validation folds or test data.
3. **One feature implementation.** `rads/features.py` must give identical results in batch (from the cache) and in streaming (frame by frame). This is tested on real videos.
4. **Synthetic data** is used for training only, never for testing. Report every result both with and without synthetic training data when the plan asks for it.
5. **Threshold choice** (P5.6) uses cross-validation normal windows only. The objective: the highest recall with false alarms per camera-hour ≤ 0.05 on those windows.
6. **P5.7 is run once per protocol** with frozen models. Record model file hashes in the run. If results miss a target, report them; do not iterate on the test set.
7. **Severity has no ground truth.** Never describe it as accurate or as predicting injuries. Report agreement with one annotator's rubric labels (kappa), self-agreement, and the rollover check, and state "single labeller".
8. **Synthetic speed check** (P6.2): the field of view is an assumption. Validate it against vehicle `extent` before using projected speeds, and report the validation result.
9. **Model files** go to `models/rads/<version>/` with `config.json` (features, hyperparameters, seed, train data, cache version) and a SHA-256 for every file.

## Completion checklist (in addition to `agents/COMMON.md` section 11)

- [ ] No test-split file opened by training or tuning code (state how you checked).
- [ ] Cross-validation results with 95% intervals.
- [ ] Batch and streaming feature parity test passes.
- [ ] Model versioned with config and hashes.
- [ ] Severity outputs include every term.
