RADS — Implement Minimal Accident-Reasoning Improvements
PHASE 8 MUST REMAIN ON HOLD

Claude has completed a forensic audit of the current RADS pipeline.

The audit found NO genuine software bugs. The current failures are limitations of the MVP rule-based reasoning system, with one overly aggressive heuristic that should be improved.

DO NOT implement Phase 8.
DO NOT create the 500-video split.
DO NOT create the Phase 8 evaluator.
DO NOT modify the P02 comparison.
DO NOT redesign the architecture.
DO NOT replace YOLO or ByteTrack.
DO NOT add optical flow, camera calibration, depth estimation, ML classifiers, transformers, or other major features.

Your task is ONLY to implement and verify the two minimal reasoning improvements below.

==================================================
P1-A — GUARD THE TRACK-LOSS EVIDENCE
==================================================

File:
rads/reasoning/accident_reasoner.py

Current problem:

The current reasoner gives a +0.6 accident-score boost whenever a track terminates shortly after an interaction.

This treats ordinary track termination, such as:
- a vehicle leaving the frame
- temporary tracking loss
- short-lived detections

as strong collision evidence.

This caused BOTH observed false positives.

Observed evidence:

FP1:
- surviving object's pre velocity ≈ 0.9 px/s
- surviving object remained essentially unaffected
- other track simply terminated

FP2:
- surviving object's velocity increased from ≈65 px/s to ≈88 px/s
- this is inconsistent with collision-induced disruption
- other track terminated

Implement a principled guard so track termination is NOT automatically treated as strong collision evidence.

The intended behavior:

If one track terminates, inspect the surviving track's kinematic behavior.

Strong track-loss evidence should require evidence that the surviving object itself was meaningfully affected, such as:
- significant deceleration
OR
- meaningful pre-interaction motion combined with post-interaction disruption.

If the surviving object is stationary or accelerates through the interaction, substantially reduce or suppress the track-loss contribution.

IMPORTANT:
Do NOT blindly copy Claude's exact suggested thresholds if the current code structure indicates a safer implementation.

Claude suggested:
- significant deceleration: post_vel < pre_vel * 0.5
- meaningful motion: pre_vel > 15 px/s
- otherwise reduce the +0.6 boost to approximately +0.1

Inspect the existing implementation first and preserve the existing scoring philosophy.

The goal is NOT to optimize the 10 videos.
The goal is to ensure that track termination is only strong evidence when the remaining object's behavior is consistent with a collision.

==================================================
P1-B — FIX THE PASSING-TRAFFIC FILTER
==================================================

File:
rads/reasoning/accident_reasoner.py

Current problem:

The existing passing-car filter is effectively bypassed whenever either track terminates.

Current logic requires:

not term_a AND not term_b

before the passing-traffic suppression logic is applied.

This is backwards for the failure mode we observed because the false positives are precisely cases where one track terminates.

Observed FP behavior:

FP1:
- one vehicle terminates
- surviving object remains essentially unaffected

FP2:
- one vehicle terminates
- surviving object accelerates after the interaction

Implement a minimal change so the surviving object's post-interaction behavior can still suppress a normal passing event even when the other track terminates.

The filter should distinguish:

NORMAL PASS:
- one object disappears/exits
- surviving object continues normally
- no meaningful deceleration/disruption

from:

POSSIBLE COLLISION:
- one object disappears
- surviving object shows meaningful kinematic disruption

Do NOT simply lower the accident threshold.
Do NOT remove the track-loss heuristic entirely.
Do NOT make arbitrary threshold changes solely to make these 10 videos pass.

==================================================
IMPORTANT IMPLEMENTATION RULES
==================================================

Before editing anything:

1. Inspect the current:
   - accident_reasoner.py
   - interaction_engine.py
   - motion_features.py
   - trajectory.py
   - relevant config
   - current experiment results

2. Understand exactly how:
   - pre_vel
   - post_vel
   - track termination
   - interaction candidates
   - accident score
   - passing filter
   are currently calculated.

3. Make the SMALLEST principled code change possible.

4. Do not modify unrelated modules.

5. Do not change the interaction thresholds.
   In particular, do NOT change:
   - normalized proximity threshold
   - min persistence frames
   - YOLO confidence threshold

6. Do not change frame_skip.
   The experiment has already established:
   frame_skip=1 is the configuration we will retain.

7. Do not modify the Phase 8 plan or Phase 8 files.

==================================================
VERIFICATION
==================================================

After implementation, rerun the SAME 10-video frame-skip experiment using:

frame_skip=1

Do NOT rerun skip=3 unless needed for debugging.

Compare against the original results.

We specifically want to verify:

Original:
- 6/10 correct
- 2 FP
- 2 FN

Expected desired behavior:
- FP1 (-6SQSDj8cYU_00.mp4 negative) should become NORMAL
- FP2 (-dmYsQc-odI_00.mp4 negative) should become NORMAL

Also verify that these true positives remain accidents:
- -6SQSDj8cYU_00.mp4 positive
- -7-vQ4obVwQ_00.mp4 positive
- -AztVDZ6cEE_00.mp4 positive

And these true negatives remain normal:
- -2UPLUV7JLg_00.mp4 negative
- -7-vQ4obVwQ_00.mp4 negative
- -9oifpjUxxM_00.mp4 negative

The two existing false negatives:
- -2UPLUV7JLg_00.mp4 positive
- -9oifpjUxxM_00.mp4 positive

are expected limitations of the current bbox-proximity/convergence approach and should NOT be artificially forced into positive predictions.

Do NOT tune the system to make those two pass.

==================================================
REGRESSION TEST
==================================================

After the original 10 videos, select 10 NEW videos:
- 5 positive
- 5 negative
- NOT used in the original frame-skip experiment.

Run the updated pipeline on these 10 videos with frame_skip=1.

Record:
- prediction
- confidence
- event time
- runtime
- any obvious false positive/false negative

This second sample is specifically to check that the reasoning changes did not simply overfit the original 10 videos.

==================================================
DOCUMENTATION
==================================================

Update the experiment documentation with:

1. What was changed.
2. Why it was changed.
3. Before/after results on the original 10 videos.
4. Results on the 10 new videos.
5. Any regression observed.
6. Whether P1-A and P1-B should be accepted.

Do NOT mark Phase 8 as complete.

Do NOT create or modify:
- picek_500_split.csv
- Phase 8 evaluator
- baseline comparison
- final benchmark results

Keep all new experiment outputs under:

rads/evaluation/experiments/

Use a clearly named directory such as:

rads/evaluation/experiments/track_loss_reasoning_fix/

==================================================
FINAL REPORT
==================================================

At the end, report:

BEFORE:
Accuracy:
FP:
FN:

AFTER:
Accuracy:
FP:
FN:

NEW 10-VIDEO REGRESSION TEST:
Accuracy:
FP:
FN:

Then provide a conclusion:

1. ACCEPT — improvements are safe enough to freeze
2. REJECT — regressions were introduced
3. INCONCLUSIVE — more testing needed

Remember:

This is a targeted reasoning improvement.
It is NOT Phase 8.
Do not implement anything beyond the scope above.