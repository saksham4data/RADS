# Reasoning Improvement Regression Test

## New 10-Video Sample (frame_skip=1)

| Video | Ground Truth | Prediction | Confidence | Event Time | Runtime |
|---|---|---|---|---|---|
| -FQxK6HdxNU_00.mp4 | accident | normal | 0.40 | N/A | 10.29s |
| -NgnSm_oEB4_00.mp4 | accident | accident | 1.00 | 0.42s | 12.22s |
| -PpBteU0p3Q_00.mp4 | accident | accident | 1.00 | 1.40s | 10.50s |
| -PpjzmhI_PE_00.mp4 | accident | accident | 1.00 | 3.73s | 12.80s |
| -Qt5bDJNT84_00.mp4 | accident | accident | 1.00 | 0.82s | 11.33s |
| -NgnSm_oEB4_00.mp4 | normal | accident | 1.00 | 0.42s | 7.58s |
| -PpBteU0p3Q_00.mp4 | normal | accident | 1.00 | 1.40s | 4.06s |
| -RE3XseZINA_00.mp4 | normal | normal | 0.00 | N/A | 8.65s |
| -RrDtLjWsT4_00.mp4 | normal | normal | 0.48 | N/A | 24.77s |
| -SNFUobKjoM_00.mp4 | normal | accident | 0.70 | 0.90s | 17.99s |

## Summary Statistics
- **Total Runtime:** 120.19s
- **Correct Predictions:** 6/10
- **False Positives:** 3
- **False Negatives:** 1