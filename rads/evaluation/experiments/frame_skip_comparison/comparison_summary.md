# Frame Skip Comparison: 1 vs 3

| Video | Ground Truth | Skip 1 Prediction | Skip 3 Prediction | Skip 1 Event Time | Skip 3 Event Time | Skip 1 Runtime | Skip 3 Runtime | Notes |
|---|---|---|---|---|---|---|---|---|
| -2UPLUV7JLg_00.mp4 | accident | normal | normal | N/A | N/A | 14.99s | 4.56s |  |
| -6SQSDj8cYU_00.mp4 | accident | accident | normal | 3.54s | N/A | 16.30s | 6.58s |  |
| -7-vQ4obVwQ_00.mp4 | accident | accident | normal | 5.07s | N/A | 15.32s | 5.57s |  |
| -9oifpjUxxM_00.mp4 | accident | normal | normal | N/A | N/A | 17.13s | 8.60s |  |
| -AztVDZ6cEE_00.mp4 | accident | accident | normal | 0.53s | N/A | 12.62s | 3.76s |  |
| -2UPLUV7JLg_00.mp4 | normal | normal | normal | N/A | N/A | 7.86s | 2.93s |  |
| -6SQSDj8cYU_00.mp4 | normal | normal | normal | N/A | N/A | 8.86s | 3.47s |  |
| -7-vQ4obVwQ_00.mp4 | normal | normal | normal | N/A | N/A | 9.38s | 3.38s |  |
| -9oifpjUxxM_00.mp4 | normal | normal | normal | N/A | N/A | 11.32s | 4.19s |  |
| -dmYsQc-odI_00.mp4 | normal | accident | normal | 3.49s | N/A | 5.03s | 1.51s |  |

## Summary Statistics
- **Total Runtime (Skip=1):** 118.82s
- **Total Runtime (Skip=3):** 44.55s
- **Speed Improvement:** 2.67x faster

- **Correct Predictions:** Skip 1 = 7/10 | Skip 3 = 5/10
- **False Positives:** Skip 1 = 1 | Skip 3 = 0
- **False Negatives:** Skip 1 = 2 | Skip 3 = 5
- **Missed by Skip=3 (detected by Skip=1):** 4
- **Event Localization Differences:** 0 cases where timing materially differed

## Recommendation

B. frame_skip=3 is risky, so use frame_skip=1