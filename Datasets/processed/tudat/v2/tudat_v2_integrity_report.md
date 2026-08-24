# TUDAT v2 Integrity Report

**Generated:** 2026-08-15 08:36:29 UTC  
**Source metadata (v1):** `Datasets/processed/global_master_metadata.csv` (111 TUDAT rows)  
**Output metadata (v2):** `Datasets/processed/tudat/v2/global_master_metadata_v2.csv`  
**Overall result:** ALL CHECKS PASSED

---

## 1. Record Counts

| Metric | Expected | Actual | Result |
|---|---|---|---|
| Total records | 93 | 93 | PASS |
| train | 65 | 65 | PASS |
| val | 13 | 13 | PASS |
| test | 15 | 15 | PASS |
| accident | 44 | 44 | PASS |
| non-accident | 49 | 49 | PASS |

## 2. Split Class Distribution

| Split | accident | non-accident | Total |
|---|---|---|---|
| train | 31 | 34 | 65 |
| val | 6 | 7 | 13 |
| test | 7 | 8 | 15 |

## 3. All Integrity Checks

| Check | Result | Detail |
|---|---|---|
| Total records == 93 | **PASS** | actual=93 |
| train count == 65 | **PASS** |  |
| val count == 13 | **PASS** |  |
| test count == 15 | **PASS** |  |
| accident count == 44 | **PASS** |  |
| non-accident count == 49 | **PASS** |  |
| No 'challenging' labels | **PASS** |  |
| Only valid labels | **PASS** |  |
| Only valid split values | **PASS** | present={'test', 'train', 'val'} |
| Every record has a split | **PASS** |  |
| No duplicate original_path | **PASS** |  |
| No train/val overlap | **PASS** |  |
| No train/test overlap | **PASS** |  |
| No val/test overlap | **PASS** |  |
| Excluded absent: ...ing-environment/motorbike2.mov | **PASS** |  |
| Excluded absent: ...ing-environment/motorbike3.mov | **PASS** |  |
| Excluded absent: ...ing-environment/motorbike4.mov | **PASS** |  |
| Excluded absent: ...ing-environment/motorbike5.mov | **PASS** |  |
| Excluded absent: ...hallenging-environment/v29.mov | **PASS** |  |
| Excluded absent: ...hallenging-environment/v30.mov | **PASS** |  |
| Excluded absent: ...hallenging-environment/v31.mov | **PASS** |  |
| Excluded absent: ...hallenging-environment/v32.mov | **PASS** |  |
| Excluded absent: ...hallenging-environment/v33.mov | **PASS** |  |
| Excluded absent: ...hallenging-environment/v34.mov | **PASS** |  |
| Excluded absent: ...hallenging-environment/v35.mov | **PASS** |  |
| Excluded absent: ...hallenging-environment/v36.mov | **PASS** |  |
| Excluded absent: ...hallenging-environment/v37.mov | **PASS** |  |
| Excluded absent: ...hallenging-environment/v38.mov | **PASS** |  |
| Excluded absent: ...hallenging-environment/v39.mov | **PASS** |  |
| Excluded absent: ...hallenging-environment/v40.mov | **PASS** |  |
| Excluded absent: ...ing-environment/wrong_way2.mov | **PASS** |  |
| Excluded absent: ...Negative_Videos/v4(1).mov | **PASS** |  |
| No challenging-environment paths | **PASS** | present=none |
| All video files exist on disk | **PASS** | missing_count=0 |
| dataset_version == 2.0 | **PASS** | compared via astype(str) due to pandas float64 inference |
| v1 still has 111 TUDAT rows | **PASS** | actual=111 |
| v1 metadata file exists | **PASS** |  |

## 4. Duplicate Exclusion Verification

18 records were excluded from v2. None appear in v2 metadata.

| Excluded path | Canonical replacement in v2 | In v2? |
|---|---|---|
| `challenging-environment/motorbike2.mov` | `Positive_Vidoes/v44.mov` | excluded=NO (correct) / canonical=YES (correct) |
| `challenging-environment/motorbike3.mov` | `Positive_Vidoes/v45.mov` | excluded=NO (correct) / canonical=YES (correct) |
| `challenging-environment/motorbike4.mov` | `Positive_Vidoes/v46.mov` | excluded=NO (correct) / canonical=YES (correct) |
| `challenging-environment/motorbike5.mov` | `Positive_Vidoes/v47.mov` | excluded=NO (correct) / canonical=YES (correct) |
| `challenging-environment/v29.mov` | `Positive_Vidoes/v29.mov` | excluded=NO (correct) / canonical=YES (correct) |
| `challenging-environment/v30.mov` | `Positive_Vidoes/v30.mov` | excluded=NO (correct) / canonical=YES (correct) |
| `challenging-environment/v31.mov` | `Positive_Vidoes/v31.mov` | excluded=NO (correct) / canonical=YES (correct) |
| `challenging-environment/v32.mov` | `Positive_Vidoes/v32.mov` | excluded=NO (correct) / canonical=YES (correct) |
| `challenging-environment/v33.mov` | `Positive_Vidoes/v33.mov` | excluded=NO (correct) / canonical=YES (correct) |
| `challenging-environment/v34.mov` | `Positive_Vidoes/v34.mov` | excluded=NO (correct) / canonical=YES (correct) |
| `challenging-environment/v35.mov` | `Positive_Vidoes/v35.mov` | excluded=NO (correct) / canonical=YES (correct) |
| `challenging-environment/v36.mov` | `Positive_Vidoes/v36.mov` | excluded=NO (correct) / canonical=YES (correct) |
| `challenging-environment/v37.mov` | `Positive_Vidoes/v37.mov` | excluded=NO (correct) / canonical=YES (correct) |
| `challenging-environment/v38.mov` | `Positive_Vidoes/v38.mov` | excluded=NO (correct) / canonical=YES (correct) |
| `challenging-environment/v39.mov` | `Positive_Vidoes/v39.mov` | excluded=NO (correct) / canonical=YES (correct) |
| `challenging-environment/v40.mov` | `Positive_Vidoes/v40.mov` | excluded=NO (correct) / canonical=YES (correct) |
| `challenging-environment/wrong_way2.mov` | `Positive_Vidoes/v42.mov` | excluded=NO (correct) / canonical=YES (correct) |
| `Negative_Videos/v4(1).mov` | `Negative_Videos/v4.mov` | excluded=NO (correct) / canonical=YES (correct) |

## 5. Cross-Split Leakage

- train/val path overlap: **0** records
- train/test path overlap: **0** records
- val/test path overlap: **0** records

**Result: No cross-split leakage.**

## 6. Backward Compatibility

- `global_master_metadata.csv` (v1): **NOT modified** (111 TUDAT rows confirmed).
- Raw video files: **NOT deleted**.
- E01-E06 experiment outputs: **NOT touched**.

---

**Final verdict: ALL 37 CHECKS PASSED**