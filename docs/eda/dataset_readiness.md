# Dataset Readiness

Generated: 2026-08-03 09:12:32 UTC
Source metadata: `E:/Rads/Datasets/processed/global_master_metadata.csv`

This is a training readiness decision document based on the completed metadata.

## Decision

- Overall Readiness Score: 78/100
- Decision: Conditionally ready; address sampling, leakage controls, and task-specific missingness before full training.

## Dataset Strengths

- All 5,338 rows have a primary `type` label.
- Bounding boxes are present for 4,238 rows and no invalid boxes were detected in rows with complete bbox coordinates.
- The metadata preserves dataset provenance and split columns for controlled training setup.

## Dataset Weaknesses

- Dataset balance is skewed: picek contributes 4238 rows (79.4%).
- Class imbalance ratio is 66.0:1.
- 18 columns have more than 50% missing values, mostly because schemas differ across datasets.

## Quality Assessment

- Metadata Quality: 87.6/100 task-weighted completeness for training-relevant metadata.
- Annotation Quality: Complete bbox annotations are concentrated in Picek; Kaggle and TUDAT do not provide bbox columns in this unified metadata.
- Class Balance: 8 classes; largest class `rear-end` has 1122 rows and smallest class `challenging` has 17 rows.
- Dataset Balance: picek: 4238 (79.4%); kaggle: 989 (18.5%); tudat: 111 (2.1%)
- Missing Metadata Assessment: High missingness is expected for schema-specific fields; training-critical missingness must be handled by filtering or task-specific loaders.
- Leakage Assessment: Exclude provenance, path, hash, split, timestamp, and processing-status fields from model inputs. Leakage-risk columns include dataset_name, source_type, original_path, processed_path, file_hash, video_id.

## Risks Before Training

- Training across all three datasets without task-aware filtering will mix dense annotation rows with video-level-only rows.
- Dataset and source-type imbalance can bias evaluation if splits are not stratified by dataset and class.
- Context columns such as weather and day_time are incomplete and should not be treated as mandatory features.
- Path, ID, hash, timestamp, and processing metadata can leak dataset provenance if accidentally encoded.

## Recommendations Before Pipeline 3

- Define the training task explicitly: classification-only can use all rows; bbox/localization training should filter to rows with complete bbox fields.
- Use `type` as the primary label and bbox/time annotation columns only where present and required by the model objective.
- Keep dataset provenance and path fields in dataloading metadata only, not model features.
- Add class-balanced or dataset-aware sampling before training.
- Treat optional context fields as ablation features after a baseline is established.
