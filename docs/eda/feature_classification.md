# Feature Classification Report

Generated: 2026-08-03 09:12:31 UTC
Source metadata: `E:/Rads/Datasets/processed/global_master_metadata.csv`

This report classifies metadata columns for model-training use. It does not modify the metadata.

## Required for Training

| Column         | Group                 | Justification                                                                           | Recommended Usage                                                                                |
|:---------------|:----------------------|:----------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------|
| type           | Required for Training | Column represents the primary class label, accident timing, or bounding-box annotation. | Use for labels or annotation targets when the training objective requires accident localization. |
| accident_time  | Required for Training | Column represents the primary class label, accident timing, or bounding-box annotation. | Use for labels or annotation targets when the training objective requires accident localization. |
| accident_frame | Required for Training | Column represents the primary class label, accident timing, or bounding-box annotation. | Use for labels or annotation targets when the training objective requires accident localization. |
| center_x       | Required for Training | Column represents the primary class label, accident timing, or bounding-box annotation. | Use for labels or annotation targets when the training objective requires accident localization. |
| center_y       | Required for Training | Column represents the primary class label, accident timing, or bounding-box annotation. | Use for labels or annotation targets when the training objective requires accident localization. |
| x1             | Required for Training | Column represents the primary class label, accident timing, or bounding-box annotation. | Use for labels or annotation targets when the training objective requires accident localization. |
| y1             | Required for Training | Column represents the primary class label, accident timing, or bounding-box annotation. | Use for labels or annotation targets when the training objective requires accident localization. |
| x2             | Required for Training | Column represents the primary class label, accident timing, or bounding-box annotation. | Use for labels or annotation targets when the training objective requires accident localization. |
| y2             | Required for Training | Column represents the primary class label, accident timing, or bounding-box annotation. | Use for labels or annotation targets when the training objective requires accident localization. |

## Optional Feature

| Column       | Group            | Justification                                                                                | Recommended Usage                                                           |
|:-------------|:-----------------|:---------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------|
| weather      | Optional Feature | Column may describe media properties or scene context but is incomplete or not a core label. | Use only after establishing a baseline and handling missingness explicitly. |
| no_frames    | Optional Feature | Column may describe media properties or scene context but is incomplete or not a core label. | Use only after establishing a baseline and handling missingness explicitly. |
| duration     | Optional Feature | Column may describe media properties or scene context but is incomplete or not a core label. | Use only after establishing a baseline and handling missingness explicitly. |
| height       | Optional Feature | Column may describe media properties or scene context but is incomplete or not a core label. | Use only after establishing a baseline and handling missingness explicitly. |
| width        | Optional Feature | Column may describe media properties or scene context but is incomplete or not a core label. | Use only after establishing a baseline and handling missingness explicitly. |
| fps          | Optional Feature | Column may describe media properties or scene context but is incomplete or not a core label. | Use only after establishing a baseline and handling missingness explicitly. |
| rollover     | Optional Feature | Column may describe media properties or scene context but is incomplete or not a core label. | Use only after establishing a baseline and handling missingness explicitly. |
| region       | Optional Feature | Column may describe media properties or scene context but is incomplete or not a core label. | Use only after establishing a baseline and handling missingness explicitly. |
| scene_layout | Optional Feature | Column may describe media properties or scene context but is incomplete or not a core label. | Use only after establishing a baseline and handling missingness explicitly. |
| day_time     | Optional Feature | Column may describe media properties or scene context but is incomplete or not a core label. | Use only after establishing a baseline and handling missingness explicitly. |
| quality      | Optional Feature | Column may describe media properties or scene context but is incomplete or not a core label. | Use only after establishing a baseline and handling missingness explicitly. |

## Metadata Only

| Column                   | Group         | Justification                                                                            | Recommended Usage                                                                                         |
|:-------------------------|:--------------|:-----------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------|
| video_id                 | Metadata Only | Column identifies provenance, paths, splits, annotation storage, or dataset bookkeeping. | Keep for loading, grouping, traceability, splitting, or post-training analysis; do not feed to the model. |
| dataset_name             | Metadata Only | Column identifies provenance, paths, splits, annotation storage, or dataset bookkeeping. | Keep for loading, grouping, traceability, splitting, or post-training analysis; do not feed to the model. |
| source_type              | Metadata Only | Column identifies provenance, paths, splits, annotation storage, or dataset bookkeeping. | Keep for loading, grouping, traceability, splitting, or post-training analysis; do not feed to the model. |
| dataset_version          | Metadata Only | Column identifies provenance, paths, splits, annotation storage, or dataset bookkeeping. | Keep for loading, grouping, traceability, splitting, or post-training analysis; do not feed to the model. |
| pipeline_version         | Metadata Only | Column identifies provenance, paths, splits, annotation storage, or dataset bookkeeping. | Keep for loading, grouping, traceability, splitting, or post-training analysis; do not feed to the model. |
| original_path            | Metadata Only | Column identifies provenance, paths, splits, annotation storage, or dataset bookkeeping. | Keep for loading, grouping, traceability, splitting, or post-training analysis; do not feed to the model. |
| processed_path           | Metadata Only | Column identifies provenance, paths, splits, annotation storage, or dataset bookkeeping. | Keep for loading, grouping, traceability, splitting, or post-training analysis; do not feed to the model. |
| video_path_mode          | Metadata Only | Column identifies provenance, paths, splits, annotation storage, or dataset bookkeeping. | Keep for loading, grouping, traceability, splitting, or post-training analysis; do not feed to the model. |
| annotation_available     | Metadata Only | Column identifies provenance, paths, splits, annotation storage, or dataset bookkeeping. | Keep for loading, grouping, traceability, splitting, or post-training analysis; do not feed to the model. |
| split                    | Metadata Only | Column identifies provenance, paths, splits, annotation storage, or dataset bookkeeping. | Keep for loading, grouping, traceability, splitting, or post-training analysis; do not feed to the model. |
| split_in_distribution    | Metadata Only | Column identifies provenance, paths, splits, annotation storage, or dataset bookkeeping. | Keep for loading, grouping, traceability, splitting, or post-training analysis; do not feed to the model. |
| split_geo_aware          | Metadata Only | Column identifies provenance, paths, splits, annotation storage, or dataset bookkeeping. | Keep for loading, grouping, traceability, splitting, or post-training analysis; do not feed to the model. |
| annotations_path         | Metadata Only | Column identifies provenance, paths, splits, annotation storage, or dataset bookkeeping. | Keep for loading, grouping, traceability, splitting, or post-training analysis; do not feed to the model. |
| map                      | Metadata Only | Column identifies provenance, paths, splits, annotation storage, or dataset bookkeeping. | Keep for loading, grouping, traceability, splitting, or post-training analysis; do not feed to the model. |
| camera_position          | Metadata Only | Column identifies provenance, paths, splits, annotation storage, or dataset bookkeeping. | Keep for loading, grouping, traceability, splitting, or post-training analysis; do not feed to the model. |
| annotations_start_offset | Metadata Only | Column identifies provenance, paths, splits, annotation storage, or dataset bookkeeping. | Keep for loading, grouping, traceability, splitting, or post-training analysis; do not feed to the model. |
| media_type               | Metadata Only | Column identifies provenance, paths, splits, annotation storage, or dataset bookkeeping. | Keep for loading, grouping, traceability, splitting, or post-training analysis; do not feed to the model. |

## Ignore During Training

| Column               | Group                  | Justification                                                                                                | Recommended Usage                     |
|:---------------------|:-----------------------|:-------------------------------------------------------------------------------------------------------------|:--------------------------------------|
| validation_status    | Ignore During Training | Column is processing metadata, technical encoding metadata, duplicate bookkeeping, or timestamp information. | Exclude from model inputs and labels. |
| processing_status    | Ignore During Training | Column is processing metadata, technical encoding metadata, duplicate bookkeeping, or timestamp information. | Exclude from model inputs and labels. |
| preprocessing_status | Ignore During Training | Column is processing metadata, technical encoding metadata, duplicate bookkeeping, or timestamp information. | Exclude from model inputs and labels. |
| file_size_bytes      | Ignore During Training | Column is processing metadata, technical encoding metadata, duplicate bookkeeping, or timestamp information. | Exclude from model inputs and labels. |
| file_hash            | Ignore During Training | Column is processing metadata, technical encoding metadata, duplicate bookkeeping, or timestamp information. | Exclude from model inputs and labels. |
| is_duplicate         | Ignore During Training | Column is processing metadata, technical encoding metadata, duplicate bookkeeping, or timestamp information. | Exclude from model inputs and labels. |
| duplicate_group_id   | Ignore During Training | Column is processing metadata, technical encoding metadata, duplicate bookkeeping, or timestamp information. | Exclude from model inputs and labels. |
| processed_at         | Ignore During Training | Column is processing metadata, technical encoding metadata, duplicate bookkeeping, or timestamp information. | Exclude from model inputs and labels. |
| metadata_source      | Ignore During Training | Column is processing metadata, technical encoding metadata, duplicate bookkeeping, or timestamp information. | Exclude from model inputs and labels. |
| weather_confidence   | Ignore During Training | Column is processing metadata, technical encoding metadata, duplicate bookkeeping, or timestamp information. | Exclude from model inputs and labels. |
| day_time_confidence  | Ignore During Training | Column is processing metadata, technical encoding metadata, duplicate bookkeeping, or timestamp information. | Exclude from model inputs and labels. |
| channels             | Ignore During Training | Column is processing metadata, technical encoding metadata, duplicate bookkeeping, or timestamp information. | Exclude from model inputs and labels. |
| codec                | Ignore During Training | Column is processing metadata, technical encoding metadata, duplicate bookkeeping, or timestamp information. | Exclude from model inputs and labels. |
| generated_at         | Ignore During Training | Column is processing metadata, technical encoding metadata, duplicate bookkeeping, or timestamp information. | Exclude from model inputs and labels. |
