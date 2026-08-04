# Dataset Schema Comparison

Generated: 2026-08-03 09:12:32 UTC
Source metadata: `E:/Rads/Datasets/processed/global_master_metadata.csv`

Schemas are compared by non-null column availability within each dataset.

## Dataset Summary

| Dataset   |   Rows |   Available Columns |   Missing Columns | Unique Metadata Fields                                                                                                                                                                                                                                                                                                                                                                                                                      |
|:----------|-------:|--------------------:|------------------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| kaggle    |    989 |                  17 |                34 | channels, day_time_confidence, generated_at, media_type, metadata_source, split_in_distribution, weather_confidence                                                                                                                                                                                                                                                                                                                         |
| picek     |   4238 |                  43 |                 8 | accident_frame, accident_time, annotation_available, annotations_path, annotations_start_offset, camera_position, center_x, center_y, duration, file_hash, fps, is_duplicate, map, no_frames, pipeline_version, preprocessing_status, processed_at, processed_path, processing_status, quality, region, rollover, scene_layout, split, split_geo_aware, split_in_distribution, validation_status, video_id, video_path_mode, x1, x2, y1, y2 |
| tudat     |    111 |                  19 |                32 | codec, day_time_confidence, duration, fps, generated_at, media_type, metadata_source, no_frames, weather_confidence                                                                                                                                                                                                                                                                                                                         |

## Column Presence

| Column                   | kaggle   | picek   | tudat   |
|:-------------------------|:---------|:--------|:--------|
| video_id                 | No       | Yes     | No      |
| dataset_name             | Yes      | Yes     | Yes     |
| source_type              | Yes      | Yes     | Yes     |
| dataset_version          | Yes      | Yes     | Yes     |
| pipeline_version         | No       | Yes     | No      |
| original_path            | Yes      | Yes     | Yes     |
| processed_path           | No       | Yes     | No      |
| video_path_mode          | No       | Yes     | No      |
| validation_status        | No       | Yes     | No      |
| processing_status        | No       | Yes     | No      |
| preprocessing_status     | No       | Yes     | No      |
| type                     | Yes      | Yes     | Yes     |
| accident_time            | No       | Yes     | No      |
| accident_frame           | No       | Yes     | No      |
| center_x                 | No       | Yes     | No      |
| center_y                 | No       | Yes     | No      |
| x1                       | No       | Yes     | No      |
| y1                       | No       | Yes     | No      |
| x2                       | No       | Yes     | No      |
| y2                       | No       | Yes     | No      |
| weather                  | Yes      | Yes     | Yes     |
| no_frames                | No       | Yes     | Yes     |
| duration                 | No       | Yes     | Yes     |
| height                   | Yes      | Yes     | Yes     |
| width                    | Yes      | Yes     | Yes     |
| fps                      | No       | Yes     | Yes     |
| file_size_bytes          | Yes      | Yes     | Yes     |
| file_hash                | No       | Yes     | No      |
| annotation_available     | No       | Yes     | No      |
| split                    | No       | Yes     | No      |
| split_in_distribution    | Yes      | Yes     | No      |
| split_geo_aware          | No       | Yes     | No      |
| rollover                 | No       | Yes     | No      |
| region                   | No       | Yes     | No      |
| scene_layout             | No       | Yes     | No      |
| day_time                 | Yes      | Yes     | Yes     |
| quality                  | No       | Yes     | No      |
| annotations_path         | No       | Yes     | No      |
| map                      | No       | Yes     | No      |
| camera_position          | No       | Yes     | No      |
| annotations_start_offset | No       | Yes     | No      |
| is_duplicate             | No       | Yes     | No      |
| duplicate_group_id       | No       | No      | No      |
| processed_at             | No       | Yes     | No      |
| media_type               | Yes      | No      | Yes     |
| metadata_source          | Yes      | No      | Yes     |
| weather_confidence       | Yes      | No      | Yes     |
| day_time_confidence      | Yes      | No      | Yes     |
| channels                 | Yes      | No      | No      |
| codec                    | No       | No      | Yes     |
| generated_at             | Yes      | No      | Yes     |

## Schema Categories

| Dataset   | Category                    | Fields                                                                                                                                                              |
|:----------|:----------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| kaggle    | Annotation Information      | type                                                                                                                                                                |
| kaggle    | Video Information           | channels, file_size_bytes, height, media_type, original_path, width                                                                                                 |
| kaggle    | Context Information         | day_time, day_time_confidence, weather, weather_confidence                                                                                                          |
| kaggle    | Bounding Box Information    | None                                                                                                                                                                |
| kaggle    | Potential Training Features | day_time, height, type, weather, width                                                                                                                              |
| picek     | Annotation Information      | accident_frame, accident_time, annotation_available, annotations_path, annotations_start_offset, quality, type                                                      |
| picek     | Video Information           | duration, file_hash, file_size_bytes, fps, height, no_frames, original_path, processed_path, video_path_mode, width                                                 |
| picek     | Context Information         | camera_position, day_time, map, region, rollover, scene_layout, weather                                                                                             |
| picek     | Bounding Box Information    | center_x, center_y, x1, x2, y1, y2                                                                                                                                  |
| picek     | Potential Training Features | accident_frame, accident_time, center_x, center_y, day_time, duration, fps, height, no_frames, region, rollover, scene_layout, type, weather, width, x1, x2, y1, y2 |
| tudat     | Annotation Information      | type                                                                                                                                                                |
| tudat     | Video Information           | codec, duration, file_size_bytes, fps, height, media_type, no_frames, original_path, width                                                                          |
| tudat     | Context Information         | day_time, day_time_confidence, weather, weather_confidence                                                                                                          |
| tudat     | Bounding Box Information    | None                                                                                                                                                                |
| tudat     | Potential Training Features | day_time, duration, fps, height, no_frames, type, weather, width                                                                                                    |

## Shared Metadata Fields

dataset_name, dataset_version, day_time, file_size_bytes, height, original_path, source_type, type, weather, width

## kaggle Details

- Available Columns: channels, dataset_name, dataset_version, day_time, day_time_confidence, file_size_bytes, generated_at, height, media_type, metadata_source, original_path, source_type, split_in_distribution, type, weather, weather_confidence, width
- Missing Columns: accident_frame, accident_time, annotation_available, annotations_path, annotations_start_offset, camera_position, center_x, center_y, codec, duplicate_group_id, duration, file_hash, fps, is_duplicate, map, no_frames, pipeline_version, preprocessing_status, processed_at, processed_path, processing_status, quality, region, rollover, scene_layout, split, split_geo_aware, validation_status, video_id, video_path_mode, x1, x2, y1, y2
- Unique Metadata Fields: channels, day_time_confidence, generated_at, media_type, metadata_source, split_in_distribution, weather_confidence

## picek Details

- Available Columns: accident_frame, accident_time, annotation_available, annotations_path, annotations_start_offset, camera_position, center_x, center_y, dataset_name, dataset_version, day_time, duration, file_hash, file_size_bytes, fps, height, is_duplicate, map, no_frames, original_path, pipeline_version, preprocessing_status, processed_at, processed_path, processing_status, quality, region, rollover, scene_layout, source_type, split, split_geo_aware, split_in_distribution, type, validation_status, video_id, video_path_mode, weather, width, x1, x2, y1, y2
- Missing Columns: channels, codec, day_time_confidence, duplicate_group_id, generated_at, media_type, metadata_source, weather_confidence
- Unique Metadata Fields: accident_frame, accident_time, annotation_available, annotations_path, annotations_start_offset, camera_position, center_x, center_y, duration, file_hash, fps, is_duplicate, map, no_frames, pipeline_version, preprocessing_status, processed_at, processed_path, processing_status, quality, region, rollover, scene_layout, split, split_geo_aware, split_in_distribution, validation_status, video_id, video_path_mode, x1, x2, y1, y2

## tudat Details

- Available Columns: codec, dataset_name, dataset_version, day_time, day_time_confidence, duration, file_size_bytes, fps, generated_at, height, media_type, metadata_source, no_frames, original_path, source_type, type, weather, weather_confidence, width
- Missing Columns: accident_frame, accident_time, annotation_available, annotations_path, annotations_start_offset, camera_position, center_x, center_y, channels, duplicate_group_id, file_hash, is_duplicate, map, pipeline_version, preprocessing_status, processed_at, processed_path, processing_status, quality, region, rollover, scene_layout, split, split_geo_aware, split_in_distribution, validation_status, video_id, video_path_mode, x1, x2, y1, y2
- Unique Metadata Fields: codec, day_time_confidence, duration, fps, generated_at, media_type, metadata_source, no_frames, weather_confidence
