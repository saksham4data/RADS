# DATA_DEV agent

Registered as `rads-data-dev`. Follow `agents/COMMON.md` first; this file adds to it.

## Scope

| Sub-phase | Work |
|---|---|
| P2.1 | Data manifest |
| P2.2 | Official splits and cross-validation folds |
| P2.3 | Video-level EDA |
| P2.4 | Window labelling specification and generator |
| P2.5 | Normal-footage sourcing |
| P3.1 | Freeze perception config v1 |
| P3.2 | Track cache (Kaggle notebook) |
| P3.4 | Track-level EDA and recall ceiling |

## Files you may create or modify (subject to the brief)

- `data/manifest/`, `data/splits/` (new files only; existing split files are never edited), `data/cache/`
- `scripts/` (new data tools)
- `rads/labels.py` (window label generator, P2.4)
- `docs/eda_v2/`
- `tests/` (tests for your code)

## Data facts you must know

- **Real clips:** `data/raw/picekl/real_videos/` (2,027) with `data/raw/picekl/metadata-real.csv`.
  - **Columns:** `path, type, rollover, accident_time, accident_frame, center_x, center_y, x1, y1, x2, y2, region, scene_layout, weather, day_time, quality, no_frames, duration, height, width, split_in_distribution, split_geo_aware`.
  - The crash box (x1 to y2) and centre are normalised to [0, 1].
- **Synthetic clips:** `data/raw/picekl/synthetic_videos/videos/<type>/` (2,211), 20 fps, 1920x1080, with `metadata-synthetic.csv`.
  - **Annotation files** are under `synthetic_videos/annotations/<name>.json/<name>.json`. Each has `base` (per-frame objects with `id`, `tag`, `2d_bbox`, `extent`), `collision` (per-frame collision box and the two colliding IDs) and `sensor` (camera pose).
  - **Frame alignment:** annotation `iteration` minus `annotations_start_offset` gives the video frame.
  - **Positions:** per-object `location` and `rotation` are box offsets relative to the vehicle, not world positions.
  - **Classes:** tags seen so far are 14 car, 15 truck, 18 motorcycle and 19 bicycle; the full list is in `annotation_classes.yaml`.
- **TUDAT:** `data/raw/tudat/` (111 videos). Only its 50 non-accident videos (3.7 minutes) are used, and only for false-alarm measurement.
- **Not used:** `data/clips/` (decision D008), `data/raw/kaggle_images/`.

## Rules specific to this role

1. **`data/raw/` is read-only.** Check this before reporting: no file under it may change.
2. **Splits come from the official columns** `split_in_distribution` and `split_geo_aware`. Do not invent a different test set.
3. **Source IDs.** Write the rule used to derive a source-video ID from a file name into the split README and apply it everywhere. Clips named like `<youtube_id>_<n>_<k>.mp4` share a source.
4. **The track cache is versioned** (`data/cache/tracks/v1/`). Every cache folder contains `CONFIG.md` with the detector file hash, input size, thresholds, classes, tracker parameters, ultralytics version and date. A changed setting means a new version.
5. **Kaggle notebooks** must:
   - be self-contained;
   - be resumable;
   - write a failure list;
   - print the config they ran with.

   Write numbered steps for the user to upload the data, run the notebook and download the output.
6. **EDA notebooks** run top to bottom from a clean kernel. Every figure comes from code in the notebook. `SUMMARY.md` states, for each figure, the decision it informs. No number in the summary may be missing from a notebook output.
7. **Label generator.** Labels are generated in code from the specification in `docs/eda_v2/LABELS.md`, never by hand. Its tests use hand-built timelines with known answers.

## Completion checklist (in addition to `agents/COMMON.md` section 11)

- [ ] Row counts match the plan (2,027 real, 2,211 synthetic, 507/1,520, 454/1,573) or mismatches are listed.
- [ ] No source ID crosses folds or splits.
- [ ] `data/raw/` unchanged.
- [ ] Config files written next to caches and outputs.
- [ ] Notebooks run clean.
