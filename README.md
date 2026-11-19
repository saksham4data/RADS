# RADS — Road Accident Detection and Severity

RADS watches road-camera video, detects accidents, estimates when they happened and how severe they look, and raises an event for a human operator to confirm. It is built to speed up the operator, not replace them.

```text
video -> YOLO + ByteTrack -> trajectories -> motion features -> object interactions
      -> accident reasoning -> event time + involved objects -> severity -> event / overlay
```

## Layout

```text
rads/                 the system
  config.yaml         every threshold and weight
  config.py           typed access to config.yaml
  perception.py       video reader, YOLO + ByteTrack tracker
  tracks.py           track history and motion features
  interactions.py     pairwise distance / overlap / closing speed, interaction candidates
  reasoning.py        rule-based accident decision and event localization
  severity.py         heuristic LOW / MEDIUM / HIGH score with evidence
  output.py           result JSON and annotated video
  pipeline.py         offline: one whole video in, one result out
  live/               streaming runtime: source, processor, events, engine, API
scripts/
  create_splits.py    builds the held-out evaluation split
  evaluate.py         runs the pipeline over a split CSV, writes JSONL results
  compare_baseline.py compares pipeline results with the ResNet18 + GRU baseline
baseline/             frozen ResNet18 + GRU clip classifier (P02), kept for comparison
tests/                unit and integration tests
data/                 videos (gitignored) and split CSVs (tracked)
  raw/                PICEK real + synthetic, TUDAT, Kaggle images
  clips/              trimmed positive / negative PICEK clips used by P02
  splits/             p02_binary, p02_test, picek_heldout_v2, tudat_v2
models/               yolo11n.pt (gitignored)
run_video.py          process one video file
run_live.py           process a file, webcam or RTSP stream continuously
```

`docs/` holds the architecture specs, research history and evaluation evidence. It is gitignored and lives only on disk.

## Usage

```bash
pip install -r requirements.txt
python run_video.py --video data/clips/positive/-PpBteU0p3Q_00.mp4 --config rads/config.yaml --output results/out.json
python run_live.py --config rads/config.yaml --source rtsp://camera/stream --api
python scripts/evaluate.py --csv data/splits/p02_test.csv --output results/p02_test.jsonl
python -m pytest tests
```

With `--api`, the live runtime serves `GET /health`, `GET /events`, `GET /events/latest`, `GET /events/{id}` and `WebSocket /ws/events` on `127.0.0.1:8100`.

## Baseline

P02, ResNet18 + GRU on 500 balanced PICEK clips: test accuracy 45.9%, AUROC 0.535, 83.8% false positives on normal clips. The paired positive and negative P02 clips share most of their footage, so this number reflects the dataset design as much as the model.

```bash
python baseline/train.py --config baseline/p02.yaml
python baseline/test.py --config baseline/p02.yaml --checkpoint baseline/outputs/latest/checkpoints/best.pt
```
