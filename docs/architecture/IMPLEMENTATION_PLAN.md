# RADS -- Deployable Runtime v1 Implementation Plan

**Project:** RADS
**Document:** Implementation Plan
**Version:** 1.0
**Status:** Active
**Date:** September 2026
**Authority:** PROD.md, MASTER_SPEC.md, SYSTEM.md

---

# 1. Overview

This document defines the implementation phases for RADS Deployable Runtime v1. Each phase is broken into small, independently verifiable steps. No phase should take more than a single focused session to complete.

The plan builds the runtime layer on top of the existing core intelligence modules without modifying them. The batch pipeline (`run_pipeline.py`) is preserved for research use.

---

# 2. Phase Map

| Phase | Name | Depends On | Estimated Effort |
|---|---|---|---|
| 0 | Repository Stabilization | -- | 30 minutes |
| 1A | Core Library Namespace | 0 | 30 minutes |
| 1B | Configuration Extension | 0 | 1 hour |
| 1C | Environment Variable Resolver | 1B | 30 minutes |
| 2A | Source Abstraction | 1B | 1.5 hours |
| 2B | Frame Processor | 1A, 2A | 2 hours |
| 2C | Runtime Engine and Main Loop | 2A, 2B | 1.5 hours |
| 2D | Health and Signal Handling | 2C | 45 minutes |
| 2E | Event Lifecycle Manager | 2B | 1 hour |
| 2F | Runtime Entry Point | 2C, 2D, 2E | 30 minutes |
| 3A | API Server and Health Endpoint | 2F | 1 hour |
| 3B | Event Endpoints | 3A, 2E | 1 hour |
| 3C | WebSocket Event Stream | 3A, 2E | 1 hour |
| 4A | Dockerfile and Build | 3A | 1 hour |
| 4B | Docker Compose and Deployment | 4A, 1C | 1 hour |
| 5 | Integration Testing and Verification | All | 2 hours |

Total estimated: 14-15 hours across multiple sessions.

---

# 3. Phase 0 -- Repository Stabilization

**Goal:** Ensure the repository is in a runnable state before any new code is written.

### 0.1 Restore deleted files

The `rads/config/`, `rads/output/`, and `rads/evaluation/` packages were previously deleted from the working tree. The user has restored them manually. Verify they are present and importable.

**Verification:**
- `python -c "from rads.config.config_loader import ConfigLoader; print('OK')"` exits 0
- `python -c "from rads.output.event_schema import build_event_result; print('OK')"` exits 0
- `python -c "from rads.output.visualizer import Visualizer; print('OK')"` exits 0

### 0.2 Verify existing tests

Run the existing test suite to establish a baseline:

```bash
python -m unittest discover -s rads/tests -t .
```

Record pass/fail count. All tests should pass before proceeding.

### 0.3 Verify batch pipeline

```bash
python run_pipeline.py --help
```

Must exit 0 and print usage. This confirms imports are functional.

### 0.4 Move model weights

Create `models/` directory at project root. Move `yolo11n.pt` from project root to `models/yolo11n.pt`. Update `pipeline_config.yaml` to reference `models/yolo11n.pt`.

Verify the batch pipeline still works after the move:

```bash
python run_pipeline.py --video <any_test_video> --config rads/config/pipeline_config.yaml --output /dev/null
```

### 0.5 Commit clean state

Commit all current changes before starting runtime development:

```bash
git add -A
git commit -m "chore: stabilize repository for runtime v1 development"
```

---

# 4. Phase 1A -- Core Library Namespace

**Goal:** Make the existing intelligence modules importable through a clean top-level namespace without modifying any of the modules themselves.

### 1A.1 Create `rads/core/__init__.py`

This file re-exports the public API of the intelligence modules:

```python
from rads.detection.detector import Detector
from rads.tracking.tracker import Tracker
from rads.motion.trajectory import TrackHistory
from rads.motion.motion_features import compute_motion_features
from rads.interaction.pairwise import compute_pairwise_metrics, enrich_with_motion
from rads.interaction.interaction_engine import detect_interactions
from rads.reasoning.accident_reasoner import evaluate_accident
from rads.severity.severity_engine import estimate_severity
from rads.output.event_schema import build_event_result
```

No other files are created or modified.

**Verification:**
- `python -c "from rads.core import Tracker, evaluate_accident, estimate_severity; print('OK')"` exits 0

---

# 5. Phase 1B -- Configuration Extension

**Goal:** Extend `ConfigLoader` and `pipeline_config.yaml` with runtime-relevant settings while preserving full backward compatibility.

### 1B.1 Add runtime sections to `pipeline_config.yaml`

Append three new top-level sections to the existing YAML. All existing sections remain unchanged.

```yaml
runtime:
  source: "file"
  source_uri: ""
  reconnect_interval_s: 5
  max_reconnect_attempts: -1
  sliding_window_s: 30
  event_confirmation_window_s: 2.0
  event_resolution_timeout_s: 10.0

device:
  compute: "auto"
  model_path: "models/yolo11n.pt"

api:
  enabled: false
  host: "0.0.0.0"
  port: 8100
  event_buffer_size: 100
```

### 1B.2 Add properties to `ConfigLoader`

Add getter properties for each new configuration key, following the existing pattern. Each property returns the YAML value with a default fallback so that existing configs without the new sections still work.

Properties to add:
- `runtime_source` (default: `"file"`)
- `runtime_source_uri` (default: `""`)
- `runtime_reconnect_interval_s` (default: `5`)
- `runtime_max_reconnect_attempts` (default: `-1`)
- `runtime_sliding_window_s` (default: `30`)
- `runtime_event_confirmation_window_s` (default: `2.0`)
- `runtime_event_resolution_timeout_s` (default: `10.0`)
- `device_compute` (default: `"auto"`)
- `device_model_path` (default: `"models/yolo11n.pt"`)
- `api_enabled` (default: `False`)
- `api_host` (default: `"0.0.0.0"`)
- `api_port` (default: `8100`)
- `api_event_buffer_size` (default: `100`)

**Verification:**
- Existing tests still pass
- `python run_pipeline.py --help` still exits 0
- Loading the config and accessing `config.runtime_source` returns `"file"`
- Loading the config and accessing `config.model` still returns the existing YOLO model value

---

# 6. Phase 1C -- Environment Variable Resolver

**Goal:** Allow Docker and deployment environments to override configuration values via environment variables.

### 1C.1 Create `rads/config/env_resolver.py`

A single function `resolve_config(config_loader)` that checks for known environment variables and overrides the corresponding config values.

Environment variable mapping:
- `RADS_SOURCE_URI` overrides `runtime.source_uri`
- `RADS_DEVICE` overrides `device.compute`
- `RADS_MODEL_PATH` overrides `device.model_path`
- `RADS_API_PORT` overrides `api.port`
- `RADS_API_ENABLED` overrides `api.enabled`

Returns a dict of overrides applied, for logging purposes.

**Verification:**
- Set `RADS_DEVICE=cpu` in shell, call `resolve_config`, verify it returns `{"device.compute": "cpu"}`

---

# 7. Phase 2A -- Source Abstraction

**Goal:** A unified frame source interface that works with files, webcams, and RTSP streams.

### 2A.1 Create `rads/runtime/__init__.py`

Empty init file.

### 2A.2 Create `rads/runtime/source.py`

Define a base class `FrameSource` and three concrete implementations:

**`FrameSource` (abstract base):**
- `open()` -- connect to the source
- `read()` -- return `(frame_index, timestamp_s, frame_ndarray)` or `None` on end
- `close()` -- release resources
- `is_open()` -- connection status
- Properties: `fps`, `width`, `height`, `source_uri`

**`FileSource(path)`:**
- Wraps `cv2.VideoCapture(path)`
- Validates file exists at construction
- `read()` returns `None` when file ends
- FPS and resolution from video metadata

**`WebcamSource(device_index)`:**
- Wraps `cv2.VideoCapture(int(device_index))`
- Continuous, never returns `None` from `read()` unless device fails
- FPS from device or default 30

**`RTSPSource(url, reconnect_interval_s, max_reconnect_attempts)`:**
- Wraps `cv2.VideoCapture(url)`
- On `read()` failure: close, wait `reconnect_interval_s`, re-open
- Tracks reconnect count. If `max_reconnect_attempts >= 0` and exceeded, raises `SourceExhaustedError`
- If `max_reconnect_attempts == -1`, reconnects indefinitely
- Logs every reconnect attempt

**Factory function `create_source(config)`:**
- Reads `runtime_source` and `runtime_source_uri` from config
- Returns the appropriate source instance

### 2A.3 Write tests for source module

Under `rads/tests/test_source.py`:
- Test `FileSource` with a known test video path from the existing test fixtures
- Test `create_source` factory with `source: "file"`
- Test `RTSPSource` reconnection logic with a mock (unit test, no real RTSP server)

**Verification:**
- `python -m pytest rads/tests/test_source.py` passes
- `FileSource` can open and read frames from a real video file

---

# 8. Phase 2B -- Frame Processor

**Goal:** Per-frame processing that wires core modules in a streaming-compatible way.

### 2B.1 Create `rads/runtime/frame_processor.py`

**`FrameProcessor` class:**

Constructor takes a `ConfigLoader` instance. Initializes:
- `Tracker` (from `rads.core`)
- `TrackHistory` (from `rads.core`)
- Internal state for pairwise metrics (rolling dict)
- Internal state for interaction candidates
- Frame counter, timestamp tracking
- Window bounds (from `runtime_sliding_window_s`)

**`process_frame(frame, frame_idx, timestamp_s)` method:**

1. Call `tracker.track(frame, frame_idx, timestamp_s)` to get tracked objects
2. Update `track_history` with new detections
3. Prune tracks older than `sliding_window_s` from active state
4. Compute pairwise metrics for currently active track pairs
5. Run interaction detection on current pairwise state
6. If interaction candidates exist, run accident reasoning
7. If accident detected, run severity estimation
8. Return a result dict or `None`

**`reset()` method:**
- Clear tracker state (`tracker.reset()`)
- Clear track history
- Clear pairwise state
- Reset frame counter

**`get_state()` method:**
- Return current active track count, frame count, last reasoning result

### 2B.2 Incremental motion features

The existing `compute_motion_features` is batch-only (iterates all tracks after all frames). For streaming, add a thin wrapper `compute_motion_features_incremental(track_history, track_ids)` that computes features for only the specified active tracks. This is a new function in a new file `rads/runtime/motion_adapter.py`, not a modification to the existing module.

### 2B.3 Windowed pairwise adapter

The existing `compute_pairwise_metrics` builds a full frame-indexed dict. For streaming, add `compute_pairwise_metrics_windowed(track_history, active_track_ids, window_frames)` in `rads/runtime/pairwise_adapter.py`. This filters the track history to only the window before calling the existing pairwise logic.

### 2B.4 Write tests

Under `rads/tests/test_frame_processor.py`:
- Test that `FrameProcessor` initializes without error
- Test `process_frame` with a synthetic frame (numpy array)
- Test `reset()` clears state
- Test window pruning removes old tracks

**Verification:**
- All new tests pass
- All existing tests still pass

---

# 9. Phase 2C -- Runtime Engine and Main Loop

**Goal:** The main run loop that reads from a source and processes frames.

### 2C.1 Create `rads/runtime/engine.py`

**`RuntimeEngine` class:**

Constructor takes a config path string. Initializes:
- `ConfigLoader`
- `FrameSource` via `create_source(config)`
- `FrameProcessor`
- Event handler list (callbacks)
- Run state flag

**`register_handler(callback)` method:**
- Append callback to handler list
- Callback signature: `callback(event_dict)` -- called on every event status transition

**`run()` method:**
- Open source
- Enter frame loop:
  - `source.read()` returns frame
  - If `None`: file ended, break. (Or for webcam/RTSP, handled by source)
  - Apply frame skip logic
  - `processor.process_frame(frame, idx, ts)`
  - If result is not None and contains an accident, pass to event lifecycle manager, which calls handlers
  - Handle keyboard interrupt (SIGINT): set run flag to False, break
- Close source
- Log summary (frames processed, events detected, runtime)

**`stop()` method:**
- Set run flag to False (for external shutdown)

### 2C.2 Write tests

Under `rads/tests/test_engine.py`:
- Test engine construction with a valid config
- Test `register_handler` stores callback
- Test `stop()` sets flag

**Verification:**
- Tests pass
- `RuntimeEngine("rads/config/pipeline_config.yaml")` constructs without error

---

# 10. Phase 2D -- Health and Signal Handling

**Goal:** Graceful shutdown and runtime health reporting.

### 2D.1 Create `rads/runtime/health.py`

**`HealthMonitor` class:**
- Tracks: start time, frames processed, events detected, last event time, source status, current FPS
- `update(frames, events)` -- called by engine each frame
- `get_health()` -- returns dict with all health fields
- `get_uptime_s()` -- seconds since start

**Signal handling (integrated into `RuntimeEngine`):**
- Register handler for `SIGINT` and `SIGTERM` on `engine.run()` entry
- On signal: call `engine.stop()`, log shutdown reason
- Restore original handlers on exit

### 2D.2 Write tests

- Test `HealthMonitor.get_health()` returns valid dict
- Test uptime calculation

---

# 11. Phase 2E -- Event Lifecycle Manager

**Goal:** Model event status transitions internally.

### 2E.1 Create `rads/runtime/event_lifecycle.py`

**`EventLifecycleManager` class:**

Tracks active events and their status transitions.

**State machine:**
```
CANDIDATE --> DETECTED --> CONFIRMED --> RESOLVED
```

**Methods:**
- `submit_detection(event_dict)` -- called by `FrameProcessor` when reasoning produces an accident
  - If new event: create with status `candidate`, assign event ID (`RADS-YYYYMMDD-NNNNN`)
  - If existing event updated with higher confidence: transition to `detected`
- `tick(current_time_s)` -- called each frame
  - Transition `detected` to `confirmed` if confirmation window has elapsed
  - Transition `confirmed` to `resolved` if resolution timeout has elapsed or all involved tracks are gone
- `get_active_events()` -- return list of events not yet resolved
- `get_recent_events(n)` -- return last N events including resolved (ring buffer)

**Event ID generation:**
- Format: `RADS-YYYYMMDD-NNNNN` where NNNNN is a zero-padded sequential counter per day

### 2E.2 Write tests

- Test event creation and ID generation
- Test status transitions: candidate to detected to confirmed to resolved
- Test timeout-based resolution
- Test ring buffer returns correct count

---

# 12. Phase 2F -- Runtime Entry Point

**Goal:** A single CLI entry point for runtime mode.

### 2F.1 Create `rads_runtime.py` at project root

```python
import argparse
from rads.runtime.engine import RuntimeEngine

def main():
    parser = argparse.ArgumentParser(description="RADS Deployable Runtime v1")
    parser.add_argument("--config", required=True, help="Path to pipeline configuration YAML")
    parser.add_argument("--source", type=str, default=None, help="Override source URI")
    parser.add_argument("--device", type=str, default=None, help="Override compute device")
    parser.add_argument("--api", action="store_true", help="Enable API server")
    args = parser.parse_args()
    # Build engine, apply overrides, run
    ...

if __name__ == "__main__":
    main()
```

### 2F.2 Console event handler

A default handler that prints structured events to stdout as JSON lines, for use without the API.

### 2F.3 Verification

```bash
python rads_runtime.py --help
```
Must exit 0 and print usage.

```bash
python rads_runtime.py --config rads/config/pipeline_config.yaml --source <path_to_test_video>
```
Must process the video, print events to stdout, and exit cleanly.

---

# 13. Phase 3A -- API Server and Health Endpoint

**Goal:** FastAPI server running alongside the processing loop.

### 3A.1 Create `rads/api/__init__.py`

Empty init.

### 3A.2 Create `rads/api/server.py`

FastAPI application with:
- `GET /health` -- returns `HealthMonitor.get_health()`
- `GET /config` -- returns current configuration (read-only)

The server runs in a background thread, started by `RuntimeEngine` when `api.enabled` is true.

### 3A.3 Create `rads/api/schemas.py`

Pydantic models for API responses:
- `HealthResponse`
- `EventResponse`
- `ConfigResponse`

### 3A.4 Add dependencies to `requirements.txt`

```
fastapi>=0.100.0
uvicorn>=0.23.0
websockets>=11.0
```

### 3A.5 Verification

- Start runtime with `--api` flag
- `curl http://localhost:8100/health` returns 200 with valid JSON

---

# 14. Phase 3B -- Event Endpoints

**Goal:** REST endpoints for querying detected events.

### 3B.1 Add to `rads/api/server.py`

- `GET /events` -- returns recent events from `EventLifecycleManager.get_recent_events()`
- `GET /events/latest` -- returns most recent event
- `GET /events/{event_id}` -- returns specific event or 404

### 3B.2 Verification

- Process a video known to contain an accident
- `curl http://localhost:8100/events` returns the detected event

---

# 15. Phase 3C -- WebSocket Event Stream

**Goal:** Live event notifications over WebSocket.

### 3C.1 Add to `rads/api/server.py`

- `WebSocket /ws/events`
- On event status transition, broadcast JSON to all connected WebSocket clients
- Handle client connect/disconnect gracefully

### 3C.2 Verification

- Connect a WebSocket client (wscat or Python script)
- Process a video with an accident
- Verify event JSON is received over the WebSocket

---

# 16. Phase 4A -- Dockerfile and Build

**Goal:** A working Docker image.

### 4A.1 Create `Dockerfile`

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY rads/ ./rads/
COPY rads_runtime.py .
COPY run_pipeline.py .
ENV RADS_MODEL_PATH=/app/models/yolo11n.pt
EXPOSE 8100
CMD ["python", "rads_runtime.py", "--config", "/app/config/pipeline_config.yaml", "--api"]
```

### 4A.2 Create `.dockerignore`

```
Datasets/
training/outputs/
wandb/
.venv/
.git/
__pycache__/
*.pyc
output_videos/
eda/
scratch/
docs/
```

### 4A.3 Verification

```bash
docker build -t rads:v1 .
```
Must complete without errors.

---

# 17. Phase 4B -- Docker Compose and Deployment

**Goal:** Single-command deployment.

### 4B.1 Create `docker-compose.yml`

```yaml
services:
  rads:
    build: .
    ports:
      - "8100:8100"
    volumes:
      - ./rads/config:/app/config
      - ./models:/app/models
    environment:
      - RADS_DEVICE=cpu
      - RADS_API_ENABLED=true
```

### 4B.2 Create `docker-compose.gpu.yml` (GPU override)

```yaml
services:
  rads:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    environment:
      - RADS_DEVICE=cuda
```

### 4B.3 Create `models/` directory structure

Move `yolo11n.pt` to `models/yolo11n.pt` if not already done in Phase 0.

### 4B.4 Verification

```bash
docker compose up --build
```
Must start and show health endpoint responding.

---

# 18. Phase 5 -- Integration Testing and Verification

**Goal:** End-to-end verification that everything works together.

### 5.1 Batch pipeline compatibility

```bash
python run_pipeline.py --video <test_video> --config rads/config/pipeline_config.yaml --output test_output.json
```
Must produce valid JSON output identical in structure to pre-runtime results.

### 5.2 Runtime file processing

```bash
python rads_runtime.py --config rads/config/pipeline_config.yaml --source <test_video>
```
Must process the video and print event JSON to stdout.

### 5.3 Runtime with API

```bash
python rads_runtime.py --config rads/config/pipeline_config.yaml --source <test_video> --api
```
Then:
```bash
curl http://localhost:8100/health
curl http://localhost:8100/events
```

### 5.4 Output comparison

Compare the event detected by the runtime with the event detected by the batch pipeline for the same video. The accident/no-accident decision, impact time, and involved objects should match.

### 5.5 Docker deployment

```bash
docker compose up --build
curl http://localhost:8100/health
```

### 5.6 All tests

```bash
python -m pytest rads/tests/ -v
```

All tests (existing and new) must pass.

---

# 19. Files Created and Modified

## New Files (14)

| File | Phase |
|---|---|
| `rads/core/__init__.py` | 1A |
| `rads/config/env_resolver.py` | 1C |
| `rads/runtime/__init__.py` | 2A |
| `rads/runtime/source.py` | 2A |
| `rads/runtime/frame_processor.py` | 2B |
| `rads/runtime/motion_adapter.py` | 2B |
| `rads/runtime/pairwise_adapter.py` | 2B |
| `rads/runtime/engine.py` | 2C |
| `rads/runtime/health.py` | 2D |
| `rads/runtime/event_lifecycle.py` | 2E |
| `rads_runtime.py` | 2F |
| `rads/api/__init__.py` | 3A |
| `rads/api/server.py` | 3A |
| `rads/api/schemas.py` | 3A |

## New Deployment Files (4)

| File | Phase |
|---|---|
| `Dockerfile` | 4A |
| `.dockerignore` | 4A |
| `docker-compose.yml` | 4B |
| `docker-compose.gpu.yml` | 4B |

## New Test Files (4)

| File | Phase |
|---|---|
| `rads/tests/test_source.py` | 2A |
| `rads/tests/test_frame_processor.py` | 2B |
| `rads/tests/test_engine.py` | 2C |
| `rads/tests/test_event_lifecycle.py` | 2E |

## Modified Files (3)

| File | Phase | Change |
|---|---|---|
| `rads/config/config_loader.py` | 1B | Add 13 new property getters with defaults |
| `rads/config/pipeline_config.yaml` | 1B | Add 3 new YAML sections |
| `requirements.txt` | 3A | Add fastapi, uvicorn, websockets |

## Preserved Unchanged (all others)

Every file under `rads/detection/`, `rads/tracking/`, `rads/motion/`, `rads/interaction/`, `rads/reasoning/`, `rads/severity/`, `rads/output/`, `rads/pipeline/`, `rads/evaluation/`, `training/`, `scripts/`, `eda/`, `Datasets/`, `run_pipeline.py`, and `docs/`.

---

# 20. Commit Strategy

Each completed phase should be committed separately:

```
Phase 0:  "chore: stabilize repository for runtime v1"
Phase 1A: "feat: add rads.core library namespace"
Phase 1B: "feat: extend configuration for runtime/device/api"
Phase 1C: "feat: add environment variable config resolver"
Phase 2A: "feat: add frame source abstraction (file/webcam/RTSP)"
Phase 2B: "feat: add streaming frame processor"
Phase 2C: "feat: add runtime engine main loop"
Phase 2D: "feat: add health monitor and signal handling"
Phase 2E: "feat: add event lifecycle manager"
Phase 2F: "feat: add runtime CLI entry point"
Phase 3A: "feat: add FastAPI server with health endpoint"
Phase 3B: "feat: add event REST endpoints"
Phase 3C: "feat: add WebSocket event stream"
Phase 4A: "feat: add Dockerfile"
Phase 4B: "feat: add docker-compose deployment"
Phase 5:  "test: integration verification"
```

---

# 21. Risk Register

| Risk | Mitigation |
|---|---|
| Tracker state leak between RTSP reconnections | `FrameProcessor.reset()` clears all state on reconnect |
| Pairwise O(n^2) scaling on busy intersections | Window pruning limits active tracks; future: spatial bucketing |
| Motion features batch API incompatible with streaming | Adapter modules in Phase 2B wrap existing functions |
| FastAPI thread conflicts with OpenCV | API runs in a daemon thread; frame processing stays on main thread |
| Docker image size (PyTorch + OpenCV) | Use slim base; consider multi-stage build if image exceeds 2 GB |
| RTSP stream latency accumulation | Drop frames if processing falls behind; configurable frame skip |
