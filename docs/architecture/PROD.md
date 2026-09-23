# RADS -- Production Deployment Specification

**Project:** RADS
**Document:** Production Deployment Specification
**Version:** 1.0
**Status:** Active -- Runtime v1 Development
**Replaces:** MVP.md
**Date:** September 2026

---

# 1. Purpose

This document defines the scope, requirements, and success criteria for RADS Deployable Runtime v1.

RADS has completed the research MVP phase. The object-centric, temporal reasoning pipeline has been demonstrated end-to-end on pre-recorded video clips. The core intelligence modules (detection, tracking, motion features, pairwise interaction, accident reasoning, severity estimation) are implemented and tested.

The next objective is to make this intelligence deployable: processing live and recorded video from real sources, exposing events over a network API, and running portably on any supported machine via Docker.

This document does not redefine the reasoning philosophy. That remains in AI.md. This document defines what the deployable runtime must do and how it will be verified.

---

# 2. Scope

RADS Deployable Runtime v1 must process video from three source types through the full reasoning pipeline and expose structured accident events over a REST and WebSocket API.

```text
Source (file / webcam / RTSP)
       |
       v
  Frame Acquisition
       |
       v
  Object Detection (YOLO)
       |
       v
  Object Tracking (ByteTrack)
       |
       v
  Track History + Motion Features
       |
       v
  Pairwise Interaction Detection
       |
       v
  Accident Reasoning
       |
       v
  Severity Estimation
       |
       v
  Structured Event (with lifecycle)
       |
       v
  API / WebSocket / Alerts
```

---

# 3. Input Sources

The runtime must accept video from:

| Source | Input Format | Behavior |
|---|---|---|
| File | Local path to MP4/AVI/MOV | Process to completion, then stop or loop |
| Webcam | Device index (0, 1, ...) | Continuous processing until shutdown |
| RTSP/IP Camera | `rtsp://` URL | Continuous processing with automatic reconnection on failure |

The source type and URI are specified in the configuration file or via environment variable override.

The source module must handle:
- Connection failure at startup (report and retry or exit with clear error)
- Mid-stream disconnection (RTSP: reconnect with configurable interval and max attempts)
- Graceful shutdown on SIGINT/SIGTERM

---

# 4. Processing Model

## 4.1 Streaming with Sliding Window

The runtime processes frames as they arrive. It does not wait for the entire video before reasoning.

Reasoning operates on a configurable sliding window (default: 30 seconds). The window bounds memory usage and ensures the system can run indefinitely on continuous sources.

When the window advances, tracks that have not been seen for longer than the window duration are pruned from active state.

## 4.2 Per-Frame Processing

For each frame, the runtime executes:

1. YOLO detection via the Tracker
2. Track history update (trajectory store)
3. Incremental pairwise metric computation for active track pairs
4. Interaction candidate evaluation
5. Accident reasoning on current candidates
6. Severity estimation if an accident is detected

Steps 1-3 execute on every processed frame. Steps 4-6 execute periodically or when an interaction trigger fires, depending on configuration.

## 4.3 Frame Skip

The runtime respects the existing `frame_skip` configuration parameter. Default is 1 (process every frame). Higher values reduce compute at the cost of temporal resolution.

---

# 5. Event Lifecycle

Detected accidents follow a structured lifecycle:

```text
CANDIDATE --> DETECTED --> CONFIRMED --> RESOLVED
```

Each event carries:

```json
{
  "event_id": "RADS-20260922-00017",
  "status": "confirmed",
  "start_time": 123.4,
  "impact_time": 125.1,
  "end_time": 132.7,
  "objects_involved": [
    {"id": 3, "class": "car"},
    {"id": 7, "class": "car"}
  ],
  "confidence": 0.87,
  "severity": "medium",
  "severity_score": 3.2,
  "severity_evidence": [
    "object_count: 2 (score: 1.0)",
    "relative_velocity: 45.2 px/s (score: 0.8)"
  ],
  "accident_type": "unknown"
}
```

**Status transitions:**
- `candidate`: Interaction detected, reasoning score exceeds initial threshold
- `detected`: Reasoning score exceeds accident threshold (0.5 by default)
- `confirmed`: Event persists for a configurable confirmation window
- `resolved`: All involved tracks have left the scene, or a timeout has elapsed

Each status transition emits an event notification to all registered handlers (API, WebSocket, log).

---

# 6. API Specification

The runtime exposes a FastAPI server on a configurable port (default: 8100).

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Runtime health: uptime, frames processed, source status, last event time |
| `/status` | GET | Current processing state: active tracks, source URI, FPS, device |
| `/events` | GET | Recent events from ring buffer (configurable depth, default 100) |
| `/events/latest` | GET | Most recent detected event |
| `/events/{event_id}` | GET | Specific event by ID |
| `/ws/events` | WebSocket | Live event stream, pushes status transitions as they occur |
| `/config` | GET | Current runtime configuration (read-only) |

The API is optional. It is enabled or disabled via configuration (`api.enabled: true/false`).

When disabled, the runtime still processes video and logs events to stdout/file.

---

# 7. Device and Hardware

The runtime must support:

| Device | Configuration |
|---|---|
| CPU | `device.compute: cpu` |
| CUDA GPU | `device.compute: cuda` or `device.compute: cuda:0` |
| Auto-detect | `device.compute: auto` (default, uses CUDA if available) |

The device setting is propagated to the YOLO model and to PyTorch operations.

The YOLO model path must be configurable and must not contain hardcoded local paths. Default: `models/yolo11n.pt` relative to the project root. Overridable via `RADS_MODEL_PATH` environment variable.

---

# 8. Configuration

All configuration is in a single YAML file, extending the existing `pipeline_config.yaml`.

New sections added for the runtime:

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

Environment variable overrides follow the pattern `RADS_<SECTION>_<KEY>` in uppercase:
- `RADS_SOURCE_URI`
- `RADS_DEVICE`
- `RADS_MODEL_PATH`
- `RADS_API_PORT`
- `RADS_API_ENABLED`

Environment variables take precedence over YAML values.

---

# 9. Docker Deployment

The runtime must be deployable as a Docker container.

Requirements:
- A `Dockerfile` that produces a working image from `python:3.11-slim` (CPU) or `nvidia/cuda` base (GPU)
- A `docker-compose.yml` for single-command startup
- Model weights mounted via volume, not baked into the image
- Configuration mounted via volume
- No hardcoded local paths anywhere in the runtime code
- `.dockerignore` excludes `Datasets/`, `training/outputs/`, `wandb/`, `.venv/`, `.git/`

```bash
# CPU
docker compose up

# GPU
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up
```

---

# 10. Scalability Architecture

Runtime v1 is single-stream. However, the internal architecture must support scaling to multiple streams by changing configuration, not code.

Design constraints for future multi-stream:
- `FrameProcessor` is instantiated per stream, carries its own Tracker, TrackHistory, and reasoning state
- The event bus is shared across processors
- The API aggregates events from all processors
- No global mutable state outside of the per-processor instances

This does not mean multi-stream is implemented in v1. It means the code does not prevent it.

---

# 11. Research System Preservation

The existing research and evaluation system is preserved unchanged:

| Component | Files | Status |
|---|---|---|
| Batch pipeline | `rads/pipeline/pipeline.py`, `run_pipeline.py` | Preserved, still works |
| Evaluator | `rads/evaluation/evaluator.py` | Preserved |
| Baseline comparison | `rads/evaluation/baseline_comparison.py` | Preserved |
| Training code | `training/` | Preserved, untouched |
| EDA | `eda/` | Preserved, untouched |
| Experiment scripts | `scripts/` | Preserved, untouched |
| Datasets | `Datasets/` | Preserved, untouched |

`run_pipeline.py` remains the entry point for batch evaluation on pre-recorded clips.
`rads_runtime.py` is the new entry point for deployable runtime mode.

Both share the same core intelligence modules.

---

# 12. Known Limitations Carried Forward

These limitations exist in the current reasoning system and are not addressed in Runtime v1. They are documented here so they are not silently forgotten.

| ID | Limitation | Impact |
|---|---|---|
| B9 | Single-vehicle accidents cannot be detected (reasoning is pairwise-only) | Misses car-vs-pole, rollover, run-off-road |
| -- | All numeric thresholds are unvalidated engineering choices | Performance depends on camera, scene, resolution |
| -- | Image-space only (pixels, not meters) | Thresholds are camera-dependent without calibration |
| -- | No accident type classification | Returns `"accident_type": "unknown"` always |

---

# 13. Success Criteria

RADS Deployable Runtime v1 is complete when:

### Pipeline
- A local MP4 file can be processed end-to-end through the runtime, producing structured event output
- A webcam (device 0) can be processed continuously with live event output
- An RTSP stream can be connected, processed, and reconnected after disconnection
- The sliding window bounds memory usage during continuous processing
- Events follow the defined lifecycle (candidate, detected, confirmed, resolved)

### API
- `GET /health` returns runtime status
- `GET /events` returns detected events
- `WebSocket /ws/events` delivers live event notifications
- API can be disabled via configuration without affecting processing

### Deployment
- `docker build` produces a working image
- `docker compose up` starts the runtime and processes video
- The container runs without any hardcoded local paths
- CPU and CUDA configurations both work

### Compatibility
- `python run_pipeline.py --video <path> --config <config> --output <out>` still works exactly as before
- All existing unit tests pass
- The research evaluation system is unaffected

### Verification Method
- Automated: unit tests, integration tests for source/reconnect/event-lifecycle
- Manual: run on MP4 and compare event output with batch pipeline output for the same video
- Manual: run on webcam, verify live processing
- Manual: `docker compose up` on a clean machine

---

# 14. What This Document Does Not Cover

- Severity model training or fitting (rule-based heuristic is retained)
- Accident type classification
- Camera calibration or world-space coordinate mapping
- Multi-stream support (architecture is ready, implementation is not in scope)
- Frontend dashboard UI
- Telegram or notification integration (the event bus supports adding handlers later)
- Performance optimization or real-time guarantees

These are future concerns documented in the roadmap, not v1 requirements.
