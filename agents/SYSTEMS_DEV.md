# SYSTEMS_DEV agent

Registered as `rads-systems-dev`. Follow `agents/COMMON.md` first; this file adds to it.

## Scope

| Sub-phase | Work |
|---|---|
| P7.1 | Model integration in the live processor, per-camera state |
| P7.2 | Alert logic: k-of-n, hysteresis, one event per crash |
| P7.3 | Severity alert behaviours and event log |
| P7.4 | Evidence clips |
| P7.5 | Operator web screen |
| P7.6 | Parity between recorded and live modes |
| P7.7 | Multi-camera scheduler |
| P8.1 to P8.4 | Export and quantisation, accuracy check, throughput and capacity, latency |
| P10.1 | Demo |

## Files you may create or modify (subject to the brief)

- `rads/live/` (all modules), `rads/output.py`, `rads/pipeline.py`
- `run_live.py`, `run_video.py`
- `models/export/<version>/`
- `scripts/` (benchmarking and demo scripts)
- `tests/`
- `docs/results/` (performance reports)

## Existing system you build on

- `rads/live/source.py`: FileSource, WebcamSource, RTSPSource with reconnect.
- `rads/live/processor.py`: per-frame processing over a sliding window.
- `rads/live/events.py`: event lifecycle. Its current promotion rule (an event moves past `candidate` only if confidence strictly increases) is replaced in P7.2.
- `rads/live/engine.py`: main loop, signal handling, health.
- `rads/live/api.py`: FastAPI with `/health`, `/config`, `/events`, `/events/latest`, `/events/{id}` and WebSocket `/ws/events`, on 127.0.0.1:8100.

## Required behaviour (decided by the user, decision D013)

| Level | Message | Repeat | Logged |
|---|---|---|---|
| LOW | "Accident detected at camera X. Call emergency services." | Once | Yes |
| MEDIUM | Same as LOW | Once | Yes |
| HIGH | "Accident detected at camera X. Call emergency services now." | Until the operator acknowledges | Yes |

**Event log.** Append-only JSON lines, one record per event and per operator action. Fields:
- event ID, camera, start, impact and end time;
- impact point;
- involved objects with classes;
- collision type;
- severity level, score and every term;
- model version;
- operator action (confirmed or dismissed) and its time.

## Rules specific to this role

1. **One processing path.** `run_video.py` and `run_live.py` call the same processing code. Parity (P7.6) is tested: same events, impact time within one frame.
2. **Measure every performance claim.**
   - Each table states the CPU model, core count, RAM, OS, Python and library versions, input size, precision, and processed fps per camera.
   - Published figures (for example Ultralytics benchmarks) are labelled as published and never mixed with measurements without a label.
3. **No backlog on live sources.** Always process the newest frame from an RTSP stream; skipped frames are counted and logged.
4. **Exported or quantised models** are accepted only if the P8.2 accuracy check shows ACCS and live-detection metrics within the FP32 95% interval. INT8 calibration uses train-split frames only.
5. **Stability.** Any change to `rads/live/` must keep `python -m pytest tests` passing. The P7.7 30-minute, 3-camera run must show memory flat within 10%.
6. **Video codec.** Evidence clips must play in a browser. Use H.264 if the installed OpenCV supports it; otherwise document the codec used and the conversion step.
7. **The operator screen** is served by the existing FastAPI app on localhost. It requires no external services.
8. **Demo clips** (P10.1) are held-out test clips chosen by a fixed, recorded list. The demo never uses clips the model was trained on.

## Completion checklist (in addition to `agents/COMMON.md` section 11)

- [ ] Tests pass, including new tests for alert logic, repetition and logging.
- [ ] Parity check passes on the stated videos.
- [ ] Performance tables include the hardware and software details.
- [ ] No accuracy regression from export without a recorded acceptance.
