import logging
import signal
import time
from rads.config import ConfigLoader
from rads.live.events import EventLifecycleManager
from rads.live.processor import FrameProcessor
from rads.live.source import create_source


logger = logging.getLogger(__name__)


class RuntimeEngine:
    def __init__(self, config_path, source_uri=None, api_enabled=None):
        self.config = ConfigLoader(config_path)
        if source_uri is not None:
            self.config.config.setdefault('runtime', {})['source_uri'] = source_uri
        if api_enabled is not None:
            self.config.config.setdefault('api', {})['enabled'] = api_enabled
        self.source = create_source(self.config)
        self.processor = FrameProcessor(self.config)
        self.health = HealthMonitor()
        self._handlers = []
        self.lifecycle = EventLifecycleManager(
            confirmation_window_s=self.config.runtime_event_confirmation_window_s,
            resolution_timeout_s=self.config.runtime_event_resolution_timeout_s,
            buffer_size=self.config.api_event_buffer_size,
            handlers=self._handlers,
        )
        self._running = False
        self._frames_processed = 0
        self._events_detected = 0

    def register_handler(self, callback):
        self._handlers.append(callback)

    def stop(self):
        self._running = False

    def _on_signal(self, signum, _frame):
        logger.info("shutdown: signal %s", signum)
        self.stop()

    def _install_signal_handlers(self):
        previous = {}
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                previous[sig] = signal.signal(sig, self._on_signal)
            except (ValueError, OSError):
                continue
        return previous

    def _restore_signal_handlers(self, previous):
        for sig, handler in previous.items():
            signal.signal(sig, handler)

    def run(self):
        self._running = True
        self._frames_processed = 0
        self._events_detected = 0
        started = time.perf_counter()
        frame_skip = self.config.frame_skip
        previous_handlers = self._install_signal_handlers()
        server = None
        try:
            if self.config.api_enabled:
                from rads.live.api import start_api
                server = start_api(self)
                deadline = time.time() + 10
                while not server.started and time.time() < deadline:
                    time.sleep(0.05)
                if not server.started:
                    raise RuntimeError("API server failed to start")
            self.source.open()
            self.health.source_status = "open"
            while self._running:
                try:
                    item = self.source.read()
                except KeyboardInterrupt:
                    self.stop()
                    break
                if item is None:
                    break
                frame_idx, timestamp_s, frame = item
                if frame_idx % frame_skip == 0:
                    try:
                        result = self.processor.process_frame(frame, frame_idx, timestamp_s)
                    except KeyboardInterrupt:
                        self.stop()
                        break
                    self._frames_processed += 1
                    if result is not None and result.get('accident'):
                        self.lifecycle.submit_detection(result)
                        self._events_detected = self.lifecycle.created_count
                self.lifecycle.active_track_ids = self.processor.track_history.get_all_track_ids()
                self.lifecycle.tick(timestamp_s)
                self.health.source_status = "open" if self.source.is_open() else "closed"
                self.health.update(self._frames_processed, self._events_detected)
        finally:
            if server is not None:
                server.should_exit = True
            self._restore_signal_handlers(previous_handlers)
            self.source.close()
            self.health.source_status = "closed"
            elapsed = time.perf_counter() - started
            logger.info(
                "frames processed=%d events detected=%d runtime_s=%.2f",
                self._frames_processed,
                self._events_detected,
                elapsed,
            )


class HealthMonitor:
    def __init__(self):
        self._start_time = time.time()
        self.frames_processed = 0
        self.events_detected = 0
        self.last_event_time = None
        self.source_status = "closed"
        self._seen_events = 0

    def update(self, frames, events):
        self.frames_processed = frames
        if events > self._seen_events:
            self.last_event_time = time.time()
        self.events_detected = events
        self._seen_events = events

    def get_uptime_s(self):
        return time.time() - self._start_time

    def get_health(self):
        uptime_s = self.get_uptime_s()
        current_fps = self.frames_processed / uptime_s if uptime_s > 0 else 0.0
        return {
            "start_time": self._start_time,
            "frames_processed": self.frames_processed,
            "events_detected": self.events_detected,
            "last_event_time": self.last_event_time,
            "source_status": self.source_status,
            "current_fps": current_fps,
        }
