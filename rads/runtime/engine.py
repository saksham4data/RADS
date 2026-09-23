import logging
import signal
import time

from rads.config.config_loader import ConfigLoader
from rads.config.env_resolver import resolve_config
from rads.runtime.frame_processor import FrameProcessor
from rads.runtime.health import HealthMonitor
from rads.runtime.source import create_source

logger = logging.getLogger(__name__)


class RuntimeEngine:
    def __init__(self, config_path):
        self.config = ConfigLoader(config_path)
        resolve_config(self.config)
        self.source = create_source(self.config)
        self.processor = FrameProcessor(self.config)
        self.health = HealthMonitor()
        self._handlers = []
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
        try:
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
                if frame_idx % frame_skip != 0:
                    continue
                try:
                    result = self.processor.process_frame(frame, frame_idx, timestamp_s)
                except KeyboardInterrupt:
                    self.stop()
                    break
                self._frames_processed += 1
                if result is not None and result.get('accident'):
                    self._events_detected += 1
                    for handler in list(self._handlers):
                        handler(result)
                self.health.source_status = "open" if self.source.is_open() else "closed"
                self.health.update(self._frames_processed, self._events_detected)
        finally:
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
