import time


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
