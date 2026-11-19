import datetime
from collections import deque


class EventLifecycleManager:
    def __init__(self, confirmation_window_s, resolution_timeout_s, buffer_size=100, handlers=None):
        self._confirmation_window_s = confirmation_window_s
        self._resolution_timeout_s = resolution_timeout_s
        self._handlers = handlers if handlers is not None else []
        self._recent = deque(maxlen=buffer_size)
        self._counter = 0
        self._counter_day = None
        self.created_count = 0
        self.active_track_ids = None

    def submit_detection(self, event_dict):
        involved = _involved_ids(event_dict)
        confidence = float(event_dict.get('confidence') or 0.0)
        existing = self._match_active(involved)
        if existing is None:
            event = self._new_event(event_dict, confidence)
            self._emit(event)
            return _public(event)
        if confidence > existing['confidence']:
            _apply_payload(existing, event_dict, confidence)
            if existing['status'] == 'candidate':
                existing['status'] = 'detected'
                existing['_detected_at'] = None
                self._emit(existing)
        return _public(existing)

    def tick(self, current_time_s):
        active = None if self.active_track_ids is None else set(self.active_track_ids)
        for event in list(self._recent):
            status = event['status']
            if status == 'detected':
                if event['_detected_at'] is None:
                    event['_detected_at'] = current_time_s
                elif current_time_s - event['_detected_at'] >= self._confirmation_window_s:
                    event['status'] = 'confirmed'
                    event['_confirmed_at'] = current_time_s
                    self._emit(event)
                    status = 'confirmed'
            if status == 'confirmed':
                timed_out = (
                    event['_confirmed_at'] is not None
                    and current_time_s - event['_confirmed_at'] >= self._resolution_timeout_s
                )
                tracks_gone = _tracks_gone(event, active)
                if timed_out or tracks_gone:
                    event['status'] = 'resolved'
                    self._emit(event)

    def get_active_events(self):
        return [_public(event) for event in self._recent if event['status'] != 'resolved']

    def get_recent_events(self, n):
        recent = list(self._recent)
        return [_public(event) for event in recent[-n:]]

    def _match_active(self, involved):
        if not involved:
            return None
        for event in self._recent:
            if event['status'] == 'resolved':
                continue
            if _involved_ids(event) == involved:
                return event
        return None

    def _new_event(self, event_dict, confidence):
        event = {
            'event_id': self._next_id(),
            'status': 'candidate',
            '_detected_at': None,
            '_confirmed_at': None,
        }
        _apply_payload(event, event_dict, confidence)
        self._recent.append(event)
        self.created_count += 1
        return event

    def _next_id(self):
        today = datetime.date.today()
        if self._counter_day != today:
            self._counter_day = today
            self._counter = 0
        self._counter += 1
        return f"RADS-{today:%Y%m%d}-{self._counter:05d}"

    def _emit(self, event):
        payload = _public(event)
        for handler in list(self._handlers):
            handler(payload)


def _involved_ids(event_dict):
    ids = []
    for obj in event_dict.get('objects_involved') or []:
        if isinstance(obj, dict) and obj.get('id') is not None:
            ids.append(obj['id'])
    return tuple(sorted(ids))


def _apply_payload(event, event_dict, confidence):
    timing = event_dict.get('event') or {}
    detail = event_dict.get('severity_detail') or {}
    event['start_time'] = timing.get('start_time', event_dict.get('start_time'))
    event['impact_time'] = timing.get('impact_time', event_dict.get('impact_time'))
    event['end_time'] = timing.get('end_time', event_dict.get('end_time'))
    event['objects_involved'] = list(event_dict.get('objects_involved') or [])
    event['confidence'] = confidence
    event['severity'] = event_dict.get('severity')
    event['severity_score'] = detail.get('score', event_dict.get('severity_score'))
    event['severity_evidence'] = list(detail.get('evidence', event_dict.get('severity_evidence') or []))
    event['accident_type'] = event_dict.get('accident_type', 'unknown')


def _tracks_gone(event, active):
    if active is None:
        return False
    involved = set(_involved_ids(event))
    if not involved:
        return False
    return involved.isdisjoint(active)


def _public(event):
    return {
        'event_id': event['event_id'],
        'status': event['status'],
        'start_time': event['start_time'],
        'impact_time': event['impact_time'],
        'end_time': event['end_time'],
        'objects_involved': event['objects_involved'],
        'confidence': event['confidence'],
        'severity': event['severity'],
        'severity_score': event['severity_score'],
        'severity_evidence': event['severity_evidence'],
        'accident_type': event['accident_type'],
    }
