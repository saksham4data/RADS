from typing import Any, List, Optional

from pydantic import BaseModel, RootModel


class HealthResponse(BaseModel):
    start_time: float
    frames_processed: int
    events_detected: int
    last_event_time: Optional[float]
    source_status: str
    current_fps: float


class EventResponse(BaseModel):
    event_id: str
    status: str
    start_time: Optional[float] = None
    impact_time: Optional[float] = None
    end_time: Optional[float] = None
    objects_involved: List[Any]
    confidence: float
    severity: Optional[str] = None
    severity_score: Optional[float] = None
    severity_evidence: List[Any]
    accident_type: str


class ConfigResponse(RootModel[dict]):
    pass
