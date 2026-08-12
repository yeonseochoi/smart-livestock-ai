"""D 에이전트가 화면과 데이터 공급자에 공개하는 안정된 반환 계약.

A/B/C의 저장 기술이 달라져도 D의 에이전트와 Streamlit 화면은 이 계약만 본다.
공식 익산시 자료가 도착하면 provider 구현만 교체하는 것이 목표다.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


DataState = Literal["fixture", "connected", "stale", "unavailable"]
Grade = Literal["낮음", "주의", "위험"]


@dataclass(frozen=True)
class SourceInfo:
    state: DataState
    name: str
    generated_at: str
    data_as_of: str | None = None
    version: str | None = None
    limitations: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SensorObservation:
    """익산악취24 예상 자료의 canonical 관측 계약.

    관측시각과 수집시각을 분리한다. 복합악취 값의 단위는 코드북을 받기 전까지
    확정하지 않고 nullable 필드로 둔다.
    """

    observation_id: str
    station_id: str
    station_name: str
    latitude: float | None
    longitude: float | None
    observed_at: str
    ingested_at: str
    h2s_ppm: float | None
    nh3_ppm: float | None
    tvoc_ppm: float | None
    complex_odor_value: float | None
    complex_odor_unit: str | None
    temperature_c: float | None
    humidity_pct: float | None
    wind_direction_text: str | None
    wind_speed_ms: float | None
    record_qc: str
    quality_flags: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class WorkWindow:
    start: str
    end: str
    window_risk: float
    recommendation_score: float
    grade: Grade
    reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GuideCard:
    farm_id: str
    work_type: str
    recommended: tuple[WorkWindow, ...]
    avoid: tuple[WorkWindow, ...]
    before_actions: tuple[str, ...]
    after_actions: tuple[str, ...]
    evidence: tuple[dict[str, Any], ...]
    assumptions: tuple[str, ...]
    limitations: tuple[str, ...]
    source_statuses: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class NotificationDraft:
    farm_id: str
    work_type: str
    work_window: dict[str, Any]
    audience_count: int
    audience_is_mock: bool
    plume_status: Literal["unverified", "unavailable"]
    message: str
    approved: bool = False
    sent: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
