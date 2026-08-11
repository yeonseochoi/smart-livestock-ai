"""D 파트의 안정된 출력 계약.

이 모듈은 A/B/C의 내부 저장 방식과 Streamlit 화면을 분리한다. 공식 데이터가
도착해도 아래 반환형을 유지하고 provider만 교체하는 것이 목표다.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


DataState = Literal["fixture", "connected", "stale", "unavailable"]
Grade = Literal["낮음", "주의", "위험"]


@dataclass(frozen=True)
class SourceInfo:
    """화면과 LLM 도구 결과에 항상 붙는 출처 정보."""

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
    """익산악취24 예상 자료를 받기 위한 canonical 측정소 관측 계약.

    복합악취 값의 의미·단위는 화면 캡처만으로 확정하지 않고 별도 필드로 둔다.
    실제 자료가 오면 컬럼명만 이 계약으로 매핑한다.
    """

    observation_id: str
    station_id: str
    station_number: str | None
    station_name: str
    station_type: str | None
    latitude: float | None
    longitude: float | None
    coord_quality: str
    observed_at: str
    ingested_at: str
    status_label: str | None
    h2s_ppm: float | None
    nh3_ppm: float | None
    tvoc_ppm: float | None
    complex_odor_value: float | None
    complex_odor_unit: str | None
    temperature_c: float | None
    humidity_pct: float | None
    wind_direction_text: str | None
    wind_direction_degree: float | None
    wind_speed_ms: float | None
    record_qc: str
    quality_flags: tuple[str, ...]
    source_row_id: str
    data_version: str

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
