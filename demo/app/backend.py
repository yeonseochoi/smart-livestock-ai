"""D가 의존하는 포트와 OpenAI 함수 도구 계약."""
from __future__ import annotations

from typing import Any, Protocol


class DecisionBackend(Protocol):
    """fixture와 B/C 실연결 어댑터가 함께 지켜야 하는 계약."""

    def get_system_status(self) -> dict[str, Any]: ...

    def get_sensor_snapshot(self, at: str | None = None) -> dict[str, Any]: ...

    def get_risk_calendar(
        self, farm_id: str, days: int, work_type: str | None
    ) -> dict[str, Any]: ...

    def get_forecast(self, farm_id: str, days: int) -> dict[str, Any]: ...

    def get_storage_days(self, farm_id: str) -> dict[str, Any]: ...

    def search_rag(
        self, question: str, query_type: str | None
    ) -> dict[str, Any]: ...

    def get_farm_config(self, farm_id: str) -> dict[str, Any]: ...

    def get_plume_assessment(self, farm_id: str, when: str) -> dict[str, Any]: ...


# Responses API의 strict function tool 형식. 선택 인자도 required에 넣고 null을
# 허용한다. 기존 agents/tools_schema.py의 Anthropic input_schema와 분리한다.
OPENAI_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "name": "get_risk_calendar",
        "description": "농가의 기간별 민원 위험 캘린더와 데이터 상태를 조회한다.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "farm_id": {"type": "string"},
                "days": {"type": "integer", "minimum": 1, "maximum": 7},
                "work_type": {
                    "type": ["string", "null"],
                    "description": "작업 유형. 단순 캘린더 조회이면 null",
                },
            },
            "required": ["farm_id", "days", "work_type"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "get_forecast",
        "description": "위험도 계산에 사용된 예보 요약과 갱신 상태를 조회한다.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "farm_id": {"type": "string"},
                "days": {"type": "integer", "minimum": 1, "maximum": 7},
            },
            "required": ["farm_id", "days"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "get_storage_days",
        "description": "농가의 분뇨 저장 경과일과 임시 14일 기준 상태를 조회한다.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {"farm_id": {"type": "string"}},
            "required": ["farm_id"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "search_rag",
        "description": "C의 법령·매뉴얼 검색 결과와 출처 메타데이터를 조회한다.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "question": {"type": "string"},
                "query_type": {"type": ["string", "null"]},
            },
            "required": ["question", "query_type"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "get_farm_config",
        "description": "선택 농가의 시연용 기본 설정을 조회한다.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {"farm_id": {"type": "string"}},
            "required": ["farm_id"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "get_plume_assessment",
        "description": "선택 시각의 풍하측 영향 후보를 미검증 참고 정보로 조회한다.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "farm_id": {"type": "string"},
                "when": {
                    "type": "string",
                    "description": "ISO 8601 형식 작업 시작 시각",
                },
            },
            "required": ["farm_id", "when"],
            "additionalProperties": False,
        },
    },
]


_DISPATCH_ALLOWLIST = {tool["name"] for tool in OPENAI_TOOLS}


def dispatch_tool(
    backend: DecisionBackend, name: str, arguments: dict[str, Any]
) -> dict[str, Any]:
    """모델이 임의 메서드를 호출하지 못하도록 허용된 도구만 실행한다."""

    if name not in _DISPATCH_ALLOWLIST:
        return {
            "status": "unavailable",
            "data": None,
            "error": f"허용되지 않은 도구: {name}",
        }
    method = getattr(backend, name)
    try:
        return method(**arguments)
    except Exception as exc:  # 도구 오류는 모델에 구조화해 돌려주고 앱은 유지한다.
        return {
            "status": "unavailable",
            "data": None,
            "error": f"{type(exc).__name__}: {exc}",
        }
