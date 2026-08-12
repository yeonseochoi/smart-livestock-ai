"""D 에이전트 도구 6종의 공급자 독립 계약과 안전한 실행기.

기존 파일의 도구 이름은 유지하되 구현을 전역 ``DEMO_FARM``에서 분리했다.
API가 없어도 ``AgentTools``가 같은 provider 메서드를 호출하므로 결과 구조가 같다.
"""
from __future__ import annotations

from typing import Any

from agents.provider import DecisionProvider


_FUNCTIONS: tuple[dict[str, Any], ...] = (
    {
        "name": "get_risk_calendar",
        "description": "기간별 민원 위험 캘린더와 출처·기준시각을 조회한다.",
        "properties": {
            "farm_id": {"type": "string"},
            "days": {"type": "integer", "minimum": 1, "maximum": 7},
            "work_type": {"type": ["string", "null"]},
        },
        "required": ["farm_id", "days", "work_type"],
    },
    {
        "name": "get_forecast",
        "description": "위험도 계산에 사용된 예보와 발표·유효시각을 조회한다.",
        "properties": {
            "farm_id": {"type": "string"},
            "days": {"type": "integer", "minimum": 1, "maximum": 7},
        },
        "required": ["farm_id", "days"],
    },
    {
        "name": "get_storage_days",
        "description": "분뇨 저장 경과일과 잠정 14일 기준 상태를 조회한다.",
        "properties": {"farm_id": {"type": "string"}},
        "required": ["farm_id"],
    },
    {
        "name": "search_rag",
        "description": "C의 법령·매뉴얼 검색 결과와 출처 메타데이터를 조회한다.",
        "properties": {
            "question": {"type": "string"},
            "query_type": {"type": ["string", "null"]},
        },
        "required": ["question", "query_type"],
    },
    {
        "name": "get_farm_config",
        "description": "선택 농가의 기본 설정과 데이터 상태를 조회한다.",
        "properties": {"farm_id": {"type": "string"}},
        "required": ["farm_id"],
    },
    {
        "name": "get_plume_assessment",
        "description": "선택 시각의 풍하측 후보를 미검증 참고 정보로 조회한다.",
        "properties": {
            "farm_id": {"type": "string"},
            "when": {"type": "string", "description": "ISO 8601 작업 시작시각"},
        },
        "required": ["farm_id", "when"],
    },
)


# OpenAI Responses API strict function tool 형식.
OPENAI_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function", "name": item["name"],
        "description": item["description"], "strict": True,
        "parameters": {
            "type": "object", "properties": item["properties"],
            "required": item["required"], "additionalProperties": False,
        },
    }
    for item in _FUNCTIONS
]

# 기존 Claude용 변수명/스키마를 유지해 외부 참조를 깨뜨리지 않는다.
TOOLS: list[dict[str, Any]] = [
    {
        "name": item["name"], "description": item["description"],
        "input_schema": {
            "type": "object", "properties": item["properties"],
            "required": item["required"], "additionalProperties": False,
        },
    }
    for item in _FUNCTIONS
]

_ALLOWLIST = {item["name"] for item in _FUNCTIONS}


class AgentTools:
    """에이전트가 보는 도구 표면. 계산·저장은 provider가 담당한다."""

    def __init__(self, provider: DecisionProvider) -> None:
        self.provider = provider

    def get_risk_calendar(
        self, farm_id: str, days: int = 3, work_type: str | None = None
    ) -> dict[str, Any]:
        return self.provider.get_risk_calendar(farm_id, days, work_type)

    def get_forecast(self, farm_id: str, days: int = 3) -> dict[str, Any]:
        return self.provider.get_forecast(farm_id, days)

    def get_storage_days(self, farm_id: str) -> dict[str, Any]:
        return self.provider.get_storage_days(farm_id)

    def search_rag(
        self, question: str, query_type: str | None = None
    ) -> dict[str, Any]:
        return self.provider.search_rag(question, query_type)

    def get_farm_config(self, farm_id: str) -> dict[str, Any]:
        return self.provider.get_farm_config(farm_id)

    def get_plume_assessment(self, farm_id: str, when: str) -> dict[str, Any]:
        return self.provider.get_plume_assessment(farm_id, when)


class LocalTools(AgentTools):
    """기존 ``LocalTools(rag_index)`` 호출을 위한 legacy adapter."""

    def __init__(self, rag_index: Any = None) -> None:
        from agents.legacy_provider import LegacyProvider

        super().__init__(LegacyProvider(rag_index=rag_index))


def dispatch_tool(
    tools: AgentTools, name: str, arguments: dict[str, Any]
) -> dict[str, Any]:
    """모델이 허용 목록 밖의 속성이나 메서드를 실행하지 못하게 한다."""

    if name not in _ALLOWLIST:
        return {"status": "unavailable", "data": None,
                "source": None, "error": f"허용되지 않은 도구: {name}"}
    try:
        return getattr(tools, name)(**arguments)
    except Exception as exc:
        return {"status": "unavailable", "data": None, "source": None,
                "error": f"{type(exc).__name__}: {exc}"}
