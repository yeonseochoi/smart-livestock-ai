"""D 작업 가이드가 사용하는 공급자 독립 도구 계약.

현재 단계에서는 애플리케이션 코드가 도구를 호출하고 Gemini는 확정 결과를
설명만 한다. ``TOOL_SPECS``와 ``dispatch_tool``은 추후 Gemini function calling을
붙일 수 있도록 공급자 중립 형태로 유지한다.
"""
from __future__ import annotations

from typing import Any

from agents.provider import DecisionProvider


TOOL_SPECS: tuple[dict[str, Any], ...] = (
    {
        "name": "get_risk_calendar",
        "description": "기간별 1시간 민원 위험도와 출처·기준시각을 조회한다.",
        "parameters": {
            "type": "object",
            "properties": {
                "farm_id": {"type": "string"},
                "days": {"type": "integer", "minimum": 1, "maximum": 7},
                "work_type": {"type": ["string", "null"]},
            },
            "required": ["farm_id", "days"],
            "additionalProperties": False,
        },
    },
    {
        "name": "get_storage_days",
        "description": "분뇨 저장 경과일과 잠정 14일 기준 상태를 조회한다.",
        "parameters": {
            "type": "object",
            "properties": {"farm_id": {"type": "string"}},
            "required": ["farm_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "search_rag",
        "description": "PR #9 법령·매뉴얼 RAG의 검색 결과와 출처 메타데이터를 조회한다.",
        "parameters": {
            "type": "object",
            "properties": {
                "question": {"type": "string"},
                "query_type": {"type": ["string", "null"]},
            },
            "required": ["question"],
            "additionalProperties": False,
        },
    },
    {
        "name": "get_farm_config",
        "description": "선택 농가의 기본 설정과 데이터 상태를 조회한다.",
        "parameters": {
            "type": "object",
            "properties": {"farm_id": {"type": "string"}},
            "required": ["farm_id"],
            "additionalProperties": False,
        },
    },
)

_ALLOWLIST = {item["name"] for item in TOOL_SPECS}


class AgentTools:
    """작업 가이드의 내부 도구 표면. 계산·저장은 provider가 담당한다."""

    def __init__(self, provider: DecisionProvider) -> None:
        self.provider = provider

    def get_risk_calendar(
        self, farm_id: str, days: int = 3, work_type: str | None = None
    ) -> dict[str, Any]:
        return self.provider.get_risk_calendar(farm_id, days, work_type)

    def get_storage_days(self, farm_id: str) -> dict[str, Any]:
        return self.provider.get_storage_days(farm_id)

    def search_rag(
        self, question: str, query_type: str | None = None
    ) -> dict[str, Any]:
        return self.provider.search_rag(question, query_type)

    def get_farm_config(self, farm_id: str) -> dict[str, Any]:
        return self.provider.get_farm_config(farm_id)


class LocalTools(AgentTools):
    """기존 ``LocalTools(rag_index)`` 호출을 위한 호환 adapter."""

    def __init__(self, rag_index: Any = None) -> None:
        from agents.legacy_provider import LegacyProvider

        super().__init__(LegacyProvider(rag_index=rag_index))


def dispatch_tool(
    tools: AgentTools, name: str, arguments: dict[str, Any]
) -> dict[str, Any]:
    """허용 목록 밖의 속성이나 메서드는 실행하지 않는다."""

    if name not in _ALLOWLIST:
        return {
            "status": "unavailable", "data": None, "source": None,
            "error": f"허용되지 않은 도구: {name}",
        }
    try:
        return getattr(tools, name)(**arguments)
    except Exception as exc:
        return {
            "status": "unavailable", "data": None, "source": None,
            "error": f"{type(exc).__name__}: {exc}",
        }
