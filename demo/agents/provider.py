"""D 에이전트가 사용하는 데이터 포트와 provider factory.

`app/`은 provider 구현을 직접 알지 않는다. 기본값은 current main adapter이며,
실데이터 연결 시 ``D_PROVIDER_FACTORY=package.module:create_provider``만 바꾼다.
연결 실패를 fixture로 조용히 대체하지 않아 출처 오인을 막는다.
"""
from __future__ import annotations

import importlib
import os
from collections.abc import Callable
from typing import Any, Protocol


class DecisionProvider(Protocol):
    def get_system_status(self) -> dict[str, Any]: ...
    def get_sensor_snapshot(self, at: str | None = None) -> dict[str, Any]: ...
    def get_risk_calendar(
        self, farm_id: str, days: int, work_type: str | None = None
    ) -> dict[str, Any]: ...
    def get_forecast(self, farm_id: str, days: int) -> dict[str, Any]: ...
    def get_storage_days(self, farm_id: str) -> dict[str, Any]: ...
    def search_rag(
        self, question: str, query_type: str | None = None
    ) -> dict[str, Any]: ...
    def get_farm_config(self, farm_id: str) -> dict[str, Any]: ...
    def get_plume_assessment(self, farm_id: str, when: str) -> dict[str, Any]: ...


ProviderFactory = Callable[..., DecisionProvider]


def _load_factory(spec: str) -> ProviderFactory:
    module_name, separator, function_name = spec.partition(":")
    if not separator or not module_name or not function_name:
        raise ValueError(
            "D_PROVIDER_FACTORY는 'package.module:function' 형식이어야 합니다."
        )
    factory: Any = getattr(importlib.import_module(module_name), function_name)
    if not callable(factory):
        raise TypeError(f"provider factory가 호출 가능하지 않습니다: {spec}")
    return factory


def create_provider(
    *, storage_days: int = 12, rag_index: Any = None
) -> DecisionProvider:
    """환경 설정에 맞는 provider를 반환한다.

    기본값은 current main의 ``serving.db``와 RAG를 연결하는 legacy adapter다.
    고정 구조 검증 데이터가 필요할 때만 ``D_PROVIDER_MODE=fixture``를 명시한다.
    """

    spec = os.getenv("D_PROVIDER_FACTORY", "").strip()
    if spec:
        return _load_factory(spec)(storage_days=storage_days, rag_index=rag_index)

    mode = os.getenv("D_PROVIDER_MODE", "legacy").strip().lower()
    if mode == "fixture":
        from agents.fixture_provider import FixtureProvider

        return FixtureProvider(storage_days=storage_days)
    if mode == "legacy":
        from agents.legacy_provider import LegacyProvider

        return LegacyProvider(rag_index=rag_index)
    raise ValueError("D_PROVIDER_MODE는 fixture 또는 legacy여야 합니다.")
