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
    *, storage_days: int = 12, rag_index: Any = None,
    farm_override: dict[str, Any] | None = None,
) -> DecisionProvider:
    """환경 설정에 맞는 provider를 반환한다.

    기본값은 current main의 ``serving.db``와 RAG를 연결하는 legacy adapter다.
    고정 구조 검증 데이터가 필요할 때만 ``D_PROVIDER_MODE=fixture``를 명시한다.

    ``farm_override``는 화면에서 사용자가 정한 농장 좌표다. 값이 있을 때만
    provider에 전달한다 — 기존 외부 factory는 ``storage_days``/``rag_index``
    두 개만 받는 계약이라, 위치를 쓰지 않는 호출에서 시그니처를 깨지 않기
    위해서다. ``fixture`` 모드는 고정 시연 데이터라 좌표를 반영하지 않는다.
    """

    extra: dict[str, Any] = {"farm_override": farm_override} if farm_override else {}

    spec = os.getenv("D_PROVIDER_FACTORY", "").strip()
    if spec:
        return _load_factory(spec)(
            storage_days=storage_days, rag_index=rag_index, **extra
        )

    mode = os.getenv("D_PROVIDER_MODE", "legacy").strip().lower()
    if mode == "fixture":
        from agents.fixture_provider import FixtureProvider

        return FixtureProvider(storage_days=storage_days)
    if mode == "legacy":
        from agents.legacy_provider import LegacyProvider

        # storage_days 를 빠뜨리면 사이드바 「분뇨 저장 경과일」이 legacy 모드에서만
        # 무동작이 된다 (바로 위 fixture 분기는 처음부터 넘기고 있었다).
        return LegacyProvider(rag_index=rag_index, storage_days=storage_days, **extra)
    raise ValueError("D_PROVIDER_MODE는 fixture 또는 legacy여야 합니다.")
