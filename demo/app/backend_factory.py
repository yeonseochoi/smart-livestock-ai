"""Streamlit과 실제 provider 구현을 분리하는 backend factory.

기본값은 결정론적 DemoBackend다. 공식 센서/B/C 어댑터가 준비되면 대시보드를
수정하지 않고 ``D_BACKEND_FACTORY=package.module:create_backend``만 설정한다.
factory는 ``storage_days`` 키워드 인자를 받고 DecisionBackend 구현을 반환한다.
"""
from __future__ import annotations

import importlib
import os
from collections.abc import Callable
from typing import Any

from .backend import DecisionBackend
from .demo_backend import DemoBackend


Factory = Callable[..., DecisionBackend]


def _load_factory(spec: str) -> Factory:
    module_name, separator, function_name = spec.partition(":")
    if not separator or not module_name or not function_name:
        raise ValueError(
            "D_BACKEND_FACTORY는 'package.module:function' 형식이어야 합니다."
        )
    module = importlib.import_module(module_name)
    factory: Any = getattr(module, function_name)
    if not callable(factory):
        raise TypeError(f"backend factory가 호출 가능하지 않습니다: {spec}")
    return factory


def create_backend(storage_days: int) -> DecisionBackend:
    """환경 설정에 맞는 backend를 만든다.

    외부 factory 로딩 실패를 fixture로 조용히 덮지 않는다. 잘못된 실연결을
    fixture 결과로 오인하는 일을 막기 위해 오류를 그대로 화면에 전달한다.
    """

    spec = os.getenv("D_BACKEND_FACTORY", "").strip()
    if not spec:
        return DemoBackend(storage_days=storage_days)
    return _load_factory(spec)(storage_days=storage_days)
