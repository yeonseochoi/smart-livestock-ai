"""OpenAI Responses API를 이용한 선택적 설명 생성 어댑터.

점수와 추천 순위는 guide_service가 먼저 확정한다. 모델은 허용된 도구로 출처를
확인하고, 그 결과를 농장주가 읽기 쉬운 문장으로 바꾸는 역할만 맡는다.
"""
from __future__ import annotations

import json
import os
from typing import Any

from .backend import DecisionBackend, OPENAI_TOOLS, dispatch_tool
from .contracts import GuideCard


SYSTEM_PROMPT = """
너는 양돈농가의 작업 의사결정을 설명하는 보조자다.
- 애플리케이션이 확정한 추천 창, 회피 창, 점수, 등급을 변경하거나 새로 계산하지 않는다.
- 도구 결과에 없는 수치·법령·효과를 만들지 않는다.
- '민원 위험도'와 '상대 위험 회피 지원'이라는 표현을 사용한다.
- fixture/mock이면 첫 문단에서 시연 데이터임을 분명히 밝힌다.
- 플룸은 미검증 참고 정보이며 위험 등급에 반영하지 않는다.
- 법령 근거에는 문서명·단원·쪽수를 보존하고, 쪽수가 없으면 미확인이라고 쓴다.
- 개별 법률 판단은 하지 않고 법률구조공단 132 또는 관할 기관 확인을 안내한다.
- 출력은 추천 창 / 대안 / 작업 전후 조치 / 판단 근거 / 한계 순서로 짧게 작성한다.
""".strip()


def is_configured() -> bool:
    """API 키와 팀이 선택한 모델명이 모두 있을 때만 연결 모드를 허용한다."""

    return bool(os.environ.get("OPENAI_API_KEY") and os.environ.get("OPENAI_MODEL"))


def compose_guide(
    backend: DecisionBackend,
    guide: GuideCard | dict[str, Any],
    max_tool_rounds: int = 4,
) -> str | None:
    """도구 호출이 끝날 때까지 Responses API 루프를 실행한다.

    키 또는 모델명이 없으면 None을 반환해 화면이 규칙 기반 설명을 사용하게 한다.
    """

    if not is_configured():
        return None
    try:
        from openai import OpenAI
    except ImportError as exc:  # 설치 누락은 fixture 시연 전체를 중단시키지 않는다.
        raise RuntimeError("openai 패키지가 설치되지 않았습니다") from exc

    guide_data = guide.to_dict() if isinstance(guide, GuideCard) else guide
    client = OpenAI()
    model = os.environ["OPENAI_MODEL"]
    history: list[Any] = [
        {
            "role": "user",
            "content": (
                "아래 결정론적 작업 계획을 변경하지 말고 설명해 주세요. 필요한 "
                "출처와 상태는 도구로 확인하세요.\n"
                + json.dumps(guide_data, ensure_ascii=False)
            ),
        }
    ]

    for _ in range(max_tool_rounds):
        response = client.responses.create(
            model=model,
            instructions=SYSTEM_PROMPT,
            tools=OPENAI_TOOLS,
            input=history,
        )
        history.extend(response.output)
        calls = [item for item in response.output if item.type == "function_call"]
        if not calls:
            return response.output_text
        for call in calls:
            try:
                arguments = json.loads(call.arguments)
            except json.JSONDecodeError as exc:
                result = {
                    "status": "unavailable",
                    "data": None,
                    "error": f"도구 인자 JSON 오류: {exc}",
                }
            else:
                result = dispatch_tool(backend, call.name, arguments)
            history.append(
                {
                    "type": "function_call_output",
                    "call_id": call.call_id,
                    "output": json.dumps(result, ensure_ascii=False),
                }
            )
    raise RuntimeError("OpenAI 도구 호출이 최대 반복 횟수를 초과했습니다")
