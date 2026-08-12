"""OpenAI Responses API를 이용한 선택적 작업 계획 설명기.

추천·회피 순위는 ``work_guide.plan_work``가 먼저 확정한다. 모델은 허용된 도구로
출처를 확인하고 문장만 작성하며, 오류가 나면 호출자가 규칙 기반 설명을 유지한다.
"""
from __future__ import annotations

import json
import os
from typing import Any

from agents.contracts import GuideCard
from agents.provider import DecisionProvider
from agents.tools_schema import AgentTools, OPENAI_TOOLS, dispatch_tool


SYSTEM_PROMPT = """
너는 양돈농가 작업 의사결정을 설명하는 보조자다.
- 애플리케이션이 확정한 추천·회피 창, 점수, 등급을 변경하거나 계산하지 않는다.
- 도구 결과에 없는 숫자·법령·효과를 만들지 않는다.
- '민원 위험도'와 '상대 위험 회피 지원'이라는 표현을 사용한다.
- fixture이면 첫 문단에서 시연 데이터임을 밝힌다.
- 플룸은 미검증 참고이며 등급과 추천 순위에 반영하지 않는다.
- 개별 법률 판단은 하지 않고 법률구조공단 132 또는 관할 기관을 안내한다.
""".strip()


def is_configured() -> bool:
    return bool(os.getenv("OPENAI_API_KEY") and os.getenv("OPENAI_MODEL"))


def compose(
    provider: DecisionProvider,
    guide: GuideCard | dict[str, Any],
    *,
    max_tool_rounds: int = 4,
) -> str | None:
    if not is_configured():
        return None
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("openai 패키지가 설치되지 않았습니다") from exc

    data = guide.to_dict() if isinstance(guide, GuideCard) else guide
    tools = AgentTools(provider)
    history: list[Any] = [{
        "role": "user",
        "content": "아래 작업 계획을 바꾸지 말고 설명하세요.\n"
                   + json.dumps(data, ensure_ascii=False),
    }]
    client = OpenAI()
    for _ in range(max_tool_rounds):
        response = client.responses.create(
            model=os.environ["OPENAI_MODEL"], instructions=SYSTEM_PROMPT,
            tools=OPENAI_TOOLS, input=history,
        )
        history.extend(response.output)
        calls = [item for item in response.output if item.type == "function_call"]
        if not calls:
            return response.output_text
        for call in calls:
            try:
                arguments = json.loads(call.arguments)
                result = dispatch_tool(tools, call.name, arguments)
            except json.JSONDecodeError as exc:
                result = {"status": "unavailable", "data": None,
                          "source": None, "error": f"도구 인자 JSON 오류: {exc}"}
            history.append({
                "type": "function_call_output", "call_id": call.call_id,
                "output": json.dumps(result, ensure_ascii=False),
            })
    raise RuntimeError("OpenAI 도구 호출이 최대 반복 횟수를 초과했습니다")
