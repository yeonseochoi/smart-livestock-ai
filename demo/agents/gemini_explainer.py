"""확정된 작업 계획을 변경하지 않고 설명만 생성하는 Gemini adapter."""
from __future__ import annotations

import json
import os
from typing import Any

import config  # demo/.env를 먼저 읽어 GOOGLE_API_KEY와 모델 설정을 반영한다.

from agents.contracts import GuideCard


GEMINI_MODEL = os.environ.get("GEMINI_MODEL_NAME", "gemini-3.5-flash")
SYSTEM_PROMPT = """
너는 양돈농가 작업 가이드의 설명을 작성하는 보조자다.
- 애플리케이션이 확정한 추천·회피 시간, 점수와 등급을 변경하거나 다시 계산하지 않는다.
- 제공된 작업 계획과 RAG 근거에 있는 내용만 사용한다.
- 자료에 없는 법적 기준, 수치, 작업 효과를 만들지 않는다.
- 법령·매뉴얼을 설명할 때 source_file, unit, page를 함께 표시한다.
- 근거가 없거나 source 상태가 unavailable이면 확인할 수 없다고 명시한다.
- fixture 데이터는 실제 관측값처럼 표현하지 않는다.
- 민원 감소나 악취 저감 효과를 보장하지 않는다.
- 추천·회피 시간표를 다시 만들지 말고 현장 작업자가 이해할 설명만 작성한다.
""".strip()


def is_configured() -> bool:
    return bool(os.environ.get("GOOGLE_API_KEY"))


def compose(guide: GuideCard | dict[str, Any]) -> str | None:
    """Gemini가 설명만 반환한다. 확정 시간표는 호출자가 별도로 출력한다."""

    if not is_configured():
        return None
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except ImportError as exc:
        raise RuntimeError("langchain-google-genai 패키지가 설치되지 않았습니다") from exc

    data = guide.to_dict() if isinstance(guide, GuideCard) else guide
    payload = {
        "guide": data,
        "rag_evidence": data.get("evidence", []),
        "source_statuses": data.get("source_statuses", {}),
        "assumptions": data.get("assumptions", []),
        "limitations": data.get("limitations", []),
    }
    prompt = (
        SYSTEM_PROMPT
        + "\n\n다음 JSON은 애플리케이션이 확정한 읽기 전용 결과다. "
          "이를 변경하지 말고 설명만 작성하라.\n"
        + json.dumps(payload, ensure_ascii=False, default=str)
    )
    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        temperature=0.2,
        google_api_key=os.environ["GOOGLE_API_KEY"],
    )
    response = llm.invoke(prompt)
    content = response.content
    if isinstance(content, str):
        return content.strip() or None
    if isinstance(content, list):
        texts = [
            str(item.get("text", ""))
            for item in content
            if isinstance(item, dict) and item.get("text")
        ]
        return "\n".join(texts).strip() or None
    return str(content).strip() or None
