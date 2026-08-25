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
- 아래 정확히 이 형태의 JSON 객체 하나만 출력한다. 코드블록이나 다른 텍스트 없이 JSON만 출력한다.
  {"summary": "<종합 설명, 문단 형태, 기존과 동일한 성격>",
   "evidence_plain": [["<rag_evidence[0] 요약 불릿1>", "<불릿2>", "<불릿3, 선택>"],
                       ["<rag_evidence[1] 요약 불릿1>", ...], ...]}
- evidence_plain 배열의 길이와 순서는 rag_evidence 배열과 정확히 같아야 한다. 원소 하나가
  근거 카드 하나(rag_evidence의 각 항목)에 대응한다.
- 각 근거 카드의 불릿은 2~3개로 쓴다. 법조문·전문용어를 그대로 옮기지 말고 농장주가 바로
  이해할 수 있게 풀어 쓴다. 불릿 하나는 30자 안팎의 짧은 문장으로, "무엇을 확인/조치해야
  하는지" 중심으로 쓴다. 원문을 그대로 옮기는 인용이 아니라 쉬운말 요약이어야 한다.
""".strip()


def is_configured() -> bool:
    return bool(os.environ.get("GOOGLE_API_KEY"))


def _parse_json_response(text: str) -> dict[str, Any] | None:
    """Gemini가 ```json 코드블록으로 감싸 보내는 경우까지 관대하게 처리한다."""
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = candidate.strip("`")
        if candidate.lower().startswith("json"):
            candidate = candidate[4:]
        candidate = candidate.strip()
    try:
        obj = json.loads(candidate)
    except (json.JSONDecodeError, TypeError):
        return None
    return obj if isinstance(obj, dict) else None


def compose(
    guide: GuideCard | dict[str, Any],
) -> tuple[str, list[list[str] | None]] | None:
    """Gemini가 종합 설명과 근거카드별 쉬운말 요약(불릿 목록)을 함께 반환한다.

    반환값은 (종합 설명, evidence와 같은 길이의 근거카드별 불릿 리스트)다.
    각 원소는 근거 카드 하나에 대응하는 2~3개짜리 문자열 리스트이거나, 그 카드에
    쉬운말이 없으면 None이다. JSON 파싱에 실패하면 근거카드별 요약 없이
    (전체 응답 텍스트, [None, None, ...])을 돌려준다 — 호출자가 최소한 종합
    설명은 쓸 수 있게 하기 위함이다.
    """

    if not is_configured():
        return None
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except ImportError as exc:
        raise RuntimeError("langchain-google-genai 패키지가 설치되지 않았습니다") from exc

    data = guide.to_dict() if isinstance(guide, GuideCard) else guide
    evidence = data.get("evidence") or []
    payload = {
        "guide": data,
        "rag_evidence": evidence,
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
    if isinstance(content, list):
        content = "\n".join(
            str(item.get("text", ""))
            for item in content
            if isinstance(item, dict) and item.get("text")
        )
    text = str(content or "").strip()
    if not text:
        return None

    parsed = _parse_json_response(text)
    if parsed is None:
        return (text, [None] * len(evidence))

    summary = str(parsed.get("summary") or "").strip() or text
    raw_plain = parsed.get("evidence_plain") or []
    evidence_plain: list[list[str] | None] = []
    for item in raw_plain:
        if isinstance(item, list):
            bullets = [str(b).strip() for b in item if str(b).strip()]
        elif item:
            # [C] 2026-08 이전 프롬프트 버전이 문자열 하나를 돌려줄 수도 있어
            # 하위호환으로 단일 불릿 리스트로 감싼다.
            bullets = [str(item).strip()]
        else:
            bullets = []
        evidence_plain.append(bullets or None)
    if len(evidence_plain) < len(evidence):
        evidence_plain += [None] * (len(evidence) - len(evidence_plain))
    else:
        evidence_plain = evidence_plain[: len(evidence)]
    return (summary, evidence_plain)
