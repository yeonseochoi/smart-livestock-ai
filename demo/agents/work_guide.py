"""S7 — 작업 가이드 에이전트.

ANTHROPIC_API_KEY 가 있으면 Claude tool use, 없으면 같은 도구를 규칙 기반으로
호출하는 오프라인 폴백. 출력 형식은 계획서 고정:
  추천 창 / 대안 창 / 작업 전·후 조치 / 판단 근거(문헌 + "익산 6년 실측" 인용)

테스트 케이스 ①: 경과일 12일 + 후반부 위험 예보 → 더 이른 저위험 창으로
앞당기고 근거 2개 이상 제시.
"""
from __future__ import annotations

import json
import os

from config import PROV
from agents.tools_schema import LocalTools, TOOLS

SYSTEM = (
    "너는 양돈농가 작업 가이드다. 반드시 도구를 호출해 근거를 수집하고 "
    "출력은 '추천 창/대안 창/작업 전·후 조치/판단 근거' 4개 절로 고정한다. "
    "판단 근거에는 문헌·법령과 '익산 6년 실측' 통계를 함께 인용한다."
)

# .env 또는 환경변수(ANTHROPIC_MODEL_NAME)로 override 가능.
# 계정에서 실제 사용 가능한 모델명인지 반드시 확인할 것
# (https://docs.claude.com/en/docs/about-claude/models) — 잘못된 모델명은
# 다른 API(제미나이)에서 겪었던 것과 동일하게 404 로 조용히 실패한다.
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL_NAME", "claude-sonnet-4-5-20250929")
MAX_TOOL_TURNS = 6  # 무한루프 방지 — 이 안에 end_turn 못 내면 오프라인 폴백


def _offline_guide(tools: LocalTools, farm_id: str, work_type: str) -> str:
    cal = tools.get_risk_calendar(farm_id, days=3, work_type=work_type)
    storage = tools.get_storage_days(farm_id)
    rag = tools.search_rag(f"{work_type} 관리 기준", query_type=work_type
                           if work_type in ("분뇨제거", "청소", "환기점검", "저감시설점검")
                           else None)
    if not cal:
        return "risk_calendar 가 비어 있습니다. run_serve.py 를 먼저 실행하세요."

    ranked = sorted(cal, key=lambda c: c["final"])  # 동률이면 이른 시각(정렬 안정성)
    best, alt = ranked[0], (ranked[1] if len(ranked) > 1 else ranked[0])

    # 플룸은 미검증 모델(S8-4) — 등급·순위에 반영하지 않고 참고로만 표시
    plume_line = None
    try:
        from datetime import datetime as _dt
        when = _dt.strptime(best["t"], "%Y-%m-%d %H시").strftime("%Y-%m-%d %H:%M")
        plume = tools.get_plume_assessment(farm_id, when)
        if "n_exposed" in plume:
            plume_line = (f"  - 참고: 추천 창 풍하측 민가 {plume['n_exposed']}동 "
                          f"(플룸 모델 — 미검증, 등급 미반영)")
    except Exception:
        pass

    src = (f"{rag['results'][0]['doc']} {rag['results'][0]['unit']}"
           if rag.get("results") else "법령·매뉴얼 검색 결과 없음")

    lines = [
        f"[추천 창] {best['t']} (final {best['final']}, 등급 {best['grade']})",
        f"[대안 창] {alt['t']} (final {alt['final']}, 등급 {alt['grade']})",
        "[작업 전·후 조치]",
        "  - 작업 전: 인근 주민 알림 초안 승인, 저감시설 약액 확인",
        "  - 작업 후: 살포지 즉시 경운(액비의 경우 배출 38% 감소), 시설 세척",
        "[판단 근거]",
        f"  - 익산 6년 실측(2020~2026): 야간·무풍·고습 블록의 민원율이 그 외 대비 유의하게 높음",
        f"  - {src}",
    ]
    if plume_line:
        lines.append(plume_line)
    if storage.get("days") is not None:
        if storage["over_2weeks"]:
            lines.append(f"  - 분뇨 저장 {storage['days']}일 경과 — 2주 임계 초과, 즉시성 가중(x1.5)")
        else:
            lines.append(
                f"  - 분뇨 저장 {storage['days']}일 — 임계까지 {storage['days_until_threshold']}일. "
                f"위험 예보 전 조기 작업 권고(앞당김)")
    return "\n".join(lines)


def _call_tool(tools: LocalTools, name: str, tool_input: dict) -> dict:
    """LocalTools 메서드 이름으로 디스패치. 도구 실행이 실패해도 예외를 올리지
    않고 {"error": ...} 로 돌려준다 — LLM 이 실패를 보고 다른 도구로 재시도하거나
    설명에 반영할 수 있게 하기 위해서다."""
    method = {
        "get_risk_calendar": tools.get_risk_calendar,
        "get_forecast": tools.get_forecast,
        "get_storage_days": tools.get_storage_days,
        "search_rag": tools.search_rag,
        "get_farm_config": tools.get_farm_config,
        "get_plume_assessment": tools.get_plume_assessment,
    }.get(name)
    if method is None:
        return {"error": f"알 수 없는 도구: {name}"}
    try:
        return method(**tool_input)
    except Exception as exc:
        return {"error": str(exc)}


def _run_api_guide(tools: LocalTools, farm_id: str, work_type: str) -> str | None:
    """Claude API tool-use 루프. TOOLS 스키마 그대로 사용하고, 도구 호출은
    LocalTools 로 위임한다(오프라인 폴백과 동일 구현 공유).

    실패/미완료 시 None 을 돌려준다 — 호출부(run())가 오프라인으로 폴백한다.
    """
    import anthropic

    client = anthropic.Anthropic()
    user_msg = (
        f"farm_id={farm_id!r}, work_type={work_type!r} 에 대한 작업 가이드를 "
        "작성해줘. get_risk_calendar 로 위험도 창부터 확인하고, get_storage_days"
        "(분뇨 저장 경과일), search_rag(법령·매뉴얼 근거), 필요하면 "
        "get_plume_assessment(참고용, 등급에는 반영 금지)까지 도구를 호출해서 "
        "근거를 모은 다음 답해."
    )
    messages = [{"role": "user", "content": user_msg}]

    for _ in range(MAX_TOOL_TURNS):
        resp = client.messages.create(
            model=ANTHROPIC_MODEL, max_tokens=1024,
            system=SYSTEM, tools=TOOLS, messages=messages,
        )
        if resp.stop_reason != "tool_use":
            text = "".join(b.text for b in resp.content if b.type == "text").strip()
            return text or None

        messages.append({"role": "assistant", "content": resp.content})
        tool_results = []
        for block in resp.content:
            if block.type != "tool_use":
                continue
            result = _call_tool(tools, block.name, block.input)
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": json.dumps(result, ensure_ascii=False, default=str),
            })
        messages.append({"role": "user", "content": tool_results})

    return None  # MAX_TOOL_TURNS 안에 end_turn 을 못 받음 — 폴백


def run(farm_id: str, work_type: str, rag_index=None) -> str:
    tools = LocalTools(rag_index)

    if not os.environ.get("ANTHROPIC_API_KEY"):
        PROV.log("S7 작업가이드", "오프라인 규칙 폴백", real=False,
                 note="ANTHROPIC_API_KEY 미설정 — 도구 6종은 동일 사용")
        return _offline_guide(tools, farm_id, work_type)

    try:
        import anthropic  # noqa: F401
    except ImportError:
        PROV.log("S7 작업가이드", "anthropic 패키지 미설치 — 오프라인 폴백", real=False)
        return _offline_guide(tools, farm_id, work_type)

    PROV.log("S7 작업가이드 LLM", f"Claude API tool use ({ANTHROPIC_MODEL})", real=True)
    try:
        result = _run_api_guide(tools, farm_id, work_type)
    except Exception as exc:
        PROV.log("S7 작업가이드", "API 호출 예외 — 오프라인 폴백", real=False, note=str(exc))
        return _offline_guide(tools, farm_id, work_type)

    if not result:
        PROV.log("S7 작업가이드", "API 루프 빈 응답 — 오프라인 폴백", real=False,
                 note=f"{MAX_TOOL_TURNS}턴 내 종료 실패 또는 빈 텍스트")
        return _offline_guide(tools, farm_id, work_type)

    return result
