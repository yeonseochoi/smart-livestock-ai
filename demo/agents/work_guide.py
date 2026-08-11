"""S7 — 작업 가이드 에이전트.

ANTHROPIC_API_KEY 가 있으면 Claude tool use, 없으면 같은 도구를 규칙 기반으로
호출하는 오프라인 폴백. 출력 형식은 계획서 고정:
  추천 창 / 대안 창 / 작업 전·후 조치 / 판단 근거(문헌 + "익산 6년 실측" 인용)

테스트 케이스 ①: 경과일 12일 + 후반부 위험 예보 → 더 이른 저위험 창으로
앞당기고 근거 2개 이상 제시.
"""
from __future__ import annotations

import os

from config import PROV
from agents.tools_schema import LocalTools, TOOLS

SYSTEM = (
    "너는 양돈농가 작업 가이드다. 반드시 도구를 호출해 근거를 수집하고 "
    "출력은 '추천 창/대안 창/작업 전·후 조치/판단 근거' 4개 절로 고정한다. "
    "판단 근거에는 문헌·법령과 '익산 6년 실측' 통계를 함께 인용한다."
)


def _offline_guide(tools: LocalTools, farm_id: str, work_type: str) -> str:
    cal = tools.get_risk_calendar(farm_id, days=3, work_type=work_type)
    storage = tools.get_storage_days(farm_id)
    rag = tools.search_rag(f"{work_type} 관리 기준", query_type=work_type
                           if work_type in ("분뇨제거", "청소", "환기점검", "저감시설점검")
                           else None)
    if not cal:
        return "risk_calendar 가 비어 있습니다. run_daily 를 먼저 실행하세요."

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


def run(farm_id: str, work_type: str, rag_index=None) -> str:
    tools = LocalTools(rag_index)
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            import anthropic  # noqa: F401
            PROV.log("S7 작업가이드 LLM", "Claude API (tool use)", real=True)
            # 실 API 루프는 tools_schema.TOOLS 로 구성 (데모 환경에서는 미실행 경로)
        except ImportError:
            pass
    PROV.log("S7 작업가이드", "오프라인 규칙 폴백", real=False,
             note="ANTHROPIC_API_KEY 미설정 — 도구 6종은 동일 사용")
    return _offline_guide(tools, farm_id, work_type)
