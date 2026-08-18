"""D 작업 가이드 에이전트.

provider 도구 결과로 추천·회피 창을 결정론적으로 확정한다. Gemini는 선택적으로
그 결과를 설명할 뿐 시간, 점수와 등급을 만들거나 변경하지 않는다.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from config import PROV

from agents.contracts import GuideCard, WorkWindow
from agents.provider import DecisionProvider, create_provider
from agents.scoring_policy import TIME_WEIGHTS, WORK_WEIGHT, storage_factor
from agents.tools_schema import AgentTools


_GRADE_ORDER = {"낮음": 0, "주의": 1, "위험": 2}


def _build_windows(
    calendar_items: list[dict[str, Any]], work_type: str, storage_days: int | None
) -> list[WorkWindow]:
    """연속된 1시간 위험값 6개에 시간 가중치를 직접 적용한다.

    일 단위 중기예보는 시간 추천 후보에서 제외한다. 존재하지 않는 시간 정밀도를
    화면에서 만들어내지 않기 위한 제한이다.
    """

    hourly = [
        item for item in calendar_items
        if item.get("resolution") == "1h" and item.get("start")
    ]
    hourly.sort(key=lambda item: item["start"])
    work_factor = WORK_WEIGHT.get(work_type, 1.0)
    storage_weight = storage_factor(storage_days)
    windows: list[WorkWindow] = []

    for index in range(max(0, len(hourly) - len(TIME_WEIGHTS) + 1)):
        segment = hourly[index:index + len(TIME_WEIGHTS)]
        starts = [datetime.fromisoformat(item["start"]) for item in segment]
        if any(
            following - current != timedelta(hours=1)
            for current, following in zip(starts, starts[1:])
        ):
            continue
        risk = sum(
            float(item["risk_score"]) * weight
            for item, weight in zip(segment, TIME_WEIGHTS)
        )
        grade = max(
            (item.get("risk_grade", "낮음") for item in segment),
            key=lambda item: _GRADE_ORDER.get(item, 0),
        )
        reasons = [
            "연속 6시간 전체에 1시간 단위 예측값이 존재",
            f"B 민원 위험지수의 6시간 가중평균 {risk:.3f}",
            f"작업유형 가중치 {work_factor:.2f} 적용 [C]",
        ]
        if storage_days is not None:
            reasons.append(
                f"저장 {storage_days}일, 저장 가중치 {storage_weight:.2f} 적용 [C]"
            )
        start = starts[0]
        windows.append(WorkWindow(
            start=start.isoformat(),
            end=(start + timedelta(hours=6)).isoformat(),
            window_risk=round(risk, 4),
            recommendation_score=round(risk * work_factor * storage_weight, 4),
            grade=grade,
            reasons=tuple(reasons),
        ))
    return windows


def plan_work(
    provider: DecisionProvider,
    farm_id: str,
    work_type: str,
    *,
    days: int = 3,
) -> GuideCard:
    """내부 도구를 호출해 추천 Top 3와 회피 Top 3를 확정한다."""

    tools = AgentTools(provider)
    farm = tools.get_farm_config(farm_id)
    calendar = tools.get_risk_calendar(farm_id, days, work_type)
    storage = tools.get_storage_days(farm_id)
    rag = tools.search_rag(f"{work_type} 작업 전후 관리 기준", work_type)
    if calendar.get("status") != "ok":
        raise RuntimeError(calendar.get("error") or "위험 캘린더를 사용할 수 없습니다")

    storage_days = None
    if storage.get("status") == "ok":
        storage_days = (storage.get("data") or {}).get("days")
    windows = _build_windows(
        (calendar.get("data") or {}).get("items", []), work_type, storage_days
    )
    if len(windows) < 6:
        raise RuntimeError("추천 Top 3/회피 Top 3를 만들 1시간 위험 구간이 부족합니다")

    recommended = tuple(sorted(
        windows, key=lambda item: (item.recommendation_score, item.start)
    )[:3])
    avoid = tuple(sorted(
        windows,
        key=lambda item: (item.recommendation_score, item.start),
        reverse=True,
    )[:3])

    rag_data = rag.get("data") or {}
    evidence = tuple(rag_data.get("results") or [])
    sources = {
        "farm": farm.get("source"),
        "risk_calendar": calendar.get("source"),
        "storage": storage.get("source"),
        "rag": rag.get("source"),
    }
    fixture_names = [
        name for name, source in sources.items()
        if (source or {}).get("state") == "fixture"
    ]
    unavailable_names = [
        name for name, source in sources.items()
        if not source or source.get("state") == "unavailable"
    ]
    assumptions = [
        "작업유형 가중치와 저장 14일 이후 1.5배는 팀 확정 전 잠정값 [C]",
        "추천 순위는 B 위험값과 기존 S5 가중식으로 코드가 계산",
    ]
    limitations = [
        "민원 발생 또는 감소를 보장하지 않고 상대 위험 회피만 지원",
    ]
    if fixture_names:
        assumptions.append("FIXTURE 포함: " + ", ".join(sorted(fixture_names)))
        limitations.insert(0, "공식 정보공개 자료 반영 전 구조 검증용 결과 포함")
    if unavailable_names:
        limitations.append("사용 불가 provider: " + ", ".join(sorted(unavailable_names)))
    if rag.get("status") == "refused":
        limitations.append("개별 법률 판단 요청으로 C 검색이 거절됨")
    elif not evidence:
        limitations.append("C 근거 검색 결과가 없어 작업 전 공식 원문 확인 필요")

    return GuideCard(
        farm_id=farm_id,
        work_type=work_type,
        recommended=recommended,
        avoid=avoid,
        before_actions=(
            "저감시설과 세척 설비의 작동 상태를 확인합니다.",
            "선택 시간의 최신 예보·데이터 갱신 상태를 다시 확인합니다.",
        ),
        after_actions=(
            "작업 장소와 이동 경로를 세척합니다.",
            "냄새·민원 발생 여부와 작업 결과를 기록합니다.",
            "문제가 있으면 원인을 남겨 다음 계획 검토에 활용합니다.",
        ),
        evidence=evidence,
        assumptions=tuple(assumptions),
        limitations=tuple(limitations),
        source_statuses=sources,
    )


def rule_based_summary(guide: GuideCard | dict[str, Any]) -> str:
    data = guide.to_dict() if isinstance(guide, GuideCard) else guide
    best, alternative = data["recommended"][:2]
    return (
        f"추천 창은 {_pretty_window(best)}이며 대안은 "
        f"{_pretty_window(alternative)}입니다. 두 창 모두 6시간 전체를 비교한 "
        "상대 민원 위험 회피 결과입니다. 실제 작업 전 최신 예보와 농장 상황을 "
        "다시 확인해야 합니다."
    )


def format_guide(guide: GuideCard | dict[str, Any]) -> str:
    """확정된 계획을 LLM 없이도 읽을 수 있는 문자열로 반환한다."""

    data = guide.to_dict() if isinstance(guide, GuideCard) else guide
    lines = ["[추천 창]"]
    for index, item in enumerate(data["recommended"], 1):
        lines.append(
            f"  {index}. {_pretty_window(item)} score {item['recommendation_score']}"
        )
    lines.append("[회피 창]")
    for index, item in enumerate(data["avoid"], 1):
        lines.append(
            f"  {index}. {_pretty_window(item)} score {item['recommendation_score']}"
        )
    lines.append("[작업 전·후 조치]")
    lines.extend(f"  - 작업 전: {item}" for item in data["before_actions"])
    lines.extend(f"  - 작업 후: {item}" for item in data["after_actions"])
    lines.append("[판단 근거]")
    if data["evidence"]:
        for item in data["evidence"][:3]:
            page_value = item.get("page")
            page = f" p.{page_value}" if page_value is not None else " (쪽수 미확인)"
            source_file = item.get("source_file") or item.get("doc") or "출처 미확인"
            unit = item.get("unit") or "단원 미확인"
            lines.append(f"  - {source_file} · {unit}{page}")
    else:
        lines.append("  - 법령·매뉴얼 검색 결과 없음: 공식 원문 확인 필요")
    lines.extend(f"  - 한계: {item}" for item in data["limitations"])
    return "\n".join(lines)


def run(farm_id: str, work_type: str, rag_index: Any = None) -> str:
    """기존 콘솔 진입점. 확정 계획 뒤에만 선택적 Gemini 설명을 붙인다."""

    try:
        provider = create_provider(rag_index=rag_index)
        guide = plan_work(provider, farm_id, work_type)
    except (RuntimeError, ValueError) as exc:
        return str(exc)

    canonical = format_guide(guide)
    from agents.gemini_explainer import GEMINI_MODEL, compose, is_configured

    if is_configured():
        try:
            explanation = compose(guide)
        except Exception as exc:
            PROV.log(
                "S7 작업가이드",
                "Gemini API 호출 실패 — 확정 계획 유지",
                real=False,
                note=str(exc),
            )
        else:
            if explanation:
                PROV.log(
                    "S7 작업가이드 LLM",
                    f"Gemini 설명 생성 ({GEMINI_MODEL})",
                    real=True,
                )
                return canonical + "\n\n[AI 설명]\n" + explanation

    PROV.log(
        "S7 작업가이드",
        "agents/work_guide.py 결정론적 오케스트레이션",
        real=False,
        note="Gemini 없이도 동일 추천 결과",
    )
    return canonical


def _pretty_window(window: dict[str, Any]) -> str:
    start = datetime.fromisoformat(window["start"])
    end = datetime.fromisoformat(window["end"])
    return f"{start:%m월 %d일 %H시}~{end:%H시}({window['grade']})"
