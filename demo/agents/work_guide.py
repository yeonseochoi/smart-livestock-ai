"""D 작업 가이드 에이전트.

기존 진입점 ``run(farm_id, work_type, rag_index)``은 유지한다. 새 핵심 함수
``plan_work``는 provider의 도구 결과만 사용해 6시간 추천/회피 창을 확정한다.
LLM은 이 결과를 바꾸지 않고 별도 모듈에서 설명만 작성한다.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from config import PROV
from constants import TIME_WEIGHTS

from agents.contracts import GuideCard, WorkWindow
from agents.provider import DecisionProvider
from agents.scoring_policy import WORK_WEIGHT, storage_factor
from agents.tools_schema import AgentTools, LocalTools


_GRADE_ORDER = {"낮음": 0, "주의": 1, "위험": 2}


def _build_windows(
    calendar_items: list[dict[str, Any]], work_type: str, storage_days: int | None
) -> list[WorkWindow]:
    """3시간 블록 두 개를 묶어 연속 6시간 창을 만든다.

    D+4~7 일 단위 값은 시간 추천 후보에서 제외한다. 이는 중기예보에 없는 시간
    정밀도를 화면에서 만들어내지 않기 위한 의도적인 제한이다.
    """

    blocks = [item for item in calendar_items
              if item.get("resolution") == "3h" and item.get("start")]
    blocks.sort(key=lambda item: item["start"])
    first_weight = sum(TIME_WEIGHTS[:3])
    second_weight = sum(TIME_WEIGHTS[3:])
    work_factor = WORK_WEIGHT.get(work_type, 1.0)
    storage_weight = storage_factor(storage_days)
    windows: list[WorkWindow] = []

    for current, following in zip(blocks, blocks[1:]):
        start = datetime.fromisoformat(current["start"])
        next_start = datetime.fromisoformat(following["start"])
        if next_start - start != timedelta(hours=3):
            continue
        risk = (float(current["risk_score"]) * first_weight
                + float(following["risk_score"]) * second_weight)
        component_grades = (current.get("risk_grade", "낮음"),
                            following.get("risk_grade", "낮음"))
        grade = max(component_grades, key=lambda item: _GRADE_ORDER.get(item, 0))
        reasons = [
            "연속 6시간 전체에 3시간 단위 예측값이 존재",
            f"B 민원 위험지수의 6시간 가중평균 {risk:.3f}",
            f"작업유형 가중치 {work_factor:.2f} 적용 [C]",
        ]
        if storage_days is not None:
            reasons.append(
                f"저장 {storage_days}일, 저장 가중치 {storage_weight:.2f} 적용 [C]"
            )
        windows.append(WorkWindow(
            start=start.isoformat(), end=(start + timedelta(hours=6)).isoformat(),
            window_risk=round(risk, 4),
            recommendation_score=round(risk * work_factor * storage_weight, 4),
            grade=grade, reasons=tuple(reasons),
        ))
    return windows


def plan_work(
    provider: DecisionProvider,
    farm_id: str,
    work_type: str,
    *,
    days: int = 3,
) -> GuideCard:
    """도구를 수집해 추천 Top 3와 회피 Top 3를 결정론적으로 반환한다."""

    tools = AgentTools(provider)
    calendar = tools.get_risk_calendar(farm_id, days, work_type)
    storage = tools.get_storage_days(farm_id)
    rag = tools.search_rag(f"{work_type} 작업 전후 관리 기준", work_type)
    if calendar.get("status") != "ok":
        raise RuntimeError(calendar.get("error") or "위험 캘린더를 사용할 수 없습니다")

    storage_days = None
    if storage.get("status") == "ok":
        storage_days = (storage.get("data") or {}).get("days")
    windows = _build_windows((calendar.get("data") or {}).get("items", []),
                             work_type, storage_days)
    if len(windows) < 6:
        raise RuntimeError("추천 Top 3/회피 Top 3를 만들 6시간 창이 부족합니다")

    recommended = tuple(sorted(
        windows, key=lambda item: (item.recommendation_score, item.start)
    )[:3])
    avoid = tuple(sorted(
        windows, key=lambda item: (item.recommendation_score, item.start),
        reverse=True,
    )[:3])

    rag_data = rag.get("data") or {}
    evidence = tuple(rag_data.get("results") or [])
    sources = {
        "risk_calendar": calendar.get("source"),
        "storage": storage.get("source"),
        "rag": rag.get("source"),
    }
    fixture_names = [name for name, source in sources.items()
                     if (source or {}).get("state") == "fixture"]
    unavailable_names = [name for name, source in sources.items()
                         if not source or source.get("state") == "unavailable"]
    assumptions = [
        "작업유형 가중치와 저장 14일 이후 1.5배는 팀 확정 전 잠정값 [C]",
        "추천 순위는 B 위험값과 기존 S5 가중식으로 코드가 계산",
    ]
    limitations = [
        "민원 발생 또는 감소를 보장하지 않고 상대 위험 회피만 지원",
        "플룸은 추천 점수와 등급에 반영하지 않음",
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
        farm_id=farm_id, work_type=work_type,
        recommended=recommended, avoid=avoid,
        before_actions=(
            "저감시설과 세척 설비의 작동 상태를 확인합니다.",
            "선택 시간의 최신 예보·데이터 갱신 상태를 다시 확인합니다.",
            "주민 알림은 문구와 대상을 확인한 뒤 승인합니다.",
        ),
        after_actions=(
            "작업 장소와 이동 경로를 세척합니다.",
            "냄새·민원 발생 여부와 작업 결과를 기록합니다.",
            "문제가 있으면 원인을 남겨 다음 계획 검토에 활용합니다.",
        ),
        evidence=evidence, assumptions=tuple(assumptions),
        limitations=tuple(limitations), source_statuses=sources,
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
    """기존 콘솔 데모와 팀원이 읽을 수 있는 문자열 표현."""

    data = guide.to_dict() if isinstance(guide, GuideCard) else guide
    lines = ["[추천 창]"]
    for index, item in enumerate(data["recommended"], 1):
        lines.append(f"  {index}. {_pretty_window(item)} score {item['recommendation_score']}")
    lines.append("[회피 창]")
    for index, item in enumerate(data["avoid"], 1):
        lines.append(f"  {index}. {_pretty_window(item)} score {item['recommendation_score']}")
    lines.append("[작업 전·후 조치]")
    lines.extend(f"  - 작업 전: {item}" for item in data["before_actions"])
    lines.extend(f"  - 작업 후: {item}" for item in data["after_actions"])
    lines.append("[판단 근거]")
    if data["evidence"]:
        for item in data["evidence"][:3]:
            page = f" p.{item['page']}" if item.get("page") else " (쪽수 미확인)"
            lines.append(f"  - {item.get('doc')} {item.get('unit')}{page}")
    else:
        lines.append("  - 법령·매뉴얼 검색 결과 없음: 공식 원문 확인 필요")
    lines.extend(f"  - 한계: {item}" for item in data["limitations"])
    return "\n".join(lines)


def run(farm_id: str, work_type: str, rag_index: Any = None) -> str:
    """기존 ``demo.py`` 호환 진입점."""

    tools = LocalTools(rag_index)
    try:
        guide = plan_work(tools.provider, farm_id, work_type)
    except RuntimeError as exc:
        return str(exc)
    PROV.log("S7 작업가이드", "agents/work_guide.py 결정론적 오케스트레이션",
             real=False, note="LLM 없이도 동일 추천 결과")
    return format_guide(guide)


def _pretty_window(window: dict[str, Any]) -> str:
    start = datetime.fromisoformat(window["start"])
    end = datetime.fromisoformat(window["end"])
    return f"{start:%m월 %d일 %H시}~{end:%H시}({window['grade']})"
