"""결정론적 작업 계획과 주민 알림 초안 서비스.

추천 시간과 점수는 코드가 계산하고, LLM은 이 결과를 바꾸지 않은 채 설명만 한다.
현재 가중치는 fixture 구조 검증용이며 B의 공개 추천 API가 생기면 교체한다.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from .backend import DecisionBackend
from .contracts import GuideCard, NotificationDraft, WorkWindow


# legacy/constants.py에서 승계한 0~5시간 감쇠 가중치. 연결 모드에서는 B/scoring이
# 계산한 값을 받아 D가 재계산하지 않는 것이 최종 목표다.
TIME_WEIGHTS = (0.30, 0.22, 0.17, 0.13, 0.10, 0.08)

# [C] 회의용 잠정값. 공식 데이터와 팀 합의 전에는 화면에 가정값으로 표시한다.
WORK_WEIGHT = {
    "분뇨제거": 1.3,
    "청소": 1.1,
    "환기점검": 1.0,
    "저감시설점검": 1.0,
    "액비살포": 1.5,
}


def _window_grade(window_risk: float) -> str:
    if window_risk < 0.18:
        return "낮음"
    if window_risk < 0.32:
        return "주의"
    return "위험"


def _build_windows(
    calendar_items: list[dict[str, Any]], work_type: str, storage_days: int | None
) -> list[WorkWindow]:
    """3시간 블록만 사용해 6시간 창을 만든다.

    D+4~7 일 단위 값은 시간대 추천 후보에서 제외해 거짓 정밀도를 만들지 않는다.
    """

    block_items = [
        item
        for item in calendar_items
        if item.get("resolution") == "3h" and item.get("start")
    ]
    block_items.sort(key=lambda item: item["start"])
    first_block_weight = sum(TIME_WEIGHTS[:3])
    second_block_weight = sum(TIME_WEIGHTS[3:])
    work_factor = WORK_WEIGHT.get(work_type, 1.0)
    storage_factor = 1.5 if storage_days is not None and storage_days >= 14 else 1.0
    windows: list[WorkWindow] = []

    for current, following in zip(block_items, block_items[1:]):
        start = datetime.fromisoformat(current["start"])
        next_start = datetime.fromisoformat(following["start"])
        if next_start - start != timedelta(hours=3):
            continue
        window_risk = (
            float(current["risk_score"]) * first_block_weight
            + float(following["risk_score"]) * second_block_weight
        )
        recommendation_score = window_risk * work_factor * storage_factor
        reasons = [
            "6시간 작업 창 전체에 3시간 블록 예보가 존재",
            f"시연용 상대 민원 위험지수 {window_risk:.3f}",
        ]
        if storage_days is not None:
            reasons.append(f"분뇨 저장 경과 {storage_days}일 반영")
        windows.append(
            WorkWindow(
                start=start.isoformat(),
                end=(start + timedelta(hours=6)).isoformat(),
                window_risk=round(window_risk, 4),
                recommendation_score=round(recommendation_score, 4),
                grade=_window_grade(window_risk),
                reasons=tuple(reasons),
            )
        )
    return windows


def plan_work(
    backend: DecisionBackend,
    farm_id: str,
    work_type: str,
    days: int = 3,
) -> GuideCard:
    """도구 데이터를 수집하고 추천 Top 3/회피 Top 3를 만든다."""

    calendar = backend.get_risk_calendar(farm_id, days, work_type)
    storage = backend.get_storage_days(farm_id)
    rag = backend.search_rag(f"{work_type} 작업 전후 관리 기준", work_type)
    if calendar.get("status") != "ok":
        raise RuntimeError(calendar.get("error") or "위험 캘린더를 사용할 수 없습니다")
    storage_days = (
        storage.get("data", {}).get("days") if storage.get("status") == "ok" else None
    )
    windows = _build_windows(calendar["data"]["items"], work_type, storage_days)
    if len(windows) < 6:
        raise RuntimeError("추천 Top 3/회피 Top 3를 만들 6시간 창이 부족합니다")

    recommended = tuple(sorted(windows, key=lambda item: item.recommendation_score)[:3])
    avoid = tuple(
        sorted(windows, key=lambda item: item.recommendation_score, reverse=True)[:3]
    )
    rag_data = rag.get("data") or {}
    evidence = tuple(rag_data.get("results") or [])
    source_statuses = {
        "risk_calendar": calendar.get("source"),
        "storage": storage.get("source"),
        "rag": rag.get("source"),
    }
    fixture_sources = [
        name
        for name, source in source_statuses.items()
        if (source or {}).get("state") == "fixture"
    ]
    stale_sources = [
        name
        for name, source in source_statuses.items()
        if (source or {}).get("state") == "stale"
    ]
    unavailable_sources = [
        name
        for name, source in source_statuses.items()
        if not source or source.get("state") == "unavailable"
    ]
    assumptions = ["작업유형 가중치와 저장 14일 이후 1.5배는 잠정 가정값 [C]"]
    limitations = [
        "민원 발생 또는 감소를 보장하지 않고 상대 위험 회피만 지원",
        "플룸 정보는 추천 점수와 등급에 반영하지 않음",
    ]
    if fixture_sources:
        assumptions.append(
            "FIXTURE provider 사용: " + ", ".join(sorted(fixture_sources))
        )
        limitations.insert(0, "공식 정보공개 자료 반영 전 구조 검증용 결과 포함")
    elif stale_sources:
        limitations.insert(
            0, "갱신 지연 provider 포함: " + ", ".join(sorted(stale_sources))
        )
    elif unavailable_sources:
        limitations.insert(
            0, "사용 불가 provider 포함: " + ", ".join(sorted(unavailable_sources))
        )
    else:
        assumptions.append("각 provider가 반환한 출처·버전·기준시각을 사용")

    return GuideCard(
        farm_id=farm_id,
        work_type=work_type,
        recommended=recommended,
        avoid=avoid,
        before_actions=(
            "저감시설과 세척 설비의 작동 상태를 확인합니다.",
            "선택 시간의 최신 예보·데이터 갱신 상태를 다시 확인합니다.",
            "주민 알림은 문구와 대상을 확인한 뒤 승인합니다.",
        ),
        after_actions=(
            "작업 장소와 이동 경로를 세척합니다.",
            "냄새·민원 발생 여부를 작업 기록에 남깁니다.",
            "문제가 있으면 다음 추천에 반영할 수 있도록 원인을 기록합니다.",
        ),
        evidence=evidence,
        assumptions=tuple(assumptions),
        limitations=tuple(limitations),
        source_statuses=source_statuses,
    )


def rule_based_summary(guide: GuideCard | dict[str, Any]) -> str:
    """OpenAI 키가 없을 때도 같은 구조로 시연하는 설명문."""

    data = guide.to_dict() if isinstance(guide, GuideCard) else guide
    best = data["recommended"][0]
    alt = data["recommended"][1]
    sources = data.get("source_statuses") or {}
    fixture_sources = [
        name
        for name, source in sources.items()
        if (source or {}).get("state") == "fixture"
    ]
    stale_sources = [
        name
        for name, source in sources.items()
        if (source or {}).get("state") == "stale"
    ]
    unavailable_sources = [
        name
        for name, source in sources.items()
        if not source or source.get("state") == "unavailable"
    ]
    if fixture_sources:
        source_note = (
            "FIXTURE provider(" + ", ".join(sorted(fixture_sources)) + ")가 포함된 "
            "구조 검증용 결과이므로"
        )
    elif stale_sources:
        source_note = (
            "갱신 지연 provider(" + ", ".join(sorted(stale_sources)) + ")가 있으므로"
        )
    elif unavailable_sources:
        source_note = (
            "사용 불가 provider(" + ", ".join(sorted(unavailable_sources)) + ")가 있으므로"
        )
    else:
        source_note = "연결된 provider의 기준시각을 사용했으며"
    return (
        f"추천 창은 {_pretty_window(best)}이며, 대안은 {_pretty_window(alt)}입니다. "
        f"두 창 모두 6시간 전체를 비교한 결과입니다. {source_note} 실제 작업 전 최신 "
        "예보와 농장 상황을 다시 확인해야 합니다."
    )


def create_notification_draft(
    backend: DecisionBackend,
    farm_id: str,
    work_type: str,
    selected_window: dict[str, Any],
) -> NotificationDraft:
    """외부 발송이나 DB 변경 없이 편집 가능한 알림 초안만 만든다."""

    plume = backend.get_plume_assessment(farm_id, selected_window["start"])
    plume_data = plume.get("data") or {}
    start = datetime.fromisoformat(selected_window["start"])
    end = datetime.fromisoformat(selected_window["end"])
    count = int(plume_data.get("n_exposed", 0))
    message = (
        f"[작업 사전 안내 초안] {start:%m월 %d일 %H시}부터 {end:%H시}까지 "
        f"선택한 농가에서 {work_type} 작업이 예정되어 있습니다. "
        "일시적으로 냄새가 느껴질 수 있어 사전에 안내드립니다. "
        "이 문구는 공모전 시연용 초안이며 실제 발송되지 않습니다."
    )
    return NotificationDraft(
        farm_id=farm_id,
        work_type=work_type,
        work_window=selected_window,
        audience_count=count,
        audience_is_mock=bool(plume_data.get("audience_is_mock", True)),
        plume_status=(
            "unverified" if plume.get("status") == "ok" else "unavailable"
        ),
        message=message,
    )


def _pretty_window(window: dict[str, Any]) -> str:
    start = datetime.fromisoformat(window["start"])
    end = datetime.fromisoformat(window["end"])
    return f"{start:%m월 %d일 %H시}~{end:%H시}({window['grade']})"
