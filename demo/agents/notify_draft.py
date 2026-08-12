"""D 주민 알림 초안 에이전트.

이 모듈은 초안만 생성한다. 기존 구현처럼 승인 전에 receptor를 삭제·삽입하거나
notification_log를 기록하지 않는다. 프로토타입 승인은 Streamlit 세션 로그에만
남고 실제 발송은 하지 않는다.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from config import DEMO_FARM, PROV

from agents.contracts import NotificationDraft
from agents.provider import DecisionProvider
from agents.tools_schema import AgentTools


def create_draft(
    provider: DecisionProvider,
    farm_id: str,
    work_type: str,
    selected_window: dict[str, Any],
) -> NotificationDraft:
    """선택한 작업 창을 변경하지 않고 편집 가능한 안내문을 만든다."""

    tools = AgentTools(provider)
    plume = tools.get_plume_assessment(farm_id, selected_window["start"])
    plume_data = plume.get("data") or {}
    start = datetime.fromisoformat(selected_window["start"])
    end = datetime.fromisoformat(selected_window["end"])
    count = int(plume_data.get("n_exposed", 0))
    message = (
        f"[작업 사전 안내 초안] {start:%m월 %d일 %H시}부터 {end:%H시}까지 "
        f"인근 농가에서 {work_type} 작업이 예정되어 있습니다. "
        "일시적으로 냄새가 느껴질 수 있어 사전에 안내드립니다. "
        "이 문구는 공모전 시연용 초안이며 실제 발송되지 않습니다."
    )
    return NotificationDraft(
        farm_id=farm_id, work_type=work_type, work_window=selected_window,
        audience_count=count,
        audience_is_mock=bool(plume_data.get("audience_is_mock", True)),
        plume_status="unverified" if plume.get("status") == "ok" else "unavailable",
        message=message,
    )


def approve_for_demo(
    draft: NotificationDraft | dict[str, Any], *, message: str | None = None
) -> dict[str, Any]:
    """실제 발송 없이 승인된 시연 로그 형태만 반환한다."""

    data = draft.to_dict() if isinstance(draft, NotificationDraft) else dict(draft)
    if message is not None:
        data["message"] = message
    data["approved"] = True
    data["sent"] = False
    data["approved_at"] = datetime.now().isoformat(timespec="seconds")
    data["approval_scope"] = "streamlit_session_demo_only"
    return data


def run(work_type: str, when: datetime) -> dict[str, Any]:
    """기존 ``demo.py`` 호환 진입점. DB 변경이나 외부 발송은 하지 않는다."""

    from agents.legacy_provider import LegacyProvider

    selected = {
        "start": when.isoformat(),
        "end": (when + timedelta(hours=6)).isoformat(),
        "grade": "미확정",
        "recommendation_score": None,
    }
    draft = create_draft(
        LegacyProvider(), DEMO_FARM["farm_id"], work_type, selected
    ).to_dict()
    PROV.log("S7 주민 알림", "agents/notify_draft.py 초안 전용",
             real=False, note="승인 전 DB 기록 없음·실발송 없음")
    print(f"  알림 대상 후보 {draft['audience_count']}동 "
          f"(플룸 {draft['plume_status']} — 등급 미반영)")
    print("  편집 가능한 초안 생성 (승인 전 — 실발송 없음)")
    return draft
