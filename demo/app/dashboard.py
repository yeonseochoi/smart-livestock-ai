"""D 파트 Streamlit 시연 화면.

실행: ``cd demo; streamlit run app/dashboard.py``
추천 계산·RAG 결합·알림 초안은 모두 ``agents/``가 담당한다.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


DEMO_DIR = Path(__file__).resolve().parents[1]
if str(DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(DEMO_DIR))

from agents.notify_draft import approve_for_demo, create_draft
from agents.openai_explainer import compose, is_configured
from agents.provider import create_provider
from agents.work_guide import plan_work, rule_based_summary


WORK_TYPES = ("분뇨제거", "청소", "환기점검", "저감시설점검", "액비살포")


def _safe(call, label: str) -> dict[str, Any]:
    try:
        return call()
    except Exception as exc:
        return {"status": "unavailable", "data": None, "source": None,
                "error": f"{label}: {type(exc).__name__}: {exc}"}


def _has_coordinate(row: dict[str, Any]) -> bool:
    return row.get("latitude") is not None and row.get("longitude") is not None


def _source_label(response: dict[str, Any]) -> str:
    source = response.get("source") or {}
    state = source.get("state", "unavailable").upper()
    return f"{state} · {source.get('name', '출처 없음')}"


def _fingerprint(
    status: dict[str, Any], calendar: dict[str, Any], work_type: str, storage_days: int
) -> str:
    status_data = status.get("data") or {}
    calendar_source = calendar.get("source") or {}
    return json.dumps({
        "snapshot_id": status_data.get("snapshot_id"),
        "calendar_version": calendar_source.get("version"),
        "work_type": work_type, "storage_days": storage_days,
    }, ensure_ascii=False, sort_keys=True)


def _reset_decision_state() -> None:
    for key in (
        "d_guide", "d_summary", "d_draft", "d_approval", "d_window",
        "d_selected_window",
    ):
        st.session_state.pop(key, None)


def _render_sensor_map(snapshot: dict[str, Any], use_basemap: bool) -> None:
    if snapshot.get("status") != "ok":
        st.info(snapshot.get("error") or "측정소 자료를 사용할 수 없습니다.")
        return
    stations = (snapshot.get("data") or {}).get("stations") or []
    if not stations:
        st.info("선택 시각에 표시할 측정소 관측이 없습니다.")
        return
    valid = [row for row in stations if _has_coordinate(row)]
    if valid:
        try:
            import pydeck as pdk

            layer = pdk.Layer(
                "ScatterplotLayer", valid, get_position="[longitude, latitude]",
                get_fill_color="[34, 158, 217, 190]", get_radius=250,
                pickable=True,
            )
            view = pdk.ViewState(
                latitude=sum(row["latitude"] for row in valid) / len(valid),
                longitude=sum(row["longitude"] for row in valid) / len(valid),
                zoom=11,
            )
            style = "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json" if use_basemap else None
            st.pydeck_chart(pdk.Deck(
                map_style=style, initial_view_state=view, layers=[layer],
                tooltip={"text": "{station_name}\n복합악취: {complex_odor_value}"},
            ))
        except ImportError:
            st.warning("pydeck 미설치로 지도 대신 표를 표시합니다.")
    else:
        st.warning("좌표가 있는 측정소가 없어 지도는 생략합니다.")

    columns = ["station_name", "observed_at", "h2s_ppm", "nh3_ppm", "tvoc_ppm",
               "complex_odor_value", "complex_odor_unit", "temperature_c",
               "humidity_pct", "wind_direction_text", "wind_speed_ms", "record_qc"]
    st.dataframe(pd.DataFrame(stations).reindex(columns=columns), width="stretch")
    st.caption("복합악취 값의 의미·단위는 공식 코드북 수령 후 확정합니다.")


def _calendar_frame(response: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for item in (response.get("data") or {}).get("items", []):
        rows.append({
            "날짜": item.get("date"),
            "시간": "일 단위" if item.get("block") is None else f"{int(item['block']) * 3:02d}시",
            "해상도": item.get("resolution"), "민원 위험지수": item.get("risk_score"),
            "등급": item.get("risk_grade"), "예측구간": item.get("horizon"),
            "유효시각": item.get("valid_at"),
        })
    return pd.DataFrame(rows)


def _window_frame(items: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for item in items:
        rows.append({
            "시작": item["start"], "종료": item["end"], "등급": item["grade"],
            "6시간 위험": item["window_risk"], "추천 비교점수": item["recommendation_score"],
        })
    return pd.DataFrame(rows)


def main() -> None:
    st.set_page_config(page_title="익산 민원 위험 작업 도우미", layout="wide")
    st.title("익산 민원 위험 작업 도우미 · D 에이전트 초안")
    st.warning("정보공개청구 자료 반영 전 구조 검증용 프로토타입입니다.")

    with st.sidebar:
        st.header("시연 설정")
        work_type = st.selectbox("작업 유형", WORK_TYPES)
        storage_days = st.slider("분뇨 저장 경과일", 0, 30, 12)
        use_basemap = st.checkbox("온라인 배경지도 사용", value=False)
        use_openai = st.checkbox(
            "OpenAI 설명 사용", value=False, disabled=not is_configured(),
            help="OPENAI_API_KEY와 OPENAI_MODEL이 모두 있을 때만 선택할 수 있습니다.",
        )

    try:
        provider = create_provider(storage_days=storage_days)
    except Exception as exc:
        st.error(f"provider 생성 실패: {type(exc).__name__}: {exc}")
        return
    status = _safe(provider.get_system_status, "상태 조회 실패")
    status_data = status.get("data") or {}
    farm_id = status_data.get("default_farm_id")
    if not farm_id:
        st.error("provider 상태에 default_farm_id가 없습니다.")
        return

    farm = _safe(lambda: provider.get_farm_config(farm_id), "농가 조회 실패")
    calendar = _safe(
        lambda: provider.get_risk_calendar(farm_id, 7, work_type),
        "위험 캘린더 조회 실패",
    )
    current_fp = _fingerprint(status, calendar, work_type, storage_days)
    if st.session_state.get("d_fingerprint") != current_fp:
        _reset_decision_state()
        st.session_state["d_fingerprint"] = current_fp

    mode = status_data.get("mode", "unknown").upper()
    cols = st.columns(4)
    cols[0].metric("데이터 모드", mode)
    cols[1].metric("공식 자료", status_data.get("official_dataset", "unknown"))
    cols[2].metric("농가", (farm.get("data") or {}).get("name", farm_id))
    cols[3].metric("위험자료", _source_label(calendar).split(" · ")[0])

    map_tab, plan_tab, evidence_tab, limits_tab = st.tabs(
        ["① 측정소 지도", "② 캘린더·작업계획", "③ 근거·알림", "④ 데이터·한계"]
    )
    with map_tab:
        st.subheader("현재·과거 측정소 관측")
        snapshot = _safe(provider.get_sensor_snapshot, "측정소 조회 실패")
        st.caption(_source_label(snapshot))
        _render_sensor_map(snapshot, use_basemap)
        st.info("이 지도는 관측 화면이며 미래 민원 위험 예측과 직접 환산하지 않습니다.")

    with plan_tab:
        st.subheader("7일 민원 위험 캘린더")
        st.caption(_source_label(calendar))
        if calendar.get("status") == "ok":
            st.dataframe(_calendar_frame(calendar), width="stretch")
        else:
            st.error(calendar.get("error") or "캘린더를 사용할 수 없습니다.")
        if st.button("6시간 작업 계획 만들기", type="primary"):
            try:
                guide_obj = plan_work(provider, farm_id, work_type, days=3)
                guide = guide_obj.to_dict()
                st.session_state["d_guide"] = guide
                st.session_state["d_summary"] = rule_based_summary(guide)
                st.session_state.pop("d_draft", None)
                st.session_state.pop("d_approval", None)
                if use_openai:
                    try:
                        generated = compose(provider, guide_obj)
                        if generated:
                            st.session_state["d_summary"] = generated
                    except Exception as exc:
                        st.warning(f"OpenAI 설명 실패: 규칙 기반 설명을 유지합니다. ({exc})")
            except Exception as exc:
                st.error(f"작업 계획 생성 실패: {exc}")

        guide = st.session_state.get("d_guide")
        if guide:
            st.markdown("#### 추천 Top 3")
            st.dataframe(_window_frame(guide["recommended"]), width="stretch")
            st.markdown("#### 회피 Top 3")
            st.dataframe(_window_frame(guide["avoid"]), width="stretch")
            st.info(st.session_state["d_summary"])

    with evidence_tab:
        guide = st.session_state.get("d_guide")
        if not guide:
            st.info("먼저 작업 계획을 만드세요.")
        else:
            st.subheader("근거 카드")
            if guide["evidence"]:
                for item in guide["evidence"]:
                    st.markdown(f"**{item.get('doc')} · {item.get('unit')}**")
                    st.caption(f"쪽수: {item.get('page') or '미확인'} · 검색점수는 신뢰확률 아님")
                    st.write(item.get("snippet", ""))
            else:
                st.warning("검색 근거가 없습니다. 공식 원문을 확인하세요.")

            options = guide["recommended"]
            labels = [f"{item['start']} ~ {item['end']} ({item['grade']})" for item in options]
            selected_index = st.selectbox(
                "확정할 작업 창", range(len(options)), format_func=lambda i: labels[i],
                key="d_window",
            )
            selected = options[selected_index]
            selected_token = f"{selected['start']}|{selected['end']}"
            if st.session_state.get("d_selected_window") != selected_token:
                st.session_state["d_selected_window"] = selected_token
                st.session_state.pop("d_draft", None)
                st.session_state.pop("d_approval", None)
            if st.button("알림 초안 만들기"):
                st.session_state["d_draft"] = create_draft(
                    provider, guide["farm_id"], guide["work_type"], selected
                ).to_dict()
                st.session_state.pop("d_approval", None)

            draft = st.session_state.get("d_draft")
            if draft:
                edited = st.text_area("주민 알림 문구", draft["message"], height=150)
                st.caption(
                    f"대상 후보 {draft['audience_count']} · 가상 여부 {draft['audience_is_mock']} · "
                    f"플룸 {draft['plume_status']} (등급 미반영)"
                )
                if st.button("농장주 승인 기록"):
                    st.session_state["d_approval"] = approve_for_demo(draft, message=edited)
                if st.session_state.get("d_approval"):
                    st.success("시연 세션에 승인만 기록했습니다. 실제 발송은 하지 않았습니다.")

    with limits_tab:
        st.subheader("데이터 상태와 교체 원칙")
        st.json(status)
        st.markdown(
            "- 측정소 관측과 미래 민원 위험은 서로 다른 데이터 흐름입니다.\n"
            "- D는 A 모델 파일이나 B DB 내부 계산을 복제하지 않고 provider 결과만 사용합니다.\n"
            "- 실제 자료 도착 후 센서 adapter와 A/B 모델을 검증한 뒤 provider를 교체합니다.\n"
            "- 플룸과 알림 대상은 검증 전 참고이며 민원 위험 등급에 반영하지 않습니다."
        )


if __name__ == "__main__":
    main()
