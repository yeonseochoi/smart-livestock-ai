"""D 파트 공모전 시연용 Streamlit 앱.

실행:
    cd demo
    streamlit run app/dashboard.py
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import streamlit as st


# 기존 demo가 평면 import 구조여서, Streamlit 실행 위치와 무관하게 demo를 추가한다.
DEMO_DIR = Path(__file__).resolve().parents[1]
if str(DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(DEMO_DIR))

from app.backend_factory import create_backend  # noqa: E402
from app.guide_service import (  # noqa: E402
    WORK_WEIGHT,
    create_notification_draft,
    plan_work,
    rule_based_summary,
)
from app.openai_guide import compose_guide, is_configured  # noqa: E402


KST = timezone(timedelta(hours=9))

STATE_LABELS = {
    "fixture": "FIXTURE",
    "connected": "연결됨",
    "stale": "갱신 지연",
    "unavailable": "사용 불가",
    "waiting": "대기 중",
    "fixture_unverified": "FIXTURE·미검증",
    "draft_only": "초안만",
}


def _display_number(value: Any, unit: str = "") -> str:
    if value is None:
        return "미측정"
    return f"{value:g}{unit}"


def _state_label(state: Any) -> str:
    text = str(state or "unavailable")
    return STATE_LABELS.get(text, text)


def _source_state(response: dict[str, Any], fallback: str = "unavailable") -> str:
    return str((response.get("source") or {}).get("state") or fallback)


def _component_state(
    system_status: dict[str, Any], component: str, fallback: str = "unavailable"
) -> str:
    components = (system_status.get("data") or {}).get("components") or {}
    value = components.get(component, fallback)
    if isinstance(value, dict):
        value = value.get("state", fallback)
    return str(value or fallback)


def _safe_backend_call(
    label: str, call: Callable[[], dict[str, Any]]
) -> dict[str, Any]:
    """UI provider 오류를 unavailable 응답으로 바꿔 다른 탭을 유지한다."""

    try:
        result = call()
        if not isinstance(result, dict):
            raise TypeError("provider 응답이 dict가 아닙니다")
        return result
    except Exception as exc:
        return {
            "status": "unavailable",
            "data": None,
            "source": {
                "state": "unavailable",
                "name": label,
                "generated_at": None,
                "data_as_of": None,
                "version": None,
                "limitations": ("provider 호출 실패",),
            },
            "error": f"{type(exc).__name__}: {exc}",
        }


def _decision_snapshot_token(
    system_status: dict[str, Any], calendar: dict[str, Any]
) -> str:
    """추천을 무효화해야 하는 데이터 기준시점/버전의 안정된 토큰."""

    source_keys = ("state", "name", "data_as_of", "version")
    item_keys = (
        "date",
        "block",
        "start",
        "resolution",
        "risk_score",
        "risk_grade",
        "model_type",
        "model_version",
        "forecast_issued_at",
        "updated_at",
    )
    payload = {
        "system": {
            "data": {
                key: (system_status.get("data") or {}).get(key)
                for key in (
                    "mode",
                    "official_dataset",
                    "components",
                    "decision_snapshot_id",
                )
            },
            "source": {
                key: (system_status.get("source") or {}).get(key)
                for key in source_keys
            },
        },
        "calendar": {
            "status": calendar.get("status"),
            "source": {
                key: (calendar.get("source") or {}).get(key) for key in source_keys
            },
            "items": [
                {key: item.get(key) for key in item_keys}
                for item in ((calendar.get("data") or {}).get("items") or [])
            ],
        },
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _has_map_coordinate(item: dict[str, Any]) -> bool:
    latitude = item.get("latitude")
    longitude = item.get("longitude")
    if (
        isinstance(latitude, bool)
        or isinstance(longitude, bool)
        or not isinstance(latitude, (int, float))
        or not isinstance(longitude, (int, float))
    ):
        return False
    return (
        math.isfinite(latitude)
        and math.isfinite(longitude)
        and -90 <= latitude <= 90
        and -180 <= longitude <= 180
    )


def _render_sensor_map(snapshot: dict[str, Any], use_basemap: bool) -> None:
    stations = [dict(item) for item in snapshot["data"]["stations"]]
    facilities = [dict(item) for item in snapshot["data"].get("facilities", [])]
    if not stations:
        st.warning("표시할 측정소 관측이 없습니다.")
        return
    map_stations = [station for station in stations if _has_map_coordinate(station)]
    map_facilities = [item for item in facilities if _has_map_coordinate(item)]
    if not map_stations:
        st.warning(
            "좌표가 확인된 측정소가 없어 지도를 그리지 못했습니다. "
            "관측 상세와 품질 표는 아래에서 확인할 수 있습니다."
        )
        return
    omitted = len(stations) - len(map_stations)
    if omitted:
        st.warning(f"좌표가 없거나 유효하지 않은 측정소 {omitted}곳은 지도에서 제외했습니다.")
    for station in map_stations:
        odor_unit = station.get("complex_odor_unit")
        station["map_label"] = (
            "–"
            if station["complex_odor_value"] is None
            else f"{station['complex_odor_value']:g}"
        )
        station["h2s_display"] = _display_number(station["h2s_ppm"])
        station["nh3_display"] = _display_number(station["nh3_ppm"])
        station["tvoc_display"] = _display_number(station["tvoc_ppm"])
        station["odor_display"] = _display_number(
            station["complex_odor_value"], f" {odor_unit}" if odor_unit else ""
        )
        station["temperature_display"] = _display_number(
            station["temperature_c"], " ℃"
        )
        station["humidity_display"] = _display_number(
            station["humidity_pct"], " %"
        )
        station["wind_speed_display"] = _display_number(
            station["wind_speed_ms"], " m/s"
        )
    try:
        import pydeck as pdk
    except ImportError:
        st.warning(
            "지도 의존성(pydeck)이 없어 표로 대체했습니다. "
            "`pip install -r requirements-d.txt` 후 지도가 표시됩니다."
        )
        st.dataframe(stations, width="stretch", hide_index=True)
        return

    station_layer = pdk.Layer(
        "ScatterplotLayer",
        data=map_stations,
        get_position="[longitude, latitude]",
        get_fill_color=[18, 158, 215, 220],
        get_line_color=[255, 255, 255, 255],
        get_radius=230,
        radius_min_pixels=16,
        radius_max_pixels=32,
        line_width_min_pixels=3,
        stroked=True,
        pickable=True,
    )
    station_text = pdk.Layer(
        "TextLayer",
        data=map_stations,
        get_position="[longitude, latitude]",
        get_text="map_label",
        get_color=[255, 255, 255, 255],
        get_size=14,
        get_alignment_baseline="center",
        pickable=False,
    )
    facility_layer = pdk.Layer(
        "ScatterplotLayer",
        data=map_facilities,
        get_position="[longitude, latitude]",
        get_fill_color=[126, 87, 160, 190],
        get_radius=150,
        radius_min_pixels=8,
        radius_max_pixels=14,
        pickable=False,
    )
    center_lat = sum(item["latitude"] for item in map_stations) / len(map_stations)
    center_lon = sum(item["longitude"] for item in map_stations) / len(map_stations)
    deck = pdk.Deck(
        map_style=(
            "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json"
            if use_basemap
            else None
        ),
        initial_view_state=pdk.ViewState(
            latitude=center_lat, longitude=center_lon, zoom=11.5, pitch=0
        ),
        layers=[facility_layer, station_layer, station_text],
        tooltip={
            "html": (
                "<b>{station_name}</b> ({station_type})<br/>"
                "측정시각: {observed_at}<br/>"
                "상태: {status_label}<hr/>"
                "황화수소 {h2s_display} ppm · 암모니아 {nh3_display} ppm<br/>"
                "TVOC {tvoc_display} ppm · 복합악취 {odor_display}<br/>"
                "온도 {temperature_display} · 습도 {humidity_display}<br/>"
                "풍향 {wind_direction_text} · 풍속 {wind_speed_display}"
            ),
            "style": {"backgroundColor": "#263238", "color": "white"},
        },
    )
    st.pydeck_chart(deck, width="stretch", height=520)
    odor_units = {
        item.get("complex_odor_unit")
        for item in stations
        if item.get("complex_odor_unit")
    }
    if odor_units:
        odor_note = "복합악취 값이며 단위는 " + ", ".join(sorted(odor_units)) + "입니다."
    else:
        odor_note = "복합악취 원값이며 의미·단위는 코드북 수령 후 확정합니다."
    st.caption(
        "파란 원은 측정소, 보라색 원은 참고 사업장 위치입니다. "
        f"파란 원 안 숫자는 {odor_note} "
        "배경 지도는 토글을 켠 경우 네트워크 연결이 필요합니다."
    )


def _render_station_detail(stations: list[dict[str, Any]]) -> None:
    selected_id = st.selectbox(
        "측정소 상세",
        [item["station_id"] for item in stations],
        format_func=lambda station_id: next(
            item["station_name"] for item in stations if item["station_id"] == station_id
        ),
    )
    station = next(item for item in stations if item["station_id"] == selected_id)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("황화수소", _display_number(station["h2s_ppm"], " ppm"))
    c2.metric("암모니아", _display_number(station["nh3_ppm"], " ppm"))
    c3.metric("TVOC", _display_number(station["tvoc_ppm"], " ppm"))
    odor_unit = station.get("complex_odor_unit")
    c4.metric(
        "복합악취" if odor_unit else "복합악취 원값",
        _display_number(
            station["complex_odor_value"], f" {odor_unit}" if odor_unit else ""
        ),
    )
    w1, w2, w3, w4 = st.columns(4)
    w1.metric("온도", _display_number(station["temperature_c"], " ℃"))
    w2.metric("습도", _display_number(station["humidity_pct"], " %"))
    w3.metric("풍향", station["wind_direction_text"] or "미측정")
    w4.metric("풍속", _display_number(station["wind_speed_ms"], " m/s"))
    st.caption(
        f"측정시각 {station['observed_at']} · 상태 {station['status_label'] or '미확인'} · "
        f"품질 {station['record_qc']} ({', '.join(station['quality_flags'])}) · "
        + (
            f"복합악취 단위 {odor_unit}"
            if odor_unit
            else "복합악취 단위는 코드북 수령 후 확정"
        )
    )


def _risk_rows(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for item in items:
        if item["resolution"] == "3h":
            start = datetime.fromisoformat(item["start"])
            period = f"{start:%m-%d %H시}"
        else:
            period = f"{item['date']} 종일"
        rows.append(
            {
                "예측 구간": period,
                "해상도": "3시간" if item["resolution"] == "3h" else "일 단위",
                "상대지수": round(float(item["risk_score"]) * 100, 1),
                "등급": item["risk_grade"],
                "모델 자리": item["model_type"],
            }
        )
    return rows


def _window_rows(windows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for rank, window in enumerate(windows, start=1):
        start = datetime.fromisoformat(window["start"])
        end = datetime.fromisoformat(window["end"])
        rows.append(
            {
                "순위": rank,
                "6시간 작업 창": f"{start:%m-%d %H시}~{end:%H시}",
                "창 위험지수": round(window["window_risk"] * 100, 1),
                "추천점수": round(window["recommendation_score"] * 100, 1),
                "등급": window["grade"],
            }
        )
    return rows


def _render_evidence(result: dict[str, Any]) -> None:
    if result.get("status") == "refused":
        st.warning(result.get("data", {}).get("answer", "개별 법률 판단 요청입니다."))
        return
    data = result.get("data") or {}
    if result.get("status") != "ok" or not data.get("results"):
        st.warning("표시할 근거가 없습니다. 근거가 없으면 답변을 생성하지 않습니다.")
        return
    if data.get("notice"):
        st.info(data["notice"])
    for evidence in data["results"]:
        page = evidence.get("page")
        page_label = f"{page}쪽" if page is not None else "쪽수 미확인"
        with st.expander(
            f"{evidence['rank']}. {evidence['doc']} · {evidence['unit']} · {page_label}"
        ):
            st.write(evidence["snippet"])
            st.caption(
                f"{evidence.get('hier', '출처')} · {evidence.get('score_kind', '검색 점수')}"
            )


def main() -> None:
    st.set_page_config(
        page_title="익산 민원 위험 작업 도우미",
        page_icon="🐷",
        layout="wide",
    )
    st.title("익산 양돈농가 민원 위험 작업 도우미")

    with st.sidebar:
        st.header("시연 설정")
        mode_slot = st.empty()
        farm_slot = st.empty()
        work_type = st.selectbox("작업 유형", list(WORK_WEIGHT), index=0)
        storage_days = st.number_input(
            "분뇨 저장 경과일", min_value=0, max_value=60, value=12, step=1
        )
        use_openai = st.checkbox(
            "OpenAI 설명 사용",
            value=False,
            disabled=not is_configured(),
            help="OPENAI_API_KEY와 OPENAI_MODEL이 모두 있을 때만 선택할 수 있습니다.",
        )
        if not is_configured():
            st.caption("현재는 규칙 기반 설명으로 끝까지 시연됩니다.")

    try:
        backend = create_backend(storage_days=int(storage_days))
        status = backend.get_system_status()
    except Exception as exc:
        st.error(f"데이터 provider를 시작할 수 없습니다: {exc}")
        st.stop()

    status_data = status.get("data") or {}
    farm_id = status_data.get("default_farm_id")
    if not farm_id:
        st.error("provider 상태에 default_farm_id가 없습니다.")
        st.stop()
    farm_config = _safe_backend_call(
        "농가 설정 provider", lambda: backend.get_farm_config(farm_id)
    )
    farm_data = farm_config.get("data") or {}
    mode = status_data.get("mode", _source_state(status))
    mode_slot.text_input("데이터 모드", value=_state_label(mode), disabled=True)
    farm_slot.text_input(
        "농가", value=str(farm_data.get("name") or farm_id), disabled=True
    )

    sensor_snapshot = _safe_backend_call(
        "측정소 관측 provider", lambda: backend.get_sensor_snapshot()
    )
    calendar = _safe_backend_call(
        "위험 캘린더 provider",
        lambda: backend.get_risk_calendar(farm_id, 7, None),
    )
    snapshot_token = _decision_snapshot_token(status, calendar)
    input_fingerprint = (
        farm_id,
        work_type,
        int(storage_days),
        snapshot_token,
    )
    previous_fingerprint = st.session_state.get("d_input_fingerprint")
    if previous_fingerprint is not None and previous_fingerprint != input_fingerprint:
        for key in (
            "d_guide",
            "d_guide_snapshot",
            "d_narrative",
            "d_notification",
            "d_notification_message",
            "d_notification_window_fingerprint",
            "d_notification_approved",
            "d_openai_warning",
            "d_rag",
        ):
            st.session_state.pop(key, None)
    st.session_state["d_input_fingerprint"] = input_fingerprint

    official_state = str(status_data.get("official_dataset") or "unavailable")
    sensor_state = _source_state(
        sensor_snapshot, _component_state(status, "sensor_map")
    )
    risk_state = _source_state(calendar, _component_state(status, "risk_calendar"))
    rag_state = _component_state(status, "rag")
    notification_state = _component_state(status, "notification")

    visible_states = {sensor_state, risk_state, rag_state}
    if "fixture" in visible_states or mode == "fixture":
        st.warning(
            "시연용 구조 검증 프로토타입입니다. FIXTURE로 표시된 값은 실제 익산시 "
            "관측·예측 결과가 아니며, 각 탭에서 출처와 기준시각을 확인할 수 있습니다."
        )
    elif official_state == "waiting":
        st.warning(
            "핵심 정보공개 자료를 기다리는 중입니다. 현재 연결 상태와 각 탭의 출처를 "
            "확인해 주세요."
        )
    elif "stale" in visible_states:
        st.warning("일부 데이터의 갱신이 지연되었습니다. 마지막 기준시각을 확인해 주세요.")
    elif status.get("status") != "ok":
        st.error(status.get("error") or "시스템 상태를 확인할 수 없습니다.")
    else:
        st.info("연결된 provider의 출처·버전·기준시각을 표시하고 있습니다.")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("공식 데이터", _state_label(official_state))
    c2.metric("측정소 지도", _state_label(sensor_state))
    c3.metric("위험 캘린더", _state_label(risk_state))
    c4.metric("법령 근거", _state_label(rag_state))
    c5.metric("알림", _state_label(notification_state))
    status_source = status.get("source") or {}
    st.caption(
        f"시스템 기준시각: {status_source.get('generated_at', '미확인')} · "
        "플룸은 미검증 참고 정보이며 등급에 반영하지 않습니다."
    )

    tab_map, tab_calendar, tab_plan, tab_evidence, tab_notice, tab_limits = st.tabs(
        [
            "① 측정소 지도",
            "② 민원 위험 캘린더",
            "③ 작업 계획",
            "④ 근거·법령",
            "⑤ 알림 초안",
            "⑥ 데이터·한계",
        ]
    )

    with tab_map:
        st.subheader("익산악취24 측정소 관측 지도")
        st.caption(
            "이 탭은 현재 관측을 보여줍니다. 미래 민원 위험 캘린더와는 별도 데이터 계층입니다."
        )
        if sensor_snapshot.get("status") != "ok":
            st.error(sensor_snapshot.get("error") or "측정소 관측을 사용할 수 없습니다.")
        else:
            stations = (sensor_snapshot.get("data") or {}).get("stations") or []
            sensor_source = sensor_snapshot.get("source") or {}
            st.caption(
                f"데이터 상태 {_state_label(sensor_source.get('state'))} · "
                f"출처 {sensor_source.get('name', '미확인')} · "
                f"관측 기준일 {sensor_source.get('data_as_of', '미확인')} · "
                f"버전 {sensor_source.get('version', '미확인')}"
            )
            if not stations:
                st.warning("해당 기준시각에 표시할 측정소 관측이 없습니다.")
            else:
                use_basemap = st.checkbox(
                    "배경 지도 사용 (인터넷 필요)", value=True, key="sensor_basemap"
                )
                _render_sensor_map(sensor_snapshot, use_basemap)
                _render_station_detail(stations)
                with st.expander("전체 측정소 표와 품질 상태"):
                    st.dataframe(stations, width="stretch", hide_index=True)
            if sensor_source.get("state") == "fixture":
                st.warning(
                    "현재 위치·명칭·수치는 화면 구조 검증용 가상 값입니다. "
                    "정보공개 자료를 받으면 sensor-observation-v1 계약으로 매핑합니다."
                )
            elif sensor_source.get("limitations"):
                st.warning(" · ".join(sensor_source["limitations"]))

    with tab_calendar:
        st.subheader("향후 7일 민원 위험 캘린더")
        st.caption(
            "D+1~3은 3시간 블록, D+4~7은 중기예보 한계 때문에 일 단위로만 표시합니다."
        )
        if calendar.get("status") != "ok":
            st.error(calendar.get("error") or "위험 캘린더를 사용할 수 없습니다.")
        else:
            calendar_items = (calendar.get("data") or {}).get("items") or []
            short = [
                row for row in calendar_items if row.get("resolution") == "3h"
            ]
            mid = [
                row for row in calendar_items if row.get("resolution") == "day"
            ]
            st.markdown("**D+1~3 · 3시간 블록**")
            st.dataframe(_risk_rows(short), width="stretch", hide_index=True)
            st.markdown("**D+4~7 · 일 단위**")
            st.dataframe(_risk_rows(mid), width="stretch", hide_index=True)

    with tab_plan:
        st.subheader("6시간 작업 창 추천")
        st.caption(
            "추천 계산은 코드가 수행하고, OpenAI는 선택 시 그 결과를 설명만 합니다."
        )
        if st.button("작업 계획 생성", type="primary"):
            try:
                guide_obj = plan_work(backend, farm_id, work_type, days=3)
                guide = guide_obj.to_dict()
                narrative = rule_based_summary(guide)
                st.session_state.pop("d_openai_warning", None)
                if use_openai:
                    try:
                        narrative = compose_guide(backend, guide_obj) or narrative
                    except Exception as exc:
                        st.session_state["d_openai_warning"] = (
                            f"OpenAI 설명을 사용할 수 없어 규칙 기반 설명으로 전환했습니다: {exc}"
                        )
                st.session_state["d_guide"] = guide
                st.session_state["d_guide_snapshot"] = input_fingerprint
                st.session_state["d_narrative"] = narrative
                for key in (
                    "d_notification",
                    "d_notification_message",
                    "d_notification_window_fingerprint",
                    "d_notification_approved",
                ):
                    st.session_state.pop(key, None)
            except Exception as exc:
                st.error(f"작업 계획을 만들 수 없습니다: {exc}")

        guide = st.session_state.get("d_guide")
        if guide:
            if st.session_state.get("d_openai_warning"):
                st.warning(st.session_state["d_openai_warning"])
            st.info(st.session_state["d_narrative"])
            left, right = st.columns(2)
            with left:
                st.markdown("**추천 Top 3**")
                st.dataframe(
                    _window_rows(guide["recommended"]),
                    width="stretch",
                    hide_index=True,
                )
            with right:
                st.markdown("**회피 Top 3**")
                st.dataframe(
                    _window_rows(guide["avoid"]),
                    width="stretch",
                    hide_index=True,
                )
            a, b = st.columns(2)
            with a:
                st.markdown("**작업 전 조치**")
                for action in guide["before_actions"]:
                    st.write(f"- {action}")
            with b:
                st.markdown("**작업 후 조치**")
                for action in guide["after_actions"]:
                    st.write(f"- {action}")
            st.caption(" · ".join(guide["assumptions"]))
        else:
            st.info("왼쪽 설정을 확인하고 ‘작업 계획 생성’을 눌러 주세요.")

    with tab_evidence:
        st.subheader("법령·매뉴얼 근거")
        question = st.text_input(
            "질문",
            value=f"{work_type} 작업 전후에 확인할 관리 기준은 무엇인가요?",
        )
        if st.button("근거 검색"):
            st.session_state["d_rag"] = _safe_backend_call(
                "RAG provider", lambda: backend.search_rag(question, work_type)
            )
        rag_result = st.session_state.get("d_rag")
        if rag_result:
            _render_evidence(rag_result)
        else:
            if rag_state == "fixture":
                st.info("현재는 C 연결 전 fixture로 반환 계약과 화면만 검증합니다.")
            else:
                st.info("질문을 입력하고 ‘근거 검색’을 눌러 주세요.")
        st.caption("검색 점수는 문서 유사도이며 법적 신뢰확률이나 정답 확률이 아닙니다.")

    with tab_notice:
        st.subheader("주민 알림 초안")
        guide = st.session_state.get("d_guide")
        if not guide:
            st.info("먼저 ③ 작업 계획에서 추천 창을 생성해 주세요.")
        else:
            options = guide["recommended"]
            labels = []
            for window in options:
                start = datetime.fromisoformat(window["start"])
                end = datetime.fromisoformat(window["end"])
                labels.append(f"{start:%m-%d %H시}~{end:%H시} · {window['grade']}")
            selected_index = st.selectbox(
                "확정할 작업 창", range(len(options)), format_func=lambda i: labels[i]
            )
            selected_window = options[selected_index]
            window_fingerprint = (
                st.session_state.get("d_guide_snapshot"),
                guide["farm_id"],
                guide["work_type"],
                selected_window["start"],
                selected_window["end"],
            )
            previous_window = st.session_state.get(
                "d_notification_window_fingerprint"
            )
            if previous_window is not None and previous_window != window_fingerprint:
                for key in (
                    "d_notification",
                    "d_notification_message",
                    "d_notification_approved",
                ):
                    st.session_state.pop(key, None)
            st.session_state["d_notification_window_fingerprint"] = window_fingerprint
            if st.button("알림 초안 생성"):
                st.session_state.pop("d_notification", None)
                st.session_state.pop("d_notification_message", None)
                st.session_state.pop("d_notification_approved", None)
                try:
                    draft = create_notification_draft(
                        backend, farm_id, guide["work_type"], selected_window
                    ).to_dict()
                    st.session_state["d_notification"] = draft
                    st.session_state["d_notification_message"] = draft["message"]
                except Exception as exc:
                    st.error(f"알림 초안을 만들 수 없습니다: {exc}")

            draft = st.session_state.get("d_notification")
            if draft:
                audience_label = (
                    "가상 주거점"
                    if draft["audience_is_mock"]
                    else "provider가 반환한 영향 후보"
                )
                st.warning(
                    f"대상 {draft['audience_count']}곳은 {audience_label}입니다. 플룸은 "
                    "미검증 참고 모델이며 추천 점수와 등급에 반영되지 않았습니다."
                )
                st.text_area("편집 가능한 문구", key="d_notification_message", height=150)
                approved = st.checkbox(
                    "농장주가 문구와 대상을 확인했습니다",
                    key="d_notification_approved",
                )
                if st.button("승인 기록 (실제 발송 안 함)"):
                    if not approved:
                        st.error("승인 확인란을 먼저 선택해 주세요.")
                    else:
                        logs = st.session_state.setdefault("d_approval_logs", [])
                        logs.append(
                            {
                                "approved_at": datetime.now(KST).isoformat(),
                                "farm_id": draft["farm_id"],
                                "work_type": draft["work_type"],
                                "work_window": draft["work_window"],
                                "decision_snapshot": snapshot_token,
                                "message": st.session_state["d_notification_message"],
                                "sent": False,
                            }
                        )
                        st.success("시연 세션에 승인만 기록했습니다. 외부 발송은 없습니다.")
                st.caption(
                    f"현재 세션 승인 기록: {len(st.session_state.get('d_approval_logs', []))}건"
                )

    with tab_limits:
        st.subheader("현재 주장할 수 있는 범위")
        if mode == "fixture":
            current_scope = (
                "현재 화면은 정보공개청구 자료 도착 전 **구조와 사용자 흐름을 "
                "검증하는 프로토타입**입니다."
            )
        else:
            current_scope = (
                "현재 화면은 provider 상태·출처·버전을 보존하는 **공모전 시연용 "
                "프로토타입**입니다."
            )
        st.markdown(
            f"""
- {current_scope}
- 측정소 지도는 **현재 관측**, 위험 캘린더는 **미래 상대 위험 예측**이며 서로 대체하지 않습니다.
- 결과는 ‘악취 발생 확률’이 아니라 **민원 위험도의 상대적 비교**로 표현합니다.
- 민원 감소를 보장하지 않고 **상대적으로 위험한 시간대 회피를 지원**합니다.
- 플룸은 미검증 참고 정보이며 위험 등급과 추천 순위에 사용하지 않습니다.
- 법령 카드는 일반 정보이며 개별 사안의 법률 판단을 제공하지 않습니다.
- 익산 외 지역은 해당 지역의 민원·기상·농가·조례 데이터로 재학습·재검증해야 합니다.
"""
        )
        st.subheader("공식 데이터 도착 후 바뀌는 부분")
        st.code(
            "sensor fixture → 익산시 측정소 시계열 adapter\n"
            "risk fixture → B risk_calendar adapter\n"
            "RAG fixture → C RagIndex adapter\n"
            "D_BACKEND_FACTORY 설정 → dashboard 수정 없이 provider 주입\n"
            "Streamlit 화면·GuideCard·NotificationDraft 계약은 유지",
            language="text",
        )


if __name__ == "__main__":
    main()
