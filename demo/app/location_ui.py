"""위치 입력과 지도 — Streamlit 위젯만 담당한다.

``dashboard.py`` 는 이 모듈의 함수 두 개만 부른다. 판단·계산은 전부
``agents/location.py`` 와 ``analysis/plume_select.py`` 에 있다.

────────────────────────────────────────────────────────────────────
왜 파일을 새로 팠나

``dashboard.py`` 는 디자인 리스킨(PR #13)으로 126줄에서 658줄이 됐고,
팀에서 2차 디자인을 이어서 작업한다. 같은 파일에 위치 위젯을 직접 넣으면
두 작업이 같은 줄에서 충돌한다. 그래서 화면 요소를 여기 모으고
``dashboard.py`` 에는 import 한 줄과 호출 두 줄만 남겼다.

기존 요소는 하나도 옮기거나 이름을 바꾸지 않았다 — 탭 이름·순서,
상단 상태카드 4개, 사이드바 기존 항목, ``_render_sensor_map()`` 전부 그대로다.
새 UI 는 사이드바 맨 아래와 탭 ① 맨 위에 '덧붙이는' 방식으로만 들어간다.

────────────────────────────────────────────────────────────────────
지도가 두 개인 이유

  위 (여기)          folium — 내 위치를 '입력'받고 부채꼴을 보여 준다
  아래 (기존 코드)   pydeck — 측정소 관측값을 '표시'만 한다

pydeck 은 임의 지점의 클릭 좌표를 돌려주지 못해 입력 위젯이 될 수 없다.
기존 pydeck 지도를 걷어내고 하나로 합치는 편이 화면상 더 낫지만, 그건
디자인 2차가 끝난 뒤 팀과 함께 정할 일이다. 지금 합치면 532줄 바뀐 파일을
또 뒤집는 셈이라 여기서는 건드리지 않는다.
"""
from __future__ import annotations

import html
from datetime import datetime, timedelta, timezone
from typing import Any

import streamlit as st

from agents.location import (
    DEFAULT_LOCATION,
    FarmLocation,
    GeocodeError,
    bbox_polygon,
    describe_overlay,
    destination,
    geocode_address,
    plume_overlay,
    receptor_points,
)
from analysis.plume_select import PLUME_MAX_KM

# Streamlit Cloud 컨테이너는 UTC 다. datetime.now() 를 그냥 쓰면 9시간 어긋나
# "지금 이후" 판정이 통째로 틀어진다. 한국 시각을 명시한다.
KST = timezone(timedelta(hours=9))

_STATE_KEY = "loc_current"
_CLICK_KEY = "loc_last_click"
_ERROR_KEY = "loc_error"

# 지도 색. dashboard.py 의 OKLCH 토큰과 같은 계열을 16진수로 옮겼다
# (folium 은 OKLCH 를 못 읽는다).
_INK_FARM = "#1F6E7E"        # 내 농장 — primaryColor
_INK_RURAL = "#C8752B"       # 농촌근거리
_INK_URBAN = "#2F6BB0"       # 시가지원거리
_INK_STATION = "#9AA3AB"     # fixture 측정소 — 회색으로 죽인다
_INK_PLUME = "#D9534F"       # 0~3km 플룸
_INK_SECTOR = "#E0A800"      # 3km 밖 섹터
_INK_BBOX = "#B03030"        # 익산 경계 상자


# ═══════════════════════════════════════════════════════════════════
# 세션 상태
# ═══════════════════════════════════════════════════════════════════

def current_location() -> FarmLocation:
    raw = st.session_state.get(_STATE_KEY)
    if not raw:
        return DEFAULT_LOCATION
    return FarmLocation(
        lat=raw["lat"], lon=raw["lon"], source=raw["source"],
        accuracy_m=raw.get("accuracy_m"), label=raw.get("label"),
    )


def _store(location: FarmLocation) -> None:
    st.session_state[_STATE_KEY] = location.to_dict()
    st.session_state.pop(_ERROR_KEY, None)
    # 위치가 바뀌면 이미 만들어 둔 작업 계획은 낡은 것이다. dashboard.py 가
    # 이미 갖고 있는 무효화 장치(d_fingerprint 불일치 -> _reset_decision_state)를
    # 그대로 쓴다. 지문만 지우면 다음 rerun 에서 알아서 다시 계산된다 —
    # dashboard.py 를 고치지 않으려고 이 방식을 택했다.
    st.session_state.pop("d_fingerprint", None)


# ═══════════════════════════════════════════════════════════════════
# 1. 사이드바 — 위치 입력 3-way
# ═══════════════════════════════════════════════════════════════════

def render_location_sidebar() -> FarmLocation:
    """사이드바 맨 아래에 「내 위치」 섹션을 그리고 현재 위치를 돌려준다."""

    location = current_location()

    st.divider()
    st.subheader("📍 내 위치")

    if location.is_default:
        st.caption("현재: 미설정 — 왕궁 축산단지 기준으로 표시 중입니다.")
    else:
        note = location.accuracy_note
        st.caption(
            f"현재: {location.source_label} · {location.lat:.5f}, {location.lon:.5f}"
            + (f"\n\n{note}" if note else "")
        )
        if not location.in_iksan:
            st.warning(
                "익산시 범위 밖입니다. 화면의 위험도는 익산 기준 값이므로 "
                "이 위치의 실제 위험도가 아닙니다.",
                icon="⚠️",
            )

    # ── ① GPS ────────────────────────────────────────────────────
    # 컴포넌트가 없거나 권한이 거부돼도 에러로 취급하지 않는다. 지도 클릭이
    # 언제나 남아 있으므로 '정상 경로가 하나 줄어든 것'일 뿐이다.
    try:
        from streamlit_geolocation import streamlit_geolocation

        fix = streamlit_geolocation()
        if fix and fix.get("latitude") is not None and fix.get("longitude") is not None:
            candidate = FarmLocation(
                lat=float(fix["latitude"]), lon=float(fix["longitude"]),
                source="gps", accuracy_m=_as_float(fix.get("accuracy")),
                label="GPS 조회 위치",
            )
            if _changed(location, candidate):
                _store(candidate)
                st.rerun()
        st.caption("위 아이콘을 누르면 브라우저가 위치 권한을 묻습니다.")
    except ImportError:
        st.caption(
            "GPS 버튼은 streamlit-geolocation 설치 후 사용할 수 있습니다. "
            "지도 클릭이나 주소 검색을 사용하세요."
        )

    # ── ② 주소 검색 ──────────────────────────────────────────────
    with st.form("loc_address_form", clear_on_submit=False):
        query = st.text_input("주소로 찾기", placeholder="예: 익산시 왕궁면 흥암리")
        submitted = st.form_submit_button("찾기")
    if submitted:
        try:
            _store(geocode_address(query))
            st.rerun()
        except GeocodeError as exc:
            st.session_state[_ERROR_KEY] = str(exc)
    if st.session_state.get(_ERROR_KEY):
        st.error(st.session_state[_ERROR_KEY])

    # ── ③ 지도 클릭 안내 + 되돌리기 ───────────────────────────────
    st.caption("또는 ① 탭의 지도를 직접 클릭하세요.")
    if not location.is_default and st.button("기본값(왕궁)으로 되돌리기"):
        st.session_state.pop(_STATE_KEY, None)
        st.session_state.pop(_CLICK_KEY, None)
        st.rerun()

    return current_location()


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _changed(current: FarmLocation, candidate: FarmLocation) -> bool:
    """같은 좌표로 계속 rerun 하지 않도록 6자리(약 0.1m)까지만 비교한다."""
    return (round(current.lat, 6), round(current.lon, 6)) != (
        round(candidate.lat, 6), round(candidate.lon, 6)
    )


# ═══════════════════════════════════════════════════════════════════
# 2. 탭 ① — 내 농장 지도
# ═══════════════════════════════════════════════════════════════════

def render_location_panel(
    location: FarmLocation, *, provider: Any, snapshot: dict[str, Any],
    use_basemap: bool,
) -> None:
    """탭 ① 맨 위에 내 농장 지도를 그린다. 실패해도 아래 기존 화면은 살아 있다."""

    st.subheader("📍 내 농장 위치")

    wind_items = _wind_items(provider)
    overlay, when = _pick_overlay(location, wind_items)

    try:
        clicked = _draw_map(location, snapshot, overlay, use_basemap)
    except ImportError:
        st.info(
            "지도 클릭 입력은 folium · streamlit-folium 설치 후 사용할 수 있습니다. "
            "(`pip install -r requirements-d.txt`)"
        )
        _manual_input(location)
        return

    if clicked is not None:
        candidate = FarmLocation(lat=clicked[0], lon=clicked[1], source="map",
                                 label="지도에서 선택한 위치")
        if _changed(location, candidate):
            _store(candidate)
            st.rerun()

    _legend(location, overlay, when)


def _wind_items(provider: Any) -> list[dict[str, Any]]:
    """예보 원값. provider 가 이 기능을 갖고 있을 때만 쓴다.

    ``DecisionProvider`` 계약에 없는 선택 기능이라 ``getattr`` 로 확인한다.
    fixture provider 에는 예보 원값이 없으므로 부채꼴 없이 핀만 표시된다.
    """
    getter = getattr(provider, "get_wind_series", None)
    if not callable(getter):
        return []
    try:
        response = getter(3)
    except Exception:
        return []
    if response.get("status") != "ok":
        return []
    return (response.get("data") or {}).get("items", [])


def _pick_overlay(
    location: FarmLocation, wind_items: list[dict[str, Any]]
) -> tuple[dict[str, Any] | None, datetime | None]:
    """시각 슬라이더를 그리고, 그 시각의 부채꼴을 계산한다.

    슬라이더를 움직이면 부채꼴이 회전한다 — "왜 이 시각을 피하라는지"를
    문장이 아니라 그림으로 설명하는 장치다.
    """
    if not wind_items:
        st.caption(
            "예보 원값(forecast_hourly)이 없어 풍하측 부채꼴은 생략합니다. "
            "`python run_serve.py` 로 채울 수 있습니다."
        )
        return None, None

    labels = [datetime.fromisoformat(item["start"]) for item in wind_items]

    # 슬라이더 기본값은 "지금 이후 첫 시각"이다. 예보 원값이 통째로 낡았으면
    # 마지막 시각을 잡고, 낡았다는 사실을 화면에 적는다.
    now = datetime.now(KST)
    future = [i for i, label in enumerate(labels) if label >= now]
    default_index = future[0] if future else len(labels) - 1
    if not future:
        st.caption(
            f"예보 원값이 {labels[-1]:%m월 %d일 %H시} 까지뿐입니다. "
            "`python run_serve.py` 로 갱신하면 최신 바람으로 부채꼴이 그려집니다."
        )

    index = st.select_slider(
        "부채꼴을 볼 시각",
        options=list(range(len(wind_items))),
        value=default_index,
        format_func=lambda i: f"{labels[i]:%m/%d %H시}",
        help="이 시각의 예보 바람으로 냄새가 가는 방향을 그립니다. "
             "슬라이더를 움직이면 부채꼴이 회전합니다.",
    )
    item = wind_items[index]
    try:
        overlay = plume_overlay(
            location.lat, location.lon,
            item["wd"], item["ws"], item["sky"], labels[index],
        )
    except Exception as exc:
        st.caption(f"부채꼴 계산 실패: {type(exc).__name__}: {exc}")
        return None, labels[index]
    return overlay, labels[index]


def _draw_map(
    location: FarmLocation, snapshot: dict[str, Any],
    overlay: dict[str, Any] | None, use_basemap: bool,
) -> tuple[float, float] | None:
    """folium 지도를 그리고 새 클릭 좌표가 있으면 돌려준다."""
    import folium
    from streamlit_folium import st_folium

    fmap = folium.Map(
        location=[location.lat, location.lon], zoom_start=11,
        tiles="OpenStreetMap" if use_basemap else None,
        control_scale=True,
    )

    # ── 레이어 0 — 익산 경계 상자 (범위 밖일 때만) ────────────────
    if not location.in_iksan:
        folium.Polygon(
            locations=bbox_polygon(), color=_INK_BBOX, weight=2, dash_array="8",
            fill=False, tooltip="익산시 대략 범위 [C] — 이 밖은 익산 기준 값이 아닙니다",
        ).add_to(fmap)

    # ── 레이어 1 — 풍하측 부채꼴 두 개 ────────────────────────────
    # 3km 안팎에서 판정 도구가 바뀐다. 색을 다르게 해 경계를 눈에 보이게 한다.
    if overlay:
        folium.Polygon(
            locations=overlay["sector_polygon"], color=_INK_SECTOR, weight=1,
            dash_array="6", fill=True, fill_color=_INK_SECTOR, fill_opacity=0.10,
            tooltip=(f"{PLUME_MAX_KM:.0f}km 밖 — 섹터 지표 ±{overlay['sector_half_angle']:.0f}° "
                     "(좌표 오차가 부채꼴보다 커 플룸으로 판정하지 않는 구간)"),
        ).add_to(fmap)
        folium.Polygon(
            locations=overlay["plume_polygon"], color=_INK_PLUME, weight=1,
            fill=True, fill_color=_INK_PLUME, fill_opacity=0.22,
            tooltip=(f"0~{PLUME_MAX_KM:.0f}km — 플룸 유효 반각 "
                     f"(안정도 {overlay['stability']}, 1km 지점 "
                     f"±{overlay['plume_half_angle_1km']:.1f}°)"),
        ).add_to(fmap)
        tip = destination(location.lat, location.lon, overlay["to_deg"], PLUME_MAX_KM)
        folium.PolyLine(
            [[location.lat, location.lon], [tip[0], tip[1]]],
            color=_INK_PLUME, weight=2, opacity=0.9,
            tooltip=f"냄새 진행 방향 {overlay['to_deg']:.0f}°",
        ).add_to(fmap)

    # ── 레이어 2 — 수용점 2곳 (학습이 쓴 실제 중심 좌표) ──────────
    for point in receptor_points():
        color = _INK_RURAL if "농촌" in point["group"] else _INK_URBAN
        folium.CircleMarker(
            [point["lat"], point["lon"]], radius=8, color=color, weight=2,
            fill=True, fill_color=color, fill_opacity=0.75,
            tooltip=f"{point['group']} · {point['note']}",
        ).add_to(fmap)

    # ── 레이어 3 — fixture 측정소 (회색 점선 + "(예시)") ──────────
    # 세 가지(회색·점선·라벨)를 동시에 걸어 실측 레이어와 확실히 구분한다.
    for row in _stations(snapshot):
        folium.CircleMarker(
            [row["latitude"], row["longitude"]], radius=6, color=_INK_STATION,
            weight=2, dash_array="4", fill=True, fill_color="#FFFFFF",
            fill_opacity=0.9,
            tooltip=f"{row.get('station_name', '측정소')} · 발표용 예시 데이터",
        ).add_to(fmap)

    # ── 레이어 4 — 내 농장 ────────────────────────────────────────
    folium.Marker(
        [location.lat, location.lon],
        tooltip=("내 농장 (기본값)" if location.is_default
                 else f"내 농장 · {location.source_label}"),
        icon=folium.Icon(color="green" if location.in_iksan else "red", icon="home",
                         prefix="fa"),
    ).add_to(fmap)

    # ── 컴포넌트 키를 '입력 경로'에 묶는다 ────────────────────────
    # st_folium 은 마지막 클릭 좌표를 계속 되돌려 준다. 키가 고정이면
    # 주소 검색으로 위치를 옮긴 뒤에도 예전 클릭이 다시 올라와 방금 찾은
    # 주소를 덮어쓴다. 그렇다고 좌표까지 키에 넣으면 클릭할 때마다 지도가
    # 새로 마운트돼 확대·이동 상태가 초기화된다.
    #
    # 그래서 키는 좌표가 아니라 source 로 만든다.
    #   · 지도를 연달아 클릭  -> 키 그대로 -> 확대 상태 유지
    #   · 주소·GPS 로 이동    -> 키 변경   -> 새 컴포넌트, 묵은 클릭 사라짐
    map_key = f"loc_map_{location.source}"
    if st.session_state.get("loc_map_key") != map_key:
        # 컴포넌트가 새로 뜨므로 남은 클릭 기록도 같이 비운다. 이게 없으면
        # 주소로 옮겼다가 아까 그 지점을 다시 클릭했을 때 무시된다.
        st.session_state["loc_map_key"] = map_key
        st.session_state.pop(_CLICK_KEY, None)

    result = st_folium(
        fmap, height=440, width=None, returned_objects=["last_clicked"],
        key=map_key,
    )
    point = (result or {}).get("last_clicked")
    if not point:
        return None
    coords = (round(float(point["lat"]), 6), round(float(point["lng"]), 6))
    if st.session_state.get(_CLICK_KEY) == coords:
        return None                       # 같은 클릭으로 무한 rerun 하지 않는다
    st.session_state[_CLICK_KEY] = coords
    return coords


def _stations(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    if snapshot.get("status") != "ok":
        return []
    rows = (snapshot.get("data") or {}).get("stations") or []
    return [r for r in rows
            if r.get("latitude") is not None and r.get("longitude") is not None]


def _manual_input(location: FarmLocation) -> None:
    """folium 이 없을 때의 최소 대체 입력. 기능이 통째로 사라지지 않게 한다."""
    col_lat, col_lon = st.columns(2)
    lat = col_lat.number_input("위도", value=float(location.lat), format="%.6f")
    lon = col_lon.number_input("경도", value=float(location.lon), format="%.6f")
    if st.button("이 좌표 사용"):
        _store(FarmLocation(lat=lat, lon=lon, source="map", label="직접 입력한 좌표"))
        st.rerun()


def _legend(
    location: FarmLocation, overlay: dict[str, Any] | None, when: datetime | None
) -> None:
    """범례와 한 줄 설명. 회색 점선이 '예시 데이터'라는 것을 글로도 못박는다."""
    items = [
        (_INK_FARM, "내 농장", "지도 클릭 · GPS · 주소로 정한 위치"),
        (_INK_RURAL, "농촌근거리", "민원 좌표 중앙값"),
        (_INK_URBAN, "시가지원거리", "민원 좌표 중앙값"),
        (_INK_STATION, "측정소(예시)", "실제 익산시 자료 아님"),
    ]
    if overlay:
        items.append((_INK_PLUME, f"0~{PLUME_MAX_KM:.0f}km 플룸", "유효 반각"))
        items.append((_INK_SECTOR, f"{PLUME_MAX_KM:.0f}km 밖 섹터", "±30° 노출 지표"))
    chips = "".join(
        f'<span style="display:inline-flex;align-items:center;gap:6px;margin-right:14px;'
        f'font-size:0.78rem;color:oklch(0.45 0.02 250);">'
        f'<span style="width:10px;height:10px;border-radius:999px;background:{color};'
        f'display:inline-block;"></span>{html.escape(name)}'
        f'<span style="color:oklch(0.62 0.02 250);">· {html.escape(note)}</span></span>'
        for color, name, note in items
    )
    st.markdown(f'<div style="margin:4px 0 8px;">{chips}</div>', unsafe_allow_html=True)

    lines = [
        f"위치 {location.lat:.5f}, {location.lon:.5f} · {location.source_label}"
        + (" · 익산시 안" if location.in_iksan else " · ⚠ 익산시 밖")
    ]
    if overlay and when is not None:
        lines.append(f"{when:%m월 %d일 %H시} 기준 — " + describe_overlay(overlay))
        if overlay["downwind_groups"]:
            lines.append(
                "이 시각에는 위 유형의 위험도를 골라 추천 순위를 매깁니다. "
                "플룸은 유형을 고르기만 하고 점수·등급은 바꾸지 않습니다."
            )
    st.markdown(
        '<div class="note-box">' + "<br>".join(html.escape(line) for line in lines)
        + "</div>",
        unsafe_allow_html=True,
    )
