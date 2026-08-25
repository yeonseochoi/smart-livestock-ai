"""D 파트 Streamlit 시연 화면.

실행: ``cd demo; streamlit run app/dashboard.py``
추천 계산·RAG 결합·선택적 Gemini 설명은 모두 ``agents/``가 담당한다.
이 파일은 화면 표시와 세션 상태만 담당한다(로직·판단은 두지 않는다).

[C] 2026-08 디자인 토큰(_CSS 안 OKLCH 값)은 팀이 Lovable로 만든 시안의
styles.css를 그대로 옮긴 것이다 — 발표용 목업과 실제 화면의 색이 어긋나면
발표 중 헷갈리기 때문에, 별도로 고안하지 않고 원본 값을 그대로 썼다.
"""
from __future__ import annotations

import html
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


DEMO_DIR = Path(__file__).resolve().parents[1]
if str(DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(DEMO_DIR))

# ── Streamlit Cloud Secrets -> 환경변수 다리 ────────────────────────────
# config.py 는 import 시점에 os.environ["DATABASE_URL"] 을 읽어 백엔드를 정한다.
# 배포 환경에는 .env 파일이 없고 값이 st.secrets 로만 들어오므로, agents 를
# import 하기 "전에" 환경변수로 옮겨 둔다. 이미 있는 값은 덮어쓰지 않는다
# (로컬 .env 가 우선). 로컬처럼 secrets 가 없으면 조용히 건너뛴다.
try:
    for _key, _value in st.secrets.items():
        if isinstance(_value, str) and _key not in os.environ:
            os.environ[_key] = _value
except Exception:
    pass

from agents.gemini_explainer import compose, is_configured
from agents.provider import create_provider
from agents.work_guide import plan_work, rule_based_summary


WORK_TYPES = ("분뇨제거", "청소", "환기점검", "저감시설점검", "액비살포")

# ── 상태값 -> 톤(ok/warn/risk/note) 매핑 ────────────────────────────────
# 톤은 _CSS의 .pill-* 클래스와 점(dot) 색을 고르는 데만 쓰고, 실제 색값은
# 전부 CSS 쪽에 모아 둔다.
_GRADE_TONE = {"낮음": "ok", "주의": "warn", "위험": "risk"}
_STATE_TONE = {
    "connected": "ok", "ready": "ok",
    "fixture": "warn", "stale": "warn", "waiting": "warn",
    "unavailable": "risk",
}
_STATE_LABEL = {
    "connected": "CONNECTED", "fixture": "FIXTURE", "stale": "STALE",
    "unavailable": "UNAVAILABLE", "waiting": "WAITING", "ready": "READY",
}

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;500;600;700;800&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'Noto Sans KR','Apple SD Gothic Neo',system-ui,sans-serif; }
.stApp { background: oklch(0.986 0.004 95); }
.block-container { padding-top: 2rem; max-width: 1180px; }
h1, h2, h3 { letter-spacing: -0.02em; margin-bottom: 0.1rem; color: oklch(0.22 0.02 250); }

div[data-testid="stMetricValue"] { font-family: 'IBM Plex Mono', ui-monospace, monospace; }

/* 탭 바 — 둥근 알약형 배경, 선택된 탭만 흰 배경 + 그림자 */
.stTabs [data-baseweb="tab-list"] {
  gap: 4px; background: oklch(0.958 0.006 240); padding: 4px; border-radius: 999px;
}
.stTabs [data-baseweb="tab"] { font-weight: 600; border-radius: 999px; padding: 6px 18px; }
.stTabs [aria-selected="true"] {
  background: oklch(1 0 0) !important;
  box-shadow: 0 1px 2px oklch(0.2 0.03 240 / 0.08), 0 4px 10px -6px oklch(0.2 0.03 240 / 0.25);
}

div[data-testid="stVerticalBlockBorderWrapper"] {
  border-radius: 16px !important; border-color: oklch(0.912 0.008 250) !important;
}
div[data-testid="stExpander"] { border-radius: 12px !important; border-color: oklch(0.912 0.008 250) !important; }

.stButton>button[kind="primary"] {
  background: oklch(0.42 0.07 200); border-radius: 10px; font-weight: 700; border: none;
}

/* ── 커스텀 컴포넌트 ── */
.stat-card {
  background: oklch(1 0 0); border: 1px solid oklch(0.912 0.008 250);
  border-radius: 14px; padding: 14px 16px;
  box-shadow: 0 1px 2px oklch(0.2 0.03 240 / 0.04), 0 8px 24px -16px oklch(0.2 0.03 240 / 0.18);
}
.stat-label {
  font-size: 0.6875rem; font-weight: 600; letter-spacing: 0.1em; text-transform: uppercase;
  color: oklch(0.52 0.02 250); margin-bottom: 6px;
}
.stat-value { font-size: 1.15rem; font-weight: 700; }
.stat-caption { font-size: 0.75rem; color: oklch(0.52 0.02 250); margin-top: 2px; }
.stat-dot { display:inline-block; width:8px; height:8px; border-radius:999px; margin-right:6px; }

.pill {
  display:inline-block; padding: 2px 12px; border-radius:999px;
  font-size:0.75rem; font-weight:700; font-family:'IBM Plex Mono',monospace;
  white-space: nowrap;
}
.pill-ok { background: oklch(0.96 0.04 155); color: oklch(0.4 0.1 155); }
.pill-warn { background: oklch(0.97 0.04 90); color: oklch(0.42 0.1 70); }
.pill-risk { background: oklch(0.965 0.03 25); color: oklch(0.44 0.17 25); }
.pill-note { background: oklch(0.94 0.02 200); color: oklch(0.3 0.05 200); }
.pill-outline { background: transparent; color: oklch(0.52 0.02 250); border: 1px solid oklch(0.9 0.008 250); }

.source-caption { font-size: 0.85rem; color: oklch(0.52 0.02 250); margin: 2px 0 10px; }

/* 이 계획의 결론 — 파란 그라데이션 카드 */
.insight-card {
  border-radius: 22px; padding: 22px 26px; margin: 4px 0 20px;
  background-image: linear-gradient(135deg, oklch(0.48 0.17 258) 0%, oklch(0.4 0.15 235) 100%);
  color: oklch(0.99 0.01 250);
  box-shadow: 0 2px 4px oklch(0.45 0.16 255 / 0.16), 0 18px 40px -22px oklch(0.45 0.16 255 / 0.55);
}
.insight-title { font-weight: 700; font-size: 1rem; margin-bottom: 14px; opacity: 0.95; }
.insight-windows { display:flex; align-items:center; gap:14px; flex-wrap: wrap; }
.insight-window { flex: 1 1 220px; background: oklch(1 0 0 / 0.12); border-radius: 14px; padding: 12px 16px; }
.insight-label { font-size: 0.7rem; letter-spacing: 0.1em; text-transform: uppercase; opacity: 0.75; margin-bottom: 4px; }
.insight-time { font-family: 'IBM Plex Mono', monospace; font-size: 1.05rem; font-weight: 700; }
.insight-arrow { font-size: 1.4rem; opacity: 0.7; padding: 0 2px; }
.insight-footer { margin-top: 16px; display:flex; align-items:center; gap:12px; flex-wrap: wrap; }
.insight-note { font-size: 0.8rem; opacity: 0.85; line-height: 1.5; }

/* 7일 캘린더 — 스크롤 리스트 */
.cal-scroll { max-height: 380px; overflow-y: auto; border: 1px solid oklch(0.912 0.008 250); border-radius: 14px; }
.cal-row {
  display: grid; grid-template-columns: 92px 48px 44px 1fr 64px 64px;
  align-items: center; gap: 10px; padding: 9px 14px;
  border-bottom: 1px solid oklch(0.94 0.006 240); font-size: 0.85rem;
}
.cal-row:last-child { border-bottom: none; }
.cal-date { color: oklch(0.52 0.02 250); font-family: 'IBM Plex Mono', monospace; font-size: 0.8rem; }
.cal-hour { font-family: 'IBM Plex Mono', monospace; font-weight: 700; }
.cal-res { color: oklch(0.52 0.02 250); font-size: 0.75rem; }
.cal-bar-wrap { display:flex; align-items:center; gap:10px; }
.cal-bar-track { background: oklch(0.94 0.006 240); border-radius: 999px; height: 8px; flex: 1; overflow: hidden; position: relative; }
.cal-bar-fill { display: block; height: 100%; border-radius: 999px; }
.cal-score { font-family: 'IBM Plex Mono', monospace; font-size: 0.8rem; min-width: 52px; text-align: right; }
.cal-horizon { color: oklch(0.52 0.02 250); font-size: 0.75rem; font-family: 'IBM Plex Mono', monospace; }

/* 추천/회피 Top3 카드 */
.rank-card {
  display:flex; align-items:center; gap:12px;
  background: oklch(1 0 0); border: 1px solid oklch(0.912 0.008 250); border-radius: 14px;
  padding: 12px 14px; margin-bottom: 10px;
}
.rank-num {
  width: 26px; height: 26px; border-radius: 999px; flex-shrink: 0;
  display:flex; align-items:center; justify-content:center;
  font-weight: 700; font-size: 0.8rem; color: white;
}
.rank-body { flex: 1; }
.rank-time { font-family: 'IBM Plex Mono', monospace; font-weight: 700; font-size: 0.95rem; }
.rank-meta { font-size: 0.75rem; color: oklch(0.52 0.02 250); margin-top: 2px; }

.note-box {
  background: oklch(0.958 0.006 240); border-radius: 12px; padding: 10px 16px;
  font-size: 0.85rem; color: oklch(0.4 0.02 250); margin-top: 12px;
}
.banner-box {
  background: oklch(0.97 0.04 90); color: oklch(0.42 0.1 70);
  border-radius: 12px; padding: 10px 16px; font-size: 0.85rem; margin: 4px 0 18px;
}
</style>
"""

_TONE_TEXT = {
    "ok": "oklch(0.4 0.1 155)", "warn": "oklch(0.42 0.1 70)",
    "risk": "oklch(0.44 0.17 25)", "note": "oklch(0.3 0.05 200)",
}
_TONE_DOT = {
    "ok": "oklch(0.62 0.13 155)", "warn": "oklch(0.72 0.14 75)",
    "risk": "oklch(0.58 0.2 25)", "note": "oklch(0.48 0.17 258)",
}


def _inject_css() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)


def _tone_pill(label: str, tone: str) -> str:
    cls = {"ok": "pill-ok", "warn": "pill-warn", "risk": "pill-risk", "note": "pill-note"}.get(
        tone, "pill-outline"
    )
    return f'<span class="pill {cls}">{html.escape(label)}</span>'


def _grade_pill(grade: str | None) -> str:
    return _tone_pill(grade or "미확인", _GRADE_TONE.get(grade or "", ""))


def _state_label_tone(state: str | None) -> tuple[str, str]:
    key = (state or "").lower()
    label = _STATE_LABEL.get(key, (state or "알 수 없음").upper())
    return label, _STATE_TONE.get(key, "risk")


def _stat_card(label: str, value: str, caption: str, tone: str | None) -> None:
    dot = _TONE_DOT.get(tone or "", "oklch(0.42 0.07 200)")
    st.markdown(
        f'<div class="stat-card"><div class="stat-label">{html.escape(label)}</div>'
        f'<div class="stat-value"><span class="stat-dot" style="background:{dot}"></span>'
        f'{html.escape(value)}</div>'
        f'<div class="stat-caption">{html.escape(caption)}</div></div>',
        unsafe_allow_html=True,
    )


def _source_caption(response: dict[str, Any]) -> None:
    """출처를 한 줄 캡션으로 보여준다 — 상태 단어만 톤 색, 나머지는 회색."""
    source = response.get("source") or {}
    label, tone = _state_label_tone(source.get("state"))
    ink = _TONE_TEXT.get(tone, "oklch(0.52 0.02 250)")
    name = source.get("name", "출처 없음")
    st.markdown(
        f'<div class="source-caption"><b style="color:{ink}">{html.escape(label)}</b> · '
        f'{html.escape(name)}</div>',
        unsafe_allow_html=True,
    )


def _info_card(title: str, state_label: str, tone: str, desc: str) -> str:
    return (
        '<div class="stat-card" style="margin-bottom:12px;">'
        '<div style="display:flex;justify-content:space-between;align-items:center;'
        f'gap:8px;margin-bottom:8px;"><b>{html.escape(title)}</b>{_tone_pill(state_label, tone)}</div>'
        f'<div style="font-size:0.85rem;color:oklch(0.4 0.02 250);line-height:1.5;">'
        f'{html.escape(desc)}</div></div>'
    )


def _safe(call, label: str) -> dict[str, Any]:
    try:
        return call()
    except Exception as exc:
        return {"status": "unavailable", "data": None, "source": None,
                "error": f"{label}: {type(exc).__name__}: {exc}"}


def _has_coordinate(row: dict[str, Any]) -> bool:
    return row.get("latitude") is not None and row.get("longitude") is not None


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
    for key in ("d_guide", "d_summary", "d_evidence_plain"):
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
                get_fill_color="[31, 110, 126, 200]", get_radius=250,
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
    with st.expander("측정소 관측값 표로 보기", expanded=False):
        st.dataframe(pd.DataFrame(stations).reindex(columns=columns), width="stretch")
    st.markdown(
        '<div class="note-box">이 지도는 관측 화면이며 미래 민원 위험 예측과 직접 환산하지 않습니다.</div>',
        unsafe_allow_html=True,
    )


def _calendar_items(response: dict[str, Any]) -> list[dict[str, Any]]:
    return (response.get("data") or {}).get("items", [])


def _render_calendar_list(items: list[dict[str, Any]]) -> None:
    if not items:
        st.info("표시할 캘린더 항목이 없습니다.")
        return
    rows_html = []
    for item in items:
        grade = item.get("risk_grade") or "미확인"
        tone = _GRADE_TONE.get(grade, "warn")
        bar_color = _TONE_DOT.get(tone, "oklch(0.72 0.14 75)")
        score = item.get("risk_score")
        try:
            score_val = float(score)
            pct = max(0.0, min(1.0, score_val)) * 100
            score_text = f"{score_val:.4f}"
        except (TypeError, ValueError):
            pct, score_text = 0.0, "-"
        hour = "일 단위" if item.get("hour") is None else f"{int(item['hour']):02d}시"
        rows_html.append(
            '<div class="cal-row">'
            f'<span class="cal-date">{html.escape(str(item.get("date") or ""))}</span>'
            f'<span class="cal-hour">{html.escape(hour)}</span>'
            f'<span class="cal-res">{html.escape(str(item.get("resolution") or ""))}</span>'
            '<span class="cal-bar-wrap">'
            f'<span class="cal-bar-track"><span class="cal-bar-fill" '
            f'style="width:{pct:.1f}%;background:{bar_color}"></span></span>'
            f'<span class="cal-score">{html.escape(score_text)}</span>'
            '</span>'
            f'<span>{_grade_pill(grade)}</span>'
            f'<span class="cal-horizon">{html.escape(str(item.get("horizon") or ""))}</span>'
            "</div>"
        )
    st.markdown(f'<div class="cal-scroll">{"".join(rows_html)}</div>', unsafe_allow_html=True)


def _fmt_window(window: dict[str, Any]) -> tuple[datetime, datetime]:
    return datetime.fromisoformat(window["start"]), datetime.fromisoformat(window["end"])


def _render_insight_card(recommended: list[dict[str, Any]], avoid: list[dict[str, Any]]) -> None:
    if not recommended or not avoid:
        return
    top = recommended[0]
    alt = recommended[1] if len(recommended) > 1 else recommended[0]
    worst = avoid[0]
    top_s, top_e = _fmt_window(top)
    alt_s, alt_e = _fmt_window(alt)
    worst_s, worst_e = _fmt_window(worst)
    st.markdown(
        '<div class="insight-card"><div class="insight-title">💡 이 계획의 결론</div>'
        '<div class="insight-windows">'
        '<div class="insight-window"><div class="insight-label">추천 작업 창</div>'
        f'<div class="insight-time">{top_s:%m월 %d일 %H시} → {top_e:%H시} ({html.escape(top["grade"])})</div></div>'
        '<div class="insight-arrow">→</div>'
        '<div class="insight-window"><div class="insight-label">대안 창</div>'
        f'<div class="insight-time">{alt_s:%m월 %d일 %H시} → {alt_e:%H시} ({html.escape(alt["grade"])})</div></div>'
        '</div>'
        '<div class="insight-footer">'
        f'<span class="pill pill-risk">회피 권고 · {worst_s:%m월 %d일 %H시} → {worst_e:%H시} '
        f'({html.escape(worst["grade"])})</span>'
        '<span class="insight-note">두 창 모두 6시간 전체를 비교한 상대 민원 위험 회피 결과입니다. '
        '실제 작업 전 최신 예보와 농장 상황을 다시 확인해야 합니다.</span>'
        "</div></div>",
        unsafe_allow_html=True,
    )


def _rank_card_html(rank: int, window: dict[str, Any], dot_color: str) -> str:
    start, end = _fmt_window(window)
    time_text = f"{start:%m.%d %H시} → {end:%m.%d %H시}"
    meta = f"6시간 위험 {window['window_risk']:.4f} · 비교점수 {window['recommendation_score']:.4f}"
    return (
        '<div class="rank-card">'
        f'<div class="rank-num" style="background:{dot_color}">{rank}</div>'
        '<div class="rank-body">'
        f'<div class="rank-time">{html.escape(time_text)}</div>'
        f'<div class="rank-meta">{html.escape(meta)}</div>'
        "</div>"
        f'{_grade_pill(window.get("grade"))}'
        "</div>"
    )


def main() -> None:
    st.set_page_config(
        page_title="익산 민원 위험 작업 도우미", layout="wide",
        page_icon="🐖",
    )
    _inject_css()

    with st.sidebar:
        st.header("시연 설정")
        st.caption("아래 값을 바꾸면 오른쪽 화면이 그 즉시 다시 계산됩니다.")

        st.subheader("농가 조건")
        work_type = st.selectbox(
            "작업 유형", WORK_TYPES,
            help="예: 분뇨 저장고를 비우는 작업이면 '분뇨제거'를 선택합니다.",
        )
        storage_days = st.slider(
            "분뇨 저장 경과일", 0, 30, 12,
            help="현재 저장조에 분뇨가 며칠째 쌓여 있는지입니다. 오래 저장할수록 악취 위험이 높게 반영됩니다.",
        )

        st.divider()
        st.subheader("화면 옵션")
        use_basemap = st.checkbox("온라인 배경지도 사용", value=False)
        use_gemini = st.checkbox(
            "Gemini 설명 사용", value=False, disabled=not is_configured(),
            help="GOOGLE_API_KEY가 있을 때만 선택할 수 있습니다. 켜면 근거카드마다 쉬운말 요약이 추가로 붙습니다.",
        )
        if not is_configured():
            st.caption("GOOGLE_API_KEY가 설정되지 않아 규칙 기반 설명만 사용합니다.")

    st.markdown(
        '<div style="font-size:0.8rem;color:oklch(0.52 0.02 250);margin-bottom:2px;">'
        "익산시 축산환경 시연</div>",
        unsafe_allow_html=True,
    )
    st.title("민원 위험 작업 도우미 · D 에이전트")
    st.markdown(
        f'<div class="banner-box"><b>{html.escape(work_type)}</b> · 분뇨 저장 {storage_days}일 경과 기준 · '
        "정보공개청구 자료 반영 전 구조 검증용 프로토타입</div>",
        unsafe_allow_html=True,
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
    snapshot = _safe(provider.get_sensor_snapshot, "측정소 조회 실패")
    current_fp = _fingerprint(status, calendar, work_type, storage_days)
    if st.session_state.get("d_fingerprint") != current_fp:
        _reset_decision_state()
        st.session_state["d_fingerprint"] = current_fp

    mode = str(status_data.get("mode", "unknown"))
    dataset = str(status_data.get("official_dataset", "unknown"))
    calendar_state_label, calendar_tone = _state_label_tone((calendar.get("source") or {}).get("state"))

    cols = st.columns(4)
    with cols[0]:
        _stat_card(
            "데이터 모드", mode.upper(),
            "구조 검증용 대체값" if mode.lower() == "legacy" else "시연용 고정값",
            "ok" if mode.lower() == "legacy" else "warn",
        )
    with cols[1]:
        _stat_card(
            "공식 자료", dataset.lower(),
            "정보공개청구 회신 대기" if dataset.lower() != "ready" else "반영 완료",
            "ok" if dataset.lower() == "ready" else "warn",
        )
    with cols[2]:
        _stat_card("농가", (farm.get("data") or {}).get("name", farm_id), farm_id, None)
    with cols[3]:
        _stat_card(
            "위험자료", calendar_state_label,
            (calendar.get("source") or {}).get("name", "출처 없음"), calendar_tone,
        )

    map_tab, plan_tab, evidence_tab, limits_tab = st.tabs(
        ["① 측정소 지도", "② 캘린더·작업계획", "③ RAG 근거", "④ 데이터·한계"]
    )
    with map_tab:
        st.subheader("현재·과거 측정소 관측")
        _source_caption(snapshot)
        _render_sensor_map(snapshot, use_basemap)

    with plan_tab:
        guide = st.session_state.get("d_guide")
        if guide:
            _render_insight_card(guide.get("recommended") or [], guide.get("avoid") or [])

        st.subheader("7일 민원 위험 캘린더")
        _source_caption(calendar)
        if calendar.get("status") == "ok":
            _render_calendar_list(_calendar_items(calendar))
        else:
            st.error(calendar.get("error") or "캘린더를 사용할 수 없습니다.")

        st.divider()
        if st.button("6시간 작업 계획 만들기", type="primary"):
            try:
                guide_obj = plan_work(provider, farm_id, work_type, days=3)
                new_guide = guide_obj.to_dict()
                st.session_state["d_guide"] = new_guide
                st.session_state["d_summary"] = rule_based_summary(new_guide)
                st.session_state["d_evidence_plain"] = [None] * len(new_guide["evidence"])
                if use_gemini:
                    try:
                        generated = compose(guide_obj)
                        if generated:
                            summary, evidence_plain = generated
                            st.session_state["d_summary"] = summary
                            st.session_state["d_evidence_plain"] = evidence_plain
                    except Exception as exc:
                        st.warning(f"Gemini 설명 실패: 규칙 기반 설명을 유지합니다. ({exc})")
                # [C] 2026-08 결론 카드를 캘린더 위(버튼보다 앞)에 두는 디자인이라,
                # 같은 rerun 안에서는 위쪽 코드가 이미 실행된 뒤라 반영되지 않는다.
                # st.rerun()으로 한 번 더 top-to-bottom을 돌려 결론 카드가 즉시 보이게 한다.
                st.rerun()
            except Exception as exc:
                st.error(f"작업 계획 생성 실패: {exc}")

        guide = st.session_state.get("d_guide")
        if guide:
            rec_col, avoid_col = st.columns(2)
            with rec_col:
                st.markdown("#### 🟢 추천 Top 3")
                for i, window in enumerate(guide["recommended"][:3], start=1):
                    st.markdown(
                        _rank_card_html(i, window, "oklch(0.62 0.13 155)"),
                        unsafe_allow_html=True,
                    )
            with avoid_col:
                st.markdown("#### 🔴 회피 Top 3")
                for i, window in enumerate(guide["avoid"][:3], start=1):
                    st.markdown(
                        _rank_card_html(i, window, "oklch(0.58 0.2 25)"),
                        unsafe_allow_html=True,
                    )
            st.info(st.session_state["d_summary"])

    with evidence_tab:
        guide = st.session_state.get("d_guide")
        if not guide:
            st.info("먼저 ② 탭에서 작업 계획을 만드세요.")
        else:
            st.subheader("근거 카드")
            evidence_items = guide["evidence"]
            if evidence_items:
                evidence_plain = st.session_state.get("d_evidence_plain") or []
                if not any(evidence_plain):
                    st.caption(
                        "쉬운말 설명은 사이드바에서 'Gemini 설명 사용'을 켜고 "
                        "작업 계획을 다시 만들면 카드마다 표시됩니다."
                    )
                paired = list(enumerate(evidence_items))
                for row_start in range(0, len(paired), 2):
                    row_cols = st.columns(2)
                    for col, (index, item) in zip(row_cols, paired[row_start:row_start + 2]):
                        with col:
                            with st.container(border=True):
                                source = item.get("source_file") or item.get("doc") or "출처 미확인"
                                unit = item.get("unit") or "단원 미확인"
                                page = item.get("page")
                                st.markdown(f"**{html.escape(source)}**")
                                st.caption(
                                    f"{unit} · 쪽수: {page if page is not None else '미확인'} · "
                                    "검색 순위 · 신뢰확률 아님"
                                )
                                bullets = evidence_plain[index] if index < len(evidence_plain) else None
                                if bullets:
                                    # [C] 2026-08 카드 요약을 한 줄 설명에서 2~3개짜리 불릿으로
                                    # 바꿨다 — 시연에서 한 줄 요약은 법조문 내용을 다 담지
                                    # 못한다는 피드백을 반영.
                                    st.markdown("\n".join(f"- {b}" for b in bullets))
                                else:
                                    st.caption("이 카드는 쉬운말 요약이 아직 없습니다.")
                                with st.expander("원문 보기", expanded=not bullets):
                                    # st.write는 마크다운으로 렌더링한다. 원문에 "1~3주"처럼
                                    # 물결표(~)가 두 번 나오면 마크다운이 그 사이 전체를
                                    # 취소선(~~...~~)으로 잘못 묶어 괄호·숫자 위치가 밀려
                                    # 보인다(work_guide.py의 "~"→"→" 수정과 같은 원인).
                                    # 근거카드는 원문을 그대로 보여줘야 하므로, 마크다운
                                    # 해석이 없는 st.text로 표시해 원문 그대로(줄바꿈 포함)
                                    # 나오게 한다.
                                    st.text(item.get("snippet", ""))
            else:
                st.warning("검색 근거가 없습니다. 공식 원문을 확인하세요.")

    with limits_tab:
        st.subheader("데이터 출처와 한계")
        st.caption("시연 화면에 표시되는 값의 근거와 아직 채워지지 않은 부분을 정리했습니다.")

        sensor_source = snapshot.get("source") or {}
        sensor_label, sensor_tone = _state_label_tone(sensor_source.get("state"))
        sensor_desc = "; ".join(sensor_source.get("limitations", ()) or ()) or (
            "익산악취24 측정소 adapter가 아직 연결되지 않아 현재·과거 관측값을 표시할 수 없습니다."
        )
        dataset_tone = "ok" if dataset.lower() == "ready" else "warn"
        cal_source = calendar.get("source") or {}
        cal_label, cal_tone = _state_label_tone(cal_source.get("state"))

        grid_col1, grid_col2 = st.columns(2)
        with grid_col1:
            st.markdown(
                _info_card("익산시 측정소 원자료", sensor_label, sensor_tone, sensor_desc),
                unsafe_allow_html=True,
            )
            st.markdown(
                _info_card(
                    "위험자료 피드", cal_label, cal_tone,
                    f'{cal_source.get("name", "D·risk_hourly")} · 1시간 해상도, D+1~3 구간 제공.',
                ),
                unsafe_allow_html=True,
            )
        with grid_col2:
            st.markdown(
                _info_card(
                    "공식 자료", dataset.upper(), dataset_tone,
                    "민원 원부·측정 원자료는 정보공개청구 회신 후 반영됩니다. "
                    "현재 지표는 구조 검증용 대체값입니다.",
                ),
                unsafe_allow_html=True,
            )
            st.markdown(
                _info_card(
                    "예측이 아닌 상대 비교", "NOTE", "note",
                    "본 화면의 지수는 절대 민원 발생 확률이 아니라 시간대 간 상대적 "
                    "회피 우선순위입니다. 작업 결정 시 현장 판단이 우선합니다.",
                ),
                unsafe_allow_html=True,
            )

        st.markdown("##### 설계 원칙")
        st.markdown(
            "- 측정소 관측과 미래 민원 위험은 서로 다른 데이터 흐름입니다.\n"
            "- D는 A 모델 파일이나 B DB 내부 계산을 복제하지 않고 provider 결과만 사용합니다.\n"
            "- 실제 자료 도착 후 센서 adapter와 A/B 모델을 검증한 뒤 provider를 교체합니다."
        )
        with st.expander("기술 상세 보기 (원본 상태 JSON)"):
            st.json(status)


if __name__ == "__main__":
    main()