"""S7 — 에이전트 도구 6종: Claude API tool use 스키마 + 로컬 구현.

ANTHROPIC_API_KEY 가 없으면 work_guide 가 오프라인 폴백으로 같은 도구를
직접 호출한다 (스키마는 실 API 전환 시 그대로 사용).
"""
from __future__ import annotations

from datetime import datetime, timedelta

from config import DEMO_FARM, PROV
from ops import db
from scoring.s5_recommend import window_risk, _load_calendar, WORK_WEIGHT, storage_factor

# legacy import (수정 금지)
from diffusion import dispersion
from plume import plume_half_angle
from residence import find_receptors
from mock_residence import mock_buildings
from geo import latlon_to_grid
import mock_forecast

TOOLS = [
    {"name": "get_risk_calendar",
     "description": "블록별 final 위험도(작업 창 가중 반영)를 돌려준다",
     "input_schema": {"type": "object", "properties": {
         "farm_id": {"type": "string"}, "days": {"type": "integer"},
         "work_type": {"type": "string"}}, "required": ["farm_id", "days"]}},
    {"name": "get_forecast",
     "description": "예보 원문", "input_schema": {"type": "object", "properties": {
         "days": {"type": "integer"}}, "required": ["days"]}},
    {"name": "get_storage_days",
     "description": "분뇨 저장 경과일과 2주 임계점 여부", "input_schema": {
         "type": "object", "properties": {"farm_id": {"type": "string"}},
         "required": ["farm_id"]}},
    {"name": "search_rag",
     "description": "법령·매뉴얼 검색 (청크+출처)", "input_schema": {
         "type": "object", "properties": {"query_type": {"type": "string"},
                                          "question": {"type": "string"}},
         "required": ["question"]}},
    {"name": "get_farm_config",
     "description": "농가 설정", "input_schema": {
         "type": "object", "properties": {"farm_id": {"type": "string"}},
         "required": ["farm_id"]}},
    {"name": "get_plume_assessment",
     "description": "지정 시각의 풍하 수용점 수·최악 지점 거리·이탈각·사유",
     "input_schema": {"type": "object", "properties": {
         "farm_id": {"type": "string"}, "datetime": {"type": "string"}},
         "required": ["farm_id", "datetime"]}},
]


class LocalTools:
    """도구 6종의 로컬 구현. API 모드/오프라인 모드가 공유한다."""

    def __init__(self, rag_index=None):
        self.rag = rag_index
        self._recs = None

    def get_risk_calendar(self, farm_id: str, days: int = 3,
                          work_type: str = "청소") -> list[dict]:
        cal = _load_calendar()
        con = db.connect()
        row = con.execute("SELECT last_manure_removal_date FROM farm_config "
                          "WHERE farm_id=?", (farm_id,)).fetchone()
        con.close()
        sdays = None
        if row and row[0]:
            sdays = (datetime.now() - datetime.strptime(row[0], "%Y-%m-%d")).days
        sf = storage_factor(sdays)
        out = []
        for t in sorted(cal):
            wr = window_risk(cal, t)
            if wr is None:
                continue
            out.append({"t": t.strftime("%Y-%m-%d %H시"), "dow": t.weekday(),
                        "final": round(wr * WORK_WEIGHT.get(work_type, 1.0) * sf, 4),
                        "grade": cal[t]["grade"]})
        return out[: days * 8]

    def get_forecast(self, days: int = 3) -> dict:
        nx, ny = latlon_to_grid(DEMO_FARM["lat"], DEMO_FARM["lon"])
        data = mock_forecast.fetch_with_fallback(nx, ny)
        return dict(list(data.items())[: days * 24])

    def get_storage_days(self, farm_id: str) -> dict:
        con = db.connect()
        row = con.execute("SELECT last_manure_removal_date FROM farm_config "
                          "WHERE farm_id=?", (farm_id,)).fetchone()
        con.close()
        if not row or not row[0]:
            return {"days": None, "over_2weeks": False}
        days = (datetime.now() - datetime.strptime(row[0], "%Y-%m-%d")).days
        return {"days": days, "over_2weeks": days >= 14,
                "days_until_threshold": max(0, 14 - days)}

    def search_rag(self, question: str, query_type: str | None = None) -> dict:
        if self.rag is None:
            return {"refused": False, "results": [],
                    "note": "RAG 인덱스 미로딩"}
        return self.rag.search(question, query_type)

    def get_farm_config(self, farm_id: str) -> dict:
        con = db.connect()
        row = con.execute("SELECT * FROM farm_config WHERE farm_id=?",
                          (farm_id,)).fetchone()
        con.close()
        if not row:
            return {}
        keys = ["farm_id", "name", "lat", "lon", "facility_type",
                "last_manure_removal_date"]
        return dict(zip(keys, row))

    def _receptors(self):
        if self._recs is None:
            self._recs, _ = find_receptors(
                DEMO_FARM["lat"], DEMO_FARM["lon"],
                buildings=mock_buildings(DEMO_FARM["lat"], DEMO_FARM["lon"]))
        return self._recs

    def get_plume_assessment(self, farm_id: str, dt_str: str) -> dict:
        dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M")
        nx, ny = latlon_to_grid(DEMO_FARM["lat"], DEMO_FARM["lon"])
        fc = mock_forecast.fetch_with_fallback(nx, ny)
        key = dt.strftime("%Y%m%d %H00")
        if key not in fc:
            return {"error": f"{key} 예보 없음"}
        v = fc[key]
        r = dispersion(v["VEC"], float(v["WSD"]), v["SKY"], dt,
                       DEMO_FARM["lat"], DEMO_FARM["lon"], self._receptors())
        out = {"n_exposed": r.n_exposed, "stability": r.stability,
               "reasons": r.reasons}
        if r.worst is not None:
            out["worst_dist_m"] = round(r.worst.dist_m)
            out["worst_angle_off"] = round(r.worst_angle_off, 1)
            out["half_angle"] = round(plume_half_angle(r.worst.dist_m, r.stability), 1)
        return out
