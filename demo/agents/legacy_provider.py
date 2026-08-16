"""기존 B/C/legacy 데모를 D 에이전트 계약에 맞추는 명시적 adapter.

이 모듈은 기존 팀 코드를 복제하지 않고 호출한다. mock을 쓰는 메서드는 source를
fixture로 표시하며, 연결 실패를 다른 데이터로 조용히 대체하지 않는다.

2026-08-16 수정 — get_risk_calendar() 가 옛 risk_calendar(3시간 블록, 그룹
미분리) 테이블을 보고 있었는데, B가 이미 risk_hourly(1시간, 농촌근거리/
시가지원거리 그룹 분리)로 파이프라인을 이전한 상태라 최신 모델 결과와
연결되지 않는 문제가 있었다. risk_hourly를 직접 읽도록 고쳤다
(그룹 간 보수적 max 조합 후 1h→3h 집계 — D 계약의 resolution='3h' 형식은 유지).
"""
from __future__ import annotations

import sqlite3
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any

from config import DEMO_FARM
from ops import db

# legacy는 기존 프로젝트 규칙에 따라 수정하지 않고 import만 한다.
from diffusion import dispersion
from geo import latlon_to_grid
from mock_residence import mock_buildings
import mock_forecast
from plume import plume_half_angle
from residence import find_receptors


KST = timezone(timedelta(hours=9))
_GRADE_ORDER = {"낮음": 0, "주의": 1, "위험": 2}


class LegacyProvider:
    def __init__(self, *, rag_index: Any = None, storage_days: int | None = None) -> None:
        self.rag = rag_index
        self.generated_at = datetime.now(KST).isoformat(timespec="seconds")

    def _source(
        self, state: str, name: str, *limitations: str, version: str | None = None
    ) -> dict[str, Any]:
        return {
            "state": state, "name": name, "generated_at": self.generated_at,
            "data_as_of": None, "version": version,
            "limitations": list(limitations),
        }

    @staticmethod
    def _response(
        status: str, data: Any, source: dict[str, Any], error: str | None = None
    ) -> dict[str, Any]:
        return {"status": status, "data": data, "source": source, "error": error}

    def get_system_status(self) -> dict[str, Any]:
        con = db.connect()
        try:
            risk_meta = con.execute(
                "SELECT COUNT(*), MAX(updated_at) FROM risk_calendar"
            ).fetchone()
        finally:
            con.close()
        row_count, latest_update = risk_meta or (0, None)
        rag_backend = getattr(self.rag, "backend", "unavailable")
        return self._response(
            "ok",
            {"mode": "legacy", "default_farm_id": DEMO_FARM["farm_id"],
             "official_dataset": "waiting",
             "snapshot_id": f"legacy:{row_count}:{latest_update}:{rag_backend}",
             "components": {"sensor": "unavailable", "risk": "connected_or_empty",
                            "rag": "connected" if self.rag is not None else "unavailable",
                            "plume": "fixture_unverified",
                            "notification": "draft_only"}},
            self._source("connected", "기존 demo B/C adapter",
                         "기상·주거점·플룸 일부는 기존 mock을 사용"),
        )

    def get_sensor_snapshot(self, at: str | None = None) -> dict[str, Any]:
        return self._response(
            "unavailable", None,
            self._source("unavailable", "익산시 측정소 adapter 미연결",
                         "공식 파일과 코드북 도착 후 구현"),
            "현재 C/B 데모에는 익산악취24 측정소 원자료가 없습니다.",
        )

    def get_risk_calendar(
        self, farm_id: str, days: int = 7, work_type: str | None = None
    ) -> dict[str, Any]:
        con = db.connect()
        try:
            rows = con.execute(
                "SELECT date, hour, grp, risk_prob, risk_grade, model_type, updated_at "
                "FROM risk_hourly ORDER BY date, hour"
            ).fetchall()
        except sqlite3.OperationalError:
            rows = []
        finally:
            con.close()
        if not rows:
            return self._response(
                "unavailable", None,
                self._source("unavailable", "B risk_hourly"),
                "risk_hourly가 비어 있습니다. 먼저 B의 serving.daily_scoring.run()을 "
                "실행하세요 (옛 risk_calendar는 현행 파이프라인이 더는 채우지 않습니다).",
            )

        # 그룹(농촌근거리/시가지원거리)별 시각 → 그룹 간 보수적 max 조합
        # (advisor.recommend.combine("max")와 동일 원칙 — 절대규칙 1: 곱하지 않는다)
        by_hour: dict[tuple[str, int], tuple] = {}
        for row in rows:
            date_value, hour, _grp, prob, grade, model_type, updated_at = row
            key = (date_value, int(hour))
            current = by_hour.get(key)
            if current is None or float(prob) > float(current[2]):
                by_hour[key] = (date_value, hour, prob, grade, model_type, updated_at)

        # 1시간 값을 3시간 블록으로 집계 (D 계약의 resolution='3h'/block 0~7 유지 —
        # work_guide._build_windows() 가 이 형태를 그대로 소비한다)
        by_block: dict[tuple[str, int], list[tuple]] = defaultdict(list)
        for (date_value, hour), row in by_hour.items():
            by_block[(date_value, hour // 3)].append(row)

        dates = sorted({date_value for date_value, _ in by_block})[
            : max(1, min(int(days), 7))
        ]
        items: list[dict[str, Any]] = []
        for index, date_text in enumerate(dates, 1):
            blocks_today = sorted(
                block for date_value, block in by_block if date_value == date_text
            )
            if index <= 3:
                for block in blocks_today:
                    block_rows = by_block[(date_text, block)]
                    prob = sum(float(r[2]) for r in block_rows) / len(block_rows)
                    grade = max(
                        (r[3] for r in block_rows),
                        key=lambda g: _GRADE_ORDER.get(g, 0),
                    )
                    start = datetime.strptime(date_text, "%Y-%m-%d").replace(
                        hour=block * 3, tzinfo=KST
                    )
                    items.append({
                        "date": date_text, "block": block, "start": start.isoformat(),
                        "resolution": "3h", "risk_score": round(prob, 4),
                        "risk_grade": grade, "horizon": "D+1~3",
                        "model_type": block_rows[0][4], "model_version": None,
                        "forecast_issued_at": block_rows[0][5],
                        "valid_at": start.isoformat(),
                    })
            else:
                day_rows = [r for block in blocks_today
                           for r in by_block[(date_text, block)]]
                prob = sum(float(r[2]) for r in day_rows) / len(day_rows)
                worst_grade = max(
                    (r[3] for r in day_rows), key=lambda g: _GRADE_ORDER.get(g, 0),
                )
                items.append({
                    "date": date_text, "block": None, "start": None,
                    "resolution": "day", "risk_score": round(prob, 4),
                    "risk_grade": worst_grade, "horizon": "D+4~7",
                    "model_type": day_rows[0][4], "model_version": None,
                    "forecast_issued_at": day_rows[0][5], "valid_at": date_text,
                })
        return self._response(
            "ok", {"farm_id": farm_id, "work_type": work_type, "items": items},
            self._source("connected", "B SQLite risk_hourly (그룹 간 max 조합, 1h→3h 집계)",
                         "그룹(농촌근거리/시가지원거리) 중 보수적으로 위험도가 더 높은 쪽 채택",
                         "D+4~7은 risk_hourly 값을 일 단위로 평균 집계"),
        )

    def get_forecast(self, farm_id: str, days: int) -> dict[str, Any]:
        farm = self._farm_row(farm_id) or DEMO_FARM
        nx, ny = latlon_to_grid(float(farm["lat"]), float(farm["lon"]))
        raw = mock_forecast.fetch_with_fallback(nx, ny)
        items = []
        for key, value in list(sorted(raw.items()))[: max(1, min(days, 7)) * 24]:
            items.append({"valid_at": datetime.strptime(key, "%Y%m%d %H%M").replace(
                              tzinfo=KST).isoformat(),
                          "temperature_c": float(value["TMP"]),
                          "wind_speed_ms": float(value["WSD"]),
                          "wind_direction_degree": float(value["VEC"]),
                          "sky": value.get("SKY")})
        return self._response(
            "ok", {"farm_id": farm_id, "items": items,
                   "forecast_issued_at": self.generated_at},
            self._source("fixture", "legacy/mock_forecast.py",
                         "기상청 실 API가 아니라 기존 고정 시나리오"),
        )

    def get_storage_days(self, farm_id: str) -> dict[str, Any]:
        farm = self._farm_row(farm_id)
        if not farm or not farm.get("last_manure_removal_date"):
            days = None
        else:
            days = (datetime.now() - datetime.strptime(
                farm["last_manure_removal_date"], "%Y-%m-%d")).days
        return self._response(
            "ok", {"farm_id": farm_id, "days": days,
                   "over_2weeks": bool(days is not None and days >= 14),
                   "days_until_threshold": None if days is None else max(0, 14 - days)},
            self._source("connected", "B farm_config",
                         "14일 기준과 1.5배 가중치는 잠정 가정값 [C]"),
        )

    def search_rag(
        self, question: str, query_type: str | None = None
    ) -> dict[str, Any]:
        if self.rag is None:
            return self._response(
                "unavailable", None,
                self._source("unavailable", "C RagIndex"),
                "RAG 인덱스가 주입되지 않았습니다.",
            )
        data = self.rag.search(question, query_type, k=3, boost=True)
        status = "refused" if data.get("refused") else "ok"
        return self._response(
            status, data,
            self._source("connected", f"C RagIndex ({data.get('backend', 'unknown')})",
                         "검색 score는 신뢰확률이 아님"),
        )

    def get_farm_config(self, farm_id: str) -> dict[str, Any]:
        farm = self._farm_row(farm_id)
        if not farm:
            return self._response(
                "unavailable", None, self._source("unavailable", "B farm_config"),
                f"농가를 찾을 수 없습니다: {farm_id}",
            )
        return self._response(
            "ok", farm, self._source("connected", "B farm_config",
                                      "현재 데모 농가는 가상 농가"),
        )

    def get_plume_assessment(self, farm_id: str, when: str) -> dict[str, Any]:
        farm = self._farm_row(farm_id) or DEMO_FARM
        dt = datetime.fromisoformat(when)
        recs, _ = find_receptors(
            float(farm["lat"]), float(farm["lon"]),
            buildings=mock_buildings(float(farm["lat"]), float(farm["lon"])),
        )
        nx, ny = latlon_to_grid(float(farm["lat"]), float(farm["lon"]))
        forecast = mock_forecast.fetch_with_fallback(nx, ny)
        key = dt.strftime("%Y%m%d %H00")
        if key not in forecast:
            return self._response(
                "unavailable", None,
                self._source("unavailable", "legacy plume"), f"{key} 예보 없음",
            )
        value = forecast[key]
        result = dispersion(value["VEC"], float(value["WSD"]), value["SKY"], dt,
                            float(farm["lat"]), float(farm["lon"]), recs)
        data: dict[str, Any] = {
            "farm_id": farm_id, "when": when, "n_exposed": result.n_exposed,
            "n_receptors": len(recs), "plume_status": "unverified",
            "affects_risk_grade": False, "audience_is_mock": True,
            "stability": result.stability, "reasons": result.reasons,
        }
        if result.worst is not None:
            data.update({"worst_dist_m": round(result.worst.dist_m),
                         "sector_half_angle": round(
                             plume_half_angle(result.worst.dist_m, result.stability), 1)})
        return self._response(
            "ok", data,
            self._source("fixture", "legacy plume + mock residence/forecast",
                         "미검증 참고이며 등급·추천 순위에 미반영"),
        )

    @staticmethod
    def _farm_row(farm_id: str) -> dict[str, Any] | None:
        con = db.connect()
        try:
            row = con.execute(
                "SELECT farm_id, name, lat, lon, facility_type, "
                "last_manure_removal_date FROM farm_config WHERE farm_id=?",
                (farm_id,),
            ).fetchone()
        finally:
            con.close()
        if not row:
            return None
        keys = ("farm_id", "name", "lat", "lon", "facility_type",
                "last_manure_removal_date")
        return dict(zip(keys, row))
