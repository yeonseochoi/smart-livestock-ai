"""기존 B/C/legacy 데모를 D 에이전트 계약에 맞추는 명시적 adapter.

이 모듈은 기존 팀 코드를 복제하지 않고 호출한다. mock을 쓰는 메서드는 source를
fixture로 표시하며, 연결 실패를 다른 데이터로 조용히 대체하지 않는다.

현행 ``serving.db``의 ``risk_hourly``를 읽어 농촌근거리/시가지원거리 중
시간별 최대 위험도를 선택한다. 단기예보 구간은 1시간 해상도를 그대로 유지하고,
중기예보 구간은 일 단위로만 노출한다.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from agents.contracts import SensorObservation
from config import DEMO_FARM
from serving import db

# legacy는 기존 프로젝트 규칙에 따라 수정하지 않고 import만 한다.
from legacy.diffusion import dispersion
from legacy.geo import latlon_to_grid
from legacy.mock_residence import mock_buildings
from legacy import mock_forecast
from legacy.plume import plume_half_angle
from legacy.residence import find_receptors


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
        con = None
        try:
            con = db.connect()
            risk_meta = con.execute(
                "SELECT COUNT(*), MAX(updated_at) FROM risk_hourly"
            ).fetchone()
        except Exception as exc:
            return self._response(
                "unavailable", None,
                self._source("unavailable", "B risk_hourly"),
                f"서빙 DB 상태를 확인할 수 없습니다: {type(exc).__name__}: {exc}",
            )
        finally:
            if con is not None:
                con.close()
        row_count, latest_update = risk_meta or (0, None)
        rag_backend = getattr(self.rag, "backend", "unavailable")
        return self._response(
            "ok",
             {"mode": "legacy", "default_farm_id": DEMO_FARM["farm_id"],
              "official_dataset": "waiting",
              "snapshot_id": f"legacy:{row_count}:{latest_update}:{rag_backend}",
              "components": {"sensor": "unavailable", "risk": "connected_or_empty",
                            "rag": "connected" if self.rag is not None else "auto"}},
            self._source("connected", f"B/C adapter · {db.describe()}"),
        )

    def get_sensor_snapshot(self, at: str | None = None) -> dict[str, Any]:
        """익산악취24 원자료 adapter가 아직 없어 발표용 고정 시연값을 쓴다.

        [C] 2026-08-26: 공식 파일·코드북 도착 전까지 이 메서드만 실측 adapter로
        바꾸면 되도록 반환 계약(SensorObservation)은 그대로 두고 값만 고정한다.
        "연결 실패를 조용히 다른 데이터로 대체하지 않는다"는 이 파일의 원칙과
        다르지 않다 — 이건 실패를 감추는 게 아니라 원래 없는 기능을 발표용으로
        명시적으로 채운 것이라, source.state를 "fixture"로 정직하게 표시해
        화면에서 CONNECTED와 구분되게 한다.
        """
        observed = datetime.now(KST).replace(minute=0, second=0, microsecond=0) - timedelta(hours=1)
        ingested = observed + timedelta(minutes=5)
        raw = (
            ("S-DEMO-01", "익산 왕궁 측정소(예시)", 35.9518, 126.9574,
             0.001, 0.002, 0.191, 1.8, 30.7, 58.9, "동남동", 2.2, "unverified"),
            ("S-DEMO-02", "익산 춘포 측정소(예시)", 35.9587, 126.9880,
             0.002, 0.004, 0.224, 2.1, 30.2, 61.4, "남동", 1.7, "unverified"),
            ("S-DEMO-03", "익산 삼기 측정소(예시)", 35.9460, 127.0140,
             None, None, None, None, 29.8, 60.1, "동남동", 2.4, "missing"),
        )
        stations = []
        for row in raw:
            station = SensorObservation(
                observation_id=f"{row[0]}:{observed.isoformat()}",
                station_id=row[0],
                station_name=row[1],
                latitude=row[2],
                longitude=row[3],
                observed_at=observed.isoformat(),
                ingested_at=ingested.isoformat(),
                h2s_ppm=row[4],
                nh3_ppm=row[5],
                tvoc_ppm=row[6],
                complex_odor_value=row[7],
                complex_odor_unit=None,
                temperature_c=row[8],
                humidity_pct=row[9],
                wind_direction_text=row[10],
                wind_speed_ms=row[11],
                record_qc=row[12],
                quality_flags=("fixture", "presentation_placeholder"),
            )
            stations.append(station.to_dict())
        return self._response(
            "ok",
            {
                "schema_version": "sensor-observation-v1",
                "requested_at": at,
                "observed_at": observed.isoformat(),
                "stations": stations,
            },
            self._source(
                "fixture", "익산악취24 예상 스키마 · 발표용 고정값",
                "측정소명·좌표·수치는 실제 익산시 자료가 아님",
                "공식 파일과 코드북 도착 후 실측 adapter로 교체 예정",
                version="presentation-fixture-v1",
            ),
        )

    def get_risk_calendar(
        self, farm_id: str, days: int = 7, work_type: str | None = None
    ) -> dict[str, Any]:
        con = None
        try:
            con = db.connect()
            rows = con.execute(
                "SELECT date, hour, grp, risk_prob, risk_grade, model_type, updated_at "
                "FROM risk_hourly "
                "WHERE updated_at=(SELECT MAX(updated_at) FROM risk_hourly) "
                "ORDER BY date, hour"
            ).fetchall()
        except Exception as exc:
            return self._response(
                "unavailable", None,
                self._source("unavailable", f"B risk_hourly · {db.describe()}"),
                f"risk_hourly 조회 실패: {type(exc).__name__}: {exc}",
            )
        finally:
            if con is not None:
                con.close()
        if not rows:
            return self._response(
                "unavailable", None,
                self._source("unavailable", "B risk_hourly"),
                "risk_hourly가 비어 있습니다. 먼저 B의 serving.daily_scoring.run()을 "
                "실행하세요 (옛 risk_calendar는 현행 파이프라인이 더는 채우지 않습니다).",
            )

        # 그룹(농촌근거리/시가지원거리) 중 시간별 위험도가 높은 행을 선택한다.
        # 두 확률을 곱하지 않는 현행 advisor의 보수적 max 원칙과 같다.
        by_hour: dict[tuple[str, int], tuple[Any, ...]] = {}
        for row in rows:
            date_value, hour, _grp, prob, grade, model_type, updated_at = row
            key = (date_value, int(hour))
            current = by_hour.get(key)
            if current is None or float(prob) > float(current[3]):
                by_hour[key] = tuple(row)

        dates = sorted({date_value for date_value, _ in by_hour})[
            : max(1, min(int(days), 7))
        ]
        items: list[dict[str, Any]] = []
        for date_text in dates:
            day_rows = [by_hour[(date_text, hour)] for hour in sorted(
                hour for day, hour in by_hour if day == date_text
            )]
            is_midterm = bool(day_rows) and all(row[5] == "reduced" for row in day_rows)
            if not is_midterm:
                for row in day_rows:
                    _, hour, _grp, prob, grade, model_type, updated_at = row
                    start = datetime.strptime(date_text, "%Y-%m-%d").replace(
                        hour=int(hour), tzinfo=KST
                    )
                    items.append({
                        "date": date_text, "hour": int(hour), "start": start.isoformat(),
                        "resolution": "1h", "risk_score": round(float(prob), 4),
                        "risk_grade": grade, "horizon": "D+1~3",
                        "model_type": model_type, "model_version": None,
                        "forecast_issued_at": updated_at,
                        "valid_at": start.isoformat(),
                    })
            else:
                prob = sum(float(row[3]) for row in day_rows) / len(day_rows)
                worst_grade = max(
                    (row[4] for row in day_rows), key=lambda g: _GRADE_ORDER.get(g, 0),
                )
                items.append({
                    "date": date_text, "hour": None, "start": None,
                    "resolution": "day", "risk_score": round(prob, 4),
                    "risk_grade": worst_grade, "horizon": "D+4~7",
                    "model_type": day_rows[0][5], "model_version": None,
                    "forecast_issued_at": day_rows[0][6], "valid_at": date_text,
                })
        return self._response(
            "ok", {"farm_id": farm_id, "work_type": work_type, "items": items},
            self._source("connected", f"B risk_hourly · {db.describe()}",
                         "가장 최근 updated_at 배치만 사용",
                         "그룹(농촌근거리/시가지원거리) 중 보수적으로 위험도가 더 높은 쪽 채택",
                         "단기는 1시간, reduced 중기는 일 단위로 제공"),
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
            try:
                from agents.rag_adapter import RagYujinAdapter

                self.rag = RagYujinAdapter()
            except Exception as exc:
                return self._response(
                    "unavailable", None,
                    self._source("unavailable", "C RAG (rag_yujin)"),
                    f"RAG 인덱스를 열 수 없습니다: {type(exc).__name__}: {exc}",
                )
        try:
            data = self.rag.search(question, query_type, k=3, boost=True)
        except Exception as exc:
            return self._response(
                "unavailable", None,
                self._source("unavailable", "C RAG (rag_yujin)"),
                f"RAG 검색 실패: {type(exc).__name__}: {exc}",
            )
        status = "refused" if data.get("refused") else "ok"
        return self._response(
            status, data,
            self._source("connected", f"C RAG ({data.get('backend', 'unknown')})",
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
