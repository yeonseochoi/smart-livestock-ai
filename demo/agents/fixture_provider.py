"""공식 정보공개 자료 도착 전 사용하는 결정론적 D 시연 provider."""
from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from typing import Any

from agents.contracts import SensorObservation, SourceInfo


KST = timezone(timedelta(hours=9))


class FixtureProvider:
    """난수 없이 같은 화면 흐름을 재현하는 명시적 fixture.

    실제 익산시 관측·민원·예측 결과로 오인되지 않도록 모든 응답의 source.state를
    ``fixture``로 고정한다.
    """

    FARM_ID = "F-DEMO-001"

    def __init__(self, *, today: date | None = None, storage_days: int = 12) -> None:
        self.today = today or datetime.now(KST).date()
        self.storage_days = int(storage_days)
        self.generated_at = datetime.combine(
            self.today, time(hour=6), tzinfo=KST
        ).isoformat()

    def _source(self, name: str, *limitations: str) -> dict[str, Any]:
        return SourceInfo(
            state="fixture",
            name=name,
            generated_at=self.generated_at,
            data_as_of=self.today.isoformat(),
            version="fixture-v2",
            limitations=(
                "정보공개청구 자료 반영 전 구조 검증용 시연 데이터",
                *limitations,
            ),
        ).to_dict()

    @staticmethod
    def _grade(score: float) -> str:
        if score < 0.18:
            return "낮음"
        if score < 0.32:
            return "주의"
        return "위험"

    def get_system_status(self) -> dict[str, Any]:
        return self._ok(
            {
                "mode": "fixture",
                "default_farm_id": self.FARM_ID,
                "official_dataset": "waiting",
                "snapshot_id": f"fixture-v2:{self.today.isoformat()}",
                "components": {
                    "sensor": "fixture",
                    "risk": "fixture",
                    "rag": "fixture",
                    "plume": "fixture_unverified",
                },
            },
            "D 고정 시연 시나리오",
        )

    def get_sensor_snapshot(self, at: str | None = None) -> dict[str, Any]:
        observed = datetime.combine(
            self.today - timedelta(days=1), time(hour=19, minute=30), tzinfo=KST
        )
        ingested = observed + timedelta(minutes=5)
        raw = (
            ("S-DEMO-01", "시연 측정소 01", 35.9518, 126.9574,
             0.001, 0.002, 0.191, 1.8, 30.7, 58.9, "동남동", 2.2, "unverified"),
            ("S-DEMO-02", "시연 측정소 02", 35.9587, 126.9880,
             0.002, 0.004, 0.224, 2.1, 30.2, 61.4, "남동", 1.7, "unverified"),
            ("S-DEMO-03", "시연 측정소 03", 35.9460, 127.0140,
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
                quality_flags=("fixture", "unit_dictionary_pending"),
            )
            stations.append(station.to_dict())
        return self._ok(
            {
                "schema_version": "sensor-observation-v1",
                "requested_at": at,
                "observed_at": observed.isoformat(),
                "stations": stations,
            },
            "익산악취24 예상 스키마 fixture",
            "측정소명·좌표·수치는 실제 익산시 자료가 아님",
            "복합악취 값의 의미·단위는 원자료 코드북 수령 후 확정",
        )

    def get_risk_calendar(
        self, farm_id: str, days: int = 7, work_type: str | None = None
    ) -> dict[str, Any]:
        if farm_id != self.FARM_ID:
            return self._missing("농가", farm_id)
        days = max(1, min(int(days), 7))
        base = (0.29, 0.23, 0.14, 0.10, 0.16, 0.25, 0.37, 0.43)
        hourly_adjust = (0.015, 0.0, -0.01)
        adjust = (0.00, 0.03, -0.02)
        items: list[dict[str, Any]] = []
        for offset in range(1, days + 1):
            target = self.today + timedelta(days=offset)
            if offset <= 3:
                for hour in range(24):
                    raw_score = base[hour // 3] + hourly_adjust[hour % 3]
                    score = max(0.01, min(0.95, raw_score + adjust[offset - 1]))
                    start = datetime.combine(
                        target, time(hour=hour), tzinfo=KST
                    )
                    items.append({
                        "date": target.isoformat(), "hour": hour,
                        "start": start.isoformat(), "resolution": "1h",
                        "risk_score": round(score, 4),
                        "risk_grade": self._grade(score), "horizon": "D+1~3",
                        "model_type": "fixture-full", "model_version": "fixture-v2",
                        "forecast_issued_at": self.generated_at,
                        "valid_at": start.isoformat(),
                    })
            else:
                score = round(sum(base) / len(base) + (offset - 5) * 0.015, 4)
                items.append({
                    "date": target.isoformat(), "hour": None, "start": None,
                    "resolution": "day", "risk_score": score,
                    "risk_grade": self._grade(score), "horizon": "D+4~7",
                    "model_type": "fixture-reduced", "model_version": "fixture-v2",
                    "forecast_issued_at": self.generated_at,
                    "valid_at": target.isoformat(),
                })
        return self._ok(
            {"farm_id": farm_id, "work_type": work_type, "items": items},
            "D 위험 캘린더 fixture",
            "점수·등급은 화면과 계약 검증용이며 모델 성능을 뜻하지 않음",
        )

    def get_forecast(self, farm_id: str, days: int) -> dict[str, Any]:
        if farm_id != self.FARM_ID:
            return self._missing("농가", farm_id)
        rows = [
            {
                "date": (self.today + timedelta(days=i)).isoformat(),
                "temperature_c": 25 + i % 3,
                "humidity_pct": 68 + i * 2,
                "wind_speed_ms": round(1.2 + i * 0.2, 1),
                "resolution": "3h" if i <= 3 else "day",
            }
            for i in range(1, max(1, min(int(days), 7)) + 1)
        ]
        return self._ok(
            {"farm_id": farm_id, "items": rows, "forecast_issued_at": self.generated_at},
            "D 기상 fixture", "기상청 API 응답이 아닌 고정 시연값",
        )

    def get_storage_days(self, farm_id: str) -> dict[str, Any]:
        if farm_id != self.FARM_ID:
            return self._missing("농가", farm_id)
        return self._ok(
            {
                "farm_id": farm_id, "days": self.storage_days,
                "over_2weeks": self.storage_days >= 14,
                "days_until_threshold": max(0, 14 - self.storage_days),
            },
            "사용자 입력 fixture", "14일 기준과 1.5배 가중치는 잠정 가정값 [C]",
        )

    def search_rag(
        self, question: str, query_type: str | None = None
    ) -> dict[str, Any]:
        legal_case = any(token in question for token in ("고발", "처벌받", "불법인가", "소송"))
        if legal_case:
            return self._response(
                "refused",
                {"refused": True, "results": [], "answer": (
                    "개별 사안의 법률 판단은 제공하지 않습니다. 법률구조공단 132 "
                    "또는 관할 행정기관에 확인해 주세요."
                )},
                "C 연결 전 RAG fixture",
                "실제 검색 결과나 법률 자문이 아님",
            )
        results = [
            {
                "rank": 1, "id": "fixture-manual-1",
                "source_file": "축산농장 악취저감시설 운영 매뉴얼(돼지)",
                "doc": "축산농장 악취저감시설 운영 매뉴얼(돼지)",
                "unit": "시연용 근거 자리표시자", "page": None,
                "hier": "매뉴얼", "score": 0.78,
                "score_kind": "검색 유사도(신뢰확률 아님)",
                "snippet": "작업 전 저감시설 상태 확인과 작업 후 세척 근거가 표시될 자리입니다.",
            },
            {
                "rank": 2, "id": "fixture-ordinance-1",
                "source_file": "익산시 악취방지 및 저감 조례",
                "doc": "익산시 악취방지 및 저감 조례",
                "unit": "시연용 근거 자리표시자", "page": None,
                "hier": "조례", "score": 0.64,
                "score_kind": "검색 유사도(신뢰확률 아님)",
                "snippet": "C 인덱스 연결 후 조문·쪽수·원문 일부로 교체됩니다.",
            },
        ]
        return self._ok(
            {"refused": False, "query_type": query_type, "backend": "fixture",
             "results": results, "notice": "법령·매뉴얼 실제 인덱스 미연결"},
            "C 연결 전 RAG fixture", "실제 검색 결과나 법률 자문이 아님",
        )

    def get_farm_config(self, farm_id: str) -> dict[str, Any]:
        if farm_id != self.FARM_ID:
            return self._missing("농가", farm_id)
        return self._ok(
            {"farm_id": farm_id, "name": "왕궁 시연 농가",
             "region": "익산시 왕궁면(가상)", "facility_type": "양돈(시연)"},
            "왕궁 가상 농가 fixture", "실제 농가를 나타내지 않음",
        )

    def get_plume_assessment(self, farm_id: str, when: str) -> dict[str, Any]:
        if farm_id != self.FARM_ID:
            return self._missing("농가", farm_id)
        datetime.fromisoformat(when)
        return self._ok(
            {"farm_id": farm_id, "when": when, "n_exposed": 8,
             "n_receptors": 24, "wind_to_degree": 245,
             "sector_half_angle": 18, "plume_status": "unverified",
             "affects_risk_grade": False, "audience_is_mock": True},
            "플룸 영향권 fixture", "가상 주거점 기반",
            "검증 미완료로 민원 위험 등급에 반영하지 않음",
        )

    def _ok(self, data: Any, name: str, *limitations: str) -> dict[str, Any]:
        return self._response("ok", data, name, *limitations)

    def _response(
        self, status: str, data: Any, name: str, *limitations: str
    ) -> dict[str, Any]:
        return {"status": status, "data": data,
                "source": self._source(name, *limitations), "error": None}

    def _missing(self, kind: str, value: str) -> dict[str, Any]:
        return {"status": "unavailable", "data": None,
                "source": self._source("D fixture"),
                "error": f"시연 {kind}을 찾을 수 없습니다: {value}"}
