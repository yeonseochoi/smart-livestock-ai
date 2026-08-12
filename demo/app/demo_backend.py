"""공식 정보공개 데이터 도착 전 사용하는 결정론적 시연 provider.

임의 난수를 사용하지 않는다. 모든 반환값에는 fixture 표시를 붙여 실제 익산시
예측 결과로 오인되지 않게 한다.
"""
from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from typing import Any

from .contracts import SensorObservation, SourceInfo


KST = timezone(timedelta(hours=9))


class DemoBackend:
    FARM_ID = "F-DEMO-001"

    def __init__(self, today: date | None = None, storage_days: int = 12) -> None:
        self.today = today or datetime.now(KST).date()
        self.storage_days = storage_days
        self._generated_at = datetime.combine(
            self.today, time(hour=6), tzinfo=KST
        ).isoformat()

    def _source(
        self,
        name: str,
        *limitations: str,
        version: str = "fixture-v1",
        data_as_of: str | None = None,
    ) -> SourceInfo:
        return SourceInfo(
            state="fixture",
            name=name,
            generated_at=self._generated_at,
            data_as_of=data_as_of or self.today.isoformat(),
            version=version,
            limitations=(
                "정보공개청구 자료 반영 전 구조 검증용 시연 데이터",
                *limitations,
            ),
        )

    @staticmethod
    def _grade(score: float) -> str:
        if score < 0.18:
            return "낮음"
        if score < 0.32:
            return "주의"
        return "위험"

    def get_system_status(self) -> dict[str, Any]:
        source = self._source("D 고정 시연 시나리오")
        return {
            "status": "ok",
            "data": {
                "mode": "fixture",
                "default_farm_id": self.FARM_ID,
                "official_dataset": "waiting",
                "components": {
                    "sensor_map": "fixture",
                    "risk_calendar": "fixture",
                    "rag": "fixture",
                    "plume": "fixture_unverified",
                    "notification": "draft_only",
                },
            },
            "source": source.to_dict(),
            "error": None,
        }

    def get_sensor_snapshot(self, at: str | None = None) -> dict[str, Any]:
        """익산악취24 화면에서 예상한 측정소 최신 관측의 고정 fixture."""

        snapshot_day = self.today - timedelta(days=1)
        observed_at = datetime.combine(
            snapshot_day, time(hour=19, minute=30), tzinfo=KST
        ).isoformat()
        ingested_at = datetime.combine(
            snapshot_day, time(hour=19, minute=35), tzinfo=KST
        ).isoformat()
        observations = (
            SensorObservation(
                observation_id=f"S-DEMO-01:{observed_at}",
                station_id="S-DEMO-01",
                station_number="01",
                station_name="시연 측정소 01",
                station_type="주거 인접",
                latitude=35.9518,
                longitude=126.9574,
                coord_quality="approximate_fixture",
                observed_at=observed_at,
                ingested_at=ingested_at,
                status_label="포집준비",
                h2s_ppm=0.001,
                nh3_ppm=0.002,
                tvoc_ppm=0.191,
                complex_odor_value=1.8,
                complex_odor_unit=None,
                temperature_c=30.7,
                humidity_pct=58.9,
                wind_direction_text="동남동",
                wind_direction_degree=None,
                wind_speed_ms=2.2,
                record_qc="unverified",
                quality_flags=("fixture", "unit_dictionary_pending"),
                source_row_id="fixture-row-01",
                data_version="fixture-v1",
            ),
            SensorObservation(
                observation_id=f"S-DEMO-02:{observed_at}",
                station_id="S-DEMO-02",
                station_number="02",
                station_name="시연 측정소 02",
                station_type="산업단지 인접",
                latitude=35.9587,
                longitude=126.9880,
                coord_quality="approximate_fixture",
                observed_at=observed_at,
                ingested_at=ingested_at,
                status_label="측정중",
                h2s_ppm=0.002,
                nh3_ppm=0.004,
                tvoc_ppm=0.224,
                complex_odor_value=2.1,
                complex_odor_unit=None,
                temperature_c=30.2,
                humidity_pct=61.4,
                wind_direction_text="남동",
                wind_direction_degree=None,
                wind_speed_ms=1.7,
                record_qc="unverified",
                quality_flags=("fixture", "unit_dictionary_pending"),
                source_row_id="fixture-row-02",
                data_version="fixture-v1",
            ),
            SensorObservation(
                observation_id=f"S-DEMO-03:{observed_at}",
                station_id="S-DEMO-03",
                station_number="03",
                station_name="시연 측정소 03",
                station_type="도심",
                latitude=35.9460,
                longitude=127.0140,
                coord_quality="approximate_fixture",
                observed_at=observed_at,
                ingested_at=ingested_at,
                status_label="측정중",
                h2s_ppm=0.001,
                nh3_ppm=0.003,
                tvoc_ppm=0.158,
                complex_odor_value=1.2,
                complex_odor_unit=None,
                temperature_c=29.8,
                humidity_pct=60.1,
                wind_direction_text="동남동",
                wind_direction_degree=None,
                wind_speed_ms=2.4,
                record_qc="unverified",
                quality_flags=("fixture", "unit_dictionary_pending"),
                source_row_id="fixture-row-03",
                data_version="fixture-v1",
            ),
            SensorObservation(
                observation_id=f"S-DEMO-04:{observed_at}",
                station_id="S-DEMO-04",
                station_number="04",
                station_name="시연 측정소 04",
                station_type="산업시설 인접",
                latitude=35.9635,
                longitude=127.0380,
                coord_quality="approximate_fixture",
                observed_at=observed_at,
                ingested_at=ingested_at,
                status_label="점검중",
                h2s_ppm=None,
                nh3_ppm=None,
                tvoc_ppm=None,
                complex_odor_value=None,
                complex_odor_unit=None,
                temperature_c=29.5,
                humidity_pct=62.0,
                wind_direction_text="남동",
                wind_direction_degree=None,
                wind_speed_ms=1.9,
                record_qc="missing",
                quality_flags=("fixture", "measurement_missing"),
                source_row_id="fixture-row-04",
                data_version="fixture-v1",
            ),
        )
        facilities = [
            {
                "site_id": "P-DEMO-01",
                "site_name": "시연 사업장 A",
                "site_type": "산업시설(가상)",
                "latitude": 35.9550,
                "longitude": 127.0010,
            },
            {
                "site_id": "P-DEMO-02",
                "site_name": "시연 사업장 B",
                "site_type": "처리시설(가상)",
                "latitude": 35.9435,
                "longitude": 127.0250,
            },
        ]
        source = self._source(
            "익산악취24 예상 스키마 fixture",
            "측정소명·좌표·수치는 실제 익산시 관측값이 아닌 화면 검증용 가상 값",
            "복합악취 값의 의미와 단위는 원자료 코드북 수령 후 확정",
            data_as_of=snapshot_day.isoformat(),
        )
        return {
            "status": "ok",
            "data": {
                "schema_version": "sensor-observation-v1",
                "requested_at": at,
                "observed_at": observed_at,
                "stations": [item.to_dict() for item in observations],
                "facilities": facilities,
            },
            "source": source.to_dict(),
            "error": None,
        }

    def get_risk_calendar(
        self, farm_id: str, days: int = 7, work_type: str | None = None
    ) -> dict[str, Any]:
        if farm_id != self.FARM_ID:
            return self._not_found("농가", farm_id)
        days = max(1, min(int(days), 7))
        source = self._source(
            "D 위험 캘린더 fixture",
            "점수·등급은 화면 및 인터페이스 검증용이며 모델 성능을 의미하지 않음",
        )
        # 새벽/오전은 낮고 저녁은 높아 시연 흐름이 매번 동일하게 재현된다.
        base = (0.29, 0.23, 0.14, 0.10, 0.16, 0.25, 0.37, 0.43)
        day_adjust = (0.00, 0.03, -0.02)
        items: list[dict[str, Any]] = []
        for offset in range(1, days + 1):
            target = self.today + timedelta(days=offset)
            if offset <= 3:
                for block, raw in enumerate(base):
                    score = min(0.95, max(0.01, raw + day_adjust[offset - 1]))
                    start = datetime.combine(
                        target, time(hour=block * 3), tzinfo=KST
                    )
                    items.append(
                        {
                            "date": target.isoformat(),
                            "block": block,
                            "start": start.isoformat(),
                            "resolution": "3h",
                            "risk_score": round(score, 4),
                            "risk_grade": self._grade(score),
                            "horizon": "D+1~3",
                            "model_type": "demo_full_placeholder",
                            "model_version": "not-trained",
                            "forecast_issued_at": self._generated_at,
                            "updated_at": self._generated_at,
                        }
                    )
            else:
                score = (0.19, 0.27, 0.34, 0.22)[offset - 4]
                items.append(
                    {
                        "date": target.isoformat(),
                        "block": None,
                        "start": None,
                        "resolution": "day",
                        "risk_score": score,
                        "risk_grade": self._grade(score),
                        "horizon": "D+4~7",
                        "model_type": "demo_reduced_placeholder",
                        "model_version": "not-trained",
                        "forecast_issued_at": self._generated_at,
                        "updated_at": self._generated_at,
                    }
                )
        return {
            "status": "ok",
            "data": {
                "farm_id": farm_id,
                "work_type": work_type,
                "items": items,
                "score_label": "시연용 상대 민원 위험지수",
            },
            "source": source.to_dict(),
            "error": None,
        }

    def get_forecast(self, farm_id: str, days: int = 7) -> dict[str, Any]:
        if farm_id != self.FARM_ID:
            return self._not_found("농가", farm_id)
        days = max(1, min(int(days), 7))
        rows = []
        for offset in range(1, days + 1):
            rows.append(
                {
                    "date": (self.today + timedelta(days=offset)).isoformat(),
                    "temperature_c": 25 + (offset % 3),
                    "humidity_pct": 68 + offset * 2,
                    "wind_speed_ms": round(1.2 + offset * 0.2, 1),
                    "resolution": "3h" if offset <= 3 else "day",
                }
            )
        source = self._source(
            "D 기상 fixture", "기상청 API 응답이 아니라 시연용 고정 값"
        )
        return {
            "status": "ok",
            "data": {"farm_id": farm_id, "items": rows},
            "source": source.to_dict(),
            "error": None,
        }

    def get_storage_days(self, farm_id: str) -> dict[str, Any]:
        if farm_id != self.FARM_ID:
            return self._not_found("농가", farm_id)
        source = self._source(
            "사용자 입력 fixture", "14일 임계와 1.5배 가중치는 잠정 가정값 [C]"
        )
        return {
            "status": "ok",
            "data": {
                "farm_id": farm_id,
                "days": self.storage_days,
                "over_2weeks": self.storage_days >= 14,
                "days_until_threshold": max(0, 14 - self.storage_days),
            },
            "source": source.to_dict(),
            "error": None,
        }

    def search_rag(
        self, question: str, query_type: str | None = None
    ) -> dict[str, Any]:
        source = self._source(
            "C 연결 전 RAG fixture",
            "아래 문구는 UI 계약 검증용 요약이며 실제 검색 결과나 법률 자문이 아님",
        )
        individualized = any(
            token in question for token in ("고발", "처벌받", "불법인가", "소송")
        )
        if individualized:
            return {
                "status": "refused",
                "data": {
                    "refused": True,
                    "answer": (
                        "개별 사안의 법률 판단은 제공하지 않습니다. 법률구조공단 "
                        "132 또는 관할 행정기관에 확인해 주세요."
                    ),
                    "results": [],
                },
                "source": source.to_dict(),
                "error": None,
            }
        results = [
            {
                "rank": 1,
                "id": "fixture-manual-1",
                "doc": "축산농장 악취저감시설 운영 매뉴얼(돼지)",
                "unit": "시연용 근거 자리표시자",
                "page": None,
                "page_end": None,
                "hier": "매뉴얼",
                "score": 0.78,
                "score_kind": "검색 유사도(신뢰확률 아님)",
                "snippet": (
                    "작업 전 저감시설 상태를 확인하고 작업 후 시설을 세척한다는 "
                    "형태의 근거가 이 위치에 표시됩니다."
                ),
            },
            {
                "rank": 2,
                "id": "fixture-ordinance-1",
                "doc": "익산시 악취방지 및 저감 조례",
                "unit": "시연용 근거 자리표시자",
                "page": None,
                "page_end": None,
                "hier": "조례",
                "score": 0.64,
                "score_kind": "검색 유사도(신뢰확률 아님)",
                "snippet": "실제 C 인덱스 연결 후 조문·쪽수·원문 일부로 교체됩니다.",
            },
        ]
        return {
            "status": "ok",
            "data": {
                "refused": False,
                "query_type": query_type,
                "backend": "fixture",
                "results": results,
                "notice": "법령·매뉴얼 실제 검색 인덱스 미연결",
            },
            "source": source.to_dict(),
            "error": None,
        }

    def get_farm_config(self, farm_id: str) -> dict[str, Any]:
        if farm_id != self.FARM_ID:
            return self._not_found("농가", farm_id)
        source = self._source("왕궁 가상 농가 fixture", "실제 농가를 나타내지 않음")
        return {
            "status": "ok",
            "data": {
                "farm_id": farm_id,
                "name": "왕궁 시연 농가",
                "region": "익산시 왕궁면(가상)",
                "facility_type": "양돈(시연)",
            },
            "source": source.to_dict(),
            "error": None,
        }

    def get_plume_assessment(self, farm_id: str, when: str) -> dict[str, Any]:
        if farm_id != self.FARM_ID:
            return self._not_found("농가", farm_id)
        # ISO 형식 여부만 검증한다. 결과는 등급과 추천 순위에 절대 반영하지 않는다.
        datetime.fromisoformat(when)
        source = self._source(
            "플룸 영향권 fixture",
            "가상 주거점 기반",
            "검증 미완료로 민원 위험 등급에 반영하지 않음",
        )
        return {
            "status": "ok",
            "data": {
                "farm_id": farm_id,
                "when": when,
                "n_exposed": 8,
                "n_receptors": 24,
                "wind_to_degree": 245,
                "sector_half_angle": 18,
                "plume_status": "unverified",
                "affects_risk_grade": False,
                "audience_is_mock": True,
            },
            "source": source.to_dict(),
            "error": None,
        }

    def _not_found(self, kind: str, value: str) -> dict[str, Any]:
        return {
            "status": "unavailable",
            "data": None,
            "source": self._source("D fixture").to_dict(),
            "error": f"시연 {kind}을 찾을 수 없습니다: {value}",
        }
