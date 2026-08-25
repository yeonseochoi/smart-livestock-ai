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
    def __init__(self, *, rag_index: Any = None, storage_days: int | None = None,
                 farm_override: dict[str, Any] | None = None) -> None:
        self.rag = rag_index
        self.generated_at = datetime.now(KST).isoformat(timespec="seconds")
        # 화면 사이드바의 「분뇨 저장 경과일」 값이다. 여기 담아 두지 않으면
        # get_storage_days() 가 참조할 곳이 없어 슬라이더가 무동작이 된다
        # (아래 메서드 주석 참조). fixture_provider 는 처음부터 이렇게 하고 있다.
        self.storage_days = storage_days
        # 사용자가 화면에서 정한 농장 좌표. 세션 메모리에만 있고 DB 에 쓰지 않는다.
        #
        # ★ DB 에 쓰지 않는 이유가 있다. 프리뷰 앱과 라이브 앱이 같은 Supabase 를
        #   읽으므로, 여기서 farm_config 를 UPDATE 하면 남의 화면이 같이 바뀐다.
        #   위치는 "이 브라우저 세션의 입력값"이지 저장할 설비 정보가 아니다.
        #
        # None 이면 지금까지와 완전히 같이 동작한다 — 위치를 정하지 않은 사용자의
        # 화면이 달라지면 안 된다.
        self.farm_override = farm_override or None

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
        # [C] 2026-08-25: 좌표를 지명과 맞췄다. 이전 값(왕궁 35.9518, 126.9574)은
        #     config.WANGGUNG(35.968937, 127.090910)에서 약 12km 서쪽이라,
        #     같은 지도에 수용점·사용자 위치를 함께 올리는 순간 "왕궁 측정소인데
        #     왜 왕궁에서 12km 떨어져 있나"가 눈에 보인다. 값 자체는 여전히
        #     발표용 고정값이며(state="fixture"), 지도에서도 회색 점선으로
        #     "(예시)" 표시해 실측 레이어와 구분한다.
        raw = (
            ("S-DEMO-01", "익산 왕궁 측정소(예시)", 35.9645, 127.0862,
             0.001, 0.002, 0.191, 1.8, 30.7, 58.9, "동남동", 2.2, "unverified"),
            ("S-DEMO-02", "익산 춘포 측정소(예시)", 35.9430, 127.0350,
             0.002, 0.004, 0.224, 2.1, 30.2, 61.4, "남동", 1.7, "unverified"),
            ("S-DEMO-03", "익산 삼기 측정소(예시)", 35.9930, 127.0180,
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

        # ── 어느 수용점 유형의 점수를 볼 것인가 ────────────────────────
        # 기본(위치 미설정)은 지금까지와 같이 두 유형 중 최댓값이다.
        # 두 확률을 곱하지 않는 현행 advisor 의 보수적 max 원칙과 같다.
        #
        # 사용자가 위치를 정했을 때만 플룸이 개입한다. 이때도 점수를 바꾸지
        # 않고 '어느 유형을 볼지'만 고른다 — 절대규칙 1. 로직은 새로 짜지 않고
        # advisor/recommend.py 가 쓰는 downwind_groups 를 그대로 재사용한다.
        wind = self._load_wind() if self.farm_override else {}
        plume_hits = 0
        by_group: dict[tuple[str, int], dict[str, tuple[Any, ...]]] = {}
        for row in rows:
            date_value, hour, grp, *_rest = row
            by_group.setdefault((date_value, int(hour)), {})[grp] = tuple(row)

        by_hour: dict[tuple[str, int], tuple[Any, ...]] = {}
        for key, per_group in by_group.items():
            picked = per_group
            if wind:
                selected = self._downwind_subset(key, per_group, wind)
                if selected:
                    picked = selected
                    plume_hits += 1
            by_hour[key] = max(picked.values(), key=lambda r: float(r[3]))

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
        limitations = ["가장 최근 updated_at 배치만 사용"]
        if plume_hits:
            limitations.append(
                f"사용자 위치 기준 풍하측 수용점 유형을 {plume_hits}개 시각에서 선택 "
                "(플룸은 유형 선택에만 쓰고 점수·등급은 바꾸지 않음)"
            )
        elif self.farm_override:
            limitations.append(
                "사용자 위치를 받았으나 풍하측에 드는 수용점 유형이 없어 "
                "두 유형의 최댓값을 그대로 사용"
            )
        else:
            limitations.append(
                "그룹(농촌근거리/시가지원거리) 중 보수적으로 위험도가 더 높은 쪽 채택"
            )
        limitations.append("단기는 1시간, reduced 중기는 일 단위로 제공")
        return self._response(
            "ok", {"farm_id": farm_id, "work_type": work_type, "items": items,
                   "plume_selected_hours": plume_hits},
            self._source("connected", f"B risk_hourly · {db.describe()}", *limitations),
        )

    # ── 플룸 유형 선택 보조 ───────────────────────────────────────
    def _load_wind(self) -> dict[tuple[str, int], dict[str, Any]]:
        """{(날짜, 시각): {wd, ws, sky}} — daily_scoring 이 저장한 예보 원값.

        advisor/recommend.py 의 _load_forecast_wind() 와 같은 자료를 읽는다.
        비어 있어도 예외를 올리지 않는다 — 부채꼴을 못 그릴 뿐 캘린더는
        지금까지와 똑같이 나와야 한다.
        """
        con = None
        try:
            con = db.connect()
            rows = con.execute(
                "SELECT date, hour, wd, ws, sky FROM forecast_hourly"
            ).fetchall()
        except Exception:
            return {}
        finally:
            if con is not None:
                con.close()
        out: dict[tuple[str, int], dict[str, Any]] = {}
        for date_text, hour, wd, ws, sky in rows:
            if wd is None or ws is None:
                continue
            out[(date_text, int(hour))] = {"wd": float(wd), "ws": float(ws), "sky": sky}
        return out

    def _downwind_subset(
        self, key: tuple[str, int], per_group: dict[str, tuple[Any, ...]],
        wind: dict[tuple[str, int], dict[str, Any]],
    ) -> dict[str, tuple[Any, ...]] | None:
        """이 시각 풍하측에 드는 유형만 남긴다. 판정 불가면 None."""
        weather = wind.get(key)
        if not weather or not self.farm_override:
            return None
        from analysis.plume_select import downwind_groups

        date_text, hour = key
        when = datetime.strptime(date_text, "%Y-%m-%d").replace(hour=hour, tzinfo=KST)
        try:
            hit, _detail = downwind_groups(
                float(self.farm_override["lat"]), float(self.farm_override["lon"]),
                weather["wd"], weather["ws"], weather["sky"], when,
            )
        except Exception:
            return None
        selected = {g: row for g, row in per_group.items() if g in hit}
        return selected or None

    def get_wind_series(self, days: int = 3) -> dict[str, Any]:
        """지도 부채꼴용 시각별 예보 원값. DecisionProvider 계약 밖의 선택 기능이다.

        Protocol 에 넣지 않은 이유는 fixture provider 가 예보 원값을 갖고 있지
        않기 때문이다. 화면은 ``getattr`` 로 있으면 쓰고 없으면 부채꼴만 생략한다.
        """
        wind = self._load_wind()
        if not wind:
            return self._response(
                "unavailable", None,
                self._source("unavailable", f"B forecast_hourly · {db.describe()}"),
                "forecast_hourly 가 비어 있습니다. serving.daily_scoring.run() 을 먼저 실행하세요.",
            )
        # ★ forecast_hourly 는 PK 가 (date, hour) 라 실행할 때마다 덮어쓰이며 쌓인다.
        #   그래서 테이블에는 지난주 예보가 그대로 남아 있다 (실측: 08-18 ~ 08-29).
        #   그냥 sorted()[:3] 하면 일주일 전 날짜가 슬라이더에 뜨고, 화면의
        #   추천 시각(모레)과 어긋나 보인다. 오늘 이후를 먼저 고르고, 그것이
        #   없을 때만(=배치가 통째로 오래됐을 때) 가장 최근 날짜로 폴백한다.
        span = max(1, min(int(days), 7))
        all_dates = sorted({date_text for date_text, _ in wind})
        today = datetime.now(KST).strftime("%Y-%m-%d")
        upcoming = [d for d in all_dates if d >= today]
        dates = upcoming[:span] if upcoming else all_dates[-span:]
        items = [
            {"date": date_text, "hour": hour,
             "start": datetime.strptime(date_text, "%Y-%m-%d").replace(
                 hour=hour, tzinfo=KST).isoformat(),
             **wind[(date_text, hour)]}
            for date_text, hour in sorted(wind)
            if date_text in dates
        ]
        return self._response(
            "ok", {"items": items},
            self._source("connected", f"B forecast_hourly · {db.describe()}",
                         "기상청 단기예보 원값이며 관측이 아님"),
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
        """분뇨 저장 경과일. DB 의 실제 반출일이 있으면 그쪽이 우선한다.

        [2026-08-25] 버그 두 개를 함께 고쳤다.

        ① 화면 슬라이더가 아무 일도 안 하고 있었다.
           dashboard.py 는 create_provider(storage_days=슬라이더값) 으로 값을
           넘기는데 __init__ 이 그걸 받기만 하고 버렸고, 이 메서드는 DB 의
           last_manure_removal_date 만 봤다. 그 컬럼이 None 이라(데모 농가는
           반출 이력이 없다) days 가 항상 None → storage_factor(None)=1.0 이었다.
           실측: 슬라이더를 0/12/20/30 으로 바꿔도 추천 점수가 1.0163 로 동일.
           화면 상단 배너는 "분뇨 저장 30일 경과 기준"이라고 적으면서 계산은
           "모름"으로 하고 있었다 — 표시와 계산이 어긋난 상태였다.
           fixture_provider 는 처음부터 self.storage_days 를 그대로 돌려주므로,
           같은 슬라이더가 모드에 따라 먹었다 안 먹었다 했다.

        ② datetime.now() 에 시간대가 없었다.
           Streamlit Cloud 컨테이너는 UTC 다. 마지막 반출일이 08-13 이고
           실제 시각이 08-25 08:00 KST(=08-24 23:00 UTC)이면 로컬은 12일,
           배포는 11일로 하루 밀린다. storage_factor 는 14일에서 1.0→1.5 로
           끊기는 계단 함수라 경계일 새벽에 로컬과 배포가 다른 값을 낸다.

        우선순위를 DB 우선으로 둔 이유는, 실제 반출 이력이 들어오는 순간
        사용자 입력보다 그쪽이 정확하기 때문이다. 어느 쪽을 썼는지는
        days_origin 과 source.limitations 에 남겨 화면에서 구분되게 한다
        (폴백을 탔다는 사실은 반드시 표시한다는 이 저장소의 규약).
        """
        farm = self._farm_row(farm_id)
        removal_date = (farm or {}).get("last_manure_removal_date")
        if removal_date:
            days = (datetime.now(KST).date()
                    - datetime.strptime(removal_date, "%Y-%m-%d").date()).days
            origin = "farm_config"
            note = f"farm_config 의 마지막 반출일 {removal_date} 기준"
        elif self.storage_days is not None:
            days = int(self.storage_days)
            origin = "user_input"
            note = "farm_config 에 마지막 반출일이 없어 화면 입력값을 사용"
        else:
            days = None
            origin = "unknown"
            note = "반출일도 입력값도 없어 저장 가중치를 적용하지 않음(1.0)"
        return self._response(
            "ok", {"farm_id": farm_id, "days": days, "days_origin": origin,
                   "over_2weeks": bool(days is not None and days >= 14),
                   "days_until_threshold": None if days is None else max(0, 14 - days)},
            self._source("connected", "B farm_config", note,
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

    def _farm_row(self, farm_id: str) -> dict[str, Any] | None:
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
        farm = dict(zip(keys, row))
        if self.farm_override:
            # 좌표와 이름만 덮어쓴다. 저장 경과일·시설유형은 DB 값을 그대로 둔다
            # — 사용자가 정한 것은 위치이지 농장의 나머지 속성이 아니다.
            farm = dict(farm)
            farm.update({
                "lat": self.farm_override["lat"], "lon": self.farm_override["lon"],
                "name": self.farm_override.get("name") or farm.get("name"),
                "coordinate_source": self.farm_override.get("source"),
            })
        return farm
