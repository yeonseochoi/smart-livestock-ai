"""농가 위치의 순수 로직 — Streamlit·folium 을 import 하지 않는다.

화면(``app/location_ui.py``)은 이 모듈이 돌려준 숫자를 그리기만 한다.
"판단은 코드, 표시는 화면"이라는 이 저장소의 역할 분담을 위치 기능에도
그대로 적용한 것이다.

────────────────────────────────────────────────────────────────────
이 모듈이 하는 일 3가지

  ① 좌표 검증      익산 경계 상자 안인가 (막지 않고 알린다)
  ② 주소 → 좌표    VWORLD 지오코딩. 키가 없으면 없다고 말한다
  ③ 부채꼴 좌표    지금 바람에서 냄새가 가는 방향의 폴리곤 꼭짓점

────────────────────────────────────────────────────────────────────
③ 은 왜 부채꼴이 두 개인가 — 3km 경계

  ``analysis/plume_select.py`` 가 이미 쓰고 있는 규칙을 화면에 그대로
  옮긴 것이다. 거리에 따라 판정 도구가 바뀐다.

    ≤ 3km   플룸 유효 반각.  익산 농가 좌표는 필지 기반[A]이라 오차가
            수십 m 인데 부채꼴 반폭은 226~1,242m 다. 좌표 오차보다
            부채꼴이 훨씬 크므로 정밀 계산이 의미를 가진다.

    > 3km   고정 섹터 ±30도.  김제 농가 좌표는 리(里) 중앙값[B]이라
            오차가 1~2km 인데 15km 지점 F등급 부채꼴 반폭은 814m 다.
            좌표 오차가 부채꼴보다 크면 적중이 나와도 우연이다.

  두 구간을 한 덩어리로 그리면 이 경계가 사라진다. 색과 투명도를 달리해
  따로 그리는 이유는, "먼 거리에서는 정밀하다고 주장하지 않는다"는 것이
  이 시스템의 설계이지 결함이 아니기 때문이다.

★ 이 부채꼴은 등급을 바꾸지 않는다 (절대규칙 1).
  ML 이 매긴 점수 중 어느 수용점 유형을 볼지 고르는 데만 쓰이며,
  화면에서는 "왜 그 유형을 골랐는지"를 보여주는 그림 역할만 한다.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

# config 를 먼저 import 해야 legacy/ 가 sys.path 에 올라간다 (config.py 참조).
from config import (
    DOWNTOWN_LAT,
    DOWNTOWN_LON,
    GROUP_CENTER,
    GROUP_RURAL,
    GROUP_URBAN,
    IKSAN_BBOX,
    WANGGUNG_LAT,
    WANGGUNG_LON,
)

from analysis.plume_select import PLUME_MAX_KM, SECTOR_DEG, downwind_groups

# legacy 는 수정하지 않고 import 만 한다 (절대규칙 3).
from plume import STABILITY_LABEL, plume_half_angle

KST = timezone(timedelta(hours=9))

_EARTH_KM = 6371.0088
SECTOR_MAX_KM = 15.0        # spatial_features 의 풍상측 탐색 반경과 같은 값

# 입력 경로별 라벨. 화면 문구를 한 곳에 모아 둔다.
SOURCE_LABEL = {
    "default": "기본값(왕궁 축산단지)",
    "gps": "GPS 자동 조회",
    "map": "지도 클릭",
    "address": "주소 검색",
}


# ═══════════════════════════════════════════════════════════════════
# 1. 위치 값
# ═══════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class FarmLocation:
    """사용자가 정한 농장 위치 하나.

    ``source`` 를 값에 붙여 두는 이유는, 정확도가 경로마다 다르기 때문이다.
    PC 브라우저의 GPS 는 IP 기반이라 수 km 오차가 나는데, 화면에서 이걸
    지도 클릭과 똑같이 보여 주면 사용자가 틀린 위치를 믿게 된다.
    """

    lat: float
    lon: float
    source: str = "default"
    accuracy_m: float | None = None
    label: str | None = None

    @property
    def is_default(self) -> bool:
        return self.source == "default"

    @property
    def in_iksan(self) -> bool:
        return in_iksan(self.lat, self.lon)

    @property
    def source_label(self) -> str:
        return SOURCE_LABEL.get(self.source, self.source)

    @property
    def accuracy_note(self) -> str | None:
        """정확도를 사람 말로. 나쁘면 나쁘다고 먼저 말한다."""
        if self.accuracy_m is None:
            return None
        if self.accuracy_m <= 100:
            return f"정확도 ±{self.accuracy_m:,.0f}m"
        if self.accuracy_m <= 1000:
            return f"정확도 ±{self.accuracy_m:,.0f}m — 건물 단위로는 부정확합니다"
        return (
            f"정확도 ±{self.accuracy_m:,.0f}m — PC 브라우저는 IP 기반이라 "
            "위치가 크게 어긋납니다. 지도를 직접 클릭하거나 주소를 입력하세요"
        )

    def as_farm_override(self) -> dict[str, Any] | None:
        """provider 에 넘길 농가 좌표 덮어쓰기 값.

        기본값이면 ``None`` 을 돌려준다 — 위치를 정하지 않은 사용자에게는
        지금까지와 완전히 같은 화면이 나와야 한다. 아무도 아무것도 안 했는데
        추천 결과가 달라지면 그게 더 나쁜 변경이다.
        """
        if self.is_default:
            return None
        return {
            "lat": float(self.lat), "lon": float(self.lon),
            "name": self.label or "사용자 지정 위치",
            "source": self.source, "in_iksan": self.in_iksan,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "lat": self.lat, "lon": self.lon, "source": self.source,
            "accuracy_m": self.accuracy_m, "label": self.label,
            "in_iksan": self.in_iksan,
        }


DEFAULT_LOCATION = FarmLocation(
    WANGGUNG_LAT, WANGGUNG_LON, "default", label="왕궁 데모농가(기본값)",
)


def in_iksan(lat: float, lon: float) -> bool:
    """익산 경계 상자 안인가. 막는 판정이 아니라 알리는 판정이다."""
    south, west, north, east = IKSAN_BBOX
    return south <= float(lat) <= north and west <= float(lon) <= east


def bbox_polygon() -> list[list[float]]:
    """경계 상자를 지도에 점선으로 그릴 때 쓰는 꼭짓점 (남서 → 시계방향)."""
    south, west, north, east = IKSAN_BBOX
    return [[south, west], [north, west], [north, east], [south, east], [south, west]]


def receptor_points() -> list[dict[str, Any]]:
    """수용점 2곳. 좌표는 학습이 산출한 민원 좌표 중앙값(config.GROUP_CENTER)."""
    return [
        {
            "group": GROUP_RURAL,
            "lat": GROUP_CENTER[GROUP_RURAL][0], "lon": GROUP_CENTER[GROUP_RURAL][1],
            "note": "축사 인접 농촌 마을 — 민원 좌표 중앙값",
        },
        {
            "group": GROUP_URBAN,
            "lat": GROUP_CENTER[GROUP_URBAN][0], "lon": GROUP_CENTER[GROUP_URBAN][1],
            "note": "축사에서 먼 시가지 — 민원 좌표 중앙값",
        },
    ]


# ═══════════════════════════════════════════════════════════════════
# 2. 기하 — 부채꼴 꼭짓점
# ═══════════════════════════════════════════════════════════════════

def destination(lat: float, lon: float, bearing_deg: float,
                dist_km: float) -> tuple[float, float]:
    """출발점에서 방위각·거리만큼 떨어진 지점 (대권 항법).

    ``legacy/geo.py`` 의 ``bearing()`` 과 역함수 관계다. legacy 는 수정하지
    않는 폴더라 반대 방향 계산이 없어 여기에 둔다.
    """
    lat1, lon1 = math.radians(lat), math.radians(lon)
    brg = math.radians(float(bearing_deg))
    ratio = float(dist_km) / _EARTH_KM
    lat2 = math.asin(
        math.sin(lat1) * math.cos(ratio)
        + math.cos(lat1) * math.sin(ratio) * math.cos(brg)
    )
    lon2 = lon1 + math.atan2(
        math.sin(brg) * math.sin(ratio) * math.cos(lat1),
        math.cos(ratio) - math.sin(lat1) * math.sin(lat2),
    )
    return math.degrees(lat2), (math.degrees(lon2) + 540) % 360 - 180


def _wedge(lat: float, lon: float, to_deg: float,
           r0_km: float, r1_km: float, half_at, steps: int = 16
           ) -> list[list[float]]:
    """반각이 거리에 따라 변하는 부채꼴의 폴리곤 꼭짓점.

    ``half_at(r_km)`` 이 그 거리에서의 반각(도)을 돌려준다. 오른쪽 가장자리를
    바깥으로 훑고 왼쪽 가장자리를 안쪽으로 되짚어 닫힌 고리를 만든다.
    """
    radii = [r0_km + (r1_km - r0_km) * i / steps for i in range(steps + 1)]
    right = [destination(lat, lon, to_deg + half_at(r), r) for r in radii]
    left = [destination(lat, lon, to_deg - half_at(r), r) for r in reversed(radii)]
    return [[p[0], p[1]] for p in right + left]


def plume_overlay(lat: float, lon: float, wd_deg: float, ws_ms: float,
                  sky: Any, when: datetime) -> dict[str, Any]:
    """지금 바람에서 냄새가 가는 방향의 부채꼴 두 개와 판정 근거.

    ``wd_deg`` 는 기상 풍향(바람이 '불어오는' 방향)이다. 냄새가 가는 방향은
    +180도이며, 이 뒤집기를 빠뜨리면 부채꼴이 정반대를 가리킨다.
    """
    groups, detail = downwind_groups(lat, lon, wd_deg, ws_ms, sky, when)
    stability = detail[0].get("stability") if detail else None
    if stability is None:
        # downwind_groups 는 안정도를 detail 에 넣지 않으므로 같은 방식으로 다시 구한다.
        from plume import pasquill_class

        stability = pasquill_class(ws_ms, sky, when, lat, lon)[0]

    to_deg = (float(wd_deg) + 180.0) % 360.0

    def _half(r_km: float) -> float:
        return plume_half_angle(max(r_km, 0.05) * 1000.0, stability)

    return {
        "to_deg": round(to_deg, 1),
        "from_deg": round(float(wd_deg) % 360.0, 1),
        "wind_speed_ms": float(ws_ms),
        "stability": stability,
        "stability_label": STABILITY_LABEL.get(stability, ""),
        "plume_half_angle_1km": round(_half(1.0), 1),
        # 0~3km — 플룸 유효 반각. 거리에 따라 좁아진다.
        "plume_polygon": _wedge(lat, lon, to_deg, 0.05, PLUME_MAX_KM, _half),
        # 3~15km — 고정 섹터. 좌표 오차가 부채꼴보다 커 플룸으로 판정하지 않는 구간.
        "sector_polygon": _wedge(
            lat, lon, to_deg, PLUME_MAX_KM, SECTOR_MAX_KM, lambda _r: SECTOR_DEG,
            steps=2,
        ),
        "sector_half_angle": SECTOR_DEG,
        "downwind_groups": list(groups),
        "detail": detail,
    }


def describe_overlay(overlay: dict[str, Any]) -> str:
    """지도 밑에 한 줄로 붙일 설명."""
    if not overlay:
        return "이 시각의 예보 원값이 없어 부채꼴을 그릴 수 없습니다."
    hit = overlay.get("downwind_groups") or []
    head = (
        f"바람 {overlay['from_deg']:.0f}도에서 불어와 냄새는 "
        f"{overlay['to_deg']:.0f}도 방향 · 안정도 {overlay['stability']}"
        f"({overlay.get('stability_label', '')})"
    )
    if not hit:
        return head + " · 풍하측에 수용점 유형 없음 — 보수적으로 두 유형의 최댓값을 씁니다."
    return head + " · 풍하측: " + ", ".join(hit)


# ═══════════════════════════════════════════════════════════════════
# 3. 주소 → 좌표 (VWORLD)
# ═══════════════════════════════════════════════════════════════════

class GeocodeError(RuntimeError):
    """주소 검색 실패. 화면에서 사용자에게 그대로 보여 준다."""


def geocode_address(query: str, *, timeout: float = 6.0) -> FarmLocation:
    """주소 한 줄 → 좌표. 실패 이유를 감추지 않고 예외 메시지로 올린다.

    도로명(ROAD) 으로 먼저 찾고 안 되면 지번(PARCEL) 으로 다시 찾는다.
    농가 주소는 지번인 경우가 많아 이 순서가 실제로 필요하다.
    """
    query = (query or "").strip()
    if not query:
        raise GeocodeError("주소를 입력하세요.")

    key = (os.environ.get("VWORLD_KEY") or "").strip()
    if not key:
        raise GeocodeError(
            "VWORLD_KEY 가 없어 주소 검색을 쓸 수 없습니다. "
            "지도를 클릭하거나 GPS 버튼을 사용하세요."
        )
    try:
        import requests
    except ImportError as exc:      # 배포 의존성에서 빠졌을 때
        raise GeocodeError(f"requests 미설치로 주소 검색을 쓸 수 없습니다: {exc}") from exc

    last_error = "검색 결과가 없습니다."
    for addr_type in ("ROAD", "PARCEL"):
        try:
            response = requests.get(
                "https://api.vworld.kr/req/address",
                params={
                    "service": "address", "request": "getcoord", "version": "2.0",
                    "crs": "epsg:4326", "type": addr_type, "format": "json",
                    "address": query, "key": key,
                },
                timeout=timeout,
            )
            payload = response.json()
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            continue
        if (payload.get("response") or {}).get("status") != "OK":
            last_error = (
                (payload.get("response") or {}).get("error", {}).get("text")
                or "검색 결과가 없습니다."
            )
            continue
        point = payload["response"]["result"]["point"]
        refined = (payload["response"].get("refined") or {}).get("text") or query
        return FarmLocation(
            lat=float(point["y"]), lon=float(point["x"]),
            source="address", label=refined,
        )
    raise GeocodeError(f"주소를 찾지 못했습니다 — {last_error}")
