"""농장 좌표만으로 인근 주거 건물을 자동 탐지한다.

2026-08-07 신규. 기존에는 주거지 1곳을 사람이 핀으로 찍어 bearing 1개를 만들었다.
  · 주거지가 여러 곳이면 누락된다
  · 농장주가 모르는 방향의 마을을 못 잡는다

데이터: 국토교통부 GIS건물통합정보 (VWorld 국가중점데이터 API)
    엔드포인트  https://api.vworld.kr/ned/wfs/getBldgisSpceWFS
    typename    dt_d010      기준일 2026-08-06      전국 약 1,439만 동
    인증        VWorld 개발키 1개. 2D데이터 API 와 같은 키로 호출된다.

핵심은 buld_prpos_code(건축물용도코드)다. 연속지적도나 도로명주소 건물에는
용도 정보가 없어 축사와 주택을 못 가르지만, 이 데이터는 건축물대장을
건물 단위로 붙여 놓아서 직접 구분된다.

────────────────────────────────────────────────────────────────────
설계 판단 — 클러스터링을 하지 않는다
    처음에는 DBSCAN 으로 '마을'을 묶었으나 제거했다.
      · 반경 2km 주거 건물이 200~1,500동. 84시점 x 1,500동 = 12.6만 회 계산인데
        1초도 걸리지 않는다. 묶을 이유가 없다.
      · 클러스터 중심을 쓰면 평균 위치 오차(실측 44~325m)가 방위 오차로
        옮겨붙는다. 개별 건물 좌표를 그대로 쓰면 그 오차가 사라진다.
      · '가장 가까운 민가 하나'만 쓰는 것도 안 된다. 최근접이 풍하측이라는
        보장이 없다. 300m 북쪽 1채보다 1km 남쪽 50채가 문제일 수 있다.
    따라서 개별 건물 전수를 그대로 반환하고, 위험도 계산에서 max() 를 취한다.

────────────────────────────────────────────────────────────────────
API 사용 시 함정 2가지
    1. bbox 순서가 좌표계에 따라 다르다.
       EPSG:4326 일 때만 (ymin, xmin, ymax, xmax) — 위도가 먼저다.
    2. maxFeatures 상한이 1000 이다. 농촌 반경 2km 가 2,500동을 넘으므로
       bbox 를 타일로 쪼개고 gis_idntfc_no 로 중복을 제거해야 한다.

────────────────────────────────────────────────────────────────────
실측 검증 (2026-08-07, 반경 2km)
  ① 용도 분류 — 지역 편차 없음
        지역            전체    주거(01·02)   축사(21)
        충남 홍성 결성   2,554      243         189
        경기 이천 율면   2,856      234         199
        전북 김제 용지   2,221      213         110
        경기 안성 일죽   3,670      705         125
     (건물명 buld_nm 으로 마을회관을 찾는 방식은 홍성에서만 작동했다.
      건물명 입력률이 지역별 0.5~5.5% 로 편차가 커서 쓸 수 없다.)

  ② 커버리지 — 8개 지역 중 7곳에서 주거 건물 탐지
        충북 진천 1,520 / 전북 정읍 323 / 충남 홍성 243 / 경기 이천 234 /
        경북 봉화 204 / 강원 정선 55 / 전남 신안(도서) 38 / 제주 중산간 0
     제주 중산간만 0인데 전체 건물이 23동뿐인 실제 무인 지대다.

────────────────────────────────────────────────────────────────────
한계
    1. 용도코드가 약 77% null 이다. 주택이 모두 미등록인 마을은 누락될 수 있다.
       include_null=True 로 두면 용도 미상 건물도 포함한다(재현율↑ 정밀도↓).
    2. dt_d010 은 대분류(끝 3자리 000)만 채워져 있다. 4개 지역 12,000여 동을
       훑었으나 01001 다중주택, 21106 양돈 같은 세부코드는 한 건도 없었다.
       나중에 채워질 경우를 대비해 prefix 매칭으로 구현했다.
    3. 결과는 '후보'다. 농장주 확인을 거쳐 확정하는 것을 전제로 한다.
"""

from __future__ import annotations

import math
import os
from collections import Counter
from dataclasses import dataclass

import requests

from constants import (
    EXCLUDE_RADIUS_M,
    LIVESTOCK_PREFIX,
    NEAR_WARN_M,
    NED_BUILDING_TYPENAME,
    NED_BUILDING_URL,
    NED_MAX_FEATURES,
    NED_TILES,
    RESIDENTIAL_PREFIX,
    SEARCH_RADIUS_M,
)
from geo import bearing

VWORLD_KEY = os.environ.get("VWORLD_KEY", "")          # 키를 코드에 박지 말 것
VWORLD_DOMAIN = os.environ.get("VWORLD_DOMAIN", "localhost")


@dataclass(frozen=True)
class Receptor:
    """악취를 받는 지점 하나 (주거 건물 1동)."""
    lat: float
    lon: float
    dist_m: float
    bearing: float          # 농장 → 이 건물 방위각 (북=0, 시계방향)
    purpose: str | None     # 건축물용도코드
    near: bool              # NEAR_WARN_M 이내 — 방위 민감도 높음

    def __repr__(self) -> str:
        mark = " [!]근거리" if self.near else ""
        return f"<Receptor {self.dist_m:.0f}m {self.bearing:.0f}°{mark}>"


def is_residential(code: str | None, include_null: bool = False) -> bool:
    """주거 건물인가. 세부코드가 채워져도 그대로 동작한다."""
    if code is None:
        return include_null
    return str(code).startswith(RESIDENTIAL_PREFIX)


def is_livestock(code: str | None) -> bool:
    """축사류인가. 판정용이 아니라 진단 출력용."""
    if code is None:
        return False
    return str(code).startswith(LIVESTOCK_PREFIX)


def _centroid(geom: dict) -> tuple[float, float]:
    """GeoJSON 지오메트리 정점 평균 → (lon, lat)."""
    xs = ys = 0.0
    n = 0

    def walk(a) -> None:
        nonlocal xs, ys, n
        if a and isinstance(a[0], (int, float)):
            xs += a[0]
            ys += a[1]
            n += 1
        else:
            for sub in a:
                walk(sub)

    walk(geom["coordinates"])
    if n == 0:
        raise ValueError("지오메트리에 좌표가 없습니다")
    return xs / n, ys / n


def fetch_buildings(lat: float, lon: float,
                    radius_m: float = SEARCH_RADIUS_M,
                    key: str | None = None,
                    tiles: int = NED_TILES) -> list[dict]:
    """반경 내 건물 전수. 반환: [{'lon','lat','purpose','area','floors'}, ...]

    bbox 를 tiles x tiles 로 쪼개 호출하고 gis_idntfc_no 로 중복을 제거한다.
    """
    key = key or VWORLD_KEY
    if not key:
        raise RuntimeError(
            "VWorld 인증키가 없습니다. 환경변수 VWORLD_KEY 를 설정하세요.\n"
            '  Windows  : setx VWORLD_KEY "발급키"   (새 터미널에서 적용)\n'
            "  mac/linux: export VWORLD_KEY=발급키"
        )

    ky = 111320.0
    kx = 111320.0 * math.cos(math.radians(lat))
    d_lat, d_lon = radius_m / ky, radius_m / kx

    seen: set[str] = set()
    out: list[dict] = []
    saturated = 0

    for i in range(tiles):
        for j in range(tiles):
            y0 = lat - d_lat + 2 * d_lat * i / tiles
            y1 = lat - d_lat + 2 * d_lat * (i + 1) / tiles
            x0 = lon - d_lon + 2 * d_lon * j / tiles
            x1 = lon - d_lon + 2 * d_lon * (j + 1) / tiles

            resp = requests.get(NED_BUILDING_URL, timeout=30, params={
                "key": key,
                "domain": VWORLD_DOMAIN,
                "typename": NED_BUILDING_TYPENAME,
                # ⚠️ EPSG:4326 은 (ymin, xmin, ymax, xmax) 순서
                "bbox": f"{y0},{x0},{y1},{x1},EPSG:4326",
                "maxFeatures": NED_MAX_FEATURES,
                "resultType": "results",
                "srsName": "EPSG:4326",
                "output": "application/json",
            })
            resp.raise_for_status()
            data = resp.json()

            if data.get("numberReturned", 0) >= NED_MAX_FEATURES:
                saturated += 1

            for feat in data.get("features", []):
                props = feat.get("properties", {})
                gid = props.get("gis_idntfc_no")
                if gid in seen:
                    continue
                seen.add(gid)
                try:
                    x, y = _centroid(feat["geometry"])
                except (KeyError, ValueError):
                    continue
                if math.hypot((x - lon) * kx, (y - lat) * ky) > radius_m:
                    continue
                out.append({
                    "lon": x, "lat": y,
                    "purpose": props.get("buld_prpos_code"),
                    "area": props.get("totar") or 0,
                    "floors": props.get("ground_floor_co") or 0,
                })

    if saturated:
        print(f"[!] {saturated}개 타일이 {NED_MAX_FEATURES}건 상한에 걸렸습니다. "
              f"tiles 를 {tiles + 2} 이상으로 올리세요.")
    return out


def find_receptors(lat: float, lon: float,
                   radius_m: float = SEARCH_RADIUS_M,
                   exclude_m: float = EXCLUDE_RADIUS_M,
                   include_null: bool = False,
                   key: str | None = None,
                   buildings: list[dict] | None = None
                   ) -> tuple[list[Receptor], Counter]:
    """농장 좌표 → (주거 수용점 리스트, 용도코드 분포). 가까운 순 정렬.

    buildings 를 직접 넘기면 API 를 호출하지 않는다(테스트·오프라인용).
    """
    blds = buildings if buildings is not None else fetch_buildings(
        lat, lon, radius_m, key)
    hist = Counter(str(b["purpose"]) if b["purpose"] is not None else "null"
                   for b in blds)

    kx = 111320.0 * math.cos(math.radians(lat))
    ky = 111320.0

    receptors: list[Receptor] = []
    for b in blds:
        if not is_residential(b["purpose"], include_null):
            continue
        dx, dy = (b["lon"] - lon) * kx, (b["lat"] - lat) * ky
        dist = math.hypot(dx, dy)
        if dist < exclude_m:                 # 농장 자기 부지 제거
            continue
        receptors.append(Receptor(
            lat=b["lat"], lon=b["lon"], dist_m=dist,
            bearing=bearing(lat, lon, b["lat"], b["lon"]),
            purpose=b["purpose"], near=dist < NEAR_WARN_M,
        ))

    receptors.sort(key=lambda r: r.dist_m)
    return receptors, hist


def summarize(receptors: list[Receptor]) -> dict:
    """수용점 목록 요약 — 출력·로그용."""
    if not receptors:
        return {"n": 0, "nearest_m": None, "near_count": 0, "bearing_spread": None}
    bearings = sorted(r.bearing for r in receptors)
    return {
        "n": len(receptors),
        "nearest_m": round(receptors[0].dist_m),
        "farthest_m": round(receptors[-1].dist_m),
        "near_count": sum(1 for r in receptors if r.near),
        "bearing_spread": (round(bearings[0]), round(bearings[-1])),
    }


if __name__ == "__main__":
    from console import use_utf8_stdout
    use_utf8_stdout()

    LAT, LON = 36.5666, 126.5500          # 충남 홍성군 결성면 (검증 지점)

    try:
        recs, hist = find_receptors(LAT, LON)
    except RuntimeError as exc:
        print(exc)
        raise SystemExit(1)

    print(f"농장 ({LAT}, {LON}) 반경 {SEARCH_RADIUS_M}m")
    print("\n용도코드 분포 (상위 8)")
    for code, cnt in hist.most_common(8):
        tag = ""
        if is_residential(code):
            tag = "← 주거"
        elif is_livestock(code):
            tag = "  (축사)"
        print(f"  {code:<8}{cnt:>6}  {tag}")

    print(f"\n주거 수용점 {len(recs)}동")
    print(f"{'거리m':>7}{'방위°':>7}  비고")
    print("─" * 30)
    for r in recs[:15]:
        print(f"{r.dist_m:>7.0f}{r.bearing:>7.0f}  "
              f"{'[!] 근거리, 방위 민감' if r.near else ''}")
    if len(recs) > 15:
        print(f"   ... 외 {len(recs) - 15}동")
    print("─" * 30)
    print("요약:", summarize(recs))
    print("\n※ 이 목록은 '후보'입니다. 농장주 확인을 거쳐 확정하세요.")
