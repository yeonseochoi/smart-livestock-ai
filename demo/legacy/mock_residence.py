"""VWorld API 없이 residence.py 를 테스트하기 위한 건물 픽스처.

사용법
    from mock_residence import mock_buildings
    from residence import find_receptors
    recs, hist = find_receptors(LAT, LON, buildings=mock_buildings(LAT, LON))

⚠️ 이 데이터는 합성입니다. 다만 아무렇게나 만든 것이 아니라
   2026-08-07 충남 홍성군 결성면(36.5666, 126.5500) 반경 2km 실측 결과에
   맞춰 보정했습니다.

   실측 (VWorld dt_d010, 건물 2,554동)
       null   1,975 (77.3%)
       01000    243 (단독주택)
       21000    189 (동물및식물관련시설 = 축사)
       17000     62 (공장)
       18000     35 (창고)
       03000     18 (제1종근린생활)
       기타      32

   마을 위치도 실측값입니다. 같은 지점에서 지목 '대' 필지를 리(里)별로
   묶었을 때 나온 방위·거리를 씁니다.
       교항리 302° 1168m │ 용호리  42°  678m │ 와리    4° 1818m
       성곡리 164° 1164m │ 형산리  69° 1656m

   난수 시드를 고정했으므로 실행할 때마다 같은 결과가 나옵니다.

실제 API 동작은 브라우저로 별도 확인했습니다(2026-08-07).
엔드포인트·파라미터·응답 스키마는 residence.py 주석 참조.
"""

from __future__ import annotations

import math
import random

# (이름, 방위각°, 거리m, 주거 건물 수)  — 방위·거리는 실측값
VILLAGES = [
    ("교항리", 302.0, 1168.0, 70),
    ("용호리",  42.0,  678.0, 60),
    ("와리",     4.0, 1818.0, 50),
    ("성곡리", 164.0, 1164.0, 40),
    ("형산리",  69.0, 1656.0, 23),
]
TOTAL_RESIDENTIAL = sum(v[3] for v in VILLAGES)      # 243

VILLAGE_SPREAD_M = 140.0        # 마을 내 건물 산포 표준편차

# 실측 분포에 맞춘 비주거 건물 수
N_BARN = 189                    # 21000
N_FACTORY = 62                  # 17000
N_WAREHOUSE = 35                # 18000
N_NEIGHBORHOOD = 18             # 03000
N_NULL = 1975                   # 용도 미상

SEED = 20260807


def _offset(lat: float, lon: float, bearing_deg: float,
            dist_m: float) -> tuple[float, float]:
    """농장에서 방위·거리만큼 떨어진 좌표. 평면 근사(2km 범위에서 오차 무시 가능)."""
    kx = 111320.0 * math.cos(math.radians(lat))
    ky = 111320.0
    th = math.radians(bearing_deg)
    return (lat + dist_m * math.cos(th) / ky,
            lon + dist_m * math.sin(th) / kx)


def mock_buildings(farm_lat: float = 36.5666, farm_lon: float = 126.5500,
                   radius_m: float = 2000.0, seed: int = SEED) -> list[dict]:
    """residence.fetch_buildings() 와 동일한 형태의 리스트를 만든다.

    반환: [{'lon','lat','purpose','area','floors'}, ...]
    """
    rng = random.Random(seed)
    kx = 111320.0 * math.cos(math.radians(farm_lat))
    ky = 111320.0
    out: list[dict] = []

    def add(lat: float, lon: float, purpose: str | None,
            area: float, floors: int) -> None:
        if math.hypot((lon - farm_lon) * kx, (lat - farm_lat) * ky) > radius_m:
            return
        out.append({"lon": lon, "lat": lat, "purpose": purpose,
                    "area": area, "floors": floors})

    # ① 마을별 주거 건물
    for _name, brg, dist, count in VILLAGES:
        clat, clon = _offset(farm_lat, farm_lon, brg, dist)
        for _ in range(count):
            dlat = rng.gauss(0.0, VILLAGE_SPREAD_M) / ky
            dlon = rng.gauss(0.0, VILLAGE_SPREAD_M) / kx
            add(clat + dlat, clon + dlon, "01000",
                round(rng.uniform(40.0, 220.0), 2), rng.choice([1, 1, 1, 2]))

    # ② 축사 — 마을과 떨어진 곳에 흩어져 있다
    for _ in range(N_BARN):
        brg = rng.uniform(0.0, 360.0)
        dist = rng.uniform(250.0, radius_m)
        lat, lon = _offset(farm_lat, farm_lon, brg, dist)
        add(lat, lon, "21000", round(rng.uniform(150.0, 900.0), 2), 1)

    # ③ 공장·창고·근린생활
    for purpose, n, lo, hi in (("17000", N_FACTORY, 200.0, 1200.0),
                               ("18000", N_WAREHOUSE, 80.0, 600.0),
                               ("03000", N_NEIGHBORHOOD, 50.0, 300.0)):
        for _ in range(n):
            brg = rng.uniform(0.0, 360.0)
            dist = rng.uniform(200.0, radius_m)
            lat, lon = _offset(farm_lat, farm_lon, brg, dist)
            add(lat, lon, purpose, round(rng.uniform(lo, hi), 2), 1)

    # ④ 용도 미상 (실측 77%)
    for _ in range(N_NULL):
        brg = rng.uniform(0.0, 360.0)
        dist = rng.uniform(50.0, radius_m)
        lat, lon = _offset(farm_lat, farm_lon, brg, dist)
        add(lat, lon, None, 0.0, 0)

    return out


if __name__ == "__main__":
    from console import use_utf8_stdout
    use_utf8_stdout()

    from collections import Counter

    blds = mock_buildings()
    hist = Counter(b["purpose"] or "null" for b in blds)
    print(f"합성 건물 {len(blds)}동  (실측 홍성 결성면 2,554동 기준)")
    for code, cnt in hist.most_common():
        print(f"  {code:<8}{cnt:>6}  ({cnt / len(blds) * 100:4.1f}%)")
    print(f"\n주거(01·02) {sum(v for k, v in hist.items() if k.startswith(('01', '02')))}동"
          f"  (실측 {TOTAL_RESIDENTIAL}동)")
