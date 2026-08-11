"""위경도 ↔ 기상청 격자 좌표 변환, 방위각 계산."""

import math


def latlon_to_grid(lat: float, lon: float) -> tuple[int, int]:
    """위경도 → 기상청 격자 (nx, ny). 격자 간격 5km. (Lambert Conformal Conic)"""
    RE, GRID = 6371.00877, 5.0
    SLAT1, SLAT2 = 30.0, 60.0        # 투영 위도 1, 2
    OLON, OLAT = 126.0, 38.0         # 기준점 경위도
    XO, YO = 43, 136                 # 기준점 격자좌표

    D = math.pi / 180.0
    re = RE / GRID
    s1, s2, ol, oa = SLAT1 * D, SLAT2 * D, OLON * D, OLAT * D

    sn = math.tan(math.pi * 0.25 + s2 * 0.5) / math.tan(math.pi * 0.25 + s1 * 0.5)
    sn = math.log(math.cos(s1) / math.cos(s2)) / math.log(sn)
    sf = math.pow(math.tan(math.pi * 0.25 + s1 * 0.5), sn) * math.cos(s1) / sn
    ro = re * sf / math.pow(math.tan(math.pi * 0.25 + oa * 0.5), sn)

    ra = re * sf / math.pow(math.tan(math.pi * 0.25 + lat * D * 0.5), sn)
    th = lon * D - ol
    if th > math.pi:
        th -= 2.0 * math.pi
    if th < -math.pi:
        th += 2.0 * math.pi
    th *= sn

    nx = int(ra * math.sin(th) + XO + 0.5)
    ny = int(ro - ra * math.cos(th) + YO + 0.5)
    return nx, ny


def bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """지점1 → 지점2 방위각(deg). 북=0, 시계방향."""
    d = math.pi / 180.0
    dlon = (lon2 - lon1) * d
    y = math.sin(dlon) * math.cos(lat2 * d)
    x = (math.cos(lat1 * d) * math.sin(lat2 * d)
         - math.sin(lat1 * d) * math.cos(lat2 * d) * math.cos(dlon))
    return (math.degrees(math.atan2(y, x)) + 360) % 360


def angle_diff(a: float, b: float) -> float:
    """두 방위각의 최소 차이 (0~180)."""
    return abs((a - b + 180) % 360 - 180)


if __name__ == "__main__":
    print(latlon_to_grid(37.5665, 126.9780))          # (60, 127) 예상
    print(bearing(0, 0, 1, 0), bearing(0, 0, 0, 1))    # 0, 90 예상
