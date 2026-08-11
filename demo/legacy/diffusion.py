"""수용점 전수에 대한 상대 확산 계산.

2026-08-07 전면 교체. 이전 구현과의 차이는 아래와 같다.

    이전                                    현재
    ────────────────────────────────────    ────────────────────────────────────
    D = 3.0 x 2.0 x 2.0 (경험 배수)          Pasquill 안정도 + Briggs 확산계수
    ±45° 계단 판정                           exp(-y²/2σy²) 연속 감쇠
    거리 미반영                              σy(x), σz(x) 가 거리 함수
    수용점 1개 (핀으로 찍은 주거지)            수용점 N개 전수 → max()
    근거등급 [C]                             [A] (국제 표준 + 20년 운영 검증)

왜 바꿨는지, 45°가 얼마나 과대했는지는 plume.py 상단 주석을 보라.

반환하는 factor 는 '상대 확산 인자' [s/m³] 이며 절대 농도가 아니다.
    농도[g/m³] = 배출률[g/s] x factor
농도 환산은 참고값으로만 쓴다(plume.concentration_ppm 주석 참조).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from geo import angle_diff
from plume import (
    STABILITY_LABEL,
    dispersion_factor,
    initial_sigmas,
    pasquill_class,
    plume_half_angle,
)
from residence import Receptor


@dataclass
class DispersionResult:
    """한 시각의 확산 판정 결과."""
    factor: float                       # 최대 상대 확산 인자 [s/m³]
    stability: str                      # Pasquill A~F
    solar_elev: float                   # 태양고도(도). 음수면 야간
    wind_to: float                      # 악취가 흘러가는 방향(도)
    worst: Receptor | None = None       # 가장 위험한 수용점
    worst_angle_off: float | None = None  # 그 수용점의 중심선 이탈각(도)
    n_exposed: int = 0                  # factor > 0 인 수용점 수
    reasons: list[str] = field(default_factory=list)


def wind_to_direction(vec: float) -> float:
    """기상청 VEC(바람이 불어오는 방향) → 악취가 흘러가는 방향."""
    return (float(vec) + 180.0) % 360.0


def dispersion(vec: float, wsd: float, sky: str, dt: datetime,
               farm_lat: float, farm_lon: float,
               receptors: list[Receptor],
               source_width_m: float = 0.0) -> DispersionResult:
    """수용점 전수를 계산하고 가장 위험한 지점의 값을 돌려준다.

    수용점이 여럿일 때 max() 를 쓰는 이유
        '가장 가까운 민가'가 곧 '가장 위험한 민가'는 아니다.
        300m 북쪽 1채보다 1km 남쪽 마을이 풍하측이면 그쪽이 문제다.
        위험도는 거리와 각도의 함수이므로 전수 계산해야 최악을 찾는다.
    """
    stability, solar_elev = pasquill_class(wsd, sky, dt, farm_lat, farm_lon)
    wind_to = wind_to_direction(vec)
    sigma_y0, sigma_z0 = initial_sigmas(source_width_m)

    best_factor = 0.0
    worst: Receptor | None = None
    worst_off: float | None = None
    n_exposed = 0

    for rec in receptors:
        off = angle_diff(wind_to, rec.bearing)          # 0~180
        f = dispersion_factor(rec.dist_m, off, stability, wsd,
                              sigma_y0, sigma_z0)
        if f > 0:
            n_exposed += 1
        if f > best_factor:
            best_factor, worst, worst_off = f, rec, off

    reasons: list[str] = [f"안정도 {stability} - {STABILITY_LABEL[stability]}"]

    if worst is None:
        reasons.append("풍하측에 주거지 없음" if receptors else "수용점 없음")
    else:
        half = plume_half_angle(worst.dist_m, stability)
        reasons.append(
            f"최악 수용점 {worst.dist_m:.0f}m, 중심선 이탈 {worst_off:.0f}deg "
            f"(플룸 유효 반각 {half:.0f}deg)"
        )
        if worst_off is not None and worst_off <= half:
            reasons.append("플룸 중심부에 위치 / 직접 영향권")
        if worst.near:
            reasons.append("근거리 수용점 / 방위 오차에 민감, 농장주 확인 필요")
        if n_exposed > 1:
            reasons.append(f"풍하측 주거 {n_exposed}동")

    return DispersionResult(
        factor=best_factor, stability=stability, solar_elev=solar_elev,
        wind_to=wind_to, worst=worst, worst_angle_off=worst_off,
        n_exposed=n_exposed, reasons=reasons,
    )


if __name__ == "__main__":
    from console import use_utf8_stdout
    use_utf8_stdout()

    from mock_residence import mock_buildings
    from residence import find_receptors

    LAT, LON = 36.5666, 126.5500
    recs, _ = find_receptors(LAT, LON, buildings=mock_buildings(LAT, LON))
    print(f"수용점 {len(recs)}동\n")

    cases = [
        ("낮·강풍·서풍",   "270", 5.0, "1", datetime(2026, 8, 8, 14)),
        ("낮·약풍·남서풍", "225", 1.5, "1", datetime(2026, 8, 8, 14)),
        ("저녁·약풍",      "225", 1.5, "1", datetime(2026, 8, 8, 21)),
        ("새벽·무풍·맑음", "225", 0.7, "1", datetime(2026, 8, 9, 3)),
        ("새벽·무풍·흐림", "225", 0.7, "4", datetime(2026, 8, 9, 3)),
    ]
    base = None
    for note, vec, wsd, sky, dt in cases:
        r = dispersion(vec, wsd, sky, dt, LAT, LON, recs)
        if base is None:
            base = r.factor or 1.0
        ratio = r.factor / base if base else 0.0
        print(f"{note:16s} 안정도 {r.stability}  factor {r.factor:.3e}  "
              f"상대 {ratio:6.2f}배  풍하 {r.n_exposed:3d}동")
        for line in r.reasons:
            print(f"      - {line}")
        print()
