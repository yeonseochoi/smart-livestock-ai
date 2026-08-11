"""대기안정도 판정과 가우시안 플룸 기반 상대 확산 계산.

2026-08-07 신규. 기존 diffusion.py 의 경험 배수(D_DIRECT 3.0 / D_STAGNANT 2.0 /
D_INVERSION 2.0, ±45° 섹터)를 대체한다.

────────────────────────────────────────────────────────────────────
왜 바꿨나
    기존 방식의 숫자 4개(3.0, 2.0, 2.0, 45°)는 전부 근거가 없었다.
    특히 ±45° 는 실제 플룸 폭과 크게 어긋난다. Briggs 계수로 계산한
    "중심선 농도의 10%까지" 유효 반각은 다음과 같다.

        거리 850m  |  A 24.4°  B 18.2°  C 12.8°  D 9.4°  E 7.0°  F 4.7°

    즉 45° 는 1.8~9.6배 과대하다. 게다가 가장 위험한 조건(F, 야간 역전층)에서
    플룸이 가장 좁아, 45° 고정 방식은 정확히 거꾸로 작동한다.
    850m 지점에서 정확히 45° 벗어난 곳의 상대농도는 A 에서 1.4e-05,
    F 에서 5.6e-148 로 사실상 0 이다.

    Pasquill 안정도 + Briggs 확산계수는 국제 표준이고, 같은 목적의
    운영 시스템(Oklahoma Dispersion Model, 1990년대~현재)도 이 체계를 쓴다.

────────────────────────────────────────────────────────────────────
쓰는 범위 — 절대 농도를 단정하지 않는다
    축산 악취에 대한 CALPUFF/ISCST3/AERMOD 검증 논문들이 일관되게 보고한다.
      · 예측/실측 비 1.40 ~ 9.37, 80% 이상이 과대예측
      · 건물 근처에서 2배 이상 과대 또는 과소
      · 평균 배출률로는 peak(순간 최대) 농도를 예측할 수 없음

    악취 민원은 1시간 평균이 아니라 순간 냄새로 발생하므로,
    "0.8 ppm 이니 법 기준 이하 ✅" 같은 판정은 위험하다.
    따라서 이 모듈은 다음만 제공한다.

      ✅ 시각 간 상대 비교 (배수)
      ✅ 안정도 등급
      ✅ 방향에 따른 연속 감쇠
      ⚠️ ppm 은 참고 구간으로만. MODEL_BIAS_RANGE 를 반드시 병기
      ❌ "법 기준 이하" 합격 판정은 하지 않는다
"""

from __future__ import annotations

import math
from datetime import datetime

from constants import (
    AREA_SOURCE_DIVISOR_Y,
    AREA_SOURCE_DIVISOR_Z,
    APPLICATION_RATE_TON_PER_HA,
    BRIGGS_SIGMA_Y,
    BRIGGS_SIGMA_Z,
    EMISSION_DURATION_H,
    MOLAR_VOLUME_25C,
    NH3_DETECTION_PPM,
    NH3_LEGAL_LIMIT_PPM,
    NH3_MOLAR_MASS,
    NH3_TARGET_INTENSITY_PPM,
    ODOR_INTENSITY_INTERCEPT,
    ODOR_INTENSITY_SLOPE,
    ODOR_INTENSITY_VALID_PPM,
    PASQUILL_TABLE,
    SKY_CLEAR,
    SOLAR_ELEV_MODERATE,
    SOLAR_ELEV_STRONG,
    SOURCE_HEIGHT_M,
    WSD_FLOOR,
)

# 수치 안정성 가드 — 물리 상수가 아니라 0 나눗셈 방지용
_MIN_DOWNWIND_M = 10.0

# 안정도 → 사람이 읽는 라벨 (Oklahoma Dispersion Model 6단계와 대응)
STABILITY_LABEL = {
    "A": "매우 불안정 / 확산 매우 좋음",
    "B": "불안정 / 확산 좋음",
    "C": "약간 불안정 / 확산 양호",
    "D": "중립 / 확산 보통",
    "E": "안정 / 확산 나쁨",
    "F": "매우 안정 / 야간 역전층, 확산 최저",
}
STABILITY_ORDER = ("A", "B", "C", "D", "E", "F")


# ═══════════════════════════════════════════════════════════════════
# 1. 태양고도
# ═══════════════════════════════════════════════════════════════════

def solar_elevation(lat: float, lon: float, dt: datetime,
                    tz_offset_h: float = 9.0) -> float:
    """태양고도(도). 음수면 일몰 후.

    dt 는 지방시(한국 표준시, UTC+9)로 받는다.
    적위는 Cooper(1969), 균시차는 Spencer 근사식을 쓴다.
    안정도 등급을 가르는 데 필요한 정밀도(±1°)면 충분하다.
    """
    n = dt.timetuple().tm_yday
    utc_hour = dt.hour + dt.minute / 60.0 - tz_offset_h

    decl = 23.45 * math.sin(math.radians(360.0 * (284 + n) / 365.0))

    b = math.radians(360.0 * (n - 81) / 364.0)
    eot_min = 9.87 * math.sin(2 * b) - 7.53 * math.cos(b) - 1.5 * math.sin(b)

    solar_time = utc_hour + lon / 15.0 + eot_min / 60.0
    hour_angle = 15.0 * (solar_time - 12.0)

    lat_r, decl_r, ha_r = map(math.radians, (lat, decl, hour_angle))
    sin_elev = (math.sin(lat_r) * math.sin(decl_r)
                + math.cos(lat_r) * math.cos(decl_r) * math.cos(ha_r))
    return math.degrees(math.asin(max(-1.0, min(1.0, sin_elev))))


# ═══════════════════════════════════════════════════════════════════
# 2. Pasquill 안정도
# ═══════════════════════════════════════════════════════════════════

def insolation_class(solar_elev: float, sky: str) -> str:
    """주간 일사강도 -> 'strong' | 'moderate' | 'slight'.

    태양고도로 기본 등급을 정하고 구름으로 낮춘다(Turner 1967 방식).
    """
    sky = str(sky)
    if solar_elev > SOLAR_ELEV_STRONG:
        base = "strong"
    elif solar_elev > SOLAR_ELEV_MODERATE:
        base = "moderate"
    else:
        base = "slight"

    if sky == SKY_CLEAR:                 # 맑음 — 그대로
        return base
    if sky == "3":                       # 구름많음 — 한 단계 하향
        return {"strong": "moderate", "moderate": "slight", "slight": "slight"}[base]
    return "slight"                      # 흐림(4) 또는 미상 — 최하


def pasquill_class(wsd: float, sky: str, dt: datetime,
                   lat: float, lon: float) -> tuple[str, float]:
    """(안정도 등급 'A'~'F', 태양고도) 를 반환한다.

    주간/야간은 태양고도 0도를 기준으로 가른다.
    야간 운량은 기상청 SKY 로 근사한다. SKY=1(맑음, 0~5/10)은
    Pasquill 표의 4/8 경계에 걸치므로 보수적으로 '맑음'(F)으로 본다.
    """
    elev = solar_elevation(lat, lon, dt)
    u = float(wsd)
    sky = str(sky)

    if elev > 0:
        col = {"strong": 0, "moderate": 1, "slight": 2}[insolation_class(elev, sky)]
    else:
        col = 4 if sky == SKY_CLEAR else 3      # 4=맑은 밤(F계열), 3=흐린 밤

    for upper in sorted(PASQUILL_TABLE):
        if u < upper:
            return PASQUILL_TABLE[upper][col], elev
    return PASQUILL_TABLE[float("inf")][col], elev


# ═══════════════════════════════════════════════════════════════════
# 3. Briggs 확산계수
# ═══════════════════════════════════════════════════════════════════

def briggs_sigma(x_m: float, stability: str) -> tuple[float, float]:
    """풍하거리 x(m) 에서의 (sigma_y, sigma_z) [m].  Open-Country 조건."""
    if stability not in BRIGGS_SIGMA_Y:
        raise ValueError(f"알 수 없는 안정도 등급: {stability!r}")
    x = max(float(x_m), _MIN_DOWNWIND_M)

    ay, by, cy = BRIGGS_SIGMA_Y[stability]
    az, bz, cz = BRIGGS_SIGMA_Z[stability]
    sigma_y = ay * x * (1.0 + by * x) ** cy
    sigma_z = az * x * (1.0 + bz * x) ** cz
    return sigma_y, sigma_z


def plume_half_angle(x_m: float, stability: str, frac: float = 0.10) -> float:
    """중심선 농도의 frac 배까지를 플룸으로 볼 때의 유효 반각(도).

    설명·검증용이며 위험도 계산에는 쓰지 않는다(연속 감쇠를 쓰므로).
    """
    if not 0.0 < frac < 1.0:
        raise ValueError("frac 은 0과 1 사이여야 합니다")
    x = max(float(x_m), _MIN_DOWNWIND_M)
    sigma_y, _ = briggs_sigma(x, stability)
    y = sigma_y * math.sqrt(2.0 * math.log(1.0 / frac))
    return math.degrees(math.atan(y / x))


# ═══════════════════════════════════════════════════════════════════
# 4. 면오염원 → 가상점원 보정
# ═══════════════════════════════════════════════════════════════════

def source_width_m(tons: float,
                   rate_ton_per_ha: float = APPLICATION_RATE_TON_PER_HA) -> float:
    """살포량(톤) → 살포지 한 변 길이(m). 정사각형 밭 가정.

    [C] 살포 기준량은 작물·지자체별로 달라 근거값이 아니다.
    실측상 수용점 거리가 500m 를 넘으면 이 보정의 효과는 2% 미만이다.
    """
    if tons <= 0 or rate_ton_per_ha <= 0:
        return 0.0
    area_m2 = (tons / rate_ton_per_ha) * 10_000.0
    return math.sqrt(area_m2)


def initial_sigmas(width_m: float,
                   height_m: float = SOURCE_HEIGHT_M) -> tuple[float, float]:
    """가상점원 보정용 초기 확산폭 (sigma_y0, sigma_z0)."""
    return (max(width_m, 0.0) / AREA_SOURCE_DIVISOR_Y,
            max(height_m, 0.0) / AREA_SOURCE_DIVISOR_Z)


# ═══════════════════════════════════════════════════════════════════
# 5. 상대 확산 인자
# ═══════════════════════════════════════════════════════════════════

def dispersion_factor(dist_m: float, angle_off_deg: float, stability: str,
                      wsd: float, sigma_y0: float = 0.0,
                      sigma_z0: float = 0.0) -> float:
    """지표 배출 · 지표 수용점의 상대 확산 인자 [s/m^3].

        농도[g/m^3] = 배출률[g/s] x 이 값

    지표 반사를 포함한 표준형:
        chi = Q / (pi * u * sy * sz) * exp(-y^2 / (2*sy^2))

    angle_off_deg 는 '악취가 흘러가는 방향'과 '농장→수용점 방위'의 차이(0~180).
    90도 이상이면 수용점이 풍상측이므로 0 을 반환한다.
    """
    d = float(dist_m)
    if d <= 0:
        return 0.0

    off = abs(float(angle_off_deg))
    if off >= 90.0:
        return 0.0

    u = max(float(wsd), WSD_FLOOR)
    th = math.radians(off)
    x_down = d * math.cos(th)          # 풍하 거리
    y_cross = d * math.sin(th)         # 중심선에서 옆으로 벗어난 거리
    if x_down <= 0:
        return 0.0

    sigma_y, sigma_z = briggs_sigma(x_down, stability)
    sigma_y = math.hypot(sigma_y, sigma_y0)
    sigma_z = math.hypot(sigma_z, sigma_z0)

    # exp 언더플로 방지 — 지수가 -700 아래면 float 로 0
    expo = -(y_cross * y_cross) / (2.0 * sigma_y * sigma_y)
    if expo < -700.0:
        return 0.0

    return math.exp(expo) / (math.pi * u * sigma_y * sigma_z)


# ═══════════════════════════════════════════════════════════════════
# 6. 농도 환산 · 악취강도  (참고값 전용)
# ═══════════════════════════════════════════════════════════════════

def emission_rate_g_s(emission_kg: float,
                      duration_h: float = EMISSION_DURATION_H) -> float:
    """총 배출량(kg NH3) → 평균 배출률(g/s).

    [B/C] 살포 직후 휘산이 가장 크고 수 시간에 걸쳐 감쇠하지만,
    실측 감쇠곡선이 없어 duration_h 동안 균등하다고 본다.
    """
    if duration_h <= 0:
        raise ValueError("duration_h 는 0보다 커야 합니다")
    return emission_kg * 1000.0 / (duration_h * 3600.0)


def concentration_ppm(emission_kg: float, factor: float,
                      duration_h: float = EMISSION_DURATION_H) -> float:
    """상대 확산 인자 → NH3 농도 추정치 [ppm, 25 °C 기준].

    ⚠️ 참고값이다. 검증 논문상 실측 대비 1.40~9.37배 과대이며
       순간 peak 는 예측하지 못한다. 합격/불합격 판정에 쓰지 말 것.
    """
    if factor <= 0:
        return 0.0
    q = emission_rate_g_s(emission_kg, duration_h)     # g/s
    chi_g_m3 = q * factor                              # g/m^3
    chi_mg_m3 = chi_g_m3 * 1000.0
    return chi_mg_m3 * MOLAR_VOLUME_25C / NH3_MOLAR_MASS


def odor_intensity(ppm: float) -> float | None:
    """악취강도 I = 1.9352 log10(C) + 0.19.

    논문 관능실험 범위(4.98~165.87 ppm) 밖이면 None 을 돌려준다.
    외삽하면 음수 강도가 나와 무의미하다.
    """
    lo, hi = ODOR_INTENSITY_VALID_PPM
    if ppm is None or ppm <= 0 or not (lo <= ppm <= hi):
        return None
    return ODOR_INTENSITY_SLOPE * math.log10(ppm) + ODOR_INTENSITY_INTERCEPT


def odor_band(ppm: float) -> str:
    """농도를 4구간으로 분류한다. 절대 판정이 아니라 위치 표시용."""
    if ppm < NH3_DETECTION_PPM:
        return "감지 불가"
    if ppm < NH3_LEGAL_LIMIT_PPM:
        return "감지되나 법 기준 이하"
    if ppm < NH3_TARGET_INTENSITY_PPM:
        return "법 기준 초과"
    return "규제 설계 목표(악취강도 2.5) 초과"


# ═══════════════════════════════════════════════════════════════════
# 자체 점검
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("── 태양고도 (서울 37.5665, 126.978)")
    for mmdd, hh in ((6, 21), (12, 21)):
        for h in (6, 9, 12, 15, 18):
            e = solar_elevation(37.5665, 126.978, datetime(2026, mmdd, hh, h))
            print(f"   {mmdd:02d}/{hh:02d} {h:02d}시  {e:6.1f}°")

    print("\n── Pasquill 안정도")
    cases = [
        (1.0, "1", datetime(2026, 8, 8, 3),  "새벽·무풍·맑음"),
        (1.0, "4", datetime(2026, 8, 8, 3),  "새벽·무풍·흐림"),
        (4.0, "1", datetime(2026, 8, 8, 13), "낮·바람·맑음"),
        (7.0, "4", datetime(2026, 8, 8, 13), "낮·강풍·흐림"),
        (1.5, "1", datetime(2026, 8, 8, 13), "낮·무풍·맑음"),
    ]
    for wsd, sky, dt, note in cases:
        stab, elev = pasquill_class(wsd, sky, dt, 36.5666, 126.55)
        print(f"   {note:16s} u={wsd:4.1f} SKY={sky}  →  {stab}  "
              f"(태양고도 {elev:5.1f}°)  {STABILITY_LABEL[stab]}")

    print("\n── 플룸 유효 반각 (중심선의 10%까지)")
    print("   거리m " + "".join(f"{s:>8}" for s in STABILITY_ORDER))
    for x in (300, 850, 2000):
        print(f"   {x:5d} " + "".join(
            f"{plume_half_angle(x, s):7.1f}°" for s in STABILITY_ORDER))

    print("\n── 상대 확산 인자 (거리 850m, 정면)")
    for s in STABILITY_ORDER:
        f = dispersion_factor(850, 0, s, 2.0)
        print(f"   {s}: {f:.3e} s/m³")

    print("\n── 45° 벗어난 지점의 상대농도 (거리 850m)")
    for s in STABILITY_ORDER:
        f0 = dispersion_factor(850, 0, s, 2.0)
        f45 = dispersion_factor(850, 45, s, 2.0)
        print(f"   {s}: {f45 / f0:.2e} 배" if f0 else f"   {s}: -")

    print("\n── 농도 구간")
    for ppm in (0.01, 0.5, 3.0, 20.0):
        i = odor_intensity(ppm)
        istr = f"{i:.2f}" if i is not None else "범위 밖"
        print(f"   {ppm:6.2f} ppm  →  {odor_band(ppm):28s} 악취강도 {istr}")
