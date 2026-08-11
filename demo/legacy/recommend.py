"""배출량(E)과 확산(factor)을 결합해 상대 위험도를 계산하고 권장/회피 시각을 추천한다.

2026-08-07 개정. 이전 대비 바뀐 점
    1. diffusion() 인터페이스 변경 — 수용점 리스트를 받아 전수 계산 후 max()
    2. 위험도를 '중앙값 시각 대비 배수'로 정규화해 출력 (2026-08-07 재수정)
    3. 절대 농도 ppm 은 참고 구간으로만 표기. 합격/불합격 판정을 하지 않음
    4. 안정도 등급을 결과에 포함

유지되는 규칙 (팀원 합의사항, 2026-08-07)
    · 배출량은 emission_kg() 반환값 그대로 사용 — 중간 반올림 없음
    · 예보는 시간순 처리 (방어적으로 재정렬)
    · 이후 6시간(TIME_WEIGHTS) 창이 실제 timestamp 기준으로 모두 존재하는
      시각만 추천/회피 후보로 삼는다. 간격을 1시간이라 가정하지 않는다.
    · 반올림은 출력 직전에만. min/max 는 원값으로 비교
    · 출력 문자열은 ASCII 기호만 (한국어 Windows cp949 콘솔 대응)
    · 동률이면 가장 이른 시각
"""

from __future__ import annotations

from datetime import datetime, timedelta
from statistics import median

from console import use_utf8_stdout as _use_utf8_stdout
from constants import MODEL_BIAS_RANGE, RISK_LEVELS, TIME_WEIGHTS
from diffusion import dispersion
from emission import advice_lines, emission_kg
from geo import latlon_to_grid
from kma import fetch_forecast as kma_fetch_forecast
from plume import concentration_ppm, odor_band, odor_intensity, source_width_m
from residence import Receptor

REQUIRED = {"VEC", "WSD", "TMP", "SKY"}


def _parse_dt(key: str) -> datetime:
    """'YYYYMMDD HHMM' → datetime."""
    return datetime.strptime(key, "%Y%m%d %H%M")


def risk_level(risk_relative: float) -> str:
    """중앙값 기준 배수 → 등급 라벨."""
    for upper, label in RISK_LEVELS:
        if risk_relative < upper:
            return label
    return RISK_LEVELS[-1][1]


# 하위호환 재수출. 실제 구현은 console.py 에 있다(순환 임포트 회피).
use_utf8_stdout = _use_utf8_stdout


def evaluate(farm_lat: float, farm_lon: float,
             receptors: list[Receptor],
             tons: float,
             method: str = "표면살포",
             tillage: bool = False,
             tn_kg_per_ton: float | None = None,
             use_time_weight: bool = True,
             use_area_source: bool = True,
             fetch_forecast_fn=kma_fetch_forecast) -> dict:
    """전체 파이프라인 실행 → 결과 dict.

    receptors 는 residence.find_receptors() 가 돌려준 주거 건물 목록이다.
    비어 있으면 확산 계산이 전부 0 이 되므로 별도 안내를 담아 돌려준다.
    """
    if tons <= 0:
        raise ValueError("살포량(tons)은 0보다 커야 합니다")

    nx, ny = latlon_to_grid(farm_lat, farm_lon)
    data = fetch_forecast_fn(nx, ny)

    keys = sorted(k for k in data if REQUIRED <= set(data[k]))
    if not keys:
        raise RuntimeError("VEC/WSD/TMP/SKY 를 모두 갖춘 예보 시점이 없습니다")

    times = {k: _parse_dt(k) for k in keys}
    width = source_width_m(tons) if use_area_source else 0.0

    # ── 시각별 확산 계산 ────────────────────────────────────────
    disp: dict[str, object] = {}
    dt_to_factor: dict[datetime, float] = {}
    for k in keys:
        v = data[k]
        res = dispersion(v["VEC"], v["WSD"], v["SKY"], times[k],
                         farm_lat, farm_lon, receptors, width)
        disp[k] = res
        dt_to_factor[times[k]] = res.factor

    # ── 시각별 배출량 · 위험도 ─────────────────────────────────
    raw_rows = []
    for k in keys:
        res = disp[k]
        e = emission_kg(tons, method, tn_kg_per_ton, tillage, times[k].month)

        if use_time_weight:
            # 살포 후 최대 6시간의 확산 인자를 가중평균.
            # offset 마다 실제 timestamp 존재를 확인한다(간격 가정 금지).
            acc = w_sum = 0.0
            found_all = True
            for j, w in enumerate(TIME_WEIGHTS):
                t_j = times[k] + timedelta(hours=j)
                if t_j in dt_to_factor:
                    acc += w * dt_to_factor[t_j]
                    w_sum += w
                else:
                    found_all = False
            f_eff = acc / w_sum if w_sum else res.factor
            insufficient = not found_all
        else:
            f_eff = res.factor
            insufficient = False

        raw_rows.append({
            "key": k, "v": data[k], "disp": res,
            "f_eff": f_eff, "emission": e, "risk": e * f_eff,
            "insufficient_forecast": insufficient,
        })

    candidates = [r for r in raw_rows if not r["insufficient_forecast"]]
    if not candidates:
        raise RuntimeError(
            "6시간 예보 창이 완전한 후보 시각이 없습니다 "
            "(예보 시계열이 너무 짧거나 시각이 누락됨)"
        )

    # ── 상대 위험도 정규화 ──────────────────────────────────────
    # ⚠️ 2026-08-07 수정. 이전에는 '0보다 큰 최소 위험도'를 1.0 기준으로 삼았으나
    #    분모가 가우시안 꼬리(exp 감쇠 극단)에서 나오면 값이 폭발했다.
    #    실측: 마을 배치(시드)만 바꿔도 회피 시각 배수가 32.5 ~ 1,978,671 로 요동.
    #    → 중앙값 기준으로 교체. 같은 실험에서 15.3 ~ 20.7 로 안정된다.
    #    해석도 더 자연스럽다 — 1.0 = "평범한 시각", 15 = "평소의 15배".
    positive = sorted(r["risk"] for r in candidates if r["risk"] > 0)
    base = median(positive) if positive else 0.0

    def rel(risk: float) -> float:
        return risk / base if base > 0 else 0.0

    best = min(candidates, key=lambda r: r["risk"])
    worst = max(candidates, key=lambda r: r["risk"])

    # ── timeline ───────────────────────────────────────────────
    timeline = []
    for r in raw_rows:
        res = r["disp"]
        ppm = concentration_ppm(r["emission"], r["f_eff"])
        intensity = odor_intensity(ppm)
        timeline.append({
            "datetime": r["key"],
            "vec": r["v"]["VEC"], "wsd": r["v"]["WSD"],
            "tmp": r["v"]["TMP"], "sky": r["v"]["SKY"],
            "stability": res.stability,
            "solar_elev": round(res.solar_elev, 1),
            "wind_to": round(res.wind_to, 1),
            "emission_kg": round(r["emission"], 3),
            "factor": r["f_eff"],
            "risk_relative": round(rel(r["risk"]), 2),
            "risk_level": risk_level(rel(r["risk"])),
            "n_exposed": res.n_exposed,
            "worst_dist_m": round(res.worst.dist_m) if res.worst else None,
            "worst_angle_off": (round(res.worst_angle_off, 1)
                                if res.worst_angle_off is not None else None),
            "ppm_reference": round(ppm, 4),
            "ppm_band": odor_band(ppm),
            "odor_intensity": round(intensity, 2) if intensity is not None else None,
            "reasons": res.reasons,
            "insufficient_forecast": r["insufficient_forecast"],
        })

    month0 = times[keys[0]].month

    return {
        "emission_kg": round(raw_rows[0]["emission"], 3),
        "recommended": {
            "datetime": best["key"],
            "risk_relative": round(rel(best["risk"]), 2),
            "risk_level": risk_level(rel(best["risk"])),
            "stability": best["disp"].stability,
            "reasons": best["disp"].reasons,
        },
        "avoid": {
            "datetime": worst["key"],
            "risk_relative": round(rel(worst["risk"]), 2),
            "risk_level": risk_level(rel(worst["risk"])),
            "stability": worst["disp"].stability,
            "reasons": worst["disp"].reasons,
        },
        "timeline": timeline,
        "advice": advice_lines(tons, method, tillage, month0),
        "meta": {
            "grid": {"nx": nx, "ny": ny},
            "n_receptors": len(receptors),
            "nearest_receptor_m": (round(min(r.dist_m for r in receptors))
                                   if receptors else None),
            "n_forecast_points": len(timeline),
            "n_candidates": len(candidates),
            "source_width_m": round(width, 1),
            "all_zero": base == 0.0,
        },
        "disclaimer": (
            f"위험도는 예보 구간의 중앙값 시각을 1.0으로 둔 상대 배수입니다. "
            f"ppm 은 참고값이며 검증 논문상 실측 대비 "
            f"{MODEL_BIAS_RANGE[0]}~{MODEL_BIAS_RANGE[1]}배 과대 가능하고 "
            f"순간 peak 는 예측하지 못합니다. 법 기준 합격 판정에 쓰지 마세요."
        ),
    }


def format_report(res: dict, top: int = 10) -> str:
    """사람이 읽는 요약 문자열.

    ⚠️ 여기서는 ASCII 기호만 쓴다. 한국어 Windows 기본 콘솔(cp949)이
       이모지·em dash 를 인코딩하지 못해 UnicodeEncodeError 로 죽기 때문이다.
       (2026-08-07 Claude Code 검증에서 발견)
    """
    m = res["meta"]
    out = [
        "=" * 68,
        " 액비 살포 권장 시각  |  향후 84시간",
        f" 인근 주거 {m['n_receptors']}동"
        + (f" | 최근접 {m['nearest_receptor_m']}m" if m["nearest_receptor_m"] else "")
        + f" | 격자 ({m['grid']['nx']}, {m['grid']['ny']})",
        f" 배출량 {res['emission_kg']} kg NH3 | 살포지 한 변 {m['source_width_m']}m",
        "=" * 68,
        "",
    ]

    if m["all_zero"]:
        out.append(" [i] 예보 전 구간에서 풍하측에 주거지가 없습니다.")
        out.append("")

    out.append(f"{'시각':<15}{'위험도':>8}{'등급':>10}{'안정도':>7}"
               f"{'풍향':>6}{'풍속':>6}{'풍하':>6}")
    out.append("-" * 68)
    rows = [r for r in res["timeline"] if not r["insufficient_forecast"]]
    rows.sort(key=lambda r: r["risk_relative"])
    for r in rows[:top]:
        out.append(
            f"{r['datetime']:<15}{r['risk_relative']:>8.2f}{r['risk_level']:>10}"
            f"{r['stability']:>7}{r['vec']:>6}{r['wsd']:>6}{r['n_exposed']:>6}"
        )
    out.append("-" * 68)

    rec, avd = res["recommended"], res["avoid"]
    out.append(f"\n[권장] {rec['datetime']}   위험도 {rec['risk_relative']} "
               f"({rec['risk_level']})")
    for line in rec["reasons"]:
        out.append(f"       - {line}")
    out.append(f"\n[회피] {avd['datetime']}   위험도 {avd['risk_relative']} "
               f"({avd['risk_level']})")
    for line in avd["reasons"]:
        out.append(f"       - {line}")

    out.append("\n[저감 방안]")
    for a in res["advice"]:
        out.append(f"       - {a}")

    out.append(f"\n[주의] {res['disclaimer']}")
    out.append("=" * 68)
    return "\n".join(out)
