"""S5 — 작업 창 추천 (시점이 아니라 '창'을 추천).

window_risk(t) = Σ w_j x risk_prob(t+j시간) / Σ w_j   (w = legacy TIME_WEIGHTS)
final(t) = window_risk x work_weight[작업유형] x storage_factor(경과일)

규칙 3개 (legacy 승계): ① 창을 못 덮는 시각 제외 ② 동률이면 이른 시각
③ min/max 는 원값 비교, 반올림은 출력 직전.

플룸 보정: 곱셈 금지(절대 규칙 1). 추천 창 대표 시각에 legacy diffusion 을 호출해
최악 수용점이 플룸 유효 반각 안이면 '등급 1단계 상향' 이산 보정만 한다.
"""
from __future__ import annotations

from datetime import datetime, timedelta

from config import DEMO_FARM, PLUME_GRADE_BUMP, PROV, finding
from ops import db

# legacy import (수정 금지)
from constants import TIME_WEIGHTS
from diffusion import dispersion
from emission import emission_kg
from plume import plume_half_angle
from residence import find_receptors
from mock_residence import mock_buildings
import mock_forecast
from geo import latlon_to_grid

# [C] 작업유형 가중치 — 구현 내용.md 는 work_weight 를 요구만 하고 값을 주지 않았다.
#     아래는 임의 설정값이므로 발표 시 한계로 명시할 것.
WORK_WEIGHT = {
    "분뇨제거": 1.3, "청소": 1.0, "환기점검": 0.7,
    "저감시설점검": 0.8, "액비살포": 1.5,
}
GRADES = ["낮음", "주의", "위험"]


def storage_factor(days: int | None) -> float:
    # 경과일 < 14일: 1.0, 이후 1.5 [C] (2주 임계점 문헌은 [A], 배수 1.5는 임의)
    if days is None:
        return 1.0
    return 1.0 if days < 14 else 1.5


def _load_calendar() -> dict[datetime, dict]:
    con = db.connect()
    rows = con.execute(
        "SELECT date, block, risk_prob, risk_grade FROM risk_calendar").fetchall()
    con.close()
    cal = {}
    for date, block, prob, grade in rows:
        t = datetime.strptime(date, "%Y-%m-%d") + timedelta(hours=3 * block)
        cal[t] = {"prob": prob, "grade": grade}
    return cal


def window_risk(cal: dict[datetime, dict], start: datetime) -> float | None:
    """TIME_WEIGHTS 는 시간 단위, 캘린더는 3시간 블록 단위 → t+j시간이 속한
    블록의 확률을 쓴다. 창(6시간)을 전부 못 덮으면 None(후보 제외, 규칙 ①)."""
    acc = w_sum = 0.0
    for j, w in enumerate(TIME_WEIGHTS):
        t = start + timedelta(hours=j)
        b = t.replace(hour=(t.hour // 3) * 3, minute=0, second=0, microsecond=0)
        if b not in cal:
            return None
        acc += w * cal[b]["prob"]
        w_sum += w
    return acc / w_sum


def recommend(work_type: str, storage_days: int | None = None,
              tons: float = 20.0, method: str = "표면살포") -> dict:
    if work_type not in WORK_WEIGHT:
        raise ValueError(f"작업유형은 {list(WORK_WEIGHT)} 중 하나")

    cal = _load_calendar()
    sf = storage_factor(storage_days)

    cands = []
    for t in sorted(cal):  # 시각 오름차순 → 동률이면 이른 시각 (규칙 ②)
        wr = window_risk(cal, t)
        if wr is None:
            continue
        cands.append({"t": t, "window_risk": wr,
                      "final": wr * WORK_WEIGHT[work_type] * sf,
                      "grade": cal[t]["grade"]})
    if not cands:
        raise RuntimeError("창을 덮을 예보가 있는 후보 시각이 없습니다")

    best = min(cands, key=lambda c: c["final"])   # 원값 비교 (규칙 ③)
    worst = max(cands, key=lambda c: c["final"])

    # ── 플룸 이산 보정 (곱셈 금지) ─────────────────────────────────
    recs, _ = find_receptors(DEMO_FARM["lat"], DEMO_FARM["lon"],
                             buildings=mock_buildings(DEMO_FARM["lat"], DEMO_FARM["lon"]))
    PROV.log("D7 주거건물", "legacy/mock_residence.py", real=False,
             note=f"VWORLD_KEY 미설정 → mock 폴백(공식 데모 경로), 수용점 {len(recs)}동")

    nx, ny = latlon_to_grid(DEMO_FARM["lat"], DEMO_FARM["lon"])
    fc = mock_forecast.fetch_with_fallback(nx, ny)
    key = best["t"].strftime("%Y%m%d %H00")
    plume_note, bumped = None, False
    if key in fc:
        v = fc[key]
        r = dispersion(v["VEC"], float(v["WSD"]), v["SKY"], best["t"],
                       DEMO_FARM["lat"], DEMO_FARM["lon"], recs)
        if r.worst is not None and r.worst_angle_off is not None:
            half = plume_half_angle(r.worst.dist_m, r.stability)
            in_plume = r.worst_angle_off <= half
            if in_plume and PLUME_GRADE_BUMP:
                # 재검증 통과 후에만 켠다 (config.PLUME_GRADE_BUMP)
                gi = GRADES.index(best["grade"]) if best["grade"] in GRADES else 0
                best["grade"] = GRADES[min(gi + 1, 2)]
                bumped = True
                plume_note = (f"풍하측 민가 {r.n_exposed}동 — 최악 수용점 "
                              f"{r.worst.dist_m:.0f}m, 이탈각 {r.worst_angle_off:.0f}도 "
                              f"(유효 반각 {half:.0f}도) → 등급 1단계 상향")
            elif in_plume:
                plume_note = (f"참고: 풍하측 민가 {r.n_exposed}동 (미검증 모델 — "
                              f"등급 반영 안 함. S8-4 재검증 후 활성화)")
            else:
                plume_note = (f"참고: 플룸 중심부 밖 (이탈각 {r.worst_angle_off:.0f}도 > "
                              f"반각 {half:.0f}도, 미검증 모델)")
    else:
        finding("추천 창 대표 시각이 예보 dict 에 없어 플룸 보정을 건너뜀 "
                "(risk_calendar 시각과 예보 시각의 정렬 문제 — 계획서 미규정)")

    out = {
        "work_type": work_type,
        "storage_days": storage_days,
        "storage_factor": sf,
        "recommended": {"t": best["t"].strftime("%Y-%m-%d %H시"),
                        "final": round(best["final"], 4), "grade": best["grade"],
                        "plume_bumped": bumped, "plume_note": plume_note},
        "avoid": {"t": worst["t"].strftime("%Y-%m-%d %H시"),
                  "final": round(worst["final"], 4), "grade": worst["grade"]},
        "n_candidates": len(cands),
    }
    # 액비 살포는 legacy 정량 배출 모델 연결
    if work_type == "액비살포":
        out["emission_kg"] = round(emission_kg(tons, method, None, False,
                                               best["t"].month), 3)
        out["emission_note"] = f"{tons}톤 {method} 기준 NH3 배출량 (legacy/emission.py)"
    return out
