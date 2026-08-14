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
from serving import db

# legacy import (수정 금지)
from constants import TIME_WEIGHTS
from diffusion import dispersion
from emission import emission_kg, advice_lines
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


# ═══════════════════════════════════════════════════════════════════
# v6 — (1시간 x 그룹) 캘린더 기반 추천
# ═══════════════════════════════════════════════════════════════════

def _load_calendar_v6() -> dict:
    """{그룹명: {시각: {prob, grade}}}"""
    con = db.connect()
    rows = con.execute(
        "SELECT date, hour, grp, risk_prob, risk_grade FROM risk_calendar_v6"
    ).fetchall()
    con.close()
    cal: dict = {}
    for date, hour, grp, prob, grade in rows:
        t = datetime.strptime(date, "%Y-%m-%d") + timedelta(hours=int(hour))
        cal.setdefault(grp, {})[t] = {"prob": prob, "grade": grade}
    return cal


def _load_forecast_wind() -> dict:
    """{시각: {wd, ws, sky}} — 플룸 그룹 선택용. daily_scoring 가 저장한 예보 원값."""
    con = db.connect()
    try:
        rows = con.execute(
            "SELECT date, hour, wd, ws, sky FROM forecast_hourly_v6").fetchall()
    except Exception:
        rows = []
    con.close()
    out = {}
    for date, hour, wd, ws, sky in rows:
        t = datetime.strptime(date, "%Y-%m-%d") + timedelta(hours=int(hour))
        out[t] = {"wd": wd, "ws": ws, "sky": sky}
    return out


def window_risk_v6(cal: dict, start: datetime) -> float | None:
    """1시간 격자이므로 TIME_WEIGHTS 6칸이 시각에 1:1 대응한다.

    v5 는 6개 가중치를 3시간 블록 2개에 뭉개 매핑했다. 1시간 격자에서는
    '살포 후 0~5시간' 이라는 원래 의도대로 작동한다.
    창(6시간)을 전부 못 덮으면 None → 후보 제외 (규칙 ①).
    """
    acc = w_sum = 0.0
    for j, w in enumerate(TIME_WEIGHTS):
        t = start + timedelta(hours=j)
        if t not in cal:
            return None
        acc += w * cal[t]["prob"]
        w_sum += w
    return acc / w_sum


def combine(vals: dict, rule: str = "max") -> float:
    """풍하측 그룹들의 위험도를 하나로 합치는 규칙.

    어느 규칙이 옳은지 검증할 데이터가 없다(농가별 작업↔민원 기록 부재).
    그래서 하나를 고르지 않고 네 개를 모두 계산해 '등급 일치율'을 보고한다.
    """
    v = list(vals.values())
    if rule == "max":
        return max(v)
    if rule == "mean":
        return sum(v) / len(v)
    if rule == "min":
        return min(v)
    if rule == "first":
        return v[0]
    raise ValueError(rule)


COMBINE_RULES = ("max", "mean", "min", "first")


def recommend_v6(work_type: str, storage_days: int | None = None,
                 tons: float = 20.0, method: str = "표면살포",
                 tillage: bool = False, species: str = "돼지",
                 farm_lat: float | None = None, farm_lon: float | None = None) -> dict:
    """v6 추천.

    ★ 배출량 절대값(kg NH3)은 출력하지 않는다.
      emission.py 는 액비(LIQUID_TN_DEFAULT = 2.0 kg-N/톤, 액비 사용 매뉴얼 실측)
      기준이라 한우·가금 분뇨에 그대로 쓸 수 없다. 민원 라벨은 '닭·돼지·소'
      통합이므로 돼지에만 정량을 붙이면 축종 간 비대칭이 생긴다.
      대신 저감 조언(advice_lines)만 제공한다 — 상대 저감률이라 축종 비교가 아니다.
      · 경운 저감(50%)  : 액비·퇴비 공통으로 유효 → 전 축종 제공
      · 공법 저감(밴드/주입식) : 액상 전용 → 돼지에만 제공
    """
    if work_type not in WORK_WEIGHT:
        raise ValueError(f"작업유형은 {list(WORK_WEIGHT)} 중 하나")
    farm_lat = DEMO_FARM["lat"] if farm_lat is None else farm_lat
    farm_lon = DEMO_FARM["lon"] if farm_lon is None else farm_lon

    cal_all = _load_calendar_v6()
    if not cal_all:
        raise RuntimeError("risk_calendar_v6 가 비어 있습니다 — daily_scoring.run_v6() 먼저")
    sf = storage_factor(storage_days)
    groups = list(cal_all)
    wind = _load_forecast_wind()          # {시각: {wd, ws, sky}}

    from analysis.plume_select import downwind_groups

    cands = []
    for t in sorted(cal_all[groups[0]]):
        vals = {}
        for g in groups:
            wr = window_risk_v6(cal_all[g], t)
            if wr is None:
                vals = {}
                break
            vals[g] = wr
        if not vals:
            continue

        # ── 플룸: 등급을 보정하지 않고 '어느 그룹을 볼지'만 고른다 ──────
        # 절대규칙 1 준수 — 곱하지 않는다. 선택만 한다.
        w = wind.get(t)
        if w is not None:
            hit, detail = downwind_groups(
                farm_lat, farm_lon, w["wd"], w["ws"], w["sky"], t)
            pmethod = detail[0]["method"] if detail else "-"
        else:
            hit, detail, pmethod = [], [], "예보 없음"   # method 는 살포방식이므로 이름 분리
        if hit:
            sel = {g: vals[g] for g in hit}
            note = "풍하측 " + "·".join(hit)
        else:
            sel = vals                      # 풍하측 그룹 없음 → 보수적으로 전체
            note = "풍하측 그룹 없음 — 전 그룹 최댓값"

        wr_max = combine(sel, "max")          # 조합 규칙: 최댓값 (보수적)
        cands.append({
            "t": t, "window_risk": wr_max, "per_group": vals, "selected": sel,
            "final": wr_max * WORK_WEIGHT[work_type] * sf,
            "grade": cal_all[max(sel, key=sel.get)][t]["grade"],
            "plume_note": note, "plume_method": pmethod,
        })
    if not cands:
        raise RuntimeError("창(6시간)을 덮을 예보가 있는 후보 시각이 없습니다")

    # ── A/B/C — 후보 집합 안에서의 상대 순위 (문서 5장 원안) ───────────
    # 캘린더 등급(위험/주의/낮음)은 '평소 대비 이 시각이 위험한가'를 말하고,
    # A/B/C 는 '내 선택지 중 최선인가'를 말한다. 둘은 다른 질문이다.
    order = sorted(cands, key=lambda c: c["final"])
    n = len(order)
    for i, c in enumerate(order):
        c["abc"] = "A" if i < n * 0.2 else ("B" if i < n * 0.5 else "C")

    best = min(cands, key=lambda c: c["final"])
    worst = max(cands, key=lambda c: c["final"])

    # 조합 규칙 민감도 — 네 규칙이 같은 시각을 고르는가
    picks = {}
    for rule in COMBINE_RULES:
        cc = [dict(c, f=combine(c["per_group"], rule) * WORK_WEIGHT[work_type] * sf)
              for c in cands]
        picks[rule] = min(cc, key=lambda c: c["f"])["t"]
    agree = sum(1 for r in COMBINE_RULES if picks[r] == picks["max"]) / len(COMBINE_RULES)

    out = {
        "work_type": work_type, "species": species,
        "storage_days": storage_days, "storage_factor": sf,
        "recommended": {"t": best["t"].strftime("%Y-%m-%d %H시"),
                        "final": round(best["final"], 4), "grade": best["grade"],
                        "abc": best["abc"],
                        "plume": best["plume_note"], "plume_method": best["plume_method"],
                        "per_group": {k: round(v, 4) for k, v in best["per_group"].items()},
                        "selected_groups": list(best["selected"])},
        "avoid": {"t": worst["t"].strftime("%Y-%m-%d %H시"),
                  "final": round(worst["final"], 4), "grade": worst["grade"],
                  "abc": worst["abc"], "plume": worst["plume_note"]},
        "abc_counts": {k: sum(1 for c in cands if c["abc"] == k) for k in "ABC"},
        "plume_selected_hours": sum(1 for c in cands
                                    if len(c["selected"]) < len(c["per_group"])),
        "n_candidates": len(cands),
        "combine_agreement": round(agree, 2),
        "combine_picks": {r: picks[r].strftime("%m-%d %H시") for r in COMBINE_RULES},
    }

    # 저감 조언 — 절대 배출량은 내보내지 않는다
    tips = []
    for line in advice_lines(tons, method, tillage, best["t"].month):
        if "경운" in line:
            tips.append(line.split("(")[0].strip())          # 전 축종 공통
        elif species == "돼지":
            tips.append(line.split("(")[0].strip())          # 액상 공법은 돼지만
    out["reduction_tips"] = tips
    out["emission_note"] = (
        "배출 절대량은 액비 실측 계수 기반이라 축종 간 비교가 불가능해 제공하지 않는다. "
        "저감률은 상대값이므로 조언으로만 제시한다.")
    return out


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
                  "final": round(worst["final"], 4), "grade": worst["grade"],
                  "abc": worst["abc"], "plume": worst["plume_note"]},
        "abc_counts": {k: sum(1 for c in cands if c["abc"] == k) for k in "ABC"},
        "plume_selected_hours": sum(1 for c in cands
                                    if len(c["selected"]) < len(c["per_group"])),
        "n_candidates": len(cands),
    }
    # 액비 살포는 legacy 정량 배출 모델 연결
    if work_type == "액비살포":
        out["emission_kg"] = round(emission_kg(tons, method, None, False,
                                               best["t"].month), 3)
        out["emission_note"] = f"{tons}톤 {method} 기준 NH3 배출량 (legacy/emission.py)"
    return out
