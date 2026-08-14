"""지시 19 — 축산농가 현황 → 좌표 매핑 → 돼지 농가 다중 발원 플룸 최종 판정.

지오코딩 우선순위:
  (a) [B] 소재지의 읍면동리를 민원 데이터 같은 지역명 좌표 중앙값으로 매핑
  (b) 매핑 실패분만 OSM Nominatim (1초 1건 제한 준수, 캐시)
산출: data/farm_coords.csv (farm_id, lat, lon, 축종, 방법)
"""
from __future__ import annotations

import json
import re
import time

import pandas as pd

from config import DATA_ROOT, MID_DIR, PROV, finding

FARM_CSV_GLOB = "*축산농가 현황*.csv"
GEOCACHE = MID_DIR / "geocode_cache.json"
OUT_CSV = MID_DIR / "farm_coords.csv"

# 판정 기준 (지시 19)
CRITERIA = {"lift": 1.5, "median_angle_off": 70.0, "downwind_rate": 0.60}


def _region_key(addr: str) -> str | None:
    """소재지 → 민원 '지역' 형식 키 ('왕궁면 흥암리' / '마동')."""
    m = re.search(r"익산시\s+(\S+?[읍면])\s+(\S+?리)", str(addr))
    if m:
        return f"{m.group(1)} {m.group(2)}"
    m = re.search(r"익산시\s+(\S+?동)", str(addr))
    if m:
        return m.group(1)
    return None


def _nominatim(region: str, cache: dict) -> tuple[float, float] | None:
    if region in cache:
        v = cache[region]
        return tuple(v) if v else None
    import requests
    try:
        r = requests.get("https://nominatim.openstreetmap.org/search",
                         params={"q": f"전북특별자치도 익산시 {region}",
                                 "format": "json", "limit": 1},
                         headers={"User-Agent": "iksan-odor-demo/1.0 (research)"},
                         timeout=10)
        r.raise_for_status()
        items = r.json()
        coord = (float(items[0]["lat"]), float(items[0]["lon"])) if items else None
    except Exception:
        coord = None
    cache[region] = list(coord) if coord else None
    time.sleep(1.1)  # Nominatim 정책: 1초 1건
    return coord


def build_farm_coords() -> pd.DataFrame:
    path = next(DATA_ROOT.glob(FARM_CSV_GLOB))
    farms = pd.read_csv(path, encoding="utf-8-sig")
    PROV.log("D8 축산농가 현황", path, real=True,
             note=f"{len(farms):,}행, 좌표 없음 → 주소 매핑")

    farms = farms[farms["영업상태"] == "정상"].copy()
    farms["region"] = farms["소재지"].map(_region_key)
    n_nokey = int(farms["region"].isna().sum())
    print(f"  정상 영업 {len(farms):,}곳 / 읍면동리 추출 실패 {n_nokey}곳")
    print(f"  사육업종 분포: {farms['사육업종'].value_counts().to_dict()}")

    # (a) 민원 지역명 → 좌표 중앙값 (전 악취민원, 신뢰 위해 3건 이상 지역만)
    comp = pd.read_parquet(MID_DIR / "complaints_clean.parquet")
    comp = comp[comp["is_iksan"]].dropna(subset=["위도", "경도"])
    reg = comp.groupby("지역").agg(lat=("위도", "median"), lon=("경도", "median"),
                                  n=("위도", "size"))
    reg = reg[reg["n"] >= 3]

    rows = []
    unmapped: list[str] = []
    for _, f in farms.dropna(subset=["region"]).iterrows():
        key = f["region"]
        if key in reg.index:
            rows.append({"farm_id": f"{f['순번']}_{f['업체명']}",
                         "lat": reg.loc[key, "lat"], "lon": reg.loc[key, "lon"],
                         "축종": f["사육업종"], "방법": "지역명매핑[B]"})
        else:
            unmapped.append(key)
            rows.append({"farm_id": f"{f['순번']}_{f['업체명']}", "lat": None,
                         "lon": None, "축종": f["사육업종"], "방법": None})
    out = pd.DataFrame(rows)
    n_a = int(out["lat"].notna().sum())

    # (b) Nominatim 보완 — 실패 지역 unique 만 조회
    uniq = sorted(set(unmapped))
    cache = json.loads(GEOCACHE.read_text(encoding="utf-8")) if GEOCACHE.exists() else {}
    if uniq:
        print(f"  지역명 매핑 실패 {len(out) - n_a}곳 (고유 지역 {len(uniq)}개) → Nominatim 보완")
    solved = {}
    for region in uniq:
        c = _nominatim(region, cache)
        if c:
            solved[region] = c
    GEOCACHE.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    farms_valid = farms.dropna(subset=["region"]).reset_index(drop=True)
    for i in range(len(out)):
        if pd.isna(out.loc[i, "lat"]):
            key = farms_valid.loc[i, "region"]
            if key in solved:
                out.loc[i, ["lat", "lon"]] = solved[key]
                out.loc[i, "방법"] = "nominatim"

    n_b = int(out["lat"].notna().sum()) - n_a
    dropped = int(out["lat"].isna().sum()) + n_nokey
    out = out.dropna(subset=["lat", "lon"]).reset_index(drop=True)
    out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"  좌표 확보: 지역명매핑 {n_a} + nominatim {n_b} = {len(out)}곳 "
          f"(제외 {dropped}곳) → {OUT_CSV.name}")
    if dropped:
        finding(f"농가 {dropped}곳 좌표 매핑 실패(주소 형식/미등록 지역) — 플룸 발원에서 제외됨")
    return out


def judge_final(res: dict) -> dict:
    c_lift = (res.get("lift") or 0) >= CRITERIA["lift"]
    c_dir = (res.get("median_angle_off") or 999) < CRITERIA["median_angle_off"]
    c_down = (res.get("downwind_rate") or 0) > CRITERIA["downwind_rate"]
    return {"lift_ok": bool(c_lift), "direction_ok": bool(c_dir),
            "downwind_ok": bool(c_down),
            "passed": bool(c_lift and c_dir and c_down)}
