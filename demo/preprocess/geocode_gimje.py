"""S0b — 김제 축산농가 리(里) 단위 재지오코딩.

김제 공공데이터는 번지가 ***로 마스킹돼 상호명 매칭으로 12.2%만 좌표를 얻는다.
그러나 '김제시 용지면 신정리 ***번지' 형태로 리까지는 살아 있다.
같은 리의 실좌표 중앙값으로 채우면 익산 v5가 [B]로 쓴 것과 같은 정밀도가 된다.

출력의 src 컬럼이 근거등급이다:  실좌표[A] / 리중앙값[B] / 읍면중앙값[C]

[구현 편차] 문서 13-3 은 SRC_CSV 를
    05_지오코딩_결과\\전북특별자치도 김제시_축산현황_20250515_geocoded.csv
로 적었으나 실제 파일은 '프로젝트 데이터' 루트에 있다. 05_지오코딩_결과 폴더에는
farm_coords_vworld.csv 하나뿐이다. → 두 경로를 모두 탐색하도록 고쳤다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import DATA_ROOT, MID_DIR, PROV, finding

_CANDIDATES = [
    DATA_ROOT / "전북특별자치도 김제시_축산현황_20250515_geocoded.csv",
    DATA_ROOT / "05_지오코딩_결과" / "전북특별자치도 김제시_축산현황_20250515_geocoded.csv",
]
SRC_CSV = next((p for p in _CANDIDATES if p.exists()), _CANDIDATES[0])
OUT_CSV = MID_DIR / "gimje_farms_ri.csv"

# '김제시 <읍면동> <리>' — 리가 없으면(동 지역) 두 번째 그룹이 NaN
ADDR_RE = r"김제시\s+(\S+)\s*(\S*리)?"


def run() -> pd.DataFrame:
    g = pd.read_csv(SRC_CSV)
    PROV.log("김제 축산현황", SRC_CSV, real=True, note=f"{len(g):,}행")

    ex = g["소재지"].str.extract(ADDR_RE)
    g["읍면"] = ex[0].fillna("")
    g["리"] = ex[1].fillna("")
    g["key"] = g["읍면"] + "|" + g["리"]

    have = g.dropna(subset=["위도"])
    anchor_ri = have.groupby("key")[["위도", "경도"]].median()
    anchor_eup = have.groupby("읍면")[["위도", "경도"]].median()

    def fill(r):
        if pd.notna(r["위도"]):
            return r["위도"], r["경도"], "실좌표[A]"
        if r["key"] in anchor_ri.index:
            a = anchor_ri.loc[r["key"]]
            return a["위도"], a["경도"], "리중앙값[B]"
        if r["읍면"] in anchor_eup.index:
            a = anchor_eup.loc[r["읍면"]]
            return a["위도"], a["경도"], "읍면중앙값[C]"
        return np.nan, np.nan, "없음"

    g[["lat", "lon", "src"]] = g.apply(fill, axis=1, result_type="expand")

    n_ok = int(g["lat"].notna().sum())
    print(f"  김제 좌표 {n_ok:,}/{len(g):,} ({n_ok/len(g):.1%}) — "
          f"{g['src'].value_counts().to_dict()}")
    if n_ok / len(g) < 0.90:
        finding(f"김제 좌표 확보율 {n_ok/len(g):.1%} 가 기대(95.9%)에 미달")

    pig = g[g["사육업종"].astype(str).str.contains("돼지", na=False)]
    print(f"  돼지농가 {len(pig):,}곳 중 좌표 보유 {int(pig['lat'].notna().sum()):,}곳 "
          f"(원좌표만이면 {int(pig['위도'].notna().sum()):,}곳)")

    g.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    return g
