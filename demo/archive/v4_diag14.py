"""지시 14 — "왕궁 3km 내 민원 10건" 재검증.

확인 항목: (a) 하버사인 단위 (b) 위경도 인자 순서 (c) 민원 부분집합
(d) 흥암리 727건 좌표 분산. 버그인지 데이터 품질 문제인지 규명한다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import MID_DIR
from analysis.figures import _dist_m

# 사용자 제공: 지역=='왕궁면 흥암리' 가축분뇨 민원 727건의 중앙값 좌표
USER_COORD = (35.968937, 127.090910)
# v1~v3 에서 쓰던 [C] 근사 좌표 (config 수정 전 값 — 기록용)
OLD_COORD = (35.977, 127.055)


def run() -> dict:
    out: dict = {}

    # (a) 하버사인 단위 검증 — 알려진 거리로 교차확인 (서울시청↔부산시청 약 325km)
    d_known = _dist_m(37.5665, 126.9780, 35.1796, 129.0756)
    out["haversine_seoul_busan_km"] = round(float(d_known) / 1000, 1)
    ok_unit = 300_000 < d_known < 350_000
    print(f"  (a) 하버사인 단위: 서울↔부산 {d_known/1000:.1f}km → "
          f"{'정상(미터 반환)' if ok_unit else '이상!'}")

    # (b) 인자 순서 검증 — 위도만/경도만 1도 이동 시 거리 (위도 1도≈111km)
    d_lat = _dist_m(35.0, 127.0, 36.0, 127.0)
    d_lon = _dist_m(35.0, 127.0, 35.0, 128.0)
    out["deg_lat_km"] = round(float(d_lat) / 1000, 1)
    out["deg_lon_km"] = round(float(d_lon) / 1000, 1)
    ok_order = abs(d_lat / 1000 - 111) < 2 and abs(d_lon / 1000 - 91) < 3
    print(f"  (b) 인자 순서: 위도1도 {d_lat/1000:.1f}km(≈111 기대), "
          f"경도1도 {d_lon/1000:.1f}km(≈91 기대, 위도35) → "
          f"{'정상' if ok_order else '이상!'}")

    # (c) 분석 부분집합 — v3 exp9 와 동일 필터 재적용
    c = pd.read_parquet(MID_DIR / "complaints_clean.parquet")
    live = c[c["is_iksan"] & c["is_livestock"]].dropna(subset=["위도", "경도"]).copy()
    live["dt"] = pd.to_datetime(live["dt"])
    sub = live[live["dt"].dt.year <= 2025]
    out["subset"] = {"전체_정제후": len(c), "익산_가축분뇨": len(live),
                     "2025이하(AWS맞춤)": len(sub)}
    print(f"  (c) 부분집합: 익산+가축분뇨 {len(live):,} (중복제거 후) → "
          f"연도≤2025 {len(sub):,} — test 한정 아님, 전체 기간 사용 확인")

    # (d) 흥암리 727건의 좌표 분산
    ha = live[live["지역"].astype(str).str.contains("흥암리", na=False)]
    out["heungam"] = {"n": len(ha)}
    if len(ha):
        med = (float(ha["위도"].median()), float(ha["경도"].median()))
        out["heungam"].update({
            "median": [round(med[0], 6), round(med[1], 6)],
            "std_lat_m": round(float(ha["위도"].std()) * 111_000, 1),
            "std_lon_m": round(float(ha["경도"].std()) * 91_000, 1),
        })
        d_self = _dist_m(med[0], med[1], ha["위도"].values, ha["경도"].values)
        out["heungam"]["within_3km_of_own_median"] = int((d_self <= 3000).sum())
        print(f"  (d) 흥암리 민원 {len(ha)}건 / 중앙값 ({med[0]:.6f}, {med[1]:.6f}) / "
              f"좌표 표준편차 위도 {out['heungam']['std_lat_m']}m, "
              f"경도 {out['heungam']['std_lon_m']}m")
        print(f"      흥암리 자기 중앙값 3km 내: "
              f"{out['heungam']['within_3km_of_own_median']}건")

    # 핵심 대조: 구좌표 vs 사용자 좌표에서의 3km 내 민원 수
    for label, (la, lo) in [("구좌표(35.977,127.055)[C]", OLD_COORD),
                            ("흥암리 중앙값(제공)", USER_COORD)]:
        d = _dist_m(la, lo, sub["위도"].values, sub["경도"].values)
        n3 = int(((d >= 100) & (d <= 3000)).sum())
        n8 = int(((d >= 100) & (d <= 8000)).sum())
        out[f"n_within_{label}"] = {"3km": n3, "8km": n8}
        print(f"  발원={label}: 3km 내 {n3}건 / 8km 내 {n8}건")

    gap = float(_dist_m(*OLD_COORD, *USER_COORD))
    out["coord_gap_km"] = round(gap / 1000, 2)
    print(f"  → 구좌표와 흥암리 중앙값의 거리: {gap/1000:.2f}km")

    out["verdict_unit_ok"] = bool(ok_unit)
    out["verdict_order_ok"] = bool(ok_order)
    return out
