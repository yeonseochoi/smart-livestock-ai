"""지시 15 — 다중 발원 플룸 검증 하네스.

발원 좌표 목록 CSV 를 입력받아, 민원별 '최근접 발원' 기준으로 플룸 적중률과
방위각 정합을 계산한다. 농가 목록 데이터가 도착하면 파일 하나로 바로 실행:

    python -m analysis.plume_multi "경로\농가목록.csv" [--radius-km 3] [--wind aws]

CSV 컬럼: lat, lon (또는 위도, 경도). farm_id/이름 컬럼은 있으면 결과에 표기.
템플릿: data/sources_template.csv
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from config import MID_DIR, OUT_DIR, WANGGUNG_LAT, WANGGUNG_LON
from analysis.s8_analysis import _dist_m, _load_cloud

# legacy import (수정 금지)
from geo import bearing, angle_diff
from plume import pasquill_class, plume_half_angle

TEMPLATE = MID_DIR / "sources_template.csv"


def load_sources(path) -> pd.DataFrame:
    s = pd.read_csv(path)
    cols = {c.strip(): c for c in s.columns}
    lat_c = cols.get("lat") or cols.get("위도")
    lon_c = cols.get("lon") or cols.get("경도")
    if not lat_c or not lon_c:
        raise ValueError(f"lat/lon(위도/경도) 컬럼이 필요합니다. 현재: {list(s.columns)}")
    id_c = cols.get("farm_id") or cols.get("이름") or cols.get("name")
    out = pd.DataFrame({
        "lat": s[lat_c].astype(float), "lon": s[lon_c].astype(float),
        "farm_id": s[id_c].astype(str) if id_c else [f"src{i}" for i in range(len(s))],
    }).dropna(subset=["lat", "lon"]).reset_index(drop=True)
    if not len(out):
        raise ValueError("유효한 발원 좌표가 없습니다")
    return out


def evaluate(sources: pd.DataFrame, radius_km: float = 3.0,
             wind: str = "aws", min_dist_m: float = 100.0) -> dict:
    """민원별 최근접 발원 기준 플룸 정합 평가."""
    c = pd.read_parquet(MID_DIR / "complaints_clean.parquet")
    c = c[c["is_iksan"] & c["is_livestock"]].dropna(subset=["위도", "경도"]).copy()
    c["dt"] = pd.to_datetime(c["dt"])

    wfile = "weather_aws.parquet" if wind == "aws" else "weather_hourly.parquet"
    w = pd.read_parquet(MID_DIR / wfile)
    if wind == "aws":
        c = c[c["dt"].dt.year <= 2025]  # AWS 커버리지
    w = w.set_index("dt")
    cloud = _load_cloud()

    # 최근접 발원 (발원 수 x 민원 수 브로드캐스트)
    dmat = np.stack([
        _dist_m(row["lat"], row["lon"], c["위도"].values, c["경도"].values)
        for _, row in sources.iterrows()])           # (S, N)
    nearest = dmat.argmin(axis=0)
    ndist = dmat.min(axis=0)
    c = c.assign(src_idx=nearest, dist_m=ndist)
    c = c[(c["dist_m"] >= min_dist_m) & (c["dist_m"] <= radius_km * 1000)]

    rows = []
    for _, row in c.iterrows():
        t = row["dt"].floor("h")
        if t not in w.index:
            continue
        met = w.loc[t]
        if pd.isna(met["wd"]) or pd.isna(met["ws"]):
            continue
        src = sources.iloc[int(row["src_idx"])]
        brg = bearing(src["lat"], src["lon"], row["위도"], row["경도"])
        if cloud is not None and t in cloud.index and not pd.isna(cloud.loc[t]):
            cl = float(cloud.loc[t])
            sky = "4" if cl >= 9 else ("3" if cl >= 6 else "1")
        else:
            sky = "1"
        stability, _ = pasquill_class(float(met["ws"]), sky, t.to_pydatetime(),
                                      src["lat"], src["lon"])
        wind_to = (float(met["wd"]) + 180) % 360
        half = plume_half_angle(float(row["dist_m"]), stability)
        off = angle_diff(wind_to, brg)
        rows.append({
            "farm_id": src["farm_id"], "dist_m": round(float(row["dist_m"])),
            "angle_off": round(off, 1), "half_angle": round(half, 1),
            "hit": off <= half,
            "placebo_hit": angle_diff((wind_to + 90) % 360, brg) <= half,
            "near_warn": row["dist_m"] < 600,   # legacy NEAR_WARN_M — 방위 민감
        })

    if not rows:
        return {"n": 0, "note": "반경 내 평가 가능한 민원이 없습니다"}
    df = pd.DataFrame(rows)
    per_src = df.groupby("farm_id").agg(
        n=("hit", "size"), hit=("hit", "mean"), placebo=("placebo_hit", "mean"),
        median_off=("angle_off", "median")).round(4).sort_values("n", ascending=False)
    per_src.to_csv(OUT_DIR / "plume_multi_results.csv", encoding="utf-8-sig")

    hit, plc = float(df["hit"].mean()), float(df["placebo_hit"].mean())
    out = {
        "n": len(df), "n_sources": len(sources), "radius_km": radius_km,
        "wind": wind,
        "hit": round(hit, 4), "placebo": round(plc, 4),
        "lift": round(hit / plc, 2) if plc else None,
        "median_angle_off": round(float(df["angle_off"].median()), 1),
        "downwind_rate": round(float((df["angle_off"] <= 90).mean()), 4),
        "near_warn_share": round(float(df["near_warn"].mean()), 4),
        "per_source_csv": str(OUT_DIR / "plume_multi_results.csv"),
    }
    print(f"  발원 {out['n_sources']}개 / 민원 {out['n']:,}건 (반경 {radius_km}km, "
          f"바람 {wind})")
    print(f"  적중 {out['hit']} (플라시보 {out['placebo']}, lift x{out['lift']}) / "
          f"이탈각 중앙값 {out['median_angle_off']}도 / 풍하측 {out['downwind_rate']:.1%}")
    print(f"  근거리(<600m, 방위 민감) 비중 {out['near_warn_share']:.1%} — "
          f"발원별 상세: {out['per_source_csv']}")
    return out


def write_template() -> None:
    TEMPLATE.write_text(
        "farm_id,lat,lon\n"
        f"왕궁_흥암리_중앙값,{WANGGUNG_LAT},{WANGGUNG_LON}\n",
        encoding="utf-8-sig")


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from console import use_utf8_stdout
    use_utf8_stdout()
    ap = argparse.ArgumentParser()
    ap.add_argument("sources_csv")
    ap.add_argument("--radius-km", type=float, default=3.0)
    ap.add_argument("--wind", choices=["aws", "jeonju"], default="aws")
    args = ap.parse_args()
    evaluate(load_sources(args.sources_csv), args.radius_km, args.wind)
