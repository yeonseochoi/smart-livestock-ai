"""S1 — 라벨 테이블 (날짜 x 3시간 블록 그리드, 라벨 3안 전부 계산)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import MID_DIR, finding


def run(complaints: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    # 1. 그리드: 2020-01-01 ~ 2026-07-31 x 블록 0~7
    dates = pd.date_range("2020-01-01", "2026-07-31", freq="D")
    grid = pd.MultiIndex.from_product(
        [dates, range(8)], names=["date", "block"]
    ).to_frame(index=False)

    # 2. 라벨 3안 (가축분뇨 & 익산 메인 셋)
    main = complaints[complaints["is_iksan"] & complaints["is_livestock"]].copy()
    main["date"] = pd.to_datetime(main["date"])
    agg = main.groupby(["date", "block"]).agg(
        y_cnt=("severity", "size"), y_sev=("severity", "max")
    ).reset_index()

    lab = grid.merge(agg, on=["date", "block"], how="left")
    lab["y_cnt"] = lab["y_cnt"].fillna(0).astype(int)
    lab["y_sev"] = lab["y_sev"].fillna(0).astype(int)
    lab["y_bin"] = (lab["y_cnt"] >= 1).astype(int)

    # 3. 기상 조인: 블록 3개 시각 평균(기온·습도·풍속·wd_sin/cos), 강수는 max
    w = weather.copy()
    w["date"] = w["dt"].dt.normalize()
    w["block"] = w["dt"].dt.hour // 3
    w["rain"] = (w["rain_mm"] > 0).astype(int)
    wb = w.groupby(["date", "block"]).agg(
        temp=("temp", "mean"), humid=("humid", "mean"), ws=("ws", "mean"),
        wd_sin=("wd_sin", "mean"), wd_cos=("wd_cos", "mean"),
        rain=("rain", "max"), n_hours=("temp", "size"),
    ).reset_index()

    lab = lab.merge(wb, on=["date", "block"], how="left")
    n_total = len(lab)
    miss = lab[["temp", "humid", "ws", "wd_sin", "wd_cos"]].isna().any(axis=1)
    n_drop = int(miss.sum())
    lab = lab[~miss].reset_index(drop=True)
    print(f"  라벨 테이블 {n_total:,}행 중 기상 결측 블록 {n_drop}개 drop → {len(lab):,}행")

    pos = lab["y_bin"].mean()
    print(f"  y_bin 양성률 {pos:.3f} (계획서 기준 0.132 내외)")
    if abs(pos - 0.132) > 0.01:
        finding(f"y_bin 양성률 {pos:.3f} 이 계획서 기준 13.2% 와 1%p 이상 차이")

    lab.to_parquet(MID_DIR / "label_table.parquet")
    return lab
