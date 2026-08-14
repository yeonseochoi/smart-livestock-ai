"""S1 — 라벨 테이블.

run()     v5 — (날짜 x 3시간 블록) 그리드.  절대 수정하지 않는다 (두 트랙 유지).
run_v6()  v6 — (그룹 x 1시간) 그리드.

[구현 편차] 문서 13-4 는 run() 을 직접 대폭 수정하라고 지시하지만, 같은 문서
13-4 수정5 가 "v5 파이프라인(demo.py, demo_v5.py)이 새 스키마를 읽고 깨지므로
두 트랙을 나란히 유지한다"고 못박는다. run() 을 고치면 그 원칙이 즉시 깨진다.
→ 지시 내용(격자·라벨·기상조인·검증문·저장)은 100% 반영하되 run_v6() 로 분리했다.
"""
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


# ═══════════════════════════════════════════════════════════════════
# v6 — (그룹, 1시간) 격자
# ═══════════════════════════════════════════════════════════════════

def run_v6(complaints: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    from config import GROUPS, GROUP_CENTER, REGION_GROUP

    # 1. 1시간 격자 x 그룹.
    # 1시간인 이유: 냄새 도달 시간 중앙값 32분(풍속 >=1.5m/s), 3시간 블록은 6배.
    #               3시간 창 안에서 풍향이 크게 바뀌는 비율 58.6%.
    #               그리고 3시간 블록에서는 'ws_lag1'을 만들 윗줄이 없다.
    # 격자 범위는 기상 커버리지에서 유도한다 (지점 교체 시 자동 대응).
    w_min = weather["dt"].min().floor("h")
    w_max = weather["dt"].max().floor("h")
    hours = pd.date_range(max(w_min, pd.Timestamp("2020-01-01")), w_max, freq="h")
    print(f"  격자 기간 {hours[0]:%Y-%m-%d} ~ {hours[-1]:%Y-%m-%d} ({len(hours):,}시각)")
    grid = pd.MultiIndex.from_product(
        [hours, GROUPS], names=["dt_h", "group"]
    ).to_frame(index=False)
    grid["date"] = grid["dt_h"].dt.normalize()
    grid["hour"] = grid["dt_h"].dt.hour
    grid["block"] = grid["hour"] // 3      # 하위호환 — s5의 3시간 창 계산이 아직 쓴다

    # 2. 라벨 — 그룹 단위 집계
    main = complaints[complaints["is_iksan"] & complaints["is_livestock"]].copy()
    main["group"] = main["지역"].map(REGION_GROUP)

    n_unmapped = int(main["group"].isna().sum())
    print(f"  그룹 미지정 민원 {n_unmapped:,}건 "
          f"({n_unmapped/len(main):.1%}) — 학습 제외")
    if n_unmapped / len(main) > 0.15:
        finding(f"그룹 미지정 비율 {n_unmapped/len(main):.1%} 가 기대(11%)보다 큼 "
                f"— config.REGION_GROUP 의 지역명이 데이터와 어긋났을 가능성")
    main = main.dropna(subset=["group"])

    # 그룹 대표 좌표를 민원 좌표 중앙값으로 갱신 (config 의 폴백[C] 를 [B] 로 승격)
    for g, d in main.groupby("group"):
        GROUP_CENTER[g] = (float(d["위도"].median()), float(d["경도"].median()))
        print(f"    {g} 중심 → ({GROUP_CENTER[g][0]:.6f}, {GROUP_CENTER[g][1]:.6f}) "
              f"[민원 {len(d):,}건 중앙값]")
    # ★ 서빙 프로세스가 같은 중심을 쓰도록 파일로 고정한다
    import json as _json
    (MID_DIR / "group_center.json").write_text(
        _json.dumps({k: list(v) for k, v in GROUP_CENTER.items()},
                    ensure_ascii=False, indent=2), encoding="utf-8")

    main["dt_h"] = pd.to_datetime(main["dt"]).dt.floor("h")
    agg = main.groupby(["dt_h", "group"]).agg(
        y_cnt=("severity", "size"), y_sev=("severity", "max"),
        region=("지역", "first"),          # F-3 grouped CV 선행조건
    ).reset_index()

    lab = grid.merge(agg, on=["dt_h", "group"], how="left")
    lab["y_cnt"] = lab["y_cnt"].fillna(0).astype(int)
    # * sample_weight 로 쓸 값이므로 0 이 아니라 1 로 채운다.
    #   0 이면 그 행의 학습 기여가 통째로 사라진다 (전체 음성 행이 무시됨).
    lab["y_sev"] = lab["y_sev"].fillna(1).clip(lower=1).astype(int)
    lab["y_bin"] = (lab["y_cnt"] >= 1).astype(int)

    # 3. 기상 조인 — 1시간 격자이므로 집계가 필요 없다.
    #    원본 풍향(wd)도 함께 가져간다 — s2b 의 풍상측 계산에 각도 원값이 필요.
    w = weather.copy()
    w["dt_h"] = w["dt"].dt.floor("h")
    w["rain"] = (w["rain_mm"] > 0).astype(int)
    wh = w[["dt_h", "temp", "humid", "ws", "wd", "wd_sin", "wd_cos", "rain"]] \
        .drop_duplicates(subset=["dt_h"])

    lab = lab.merge(wh, on="dt_h", how="left")
    n_total = len(lab)
    miss = lab[["temp", "humid", "ws", "wd_sin", "wd_cos"]].isna().any(axis=1)
    n_drop = int(miss.sum())
    lab = lab[~miss].reset_index(drop=True)
    print(f"  라벨 테이블 {n_total:,}행 중 기상 결측 {n_drop:,}행 drop → {len(lab):,}행")

    pos = lab["y_bin"].mean()
    print(f"  y_bin 양성률 {pos:.4f}  (그리드 변경으로 v5 기준 0.132 와 직접 비교 불가)")
    for gname, gdf in lab.groupby("group"):
        print(f"    {gname}: 양성 {int(gdf['y_bin'].sum()):,}행 / "
              f"{len(gdf):,}행 = {gdf['y_bin'].mean():.4f}")

    lab.to_parquet(MID_DIR / "label_table_v6.parquet")
    return lab


# ═══════════════════════════════════════════════════════════════════
# v6r — (지역, 1시간) 격자.  그룹은 '모델을 나누는 축'으로만 남는다.
# ═══════════════════════════════════════════════════════════════════

def run_v6r(complaints: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    from config import REGIONS, REGION_GROUP_R, REGION_CENTER

    w_min = weather["dt"].min().floor("h")
    w_max = weather["dt"].max().floor("h")
    hours = pd.date_range(max(w_min, pd.Timestamp("2020-01-01")), w_max, freq="h")
    grid = pd.MultiIndex.from_product(
        [hours, REGIONS], names=["dt_h", "region"]
    ).to_frame(index=False)
    grid["group"] = grid["region"].map(REGION_GROUP_R)
    grid["date"] = grid["dt_h"].dt.normalize()
    grid["hour"] = grid["dt_h"].dt.hour

    main = complaints[complaints["is_iksan"] & complaints["is_livestock"]].copy()
    n_all = len(main)
    main = main[main["지역"].isin(REGIONS)].copy()
    print(f"  지역 {len(REGIONS)}개 커버 {len(main):,}/{n_all:,} = {len(main)/n_all:.1%}")

    # 지역 중심 = 그 지역 민원 좌표 중앙값 [B]
    for r, d in main.groupby("지역"):
        REGION_CENTER[r] = (float(d["위도"].median()), float(d["경도"].median()))
    import json as _json
    (MID_DIR / "region_center.json").write_text(
        _json.dumps({k: list(v) for k, v in REGION_CENTER.items()},
                    ensure_ascii=False, indent=2), encoding="utf-8")

    main["dt_h"] = pd.to_datetime(main["dt"]).dt.floor("h")
    agg = main.groupby(["dt_h", "지역"]).agg(
        y_cnt=("severity", "size"), y_sev=("severity", "max")
    ).reset_index().rename(columns={"지역": "region"})

    lab = grid.merge(agg, on=["dt_h", "region"], how="left")
    lab["y_cnt"] = lab["y_cnt"].fillna(0).astype(int)
    lab["y_sev"] = lab["y_sev"].fillna(1).clip(lower=1).astype(int)
    lab["y_bin"] = (lab["y_cnt"] >= 1).astype(int)

    w = weather.copy()
    w["dt_h"] = w["dt"].dt.floor("h")
    w["rain"] = (w["rain_mm"] > 0).astype(int)
    wh = w[["dt_h", "temp", "humid", "ws", "wd", "wd_sin", "wd_cos", "rain"]] \
        .drop_duplicates(subset=["dt_h"])
    lab = lab.merge(wh, on="dt_h", how="left")

    n_total = len(lab)
    miss = lab[["temp", "humid", "ws", "wd_sin", "wd_cos"]].isna().any(axis=1)
    lab = lab[~miss].reset_index(drop=True)
    print(f"  라벨 테이블 {n_total:,}행 중 기상 결측 {int(miss.sum()):,}행 drop "
          f"→ {len(lab):,}행")
    print(f"  전체 양성 {int(lab['y_bin'].sum()):,}행 "
          f"(양성률 {lab['y_bin'].mean():.4f})")
    for g, gd in lab.groupby("group"):
        print(f"    {g}: 지역 {gd['region'].nunique()}개 · {len(gd):,}행 · "
              f"양성 {int(gd['y_bin'].sum()):,} ({gd['y_bin'].mean():.4f})")

    lab.to_parquet(MID_DIR / "label_table_v6r.parquet")
    return lab
