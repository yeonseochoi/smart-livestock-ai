"""S0 — 데이터 정제 (구현 내용.md S0 절차 그대로).

입력: D1 민원 xlsx, D2 기상 csv (실데이터)
출력: data/complaints_clean.parquet, data/weather_hourly.parquet
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import (
    COMPLAINTS_XLSX, WEATHER_CSV, MID_DIR, PROV,
    EXPECT_RAW_ROWS, EXPECT_IKSAN_ROWS, EXPECT_LIVESTOCK_ROWS,
    LIVESTOCK_CODE, finding,
)


def clean_complaints() -> pd.DataFrame:
    df = pd.read_excel(COMPLAINTS_XLSX)
    PROV.log("D1 민원 원본", COMPLAINTS_XLSX, real=True, note=f"{len(df):,}행")

    if len(df) != EXPECT_RAW_ROWS:
        finding(f"D1 원본 행수 {len(df):,} ≠ 계획서 기준 {EXPECT_RAW_ROWS:,}")

    # 1. 발생일시 파싱 (계획서: 결측 0건 확인됨)
    df["dt"] = pd.to_datetime(df["악취발생일시"], errors="coerce")
    n_bad = int(df["dt"].isna().sum())
    if n_bad:
        finding(f"악취발생일시 파싱 실패 {n_bad}건 — 계획서는 '결측 0건 확인됨'이라 했으나 불일치")
        df = df.dropna(subset=["dt"])

    # 2. 익산시 필터 — 제외 행은 버리지 않고 플래그 보존
    df["is_iksan"] = df["시군구"].eq("익산시")
    n_iksan = int(df["is_iksan"].sum())
    print(f"  익산시 필터: {n_iksan:,} / {len(df):,} (기준 {EXPECT_IKSAN_ROWS:,})")
    if n_iksan != EXPECT_IKSAN_ROWS:
        finding(f"익산 행수 {n_iksan:,} ≠ 계획서 기준 {EXPECT_IKSAN_ROWS:,}")

    # 3. 가축분뇨(101) 메인 셋 플래그 (전체 셋 보존)
    df["is_livestock"] = df["악취종류코드"].eq(LIVESTOCK_CODE)
    n_live = int((df["is_iksan"] & df["is_livestock"]).sum())
    print(f"  익산+가축분뇨(101): {n_live:,} (기준 {EXPECT_LIVESTOCK_ROWS:,})")
    if n_live != EXPECT_LIVESTOCK_ROWS:
        finding(f"가축분뇨 건수 {n_live:,} ≠ 계획서 기준 {EXPECT_LIVESTOCK_ROWS:,}")

    # 4. 중복 제거: 좌표 4자리 반올림 동일 지점 + 10분 이내 재신고 → 1건
    df = df.sort_values("dt").reset_index(drop=True)
    df["rlat"] = df["위도"].round(4)
    df["rlon"] = df["경도"].round(4)
    gap = df.groupby(["rlat", "rlon"])["dt"].diff()
    dup = gap.notna() & (gap <= pd.Timedelta(minutes=10))
    n_dup = int(dup.sum())
    df = df[~dup].reset_index(drop=True)
    print(f"  중복 제거: {n_dup:,}건 제거 → {len(df):,}행")

    # 5. 파생 컬럼
    df["date"] = df["dt"].dt.date
    df["hour"] = df["dt"].dt.hour
    df["block"] = df["hour"] // 3
    df["severity"] = df["악취강도코드"]

    keep = ["dt", "date", "hour", "block", "severity", "위도", "경도",
            "is_iksan", "is_livestock", "악취종류코드", "지역"]
    out = df[keep].copy()
    out.to_parquet(MID_DIR / "complaints_clean.parquet")
    return out


def clean_weather() -> pd.DataFrame:
    w = pd.read_csv(WEATHER_CSV)
    PROV.log("D2 기상 시간자료", WEATHER_CSV, real=True, note=f"{len(w):,}행")

    w["dt"] = pd.to_datetime(w["일시"])
    w = w.rename(columns={"기온C": "temp", "습도pct": "humid",
                          "풍향deg": "wd", "풍속ms": "ws", "강수량mm": "rain_mm"})
    # ASOS 강수량 결측 = 무강수
    w["rain_mm"] = w["rain_mm"].fillna(0.0)

    # 풍향 인코딩 (359° vs 1° 문제 회피). 원본 각도도 보존.
    w["wd_sin"] = np.sin(np.radians(w["wd"]))
    w["wd_cos"] = np.cos(np.radians(w["wd"]))

    # 결측: 3시간 이하 선형보간, 그 이상 결측 유지.
    # 풍향은 각도를 직접 보간하면 순환성이 깨지므로 sin/cos 를 보간 후 재구성한다.
    # (계획서는 이 순환성 문제를 보간 단계에 명시하지 않았다 — validation_report 참조)
    w = w.set_index("dt").sort_index()
    before = int(w[["temp", "humid", "ws", "wd_sin", "wd_cos"]].isna().any(axis=1).sum())
    for col in ["temp", "humid", "ws", "wd_sin", "wd_cos"]:
        w[col] = w[col].interpolate(limit=3, limit_area="inside")
    w["wd"] = (np.degrees(np.arctan2(w["wd_sin"], w["wd_cos"])) + 360) % 360
    after = int(w[["temp", "humid", "ws", "wd_sin", "wd_cos"]].isna().any(axis=1).sum())
    print(f"  기상 결측 시각: {before} → 보간 후 {after}")

    w = w.reset_index()
    w.to_parquet(MID_DIR / "weather_hourly.parquet")
    return w


def run() -> tuple[pd.DataFrame, pd.DataFrame]:
    c = clean_complaints()
    w = clean_weather()
    return c, w
