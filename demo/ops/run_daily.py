"""S4 — 일일 배관: 예보 → 모델 → 확률 → 등급 → risk_calendar upsert.

구현 순서 요령대로 두 단계를 지원한다:
  dummy=True  : 모델 없이 random.uniform 확률로 배관 전체를 먼저 관통 (②)
  dummy=False : 학습된 full/reduced 모델로 교체 (⑦ 실모델 교체)

예보는 KMA_KEY 가 설정돼 있으면 legacy/kma.py 실 API, 없으면 legacy/mock_forecast
(공식 데모 폴백 경로). 중기예보(D+4~7)는 kma_mid.py 신규 작성 대상이지만 서비스키가
없어 데모에서는 단기 mock 시나리오의 일 단위 집계로 대체한다 — 어떤 걸 썼는지 로깅.
"""
from __future__ import annotations

import json
import os
import pickle
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from config import DEMO_FARM, MID_DIR, PROV, SEED, finding
from ops import db

# legacy import (수정 금지, import 만)
from geo import latlon_to_grid
import mock_forecast

GRADE = ("낮음", "주의", "위험")

# [C] mock 예보에는 습도(REH)가 없어 SKY 로 근사한다. 실 API 전환 시 제거.
SKY_TO_HUMID = {"1": 60.0, "3": 72.0, "4": 85.0}


def _fetch_forecast(nx: int, ny: int) -> tuple[dict, bool]:
    if os.environ.get("KMA_KEY"):
        import kma
        try:
            return kma.fetch_with_fallback(nx, ny), True
        except Exception as e:
            finding(f"KMA 실 API 호출 실패({e!r}) — mock 폴백")
    return mock_forecast.fetch_with_fallback(nx, ny), False


def _grade(prob: float, cuts: dict) -> str:
    if prob >= cuts["risk"]:
        return "위험"
    if prob >= cuts["watch"]:
        return "주의"
    return "낮음"


def run(dummy: bool = True, now: datetime | None = None) -> dict:
    con = db.connect()
    db.upsert_farm(con, DEMO_FARM)
    now = now or datetime.now()
    nx, ny = latlon_to_grid(DEMO_FARM["lat"], DEMO_FARM["lon"])

    data, real_api = _fetch_forecast(nx, ny)
    PROV.log("D3 단기예보", "기상청 API" if real_api else "legacy/mock_forecast.py",
             real=real_api, note=f"{len(data)}시점, 격자({nx},{ny})")

    # 예보 시간행 → (date, block) 집계
    rows = []
    for k, v in data.items():
        dt = datetime.strptime(k, "%Y%m%d %H%M")
        rows.append({
            "date": dt.strftime("%Y-%m-%d"), "block": dt.hour // 3, "dt": dt,
            "temp": float(v["TMP"]), "ws": float(v["WSD"]), "wd": float(v["VEC"]),
            "humid": SKY_TO_HUMID.get(str(v.get("SKY", "1")), 70.0),
            "rain": 1 if float(v.get("POP", 0)) >= 50 else 0,
        })
    fc = pd.DataFrame(rows)
    fc["wd_sin"] = np.sin(np.radians(fc["wd"]))
    fc["wd_cos"] = np.cos(np.radians(fc["wd"]))
    blk = fc.groupby(["date", "block"]).agg(
        temp=("temp", "mean"), humid=("humid", "mean"), ws=("ws", "mean"),
        wd_sin=("wd_sin", "mean"), wd_cos=("wd_cos", "mean"), rain=("rain", "max"),
    ).reset_index()

    stamp = now.strftime("%Y-%m-%d %H:%M:%S")
    out: list[tuple] = []

    if dummy:
        rng = random.Random(SEED)
        cuts = {"risk": 0.8, "watch": 0.5}
        for _, r in blk.iterrows():
            p = rng.uniform(0, 1)
            out.append((r["date"], int(r["block"]), round(p, 4), _grade(p, cuts), "dummy", stamp))
        model_note = "dummy(random.uniform)"
    else:
        with open(MID_DIR / "model_full.pkl", "rb") as fh:
            full = pickle.load(fh)
        with open(MID_DIR / "model_reduced.pkl", "rb") as fh:
            reduced = pickle.load(fh)
        with open(MID_DIR / "grade_cuts.json", encoding="utf-8") as fh:
            cuts = json.load(fh)

        # D+1~3: full 모델 (블록별)
        b = blk.copy()
        d = pd.to_datetime(b["date"])
        b["calm"] = (b["ws"] < 1.5).astype(int)
        b["humid80"] = (b["humid"] > 80).astype(int)
        b["night"] = b["block"].isin([0, 1, 7]).astype(int)
        b["night_calm"] = b["night"] * b["calm"]
        b["night_ws"] = b["night"] * b["ws"]
        b["night_humid"] = b["night"] * b["humid"]
        b["month_sin"] = np.sin(2 * np.pi * d.dt.month / 12)
        b["month_cos"] = np.cos(2 * np.pi * d.dt.month / 12)
        b["dow"] = d.dt.dayofweek
        b["year"] = d.dt.year
        p_full = full["model"].predict_proba(b[full["features"]])[:, 1]
        for (_, r), p in zip(b.iterrows(), p_full):
            out.append((r["date"], int(r["block"]), round(float(p), 4),
                        _grade(float(p), cuts), "full", stamp))

        # D+4~7: 중기예보 (ops/kma_mid — KMA_KEY 있으면 실 API) + reduced 모델.
        # 중기예보는 일 단위라 블록 해상도가 없다 → 일 확률을 8블록에 복제.
        from ops import kma_mid
        mid = kma_mid.fetch_mid(now)
        mid_rows = []
        if mid is not None:
            for day, v in mid.items():
                # [C] 중기기온은 min/max 2값 — 평균으로 대표 (계획서 미규정, v2 허점 2)
                temp = (v["tmin"] + v["tmax"]) / 2
                rain = 1 if (v["pop"] or 0) >= 50 else 0
                for block in range(8):
                    mid_rows.append({"date": day, "block": block,
                                     "temp": temp, "rain": rain})
            mid_src, mid_real = f"기상청 중기예보 API (육상 {kma_mid.MID_LAND_REG_ID}, " \
                                f"기온 {kma_mid.MID_TA_REG_ID})", True
        else:
            day_agg = blk.groupby("date").agg(temp=("temp", "mean"),
                                              rain=("rain", "max")).reset_index()
            last_short = pd.to_datetime(blk["date"]).max()
            for offset in range(4, 8):
                day = (last_short + timedelta(days=offset - 3)).strftime("%Y-%m-%d")
                src = day_agg.iloc[offset % len(day_agg)]
                for block in range(8):
                    mid_rows.append({"date": day, "block": block,
                                     "temp": src["temp"], "rain": src["rain"]})
            mid_src, mid_real = "mock 일집계 (KMA_KEY 미설정 — kma_mid 폴백)", False
        mb = pd.DataFrame(mid_rows)
        md = pd.to_datetime(mb["date"])
        mb["month_sin"] = np.sin(2 * np.pi * md.dt.month / 12)
        mb["month_cos"] = np.cos(2 * np.pi * md.dt.month / 12)
        mb["dow"] = md.dt.dayofweek
        p_red = reduced["model"].predict_proba(mb[reduced["features"]])[:, 1]
        for (_, r), p in zip(mb.iterrows(), p_red):
            out.append((r["date"], int(r["block"]), round(float(p), 4),
                        _grade(float(p), cuts), "reduced", stamp))
        PROV.log("D4 중기예보", mid_src, real=mid_real, note="D+4~7 reduced 모델 입력")
        model_note = f"full {len(p_full)}블록 + reduced {len(p_red)}블록"

    db.upsert_risk(con, out)
    n = con.execute("SELECT COUNT(*) FROM risk_calendar").fetchone()[0]
    print(f"  run_daily({'dummy' if dummy else 'real model'}): {len(out)}블록 upsert, "
          f"risk_calendar 총 {n}행 — {model_note}")
    con.close()
    return {"n_upsert": len(out), "model": model_note}
