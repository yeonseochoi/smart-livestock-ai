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
    """실 API 우선, 실패 시에만 mock 폴백.

    ★ 수정 — 기존에는 환경변수 KMA_KEY 가 있을 때만 kma 를 import 했다.
      그런데 legacy/kma.py 는 KMA_KEY 미설정 시 하드코딩된 폴백 서비스키를 쓰도록
      이미 되어 있다(SERVICE_KEY = os.environ.get("KMA_KEY", _FALLBACK_SERVICE_KEY)).
      따라서 키가 있는데도 게이트에 막혀 항상 mock 으로 빠지고 있었다.
      → 무조건 실 API 를 먼저 시도하고, 예외일 때만 mock 으로 내려간다.
    """
    try:
        import kma
        data = kma.fetch_with_fallback(nx, ny)
        if data:
            return data, True
        finding("KMA 실 API 가 빈 응답 — mock 폴백")
    except Exception as e:
        finding(f"KMA 실 API 호출 실패({e!r}) — mock 폴백")
    return mock_forecast.fetch_with_fallback(nx, ny), False


def _grade(prob: float, cuts: dict) -> str:
    if prob >= cuts["risk"]:
        return "위험"
    if prob >= cuts["watch"]:
        return "주의"
    return "낮음"


def _grade_v6(prob: float, cuts: dict, month: int | None = None) -> str:
    """v6 등급 — 그 달의 컷으로 판정한다 (계절 드리프트 대응).

    연간 분위수 하나로 고정하면 여름엔 거의 전부 '위험', 겨울엔 거의 전부 '낮음'이
    되어 등급이 정보를 잃는다. 월별 컷이 있으면 그것을 쓰고, 없으면 연간 컷으로
    폴백한다.

    ⚠️ 절대 하한(abs_safe)은 쓰지 않는다. scale_pos_weight x sample_weight 이중
      가중으로 확률이 양성률 대비 8~16배 부풀려져 있어 절대 임계값에 의미가 없다.
      확률은 '순위 점수'로만 해석해야 한다.
    """
    c = cuts
    if month is not None:
        mc = (cuts.get("monthly") or {}).get(str(int(month)))
        if mc:
            c = mc
    if prob >= c["risk"]:
        return "위험"
    if prob >= c["watch"]:
        return "주의"
    return "낮음"


def run_v6(now: datetime | None = None, hist_tail_h: int = 48) -> dict:
    """v6 서빙 — 예보 → (1시간 x 그룹) 피처 → 그룹별 모델 → risk_calendar_v6.

    hist_tail_h: 예보 앞에 이어붙일 관측 시간 수.
      시차(ws_lag1)와 무풍 연속시간(calm_streak)은 '직전 값'이 있어야 계산된다.
      예보 첫 시각에는 그게 없으므로 관측 마지막 N시간을 붙여 계산한 뒤 잘라낸다.
      안 하면 서빙에서 calm_streak 이 항상 0 에서 시작해 학습 분포와 어긋난다.
    """
    from config import GROUPS
    from etl import s2b_spatial
    from etl.s2_features import build_serving_features

    con = db.connect()
    db.upsert_farm(con, DEMO_FARM)
    now = now or datetime.now()
    nx, ny = latlon_to_grid(DEMO_FARM["lat"], DEMO_FARM["lon"])
    data, real_api = _fetch_forecast(nx, ny)
    PROV.log("D3 단기예보", "기상청 API" if real_api else "legacy/mock_forecast.py",
             real=real_api, note=f"{len(data)}시점, 격자({nx},{ny})")

    # 실 API 는 REH(습도)를 준다. mock 에는 없어 SKY 로 근사한다.
    n_reh = sum(1 for v in data.values() if "REH" in v)
    rows = []
    for k, v in data.items():
        dt = datetime.strptime(k, "%Y%m%d %H%M")
        humid = (float(v["REH"]) if "REH" in v
                 else SKY_TO_HUMID.get(str(v.get("SKY", "1")), 70.0))
        rows.append({
            "dt_h": pd.Timestamp(dt), "temp": float(v["TMP"]),
            "ws": float(v["WSD"]), "wd": float(v["VEC"]),
            "humid": humid, "sky": str(v.get("SKY", "1")),
            "rain": 1 if float(v.get("POP", 0)) >= 50 else 0,
        })
    print(f"  습도: REH 실측 {n_reh}/{len(data)}시점"
          + ("" if n_reh == len(data) else " · 나머지는 SKY 근사 [C]"))
    fc = pd.DataFrame(rows).drop_duplicates(subset=["dt_h"]).sort_values("dt_h")
    fc_start = fc["dt_h"].min()

    # ── 관측 꼬리 이어붙이기 (시차·무풍 연속시간용) ────────────────────
    obs = pd.read_parquet(MID_DIR / "weather_hourly.parquet")
    obs["dt_h"] = obs["dt"].dt.floor("h")
    obs = obs[obs["dt_h"] < fc_start].sort_values("dt_h").tail(hist_tail_h)
    obs = obs.assign(rain=(obs["rain_mm"] > 0).astype(int))[
        ["dt_h", "temp", "humid", "ws", "wd", "rain"]]
    n_tail = len(obs)
    if n_tail == 0:
        finding("관측 꼬리가 비어 있어 calm_streak 이 0에서 시작합니다 "
                "— 학습 분포와 어긋납니다")
    hourly = pd.concat([obs, fc], ignore_index=True).sort_values("dt_h")
    hourly["wd_sin"] = np.sin(np.radians(hourly["wd"]))
    hourly["wd_cos"] = np.cos(np.radians(hourly["wd"]))

    # 그룹 축을 곱한다
    b = pd.concat([hourly.assign(group=g) for g in GROUPS], ignore_index=True)
    b = build_serving_features(b)
    b = s2b_spatial.run_serving(b)
    b = s2b_spatial.add_prior_rate_serving(b, asof=pd.Timestamp(now).floor("h"))
    b = b[b["dt_h"] >= fc_start].copy()          # 관측 꼬리 제거
    b["date"] = b["dt_h"].dt.strftime("%Y-%m-%d")
    b["hour"] = b["dt_h"].dt.hour

    stamp = now.strftime("%Y-%m-%d %H:%M:%S")
    out: list[tuple] = []
    for g in GROUPS:
        with open(MID_DIR / f"model_{g}_full.pkl", "rb") as fh:
            full = pickle.load(fh)
        with open(MID_DIR / f"grade_cuts_{g}.json", encoding="utf-8") as fh:
            cuts = json.load(fh)
        gb = b[b["group"] == g]
        missing = [c for c in full["features"] if c not in gb.columns]
        if missing:
            raise KeyError(f"서빙 피처 누락 {missing} — 학습/서빙 불일치")
        p = full["model"].predict_proba(gb[full["features"]])[:, 1]
        for (_, r), pv in zip(gb.iterrows(), p):
            out.append((r["date"], int(r["hour"]), g, round(float(pv), 4),
                        _grade_v6(float(pv), cuts, r["dt_h"].month),
                        "full_v6", stamp))

    # 예보 원값 저장 — s5 의 플룸 그룹 선택이 시각별 바람을 쓴다
    src = "kma" if real_api else "mock"
    db.upsert_forecast_v6(con, [
        (r["dt_h"].strftime("%Y-%m-%d"), int(r["dt_h"].hour), float(r["wd"]),
         float(r["ws"]), str(r.get("sky", "1")), float(r["temp"]),
         float(r["humid"]), int(r["rain"]), src, stamp)
        for _, r in fc.iterrows()])

    # ── D+4~7 중기예보 (일 단위) → reduced_v6 모델 ──────────────────
    n_mid = 0
    try:
        n_mid = _append_mid_v6(con, now, stamp, fc["dt_h"].max())
    except FileNotFoundError:
        finding("reduced_v6 모델 없음 — 중기예보 구간 생략 (s3_train.run_v6 재실행 필요)")
    except Exception as e:
        finding(f"중기예보 처리 실패({e!r}) — 단기예보 구간만 제공")

    db.upsert_risk_v6(con, out)
    n = con.execute("SELECT COUNT(*) FROM risk_calendar_v6").fetchone()[0]
    print(f"  run_v6: 단기 {len(fc)}시각 + 관측꼬리 {n_tail}시각 → {len(out)}행"
          f" · 중기 {n_mid}행 · 누적 {n}행")
    con.close()
    return {"n_upsert": len(out), "n_forecast_hours": len(fc),
            "tail_hours": n_tail, "n_mid_rows": n_mid, "real_api": real_api}


def _append_mid_v6(con, now, stamp, last_short) -> int:
    """중기예보 D+4~7 → reduced_v6 모델 → risk_calendar_v6.

    중기예보는 일 단위(최저/최고기온·강수확률)라 풍향·풍속이 없다.
    그래서 바람 없이 학습한 reduced_v6 모델을 따로 쓴다.
    일 확률을 24시간에 복제하되, 시간 프로파일은 모델의 hour 피처가 만든다.
    """
    from config import GROUPS
    from etl.s2_features import REDUCED_FEATURES_V6
    from ops import kma_mid

    mid = kma_mid.fetch_mid(now)
    if mid is not None:
        mid_src, mid_real = (f"기상청 중기예보 API (육상 {kma_mid.MID_LAND_REG_ID}, "
                             f"기온 {kma_mid.MID_TA_REG_ID})"), True
    else:
        # 폴백 — 단기예보 마지막 날을 D+4~7 로 복제 (mock 경로)
        finding("중기예보 API 미승인/실패 — 단기예보 일집계로 폴백 "
                "(data.go.kr 에서 MidFcstInfoService 활용신청 필요)")
        base = pd.read_sql("SELECT * FROM forecast_hourly_v6", con)
        day = base.groupby("date").agg(temp=("temp", "mean"),
                                       rain=("rain", "max")).reset_index()
        mid = {}
        for k in range(4, 8):
            d = (pd.Timestamp(last_short).normalize() + pd.Timedelta(days=k - 3))
            r = day.iloc[k % len(day)]
            mid[d.strftime("%Y-%m-%d")] = {"tmin": r["temp"] - 4, "tmax": r["temp"] + 4,
                                           "pop": 60.0 if r["rain"] else 10.0}
        mid_src, mid_real = "mock 일집계 (중기 API 미승인 폴백)", False
    PROV.log("D4 중기예보", mid_src, real=mid_real, note="D+4~7 reduced_v6 모델 입력")

    rows = []
    for day, v in mid.items():
        temp = (v["tmin"] + v["tmax"]) / 2.0      # [C] 계획서 미규정 — 평균 대표
        rain = 1 if (v.get("pop") or 0) >= 50 else 0
        for h in range(24):
            rows.append({"date": day, "hour": h, "temp": temp, "rain": rain})
    mb = pd.DataFrame(rows)
    md = pd.to_datetime(mb["date"])
    mb["month_sin"] = np.sin(2 * np.pi * md.dt.month / 12)
    mb["month_cos"] = np.cos(2 * np.pi * md.dt.month / 12)
    mb["dow"] = md.dt.dayofweek
    mb["night"] = (~mb["hour"].between(7, 17)).astype(int)

    out = []
    for g in GROUPS:
        path = MID_DIR / f"model_{g}_reduced.pkl"
        if not path.exists():
            raise FileNotFoundError(path)
        with open(path, "rb") as fh:
            red = pickle.load(fh)
        with open(MID_DIR / f"grade_cuts_{g}.json", encoding="utf-8") as fh:
            cuts = json.load(fh)
        p = red["model"].predict_proba(mb[REDUCED_FEATURES_V6])[:, 1]
        for (_, r), pv in zip(mb.iterrows(), p):
            out.append((r["date"], int(r["hour"]), g, round(float(pv), 4),
                        _grade_v6(float(pv), cuts, int(md.dt.month.iloc[0])),
                        "reduced_v6", stamp))
    db.upsert_risk_v6(con, out)
    return len(out)


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
