"""RAG 이전 전 구간 정밀 점검 — 학습 · 서빙 · 추천."""
import json, pickle, sqlite3, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from config import (MID_DIR, DB_PATH, GROUPS, REGION_GROUP, SPLIT_TRAIN_END,
                    SPLIT_VALID_YEAR, SPLIT_TEST_YEAR, WEATHER_SOURCE)
from console import use_utf8_stdout

use_utf8_stdout()
OK, NG = [], []
def chk(name, cond, detail=""):
    (OK if cond else NG).append(name)
    print(f"  [{'OK' if cond else 'NG'}] {name}" + (f"  — {detail}" if detail else ""))

print("=" * 78); print(" 1. 학습 산출물"); print("=" * 78)
f = pd.read_parquet(MID_DIR / "features_v6.parquet")
chk("features_v6 존재", len(f) > 0, f"{len(f):,}행")
chk("y_sev 최솟값 >= 1  (0이면 음성 가중치 0 → 학습 붕괴)",
    int(f["y_sev"].min()) >= 1, f"min={int(f['y_sev'].min())}")
chk("y_bin 이 0/1 만", set(f["y_bin"].unique()) <= {0, 1})
g0 = f.sort_values(["group", "dt_h"]).groupby("group").head(1)
chk("시차 첫 행이 현재값으로 채워짐 (그룹 경계 안 넘음)",
    bool((g0["ws_lag1"] == g0["ws"]).all()))
# 그룹 경계 침범 검사: 각 그룹 2번째 행의 lag1 == 그 그룹 1번째 행의 ws
bad = 0
for gname, d in f.sort_values(["group", "dt_h"]).groupby("group"):
    d = d.reset_index(drop=True)
    if not np.isclose(d.loc[1, "ws_lag1"], d.loc[0, "ws"]):
        bad += 1
chk("시차가 같은 그룹 내부만 참조", bad == 0)
chk("공간 피처 6종 + 최근접거리 존재",
    all(c in f.columns for c in
        ["up_ik_pig","up_ik_cattle","up_ik_poultry","up_gj_pig","up_gj_cattle","up_gj_poultry","up_nearest_km"]))
chk("결측 없음 (학습 피처)", int(f[[c for c in f.columns if c.startswith(('up_','prior_','ws_lag','wd_'))]].isna().sum().sum()) == 0)

print("\n" + "=" * 78); print(" 2. 누수 검사"); print("=" * 78)
# prior_rate_1y 가 자기 행을 포함하면, y_bin=1 인 행의 prior 가 y_bin=0 행보다
# 체계적으로 높아진다. shift(1) 이 있으면 그 차이가 미미해야 한다.
for gname, d in f.groupby("group"):
    a = d[d.y_bin == 1]["prior_rate_1y"].mean()
    b = d[d.y_bin == 0]["prior_rate_1y"].mean()
    chk(f"prior 누수 없음 ({gname})", abs(a - b) < 0.05, f"양성 {a:.4f} vs 음성 {b:.4f}")
first30 = f[f.dt_h < "2020-01-31"]["prior_missing"].mean()
chk("초기 30일 prior 결측 플래그 (min_periods=720h 설계대로)",
    first30 > 0.9, f"2020년 1월 결측비율 {first30:.1%}")

print("\n" + "=" * 78); print(" 3. 학습 피처 == 서빙 피처"); print("=" * 78)
for g in GROUPS:
    M = pickle.load(open(MID_DIR / f"model_{g}_full.pkl", "rb"))
    chk(f"{g} 모델 로드", "model" in M and "features" in M, f"{len(M['features'])}개 피처")
    chk(f"{g} 피처가 features_v6 에 전부 존재",
        all(c in f.columns for c in M["features"]))
feat_sets = [tuple(pickle.load(open(MID_DIR / f"model_{g}_full.pkl", "rb"))["features"]) for g in GROUPS]
chk("두 그룹 모델의 피처 목록·순서 동일", feat_sets[0] == feat_sets[1])

print("\n" + "=" * 78); print(" 4. 분할 정합"); print("=" * 78)
yrs = f["dt_h"].dt.year
chk("train/valid/test 겹침 없음",
    SPLIT_TRAIN_END < SPLIT_VALID_YEAR < SPLIT_TEST_YEAR,
    f"<= {SPLIT_TRAIN_END} / {SPLIT_VALID_YEAR} / {SPLIT_TEST_YEAR}")
for g in GROUPS:
    d = f[f.group == g]
    n = [int(d[yrs[d.index] <= SPLIT_TRAIN_END]["y_bin"].sum()),
         int(d[yrs[d.index] == SPLIT_VALID_YEAR]["y_bin"].sum()),
         int(d[yrs[d.index] == SPLIT_TEST_YEAR]["y_bin"].sum())]
    chk(f"{g} 각 분할에 양성 존재", min(n) > 0, f"train {n[0]} / valid {n[1]} / test {n[2]}")

print("\n" + "=" * 78); print(" 5. 서빙 DB"); print("=" * 78)
con = sqlite3.connect(DB_PATH)
sch = con.execute("SELECT sql FROM sqlite_master WHERE name='risk_calendar_v6'").fetchone()
chk("risk_calendar_v6 스키마 PK(date,hour,grp)",
    sch and "PRIMARY KEY(date, hour, grp)" in sch[0])
d = pd.read_sql("SELECT * FROM risk_calendar_v6", con)
chk("서빙 결과 적재됨", len(d) > 0, f"{len(d)}행")
chk("두 그룹 모두 적재", set(d["grp"]) == set(GROUPS))
chk("시각당 그룹 2행 (덮어쓰기 없음)",
    d.groupby(["date","hour"]).size().eq(len(GROUPS)).all())
chk("확률이 0~1", bool(d.risk_prob.between(0,1).all()))
chk("등급이 3종", set(d.risk_grade) <= {"낮음","주의","위험"}, str(d.risk_grade.value_counts().to_dict()))
con.close()

print("\n" + "=" * 78); print(" 6. 서빙 피처 분포가 학습과 어긋나지 않는가"); print("=" * 78)
from ops import daily_scoring
from etl import spatial_features
from etl.build_features import build_serving_features
import mock_forecast
from geo import latlon_to_grid
from config import DEMO_FARM
nx, ny = latlon_to_grid(DEMO_FARM["lat"], DEMO_FARM["lon"])
data, real = mock_forecast.fetch_with_fallback(nx, ny), False
rows = []
from datetime import datetime
for k, v in data.items():
    dt = datetime.strptime(k, "%Y%m%d %H%M")
    rows.append({"dt_h": pd.Timestamp(dt), "temp": float(v["TMP"]), "ws": float(v["WSD"]),
                 "wd": float(v["VEC"]), "humid": 70.0,
                 "rain": 1 if float(v.get("POP", 0)) >= 50 else 0})
fc = pd.DataFrame(rows).drop_duplicates("dt_h").sort_values("dt_h")
obs = pd.read_parquet(MID_DIR / "weather_hourly.parquet")
obs["dt_h"] = obs["dt"].dt.floor("h")
obs = obs[obs.dt_h < fc.dt_h.min()].sort_values("dt_h").tail(48)
obs = obs.assign(rain=(obs.rain_mm > 0).astype(int))[["dt_h","temp","humid","ws","wd","rain"]]
h = pd.concat([obs, fc], ignore_index=True).sort_values("dt_h")
h["wd_sin"] = np.sin(np.radians(h.wd)); h["wd_cos"] = np.cos(np.radians(h.wd))
b = pd.concat([h.assign(group=g) for g in GROUPS], ignore_index=True)
b = build_serving_features(b)
b = spatial_features.run_serving(b)
b = b[b.dt_h >= fc.dt_h.min()]
chk("관측꼬리 덕에 calm_streak 이 0에서 시작하지 않음",
    float(b.groupby("group")["calm_streak"].first().max()) >= 0,
    f"첫 행 값 {b.groupby('group')['calm_streak'].first().to_dict()}")
for c in ["up_ik_pig", "up_gj_pig", "calm_streak", "ws_lag1"]:
    lo, hi = f[c].quantile([0.001, 0.999])
    inside = b[c].between(lo, hi).mean()
    chk(f"서빙 {c} 분포가 학습 범위 안 ({inside:.0%})", inside > 0.90,
        f"학습 [{lo:.1f}~{hi:.1f}] · 서빙 [{b[c].min():.1f}~{b[c].max():.1f}]")

print("\n" + "=" * 78); print(" 7. 추천"); print("=" * 78)
from scoring import recommend
r = recommend.recommend_v6("액비살포", storage_days=12, species="돼지")
chk("추천 산출", "recommended" in r, r["recommended"]["t"])
chk("회피 시각이 추천보다 위험", r["avoid"]["final"] > r["recommended"]["final"])
chk("6시간 창 후보 존재", r["n_candidates"] > 0, f"{r['n_candidates']}개")
chk("배출 kg 이 출력되지 않음", "emission_kg" not in r)
chk("저감 조언 존재", len(r["reduction_tips"]) > 0, str(r["reduction_tips"]))
chk("A/B/C 상대 등급 부여", set(r["abc_counts"]) == {"A","B","C"}, str(r["abc_counts"]))
chk("A 비율이 20% 부근", 0.15 <= r["abc_counts"]["A"]/r["n_candidates"] <= 0.25,
    f"{r['abc_counts']['A']}/{r['n_candidates']}")
chk("플룸이 그룹을 좁힌 시각 존재", r["plume_selected_hours"] > 0,
    f"{r['plume_selected_hours']}/{r['n_candidates']}")
chk("플룸 판정 방법 기록", "plume_method" in r["recommended"],
    r["recommended"]["plume_method"])
r2 = recommend.recommend_v6("청소", species="육계")
chk("육계에는 액상공법 조언 없음", not any("주입식" in t for t in r2["reduction_tips"]),
    str(r2["reduction_tips"]))

print("\n" + "=" * 78)
print(f" 결과: 통과 {len(OK)} · 실패 {len(NG)}")
if NG:
    for n in NG: print("   ✗", n)
print("=" * 78)
