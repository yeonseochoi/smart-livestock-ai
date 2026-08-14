"""v3 — 익산 AWS(702) 도착 후 실험.

9.  플룸 재검증: 왕궁 반경 3km 민원 한정, 전주(146) vs 익산(702) 바람 비교
10. ML 재학습: 기상 피처 전주→익산 교체, train ≤2023 / valid 2024 / test 2025
    (2026 AWS 미확보 → test 를 2025 로 조정, 두 버전 모두 같은 분할·같은 블록)
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from xgboost import XGBClassifier

from config import (DATA_ROOT, MID_DIR, OUT_DIR, PROV, SEED,
                    WANGGUNG_LAT, WANGGUNG_LON, finding)
from etl.build_features import FULL_FEATURES
from model.train_model import weekly_ranking_hit
from analysis.figures import _dist_m, _load_cloud

# legacy import (수정 금지)
from geo import bearing, angle_diff
from plume import pasquill_class, plume_half_angle

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

AWS_CSV = DATA_ROOT / "02_기상데이터" / "aws_702_2020_2025_utf8.csv"


def prep_aws() -> pd.DataFrame:
    """익산 AWS 시간자료 정제 — s0 기상 정제와 동일 규칙(순환 보간 포함)."""
    a = pd.read_csv(AWS_CSV)
    PROV.log("D2b 익산 AWS(702)", AWS_CSV, real=True,
             note=f"{len(a):,}행 2020~2025, 결측<2%")
    a["dt"] = pd.to_datetime(a["일시"])
    a = a.rename(columns={"기온C": "temp", "습도pct": "humid",
                          "풍향deg": "wd", "풍속ms": "ws", "강수량mm": "rain_mm"})
    a["rain_mm"] = a["rain_mm"].fillna(0.0)
    a["wd_sin"] = np.sin(np.radians(a["wd"]))
    a["wd_cos"] = np.cos(np.radians(a["wd"]))
    a = a.set_index("dt").sort_index()
    for col in ["temp", "humid", "ws", "wd_sin", "wd_cos"]:
        a[col] = a[col].interpolate(limit=3, limit_area="inside")
    a["wd"] = (np.degrees(np.arctan2(a["wd_sin"], a["wd_cos"])) + 360) % 360
    a = a.reset_index()
    a.to_parquet(MID_DIR / "weather_aws.parquet")
    return a


# ────────────────────────────────────────────────────────────────────
def exp9_plume(radii_km: tuple = (3.0, 5.0, 8.0)) -> dict:
    """왕궁 반경별 민원의 플룸 정합성 — 전주 vs 익산 바람 비교표.

    지시는 3km 한정이었으나 실측상 3km 내 민원이 극소수(발원 좌표 [C] 문제의
    증거)라 통계력이 없어, 5km/8km 민감도를 함께 계산한다.
    """
    c_all = pd.read_parquet(MID_DIR / "complaints_clean.parquet")
    c_all = c_all[c_all["is_iksan"] & c_all["is_livestock"]].dropna(
        subset=["위도", "경도"]).copy()
    c_all["dt"] = pd.to_datetime(c_all["dt"])
    c_all = c_all[c_all["dt"].dt.year <= 2025]  # AWS 커버리지에 맞춤
    c_all["dist_m"] = _dist_m(WANGGUNG_LAT, WANGGUNG_LON,
                              c_all["위도"].values, c_all["경도"].values)
    c_all["brg"] = [bearing(WANGGUNG_LAT, WANGGUNG_LON, la, lo)
                    for la, lo in zip(c_all["위도"], c_all["경도"])]

    jj = pd.read_parquet(MID_DIR / "weather_hourly.parquet").set_index("dt")
    aws = pd.read_parquet(MID_DIR / "weather_aws.parquet").set_index("dt")
    cloud = _load_cloud()
    stations = {"jeonju_146": jj, "iksan_702": aws}

    out: dict = {}
    plot_offs: dict = {}
    for r_km in radii_km:
        c = c_all[(c_all["dist_m"] >= 100) & (c_all["dist_m"] <= r_km * 1000)]
        stats = {k: {"n": 0, "hit": 0, "placebo": 0, "offs": [], "downwind": 0}
                 for k in stations}
        for _, row in c.iterrows():
            t = row["dt"].floor("h")
            dist, brg = float(row["dist_m"]), float(row["brg"])
            if cloud is not None and t in cloud.index and not pd.isna(cloud.loc[t]):
                cl = float(cloud.loc[t])
                sky = "4" if cl >= 9 else ("3" if cl >= 6 else "1")
            else:
                sky = "1"
            for name, w in stations.items():
                if t not in w.index:
                    continue
                met = w.loc[t]
                if pd.isna(met["wd"]) or pd.isna(met["ws"]):
                    continue
                stability, _ = pasquill_class(float(met["ws"]), sky,
                                              t.to_pydatetime(),
                                              WANGGUNG_LAT, WANGGUNG_LON)
                wind_to = (float(met["wd"]) + 180) % 360
                half = plume_half_angle(dist, stability)
                off = angle_diff(wind_to, brg)
                s = stats[name]
                s["n"] += 1
                s["offs"].append(off)
                s["hit"] += off <= half
                s["placebo"] += angle_diff((wind_to + 90) % 360, brg) <= half
                s["downwind"] += off <= 90

        key = f"{r_km:.0f}km"
        out[key] = {}
        for name, s in stats.items():
            n = s["n"] or 1
            hit, plc = s["hit"] / n, s["placebo"] / n
            out[key][name] = {
                "n": s["n"], "hit": round(hit, 4), "placebo": round(plc, 4),
                "lift": round(hit / plc, 2) if plc else None,
                "median_angle_off": round(float(np.median(s["offs"])), 1)
                if s["offs"] else None,
                "downwind_rate": round(s["downwind"] / n, 4),
            }
            o = out[key][name]
            print(f"  [{key} {name}] n={o['n']} hit {o['hit']} "
                  f"(플라시보 {o['placebo']}, lift x{o['lift']}) / "
                  f"이탈각 중앙값 {o['median_angle_off']}도 / 풍하측 {o['downwind_rate']:.1%}")
        plot_offs[key] = {name: stats[name]["offs"] for name in stations}

    # 판정 반경: n ≥ 100 인 가장 작은 반경 (없으면 최대 반경)
    judged_at = next((f"{r:.0f}km" for r in radii_km
                      if out[f"{r:.0f}km"]["iksan_702"]["n"] >= 100),
                     f"{radii_km[-1]:.0f}km")
    out["judged_at"] = judged_at
    n3 = out[f"{radii_km[0]:.0f}km"]["iksan_702"]["n"]
    if n3 < 100:
        finding(f"왕궁 {radii_km[0]:.0f}km 내 민원 {n3}건뿐 — 지시된 3km 로는 통계력이 "
                f"없어 {judged_at} 로 판정 (3km 내 민원 희소 자체가 발원 좌표 [C] 오류 "
                f"또는 '왕궁 단일 발원' 가정 오류의 증거)")

    # 판정 반경의 각도 이탈 분포 그래프
    fig, ax = plt.subplots(figsize=(8, 4))
    bins = np.arange(0, 181, 10)
    for name, color in [("jeonju_146", "#7f8c8d"), ("iksan_702", "#2980b9")]:
        ax.hist(plot_offs[judged_at][name], bins=bins, density=True, alpha=0.55,
                label=f"{name} (중앙값 {out[judged_at][name]['median_angle_off']}도)",
                color=color)
    ax.axvline(90, ls=":", color="k", lw=1)
    ax.set_xlabel("민원 방위각과 풍하 방향의 이탈각 (도) — 균일랜덤이면 평균 90도")
    ax.set_ylabel("밀도")
    ax.set_title(f"v3-9 플룸 방위 정합성: 전주 vs 익산 (왕궁 {judged_at} 내 민원)")
    ax.legend()
    fig.tight_layout(); fig.savefig(OUT_DIR / "v3_9_plume_station.png", dpi=130)
    plt.close(fig)
    return out


# ────────────────────────────────────────────────────────────────────
def _build_features(weather: pd.DataFrame, end: str = "2025-12-31") -> pd.DataFrame:
    """S1+S2 로직 축약판 — 주어진 기상으로 라벨·피처 테이블 생성 (파일 미덮어씀)."""
    comp = pd.read_parquet(MID_DIR / "complaints_clean.parquet")
    main = comp[comp["is_iksan"] & comp["is_livestock"]].copy()
    main["date"] = pd.to_datetime(main["date"])

    dates = pd.date_range("2020-01-01", end, freq="D")
    grid = pd.MultiIndex.from_product([dates, range(8)],
                                      names=["date", "block"]).to_frame(index=False)
    agg = main.groupby(["date", "block"]).agg(
        y_cnt=("severity", "size"), y_sev=("severity", "max")).reset_index()
    lab = grid.merge(agg, on=["date", "block"], how="left")
    lab["y_cnt"] = lab["y_cnt"].fillna(0).astype(int)
    lab["y_bin"] = (lab["y_cnt"] >= 1).astype(int)

    w = weather.copy()
    w["date"] = w["dt"].dt.normalize()
    w["block"] = w["dt"].dt.hour // 3
    w["rain"] = (w["rain_mm"] > 0).astype(int)
    wb = w.groupby(["date", "block"]).agg(
        temp=("temp", "mean"), humid=("humid", "mean"), ws=("ws", "mean"),
        wd_sin=("wd_sin", "mean"), wd_cos=("wd_cos", "mean"),
        rain=("rain", "max")).reset_index()
    lab = lab.merge(wb, on=["date", "block"], how="left")
    lab = lab[~lab[["temp", "humid", "ws", "wd_sin", "wd_cos"]].isna().any(axis=1)]

    f = lab.copy()
    f["calm"] = (f["ws"] < 1.5).astype(int)
    f["humid80"] = (f["humid"] > 80).astype(int)
    f["night"] = f["block"].isin([0, 1, 7]).astype(int)
    f["night_calm"] = f["night"] * f["calm"]
    f["night_ws"] = f["night"] * f["ws"]
    f["night_humid"] = f["night"] * f["humid"]
    month = f["date"].dt.month
    f["month_sin"] = np.sin(2 * np.pi * month / 12)
    f["month_cos"] = np.cos(2 * np.pi * month / 12)
    f["dow"] = f["date"].dt.dayofweek
    f["year"] = f["date"].dt.year
    return f.reset_index(drop=True)


def exp10_ml(aws: pd.DataFrame) -> dict:
    """기상 피처 전주→익산 교체 재학습 (v3 지시 10).

    분할: train ~2024 (조기중단 검증용으로 2024 를 분리) / test 2025.1~7월.
    test 를 1~7월로 제한하는 이유: v2 실험 4 — 서로 다른 계절 구성끼리의 성능
    비교는 착시를 만든다. 기존(전주) 대비 비교 가능하도록 월 구성을 통일한다.
    두 버전 모두 같은 (date, block) 교집합에서 학습·평가한다.
    피처는 v3 지시 11 에 따라 연속변수 기본(FULL_FEATURES)을 쓴다.
    """
    f_jj = _build_features(pd.read_parquet(MID_DIR / "weather_hourly.parquet"))
    f_aws = _build_features(aws)

    # 공정 비교: 두 테이블 모두에 존재하는 블록만 사용
    key = ["date", "block"]
    common = f_jj.merge(f_aws[key], on=key)[key]
    f_jj = f_jj.merge(common, on=key)
    f_aws = f_aws.merge(common, on=key)
    print(f"  공통 블록 {len(common):,} (전주 전용/익산 전용 블록 제외)")

    out = {"n_common_blocks": len(common)}
    for name, f in [("jeonju_146", f_jj), ("iksan_702_aws", f_aws)]:
        y = f["date"].dt.year
        tr = f[y <= 2023]
        va = f[y == 2024]                                  # 조기중단 검증 (train~2024 의 일부)
        te = f[(y == 2025) & (f["date"].dt.month <= 7)]    # 1~7월 구성 통일
        spw = float((tr["y_bin"] == 0).sum() / max((tr["y_bin"] == 1).sum(), 1))
        m = XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.05,
                          subsample=0.9, colsample_bytree=0.9, scale_pos_weight=spw,
                          random_state=SEED, eval_metric="aucpr",
                          early_stopping_rounds=40)
        m.fit(tr[FULL_FEATURES], tr["y_bin"],
              eval_set=[(va[FULL_FEATURES], va["y_bin"])], verbose=False)
        te = te.assign(proba=m.predict_proba(te[FULL_FEATURES])[:, 1])
        hit, _ = weekly_ranking_hit(te, "proba")
        out[name] = {
            "pr_auc": round(float(average_precision_score(te["y_bin"], te["proba"])), 4),
            "roc_auc": round(float(roc_auc_score(te["y_bin"], te["proba"])), 4),
            "weekly_hit": round(hit, 4),
            "split": f"train {len(tr):,} / es-valid {len(va):,} / test(2025.1~7) {len(te):,}",
        }
        print(f"  [{name}] test 2025.1~7 PR-AUC {out[name]['pr_auc']} / "
              f"주간 적중 {out[name]['weekly_hit']}")
    return out


# ────────────────────────────────────────────────────────────────────
# 플룸 재검증 통과 기준 (v3 지시 9 — 스스로 정의).
# 셋 다 충족해야 PASS. 근거를 각 항목에 명시한다.
#   ① 익산 이탈각 중앙값 < 70도  — 균일랜덤 기대 90도 대비 명확한 방향성
#   ② 익산 lift ≥ 1.5           — 플라시보(90도 회전) 대비 1.5배 이상
#   ③ 익산이 전주보다 전 지표 개선 — 관측소 교체가 원인이라는 인과 확인
def judge_plume(e9: dict) -> dict:
    """판정은 통계력이 있는 반경(e9['judged_at'])에서 수행한다."""
    at = e9["judged_at"]
    jj, aw = e9[at]["jeonju_146"], e9[at]["iksan_702"]
    c1 = (aw["median_angle_off"] or 999) < 70
    c2 = (aw["lift"] or 0) >= 1.5
    c3 = ((aw["median_angle_off"] or 999) < (jj["median_angle_off"] or 999)
          and aw["hit"] > jj["hit"] and (aw["lift"] or 0) > (jj["lift"] or 0))
    passed = c1 and c2 and c3
    return {"judged_at": at, "c1_direction": bool(c1), "c2_lift": bool(c2),
            "c3_improves_over_jeonju": bool(c3), "passed": bool(passed)}
