"""v2 실험 4종 — 파이프라인 비판 검증.

1. 기후학 베이스라인 3자 대결 (month x block / block-only / full 모델)
2. 조건부 가치: 기상이 평년에서 크게 벗어난 블록에서만 재비교
3. 강건성: 이진 플래그 → 연속 변수 변형 학습 + 풍속 ±0.5m/s 섭동
4. 2026 드리프트: valid 0.42 → test 0.30 하락 원인 분해
"""
from __future__ import annotations

import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier

from config import MID_DIR, OUT_DIR, SEED
from preprocess.build_features import FULL_FEATURES_LEGACY
from model.train_model import weekly_ranking_hit_legacy

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False


def _load():
    feat = pd.read_parquet(MID_DIR / "features.parquet")
    with open(MID_DIR / "model_full.pkl", "rb") as fh:
        full = pickle.load(fh)
    return feat, full


def _splits(feat):
    y = feat["date"].dt.year
    return feat[y <= 2024].copy(), feat[y == 2025].copy(), feat[y == 2026].copy()


def _metrics(d: pd.DataFrame, score_col: str) -> dict:
    ap = float(average_precision_score(d["y_bin"], d[score_col]))
    hit, _ = weekly_ranking_hit_legacy(d, score_col)
    return {"pr_auc": round(ap, 4), "weekly_hit": round(hit, 4)}


# ────────────────────────────────────────────────────────────────────
def exp1_baselines() -> dict:
    """month x block / block-only 기후학 vs full 모델 — 같은 test 구간."""
    feat, full = _load()
    tr, _, te = _splits(feat)
    te = te.copy()
    te["month"] = te["date"].dt.month
    tr = tr.assign(month=tr["date"].dt.month)

    climo_mb = tr.groupby(["month", "block"])["y_bin"].mean()
    climo_b = tr.groupby("block")["y_bin"].mean()

    te["s_climo_mb"] = te.set_index(["month", "block"]).index.map(climo_mb).values
    te["s_climo_b"] = te["block"].map(climo_b).values
    te["s_full"] = full["model"].predict_proba(te[full["features"]])[:, 1]

    out = {
        "climo_month_block": _metrics(te, "s_climo_mb"),
        "climo_block_only": _metrics(te, "s_climo_b"),
        "full_model": _metrics(te, "s_full"),
    }
    # 그래프
    fig, ax = plt.subplots(figsize=(7, 4))
    names = ["block-only\n기후학", "month x block\n기후학", "XGB full\n(기상 포함)"]
    hits = [out["climo_block_only"]["weekly_hit"],
            out["climo_month_block"]["weekly_hit"], out["full_model"]["weekly_hit"]]
    aps = [out["climo_block_only"]["pr_auc"],
           out["climo_month_block"]["pr_auc"], out["full_model"]["pr_auc"]]
    x = np.arange(3)
    ax.bar(x - 0.18, hits, 0.36, label="주간 랭킹 적중률", color="#2980b9")
    ax.bar(x + 0.18, aps, 0.36, label="PR-AUC", color="#e67e22")
    ax.axhline(0.2, ls="--", color="#c0392b", lw=1, label="랜덤 기대(랭킹) 0.20")
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_title("v2-1 기후학 베이스라인 3자 대결 (test 2026.1~7)")
    ax.legend(fontsize=8)
    for i, (h, a) in enumerate(zip(hits, aps)):
        ax.text(i - 0.18, h, f"{h:.3f}", ha="center", va="bottom", fontsize=8)
        ax.text(i + 0.18, a, f"{a:.3f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout(); fig.savefig(OUT_DIR / "v2_1_baselines.png", dpi=130); plt.close(fig)

    print(f"  [1] block-only {out['climo_block_only']} / month x block {out['climo_month_block']}"
          f" / full {out['full_model']}")
    return out


# ────────────────────────────────────────────────────────────────────
def exp2_conditional() -> dict:
    """기상 이상(평년 대비 풍속·습도 상하위 20%) 블록에서 full vs 기후학."""
    feat, full = _load()
    tr, _, te = _splits(feat)
    tr = tr.assign(month=tr["date"].dt.month)
    te = te.assign(month=te["date"].dt.month)

    clim = tr.groupby(["month", "block"])[["ws", "humid"]].mean()
    climo_mb = tr.groupby(["month", "block"])["y_bin"].mean()
    idx = te.set_index(["month", "block"]).index
    te["ws_anom"] = te["ws"].values - idx.map(clim["ws"]).values
    te["humid_anom"] = te["humid"].values - idx.map(clim["humid"]).values
    te["s_climo"] = idx.map(climo_mb).values
    te["s_full"] = full["model"].predict_proba(te[full["features"]])[:, 1]

    def pct_extreme(s):
        lo, hi = s.quantile(0.2), s.quantile(0.8)
        return (s <= lo) | (s >= hi)

    extreme = pct_extreme(te["ws_anom"]) | pct_extreme(te["humid_anom"])
    out = {}
    for name, sub in [("anomalous", te[extreme]), ("normal", te[~extreme])]:
        k = max(1, int(0.2 * len(sub)))
        top_full = sub.nlargest(k, "s_full")
        top_climo = sub.nlargest(k, "s_climo")
        out[name] = {
            "n": len(sub), "pos_rate": round(float(sub["y_bin"].mean()), 4),
            "full_pr_auc": round(float(average_precision_score(sub["y_bin"], sub["s_full"])), 4),
            "climo_pr_auc": round(float(average_precision_score(sub["y_bin"], sub["s_climo"])), 4),
            "full_prec_at20": round(float(top_full["y_bin"].mean()), 4),
            "climo_prec_at20": round(float(top_climo["y_bin"].mean()), 4),
        }
    print(f"  [2] 이상 블록(n={out['anomalous']['n']}): full AP {out['anomalous']['full_pr_auc']}"
          f" vs 기후학 AP {out['anomalous']['climo_pr_auc']} / "
          f"평년 블록: {out['normal']['full_pr_auc']} vs {out['normal']['climo_pr_auc']}")
    return out


# ────────────────────────────────────────────────────────────────────
VARIANT_FEATURES = [
    # calm/humid80/night_calm 이진 플래그 제거, 연속 상호작용으로 교체
    "temp", "humid", "ws", "wd_sin", "wd_cos", "rain",
    "night", "night_ws", "night_humid",
    "block", "month_sin", "month_cos", "dow", "year",
]


def exp3_robustness() -> dict:
    feat, full = _load()
    feat = feat.copy()
    feat["night_ws"] = feat["night"] * feat["ws"]
    feat["night_humid"] = feat["night"] * feat["humid"]
    tr, va, te = _splits(feat)

    # (a) 연속 변수 변형 모델
    spw = float((tr["y_bin"] == 0).sum() / max((tr["y_bin"] == 1).sum(), 1))
    m = XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.05,
                      subsample=0.9, colsample_bytree=0.9, scale_pos_weight=spw,
                      random_state=SEED, eval_metric="aucpr", early_stopping_rounds=40)
    m.fit(tr[VARIANT_FEATURES], tr["y_bin"],
          eval_set=[(va[VARIANT_FEATURES], va["y_bin"])], verbose=False)
    te = te.copy()
    te["s_variant"] = m.predict_proba(te[VARIANT_FEATURES])[:, 1]
    te["s_full"] = full["model"].predict_proba(te[full["features"]])[:, 1]
    variant = _metrics(te, "s_variant")
    orig = _metrics(te, "s_full")

    # (b) 풍속 ±0.5 m/s 섭동 → 주간 top-k 세트 교체율
    def perturbed_score(delta: float) -> np.ndarray:
        p = te.copy()
        p["ws"] = (p["ws"] + delta).clip(lower=0)
        p["calm"] = (p["ws"] < 1.5).astype(int)
        p["night_calm"] = p["night"] * p["calm"]
        return full["model"].predict_proba(p[full["features"]])[:, 1]

    pert = {}
    te["week"] = te["date"].dt.to_period("W-SUN")
    for delta in (+0.5, -0.5):
        te["s_pert"] = perturbed_score(delta)
        turnover, hit_flip, n_wk = [], 0, 0
        for _, g in te.groupby("week"):
            if g["y_bin"].sum() == 0:
                continue
            n_wk += 1
            k = max(1, int(np.ceil(0.2 * len(g))))
            base = set(g.nlargest(k, "s_full").index)
            new = set(g.nlargest(k, "s_pert").index)
            turnover.append(1 - len(base & new) / k)
            hit_b = g.loc[list(base), "y_bin"].sum() > 0 and \
                g[g["y_bin"] == 1].index.isin(base).mean()
            hit_p = g[g["y_bin"] == 1].index.isin(new).mean()
            if abs(hit_b - hit_p) > 1e-9:
                hit_flip += 1
        pert[f"{delta:+.1f}"] = {
            "mean_topk_turnover": round(float(np.mean(turnover)), 4),
            "weeks_hit_changed": round(hit_flip / n_wk, 4), "n_weeks": n_wk,
        }

    out = {"variant_continuous": variant, "original_full": orig, "perturbation": pert}
    print(f"  [3] 변형(연속) {variant} vs 원본 {orig}")
    print(f"      섭동 +0.5: top-k 교체율 {pert['+0.5']['mean_topk_turnover']:.1%}, "
          f"주간적중 변동 주 비율 {pert['+0.5']['weeks_hit_changed']:.1%} / "
          f"-0.5: {pert['-0.5']['mean_topk_turnover']:.1%}, {pert['-0.5']['weeks_hit_changed']:.1%}")
    return out


# ────────────────────────────────────────────────────────────────────
def exp4_drift() -> dict:
    """valid 0.42 → test 0.30 하락 분해: 평가 구간 구성 효과 vs 신고 패턴 vs 기상."""
    feat, full = _load()
    feat = feat.copy()
    feat["year"] = feat["date"].dt.year
    feat["month"] = feat["date"].dt.month
    _, va, te = _splits(feat)
    va = va.assign(month=va["date"].dt.month)
    te = te.assign(month=te["date"].dt.month)
    va["proba"] = full["model"].predict_proba(va[full["features"]])[:, 1]
    te["proba"] = full["model"].predict_proba(te[full["features"]])[:, 1]

    # (a) 구성 효과: valid 를 test 와 같은 1~7월로 제한하면?
    va17 = va[va["month"] <= 7]
    comp = {
        "valid_full_year": _metrics(va, "proba"),
        "valid_jan_jul": _metrics(va17, "proba"),
        "test_jan_jul": _metrics(te, "proba"),
        "valid_jan_jul_pos_rate": round(float(va17["y_bin"].mean()), 4),
        "test_pos_rate": round(float(te["y_bin"].mean()), 4),
    }

    # (b) 연·월별 신고율 추이 (라벨 테이블 전체)
    monthly = feat.groupby([feat["year"], feat["month"]])["y_bin"].mean().rename("pos_rate")
    yearly = feat.groupby("year")["y_bin"].agg(["mean", "sum"])

    # (c) 2025 vs 2026 1~7월: 신고율·기상 비교
    cmp_rows = {}
    for yr, d in [(2025, va17), (2026, te)]:
        cmp_rows[yr] = {
            "pos_rate": round(float(d["y_bin"].mean()), 4),
            "complaints": int(d["y_cnt"].sum()) if "y_cnt" in d else None,
            "temp_mean": round(float(d["temp"].mean()), 2),
            "ws_mean": round(float(d["ws"].mean()), 2),
            "humid_mean": round(float(d["humid"].mean()), 2),
        }

    # (d) 월별 AP (양성 5블록 이상인 달만)
    monthly_ap = {}
    for yr, d in [(2025, va), (2026, te)]:
        for mth, g in d.groupby("month"):
            if g["y_bin"].sum() >= 5:
                monthly_ap[f"{yr}-{mth:02d}"] = round(
                    float(average_precision_score(g["y_bin"], g["proba"])), 4)

    # 그래프: 월별 신고율 + 월별 AP
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=False)
    m_idx = [f"{y}-{m:02d}" for (y, m) in monthly.index]
    axes[0].plot(m_idx, monthly.values, marker=".", color="#2c3e50")
    axes[0].set_xticks(m_idx[::6]); axes[0].tick_params(axis="x", labelsize=7)
    axes[0].set_ylabel("블록 양성률"); axes[0].set_title("v2-4a 월별 신고율 추이 2020~2026")
    keys = sorted(monthly_ap)
    axes[1].bar(keys, [monthly_ap[k] for k in keys],
                color=["#7f8c8d" if k.startswith("2025") else "#2980b9" for k in keys])
    axes[1].tick_params(axis="x", rotation=45, labelsize=7)
    axes[1].set_ylabel("월별 PR-AUC"); axes[1].set_title("v2-4b 월별 PR-AUC (2025 회색 / 2026 파랑)")
    fig.tight_layout(); fig.savefig(OUT_DIR / "v2_4_drift.png", dpi=130); plt.close(fig)

    out = {"composition": comp, "year_2025_vs_2026": cmp_rows,
           "monthly_ap": monthly_ap,
           "yearly_pos_rate": {int(y): round(float(r["mean"]), 4)
                               for y, r in yearly.iterrows()}}
    print(f"  [4] valid 전체 {comp['valid_full_year']} / valid 1~7월 {comp['valid_jan_jul']}"
          f" / test {comp['test_jan_jul']}")
    return out
