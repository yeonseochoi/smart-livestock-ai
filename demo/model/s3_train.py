"""S3 — 모델 학습·평가.

시계열 고정 분할: train 2020~2024 / valid 2025 / test 2026.1~7 (랜덤 분할 금지).
XGBoost full/reduced + 로지스틱 베이스라인, PR-AUC + 주간 랭킹 적중률.
등급 컷은 valid 예측 확률 분위수로 고정해 파일로 저장한다.
"""
from __future__ import annotations

import json
import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from config import MID_DIR, OUT_DIR, SEED, finding
from etl.s2_features import BINARY_FULL_FEATURES, FULL_FEATURES, REDUCED_FEATURES


def weekly_ranking_hit(df: pd.DataFrame, proba_col: str, frac: float = 0.2) -> tuple[float, pd.DataFrame]:
    """각 주의 블록을 위험도 내림차순 정렬 → 실제 민원 블록이 상위 20%에 든 비율.

    계획서는 '56개 블록, k=11'로 고정했지만 기상 결측 drop 으로 주당 블록 수가
    변동하므로 k = ceil(0.2 x 해당 주 블록 수) 로 일반화했다.
    """
    d = df.copy()
    d["week"] = d["date"].dt.to_period("W-SUN")
    rows = []
    for wk, g in d.groupby("week"):
        pos = g[g["y_bin"] == 1]
        if len(pos) == 0:
            continue
        k = max(1, int(np.ceil(frac * len(g))))
        top = g.nlargest(k, proba_col)
        hit = pos.index.isin(top.index).mean()
        rows.append({"week": str(wk), "n_blocks": len(g), "n_pos": len(pos), "hit": hit})
    wk_df = pd.DataFrame(rows)
    return (float(wk_df["hit"].mean()) if len(wk_df) else float("nan")), wk_df


def run(feat: pd.DataFrame) -> dict:
    f = feat.copy()
    year = f["date"].dt.year
    tr = f[year <= 2024]
    va = f[year == 2025]
    te = f[year == 2026]
    print(f"  분할: train {len(tr):,} / valid {len(va):,} / test {len(te):,}")

    spw = float((tr["y_bin"] == 0).sum() / max((tr["y_bin"] == 1).sum(), 1))
    results: dict = {"split": {"train": len(tr), "valid": len(va), "test": len(te)}}
    models: dict = {}

    # "full" = 연속변수 기본 모델 (v3 지시 11 승격), "full_binary" = 이진 플래그 백업
    for name, feats in [("full", FULL_FEATURES), ("full_binary", BINARY_FULL_FEATURES),
                        ("reduced", REDUCED_FEATURES)]:
        m = XGBClassifier(
            n_estimators=400, max_depth=5, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9,
            scale_pos_weight=spw, random_state=SEED,
            eval_metric="aucpr", early_stopping_rounds=40,
        )
        m.fit(tr[feats], tr["y_bin"], eval_set=[(va[feats], va["y_bin"])], verbose=False)
        models[name] = m

        for split_name, d in [("valid", va), ("test", te)]:
            p = m.predict_proba(d[feats])[:, 1]
            d = d.assign(proba=p)
            prauc = float(average_precision_score(d["y_bin"], p))
            hit, wk = weekly_ranking_hit(d, "proba")
            results[f"{name}_{split_name}"] = {
                "pr_auc": round(prauc, 4), "weekly_hit": round(hit, 4),
                "pos_rate": round(float(d["y_bin"].mean()), 4),
            }
            if name == "full":
                wk.to_csv(OUT_DIR / f"weekly_hit_{split_name}.csv", index=False)  # S8-5용
            if name == "full" and split_name == "valid":
                results["_valid_proba"] = p

        with open(MID_DIR / f"model_{name}.pkl", "wb") as fh:
            pickle.dump({"model": m, "features": feats}, fh)

    # 베이스라인: 로지스틱 회귀 (full 피처)
    sc = StandardScaler().fit(tr[FULL_FEATURES])
    lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED)
    lr.fit(sc.transform(tr[FULL_FEATURES]), tr["y_bin"])
    for split_name, d in [("valid", va), ("test", te)]:
        p = lr.predict_proba(sc.transform(d[FULL_FEATURES]))[:, 1]
        d = d.assign(proba=p)
        hit, _ = weekly_ranking_hit(d, "proba")
        results[f"logit_{split_name}"] = {
            "pr_auc": round(float(average_precision_score(d["y_bin"], p)), 4),
            "weekly_hit": round(hit, 4),
        }

    # 등급 컷: valid 확률 분위수 고정 (상위 20% 위험 / 20~50% 주의 / 나머지 낮음)
    vp = results.pop("_valid_proba")
    cuts = {"risk": float(np.quantile(vp, 0.8)), "watch": float(np.quantile(vp, 0.5))}
    with open(MID_DIR / "grade_cuts.json", "w", encoding="utf-8") as fh:
        json.dump(cuts, fh, indent=2)
    results["grade_cuts"] = {k: round(v, 4) for k, v in cuts.items()}

    # 해석: SHAP 상위 5개.
    # shap 0.49 는 xgboost 3.1 의 base_score 문자열("[5E-1]")을 파싱하지 못해
    # TreeExplainer 가 죽는다 → xgboost 내장 pred_contribs(TreeSHAP 동일 알고리즘) 사용.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import xgboost as xgb
        plt.rcParams["font.family"] = "Malgun Gothic"
        plt.rcParams["axes.unicode_minus"] = False
        sample = va[FULL_FEATURES].sample(min(2000, len(va)), random_state=SEED)
        contrib = models["full"].get_booster().predict(
            xgb.DMatrix(sample), pred_contribs=True)
        sv = contrib[:, :-1]  # 마지막 열은 bias
        mean_abs = np.abs(sv).mean(axis=0)
        order = np.argsort(mean_abs)[::-1][:5]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.barh([FULL_FEATURES[i] for i in order][::-1], mean_abs[order][::-1])
        ax.set_title("SHAP 평균 |기여도| 상위 5 (full 모델, valid 표본)")
        ax.set_xlabel("mean |SHAP value|")
        fig.tight_layout()
        fig.savefig(OUT_DIR / "s3_shap_top5.png", dpi=130)
        plt.close(fig)
        results["shap_top5"] = [FULL_FEATURES[i] for i in order]
    except Exception as e:  # SHAP 실패해도 파이프라인은 계속
        finding(f"SHAP 계산 실패({e!r}) — 그래프 생략")

    # 랜덤 기대치 대비 확인 (완료 기준)
    hit_test = results["full_test"]["weekly_hit"]
    print(f"  full 모델 test 주간 랭킹 적중률 {hit_test:.3f} (랜덤 기대 ~0.20)")
    if hit_test <= 0.20:
        finding(f"test 주간 랭킹 적중률 {hit_test:.3f} 이 랜덤 기대치(0.20)를 넘지 못함 — 완료 기준 미달")

    return results
