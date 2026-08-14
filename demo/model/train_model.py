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
from etl.build_features import BINARY_FULL_FEATURES, FULL_FEATURES, REDUCED_FEATURES


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


# ═══════════════════════════════════════════════════════════════════
# v6 — 그룹별 학습 + grouped CV
# ═══════════════════════════════════════════════════════════════════

def weekly_ranking_hit_v6(df: pd.DataFrame, proba_col: str,
                          frac: float = 0.2) -> tuple[float, pd.DataFrame]:
    """v5 와 동일하지만 date 대신 dt_h 로 주를 만든다."""
    d = df.copy()
    d["week"] = d["dt_h"].dt.to_period("W-SUN")
    rows = []
    for wk, g in d.groupby("week"):
        pos = g[g["y_bin"] == 1]
        if len(pos) == 0:
            continue
        k = max(1, int(np.ceil(frac * len(g))))
        top = g.nlargest(k, proba_col)
        rows.append({"week": str(wk), "n_blocks": len(g), "n_pos": len(pos),
                     "hit": pos.index.isin(top.index).mean()})
    wk_df = pd.DataFrame(rows)
    return (float(wk_df["hit"].mean()) if len(wk_df) else float("nan")), wk_df


def grouped_cv(feat: pd.DataFrame, feats: list, n_folds: int = 4) -> dict:
    """지역(그룹 내 세부 지역)을 통째로 빼고 학습 → 그 지역을 맞히게 한다.

    시간 분할만으로는 '모델이 지역을 외운 것'과 '일반 규칙을 배운 것'을
    구분할 수 없다. 같은 지역이 학습·검증 양쪽에 들어가기 때문이다.
    """
    from config import REGION_GROUP
    regions = sorted(set(REGION_GROUP))
    folds = [regions[i::n_folds] for i in range(n_folds)]
    scores, detail = [], []
    for held in folds:
        tr = feat[~feat["region"].isin(held)]
        te = feat[feat["region"].isin(held)]
        if te["y_bin"].sum() < 10:
            continue
        m = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                          random_state=SEED, eval_metric="aucpr")
        m.fit(tr[feats], tr["y_bin"], sample_weight=tr["y_sev"], verbose=False)
        p = m.predict_proba(te[feats])[:, 1]
        ap = float(average_precision_score(te["y_bin"], p))
        scores.append(ap)
        detail.append({"held": held, "n_test": int(len(te)),
                       "test_pos_rate": round(float(te["y_bin"].mean()), 4),
                       "pr_auc": round(ap, 4)})
        print(f"    fold {held} → test {len(te):,}행 "
              f"양성률 {te['y_bin'].mean():.4f}  PR-AUC {ap:.4f}")
    return {"folds": len(scores),
            "pr_auc_mean": round(float(np.mean(scores)), 4) if scores else None,
            "pr_auc_std": round(float(np.std(scores)), 4) if scores else None,
            "detail": detail}


def _run_one_group(f: pd.DataFrame, gname: str, feats: list) -> dict:
    from config import SPLIT_TRAIN_END, SPLIT_VALID_YEAR, SPLIT_TEST_YEAR
    year = f["dt_h"].dt.year
    tr = f[year <= SPLIT_TRAIN_END]
    va = f[year == SPLIT_VALID_YEAR]
    te = f[year == SPLIT_TEST_YEAR]
    print(f"  [{gname}] 분할: train {len(tr):,} / valid {len(va):,} / test {len(te):,}")
    print(f"          양성: train {int(tr['y_bin'].sum()):,} / "
          f"valid {int(va['y_bin'].sum()):,} / test {int(te['y_bin'].sum()):,}")

    spw = float((tr["y_bin"] == 0).sum() / max((tr["y_bin"] == 1).sum(), 1))
    res: dict = {"split": {"train": len(tr), "valid": len(va), "test": len(te)},
                 "scale_pos_weight": round(spw, 1)}

    m = XGBClassifier(
        n_estimators=400, max_depth=5, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9,
        scale_pos_weight=spw, random_state=SEED,
        eval_metric="aucpr", early_stopping_rounds=40,
    )
    # 강도(y_sev 1~5)를 학습 가중치로. 심한 민원을 놓치는 것을 더 크게 벌준다.
    m.fit(tr[feats], tr["y_bin"],
          sample_weight=tr["y_sev"],
          eval_set=[(va[feats], va["y_bin"])],
          sample_weight_eval_set=[va["y_sev"]],
          verbose=False)

    vp = None
    for split_name, d in [("valid", va), ("test", te)]:
        p = m.predict_proba(d[feats])[:, 1]
        d = d.assign(proba=p)
        hit, _ = weekly_ranking_hit_v6(d, "proba")
        res[split_name] = {
            "pr_auc": round(float(average_precision_score(d["y_bin"], p)), 4),
            "weekly_hit": round(hit, 4),
            "pos_rate": round(float(d["y_bin"].mean()), 4),
            "proba_mean": round(float(p.mean()), 4),
            "proba_median": round(float(np.median(p)), 4),
            "proba_p05": round(float(np.quantile(p, 0.05)), 4),
        }
        if split_name == "valid":
            vp = p

    # ── 등급 컷 — 월별로 따로 잡는다 (계절 드리프트 대응) ──────────────
    # 연간 분위수 하나로 고정하면 여름처럼 위험한 계절엔 거의 전부 '위험'이 되고
    # 겨울엔 거의 전부 '낮음'이 되어 등급이 정보를 잃는다.
    # 실측: 8월 중순 실예보 166행에서 위험 91 / 주의 52 / 낮음 23.
    # 절대 하한(abs_safe)은 폐기 — scale_pos_weight x sample_weight 이중 가중으로
    # 확률이 양성률 대비 8~16배 부풀려져 있어 0.15 같은 절대값에 의미가 없다.
    cuts = {"risk": float(np.quantile(vp, 0.8)),
            "watch": float(np.quantile(vp, 0.5)),
            "monthly": {}}
    vmonth = va["dt_h"].dt.month.to_numpy()
    for mm in range(1, 13):
        sel = vp[vmonth == mm]
        if len(sel) >= 200:                     # 분위수가 안정될 최소 표본
            cuts["monthly"][str(mm)] = {
                "risk": float(np.quantile(sel, 0.8)),
                "watch": float(np.quantile(sel, 0.5)),
                "n": int(len(sel)),
            }
    with open(MID_DIR / f"grade_cuts_{gname}.json", "w", encoding="utf-8") as fh:
        json.dump(cuts, fh, indent=2, ensure_ascii=False)
    res["grade_cuts"] = {"risk": round(cuts["risk"], 4),
                         "watch": round(cuts["watch"], 4),
                         "n_monthly": len(cuts["monthly"])}
    res["monthly_watch"] = {k: round(v["watch"], 3)
                            for k, v in cuts["monthly"].items()}

    with open(MID_DIR / f"model_{gname}_full.pkl", "wb") as fh:
        pickle.dump({"model": m, "features": feats}, fh)

    # ── reduced 모델 (중기예보 D+4~7 용) — 바람 없이 학습 ──────────────
    from etl.build_features import REDUCED_FEATURES_V6
    mr = XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9, scale_pos_weight=spw,
        random_state=SEED, eval_metric="aucpr", early_stopping_rounds=40,
        tree_method="hist", n_jobs=-1,
    )
    mr.fit(tr[REDUCED_FEATURES_V6], tr["y_bin"], sample_weight=tr["y_sev"],
           eval_set=[(va[REDUCED_FEATURES_V6], va["y_bin"])],
           sample_weight_eval_set=[va["y_sev"]], verbose=False)
    pr_te = mr.predict_proba(te[REDUCED_FEATURES_V6])[:, 1]
    res["reduced"] = {
        "features": REDUCED_FEATURES_V6,
        "test_pr_auc": round(float(average_precision_score(te["y_bin"], pr_te)), 4),
        "test_lift": round(float(average_precision_score(te["y_bin"], pr_te))
                           / float(te["y_bin"].mean()), 2),
    }
    with open(MID_DIR / f"model_{gname}_reduced.pkl", "wb") as fh:
        pickle.dump({"model": mr, "features": REDUCED_FEATURES_V6}, fh)
    print(f"          reduced(중기용) test PR-AUC {res['reduced']['test_pr_auc']} "
          f"(lift {res['reduced']['test_lift']}배) · 월별 컷 {len(cuts['monthly'])}개")
    return res


def _fit(tr, va, feats, spw, seed: int = SEED):
    m = XGBClassifier(
        n_estimators=400, max_depth=5, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9, scale_pos_weight=spw,
        random_state=seed, eval_metric="aucpr", early_stopping_rounds=40,
        tree_method="hist", n_jobs=-1,
    )
    m.fit(tr[feats], tr["y_bin"], sample_weight=tr["y_sev"],
          eval_set=[(va[feats], va["y_bin"])],
          sample_weight_eval_set=[va["y_sev"]], verbose=False)
    return m


def grouped_cv_region(f: pd.DataFrame, feats: list, n_folds: int = 3) -> dict:
    """지역을 통째로 빼고 학습 → 그 지역을 맞힌다. v6r 에서는 실제로 작동한다.

    v6(그룹 격자)에서는 음성 행에 region 이 없어 test 가 전부 양성이 되고
    PR-AUC 가 항상 1.0 이 나왔다. 지역 격자에서는 음성 행에도 region 이 있다.
    """
    regions = sorted(f["region"].unique())
    folds = [regions[i::n_folds] for i in range(n_folds)]
    rows = []
    for held in folds:
        tr = f[~f["region"].isin(held)]
        te = f[f["region"].isin(held)]
        if te["y_bin"].sum() < 10:
            continue
        from config import SPLIT_TRAIN_END, SPLIT_VALID_YEAR
        va = tr[tr["dt_h"].dt.year == SPLIT_VALID_YEAR]
        tr2 = tr[tr["dt_h"].dt.year <= SPLIT_TRAIN_END]
        spw = float((tr2["y_bin"] == 0).sum() / max((tr2["y_bin"] == 1).sum(), 1))
        m = _fit(tr2, va, feats, spw)
        p = m.predict_proba(te[feats])[:, 1]
        ap = float(average_precision_score(te["y_bin"], p))
        base = float(te["y_bin"].mean())
        rows.append({"held": held, "n_test": int(len(te)),
                     "test_pos_rate": round(base, 5),
                     "pr_auc": round(ap, 5), "lift": round(ap / base, 2)})
        print(f"    빼놓은 지역 {held}")
        print(f"      test {len(te):,}행 · 양성률 {base:.5f} · "
              f"PR-AUC {ap:.5f} · lift {ap/base:.2f}배")
    return {"folds": len(rows),
            "lift_mean": round(float(np.mean([r["lift"] for r in rows])), 2) if rows else None,
            "lift_min": round(float(np.min([r["lift"] for r in rows])), 2) if rows else None,
            "detail": rows}


def run_v6r(feat: pd.DataFrame) -> dict:
    """지역 격자 학습. 모델은 여전히 그룹당 1개(총 2개)."""
    from config import (GROUPS, REGION_GROUP, SPLIT_TRAIN_END,
                        SPLIT_VALID_YEAR, SPLIT_TEST_YEAR)
    from etl.build_features import FULL_FEATURES_V6

    feats = FULL_FEATURES_V6
    out: dict = {}

    for g in GROUPS:
        f = feat[feat["group"] == g].copy()
        year = f["dt_h"].dt.year
        tr = f[year <= SPLIT_TRAIN_END]
        va = f[year == SPLIT_VALID_YEAR]
        te = f[year == SPLIT_TEST_YEAR]
        spw = float((tr["y_bin"] == 0).sum() / max((tr["y_bin"] == 1).sum(), 1))
        print(f"  [{g}] 지역 {f['region'].nunique()}개 · "
              f"train {len(tr):,} / valid {len(va):,} / test {len(te):,}")
        print(f"          양성: {int(tr['y_bin'].sum()):,} / "
              f"{int(va['y_bin'].sum()):,} / {int(te['y_bin'].sum()):,}  spw {spw:.0f}")

        m = _fit(tr, va, feats, spw)
        res = {"regions": int(f["region"].nunique()), "scale_pos_weight": round(spw, 1)}

        for name, d in [("valid", va), ("test", te)]:
            p = m.predict_proba(d[feats])[:, 1]
            base = float(d["y_bin"].mean())
            ap = float(average_precision_score(d["y_bin"], p))
            res[name] = {"pr_auc": round(ap, 5), "pos_rate": round(base, 5),
                         "lift": round(ap / base, 2)}
            # 지역별 성능 분리 보고 (문서 허점 ③ 대응)
            per = []
            for r, rd in d.assign(proba=p).groupby("region"):
                if rd["y_bin"].sum() < 5:
                    continue
                b = float(rd["y_bin"].mean())
                a = float(average_precision_score(rd["y_bin"], rd["proba"]))
                per.append({"region": r, "pos": int(rd["y_bin"].sum()),
                            "pr_auc": round(a, 4), "lift": round(a / b, 2)})
            res[name]["per_region"] = sorted(per, key=lambda x: -x["lift"])

        with open(MID_DIR / f"model_v6r_{g}_full.pkl", "wb") as fh:
            pickle.dump({"model": m, "features": feats}, fh)

        # ── 동일 시험지 비교: 지역 확률 → noisy-OR → 그룹 확률 ──────────
        # 원래 14개 매핑 지역만 합쳐야 v6 의 그룹 라벨과 정확히 같아진다.
        keep = [r for r, gg in REGION_GROUP.items() if gg == g]
        cmp_rows = {}
        for name, d in [("valid", va), ("test", te)]:
            sub = d[d["region"].isin(keep)].copy()
            sub["proba"] = m.predict_proba(sub[feats])[:, 1]
            agg = sub.groupby("dt_h").agg(
                y_bin=("y_bin", "max"),
                nor=("proba", lambda s: 1.0 - np.prod(1.0 - s.to_numpy())),
                pmax=("proba", "max"))
            agg = agg.reset_index()
            hit_nor, _ = weekly_ranking_hit_v6(agg, "nor")
            hit_max, _ = weekly_ranking_hit_v6(agg, "pmax")
            cmp_rows[name] = {
                "n_hours": int(len(agg)), "pos_rate": round(float(agg["y_bin"].mean()), 5),
                "pr_auc_noisyOR": round(float(average_precision_score(agg["y_bin"], agg["nor"])), 5),
                "pr_auc_max": round(float(average_precision_score(agg["y_bin"], agg["pmax"])), 5),
                "weekly_hit_noisyOR": round(hit_nor, 4),
                "weekly_hit_max": round(hit_max, 4),
            }
        res["group_level_compare"] = cmp_rows

        print(f"\n    [{g}] 지역 확률 → noisy-OR → 그룹 시험지 (v6 와 동일 라벨)")
        for name, v in cmp_rows.items():
            print(f"      {name}: PR-AUC {v['pr_auc_noisyOR']:.4f} "
                  f"(max규칙 {v['pr_auc_max']:.4f}) · "
                  f"weekly_hit {v['weekly_hit_noisyOR']:.4f} · 양성률 {v['pos_rate']:.4f}")

        print(f"\n    [{g}] grouped CV — 지역을 통째로 빼고 학습")
        res["grouped_cv"] = grouped_cv_region(f, feats)
        out[g] = res
        print()

    return out


def run_v6(feat: pd.DataFrame) -> dict:
    from config import GROUPS
    from etl.build_features import FULL_FEATURES_V6

    feats = FULL_FEATURES_V6
    results = {}
    for g in GROUPS:
        sub = feat[feat["group"] == g].copy()
        results[g] = _run_one_group(sub, g, feats)

    print("\n  [grouped CV — 지역을 통째로 빼고 학습]")
    results["grouped_cv"] = grouped_cv(feat, feats)
    return results
