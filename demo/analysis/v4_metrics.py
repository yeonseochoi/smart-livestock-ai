"""지시 18 — 커뮤니케이션용 지표: ROC-AUC 병기 + 리프트 차트.

메인 프레임은 여전히 '기후학 대비 증분'. 여기서는 첫인상용 지표만 준비한다.
리프트 차트: 블록을 위험도 내림차순으로 상위 x% 추천했을 때 포착되는
민원 건수(y_cnt) 비율. 상위 20% 지점을 표기한다.
"""
from __future__ import annotations

import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from config import MID_DIR, OUT_DIR

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False


def run() -> dict:
    feat = pd.read_parquet(MID_DIR / "features.parquet")
    with open(MID_DIR / "model_full.pkl", "rb") as fh:
        full = pickle.load(fh)

    te = feat[feat["date"].dt.year == 2026].copy()
    te["proba"] = full["model"].predict_proba(te[full["features"]])[:, 1]

    out = {
        "full_test2026": {
            "pr_auc": round(float(average_precision_score(te["y_bin"], te["proba"])), 4),
            "roc_auc": round(float(roc_auc_score(te["y_bin"], te["proba"])), 4),
            "pos_rate": round(float(te["y_bin"].mean()), 4),
        }
    }

    # 리프트 곡선: 상위 x% 블록 추천 시 포착되는 민원(y_cnt) 비율
    d = te.sort_values("proba", ascending=False).reset_index(drop=True)
    cum_complaints = d["y_cnt"].cumsum() / d["y_cnt"].sum()
    frac_blocks = np.arange(1, len(d) + 1) / len(d)
    idx20 = int(np.searchsorted(frac_blocks, 0.2))
    capture20 = float(cum_complaints.iloc[idx20])
    out["lift_capture_at_20pct"] = round(capture20, 4)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(frac_blocks * 100, cum_complaints * 100, color="#2980b9", lw=2,
            label="연속 full 모델 (test 2026.1~7)")
    ax.plot([0, 100], [0, 100], ls="--", color="#7f8c8d", lw=1, label="랜덤 추천")
    ax.axvline(20, ls=":", color="#c0392b", lw=1)
    ax.annotate(f"상위 20% 추천 시\n민원의 {capture20:.0%} 포착",
                xy=(20, capture20 * 100), xytext=(32, capture20 * 100 - 12),
                arrowprops={"arrowstyle": "->"}, fontsize=10)
    ax.set_xlabel("추천(회피 경고) 블록 비율 (%)")
    ax.set_ylabel("포착되는 가축분뇨 민원 비율 (%)")
    ax.set_title("v4 리프트 차트 — 상위 위험 블록 추천의 민원 포착률")
    ax.legend()
    fig.tight_layout(); fig.savefig(OUT_DIR / "v4_lift_chart.png", dpi=130)
    plt.close(fig)

    m = out["full_test2026"]
    print(f"  연속 full (test 2026): PR-AUC {m['pr_auc']} / ROC-AUC {m['roc_auc']}"
          f" / 상위 20% 추천 시 민원 포착 {capture20:.1%}")
    return out
