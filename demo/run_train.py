"""v6 — 격자 재정의 라운드 (모델링까지. RAG·추천·서빙 제외).

python run_train.py

1. S0   정제 (complaints_clean / weather_hourly 없으면 자동 생성)
2. S0b  김제 리 재지오코딩
3. S1'  (그룹, 1시간) 라벨 테이블
4. S2b  풍상측 노출 + 직전1년 민원율
5. S2'  시차·무풍 피처
6. S3'  그룹별 XGBoost + grouped CV
7. 결과 요약 → out/v6_results.json

[구현 편차] 문서 13-16 은 "demo.py 선실행 필요"라고 했지만 demo.py 는 RAG 인덱싱과
예보 API 호출까지 전부 돌린다(이번 범위 밖). → S0 만 필요하므로 직접 호출한다.
"""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

from config import MID_DIR, OUT_DIR, FINDINGS, section
from console import use_utf8_stdout      # legacy import (수정 금지)

RESULTS: dict = {}


def bootstrap_s0():
    """complaints_clean.parquet / weather_hourly.parquet 확보."""
    cpath = MID_DIR / "complaints_clean.parquet"
    wpath = MID_DIR / "weather_hourly.parquet"
    if cpath.exists() and wpath.exists():
        print("  기존 S0 산출물 사용")
        return pd.read_parquet(cpath), pd.read_parquet(wpath)
    from preprocess import clean_data
    return clean_data.run()


def main():
    use_utf8_stdout()
    print("악취·분뇨 프로젝트 — v6 격자 재정의 (모델링까지)")

    from preprocess import geocode_gimje, build_grid, build_features, spatial_features
    from model import train_model

    section("v6-0 S0 — 민원·기상 정제")
    c, w = bootstrap_s0()
    RESULTS["s0"] = {"complaints": len(c), "weather": len(w)}

    section("v6-1 S0b — 김제 리 재지오코딩")
    gj = geocode_gimje.run()
    RESULTS["geocode_gimje"] = {
        "rows": len(gj), "with_coord": int(gj["lat"].notna().sum()),
        "src": {k: int(v) for k, v in gj["src"].value_counts().items()},
    }

    section("v6-2 S1' — (그룹, 1시간) 라벨 테이블")
    lab = build_grid.run_v6(c, w)
    RESULTS["s1_v6"] = {
        "rows": len(lab),
        "pos_rate": round(float(lab["y_bin"].mean()), 4),
        "by_group": {g: {"rows": len(d), "pos": int(d["y_bin"].sum())}
                     for g, d in lab.groupby("group")},
    }

    section("v6-3 S2b — 풍상측 노출 + 직전1년 민원율")
    lab = spatial_features.run(lab)
    lab = spatial_features.add_prior_rate(lab)

    section("v6-4 S2' — 시차·무풍 피처")
    feat = build_features.run_v6(lab)
    RESULTS["s2_v6"] = {"rows": len(feat),
                        "n_features": len(build_features.FULL_FEATURES_V6),
                        "features": build_features.FULL_FEATURES_V6}

    section("v6-5 S3' — 그룹별 학습 + grouped CV")
    RESULTS["s3_v6"] = train_model.run_v6(feat)

    section("결과 저장")
    RESULTS["findings"] = FINDINGS
    with open(OUT_DIR / "v6_results.json", "w", encoding="utf-8") as fh:
        json.dump(RESULTS, fh, ensure_ascii=False, indent=2, default=str)
    print(f"  저장: {OUT_DIR / 'v6_results.json'}")
    print(f"  실행 중 발견 {len(FINDINGS)}건")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
