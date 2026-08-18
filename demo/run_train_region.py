"""지역 격자 트랙 (병행 검증용).

한 줄 = (시각, 지역 21개).  모델은 여전히 그룹당 1개(근거리형/원거리형 = 2개).
지역은 '행을 나누는 축', 그룹은 '모델을 나누는 축'.

python run_train_region.py   (run_train.py 선실행 권장 — gimje_farms_ri.csv 를 쓴다)
"""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

from config import MID_DIR, OUT_DIR, FINDINGS, section
from console import use_utf8_stdout

RESULTS: dict = {}


def main():
    use_utf8_stdout()
    print("악취·분뇨 프로젝트 — 지역 격자 트랙")

    from preprocess import clean_data, geocode_gimje, build_grid, build_features, spatial_features
    from model import train_model

    section("1. 민원·기상")
    cp, wp = MID_DIR / "complaints_clean.parquet", MID_DIR / "weather_hourly.parquet"
    if cp.exists() and wp.exists():
        c, w = pd.read_parquet(cp), pd.read_parquet(wp)
        print("  기존 S0 산출물 사용")
    else:
        c, w = clean_data.run()

    if not (MID_DIR / "gimje_farms_ri.csv").exists():
        section("2. 김제 리(里) 재지오코딩")
        geocode_gimje.run()

    section("3. (1시간 x 지역) 라벨 테이블")
    lab = build_grid.run_regional(c, w)
    RESULTS["grid"] = {
        "rows": len(lab), "regions": int(lab["region"].nunique()),
        "pos": int(lab["y_bin"].sum()),
        "pos_rate": round(float(lab["y_bin"].mean()), 5),
    }

    section("4. 지역별 풍상측 노출")
    lab = spatial_features.run_regions(lab)

    section("5. 지역별 직전1년 민원율")
    lab = spatial_features.add_prior_rate_regional(lab)

    section("6. 시차·무풍 피처 (지역 경계 유지)")
    feat = build_features.run_regional(lab)

    section("7. 그룹별 학습 + 동일시험지 비교 + grouped CV")
    RESULTS["train"] = train_model.run_regional(feat)

    section("결과 저장")
    RESULTS["findings"] = FINDINGS
    with open(OUT_DIR / "training_results_regional.json", "w", encoding="utf-8") as fh:
        json.dump(RESULTS, fh, ensure_ascii=False, indent=2, default=str)
    print(f"  저장: {OUT_DIR / 'training_results_regional.json'}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
