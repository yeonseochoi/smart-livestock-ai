"""S2 — 피처 엔지니어링.

절대 규칙 2 준수: 서빙 피처는 예보 API 제공 변수만.
NH3·CO2 는 넣지 않는다. 방향성 피처(풍향-축산단지 방위각 차)도 넣지 않는다
(S8-2 분석 전용 — 이중 계산 금지).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import MID_DIR

# 단기예보(full)용 — TMP/REH/WSD/VEC/PTY 와 1:1 매핑되는 것만.
# v3 지시 11: 연속변수 버전(임계값 없는 night x ws / night x humid)이 기본.
# 이진 플래그 버전은 백업(model_full_binary.pkl)으로만 보존한다.
# 근거: v2 실험 3 — 성능 동급(적중률 0.477 vs 0.467)이면서 예보 오차 섭동에
# 임계값(1.5/80) 경계 뒤집힘이 없는 연속 버전이 안전.
FULL_FEATURES = [
    "temp", "humid", "ws", "wd_sin", "wd_cos", "rain",
    "night", "night_ws", "night_humid",
    "block", "month_sin", "month_cos", "dow", "year",
]
BINARY_FULL_FEATURES = [
    "temp", "humid", "ws", "wd_sin", "wd_cos", "rain",
    "calm", "humid80", "night", "night_calm",
    "block", "month_sin", "month_cos", "dow", "year",
]
# 중기예보(reduced)용 — 중기예보에는 풍향·풍속·습도가 없다
REDUCED_FEATURES = ["temp", "rain", "month_sin", "month_cos", "block", "dow"]


def run(lab: pd.DataFrame) -> pd.DataFrame:
    f = lab.copy()
    f["calm"] = (f["ws"] < 1.5).astype(int)
    f["humid80"] = (f["humid"] > 80).astype(int)
    f["night"] = f["block"].isin([0, 1, 7]).astype(int)
    f["night_calm"] = f["night"] * f["calm"]
    f["night_ws"] = f["night"] * f["ws"]        # 연속 상호작용 (기본 모델용)
    f["night_humid"] = f["night"] * f["humid"]
    month = f["date"].dt.month
    f["month_sin"] = np.sin(2 * np.pi * month / 12)
    f["month_cos"] = np.cos(2 * np.pi * month / 12)
    f["dow"] = f["date"].dt.dayofweek
    f["year"] = f["date"].dt.year
    f.to_parquet(MID_DIR / "features.parquet")
    print(f"  피처 테이블 {len(f):,}행 / full {len(FULL_FEATURES)}개, reduced {len(REDUCED_FEATURES)}개")
    return f
