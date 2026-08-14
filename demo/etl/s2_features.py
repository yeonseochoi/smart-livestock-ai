"""S2 — 피처 엔지니어링.

절대 규칙 2 준수: 서빙 피처는 예보 API 제공 변수만. NH3·CO2 는 넣지 않는다.

v6 변경 — 방향성 피처 금지를 해제한다.
  기존 금지 사유는 PLUME_GRADE_BUMP(등급 보정)와의 이중 계산이었다.
  v6에서 BUMP 를 영구 OFF 로 확정했으므로(config.py 참조) 겹침이 없다.
  물리 정보는 ①피처(풍상측 노출) + ②조합(플룸 지역 선택) 두 자리에서만 쓴다.
  ①은 수용점 기준 '어디서 오나', ②는 발원 기준 '어디로 가나' — 방향도 단계도 다르다.

[구현 편차] 문서 13-6 E-1 은 FULL_FEATURES 자체를 v6 목록으로 바꾸라고 하지만,
model/s3_train.py 가 이 이름을 그대로 import 해 v5 를 학습한다. 바꾸면 v5 트랙이
즉시 깨지고 회귀 체크리스트 2번("v5 weekly_hit 0.467 유지")을 통과할 수 없다.
→ FULL_FEATURES 는 v5 그대로 두고 FULL_FEATURES_V6 를 새로 추가했다.
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


# ═══════════════════════════════════════════════════════════════════
# v6 — 시차·무풍 + 공간 + 직전1년 민원율
# ═══════════════════════════════════════════════════════════════════

from etl.s2b_spatial import SPATIAL_FEATURES, PRIOR_FEATURES  # noqa: E402

# v5 유산 — 기존 트랙 재현용. 절대 수정하지 말 것. (= 위의 FULL_FEATURES)
FULL_FEATURES_V5 = list(FULL_FEATURES)

# v6 신규 시계열 — 냄새 도달 시간이 중앙값 32분이므로
# 민원 시각의 바람보다 '직전 시각의 바람'이 실제 원인에 가깝다.
LAG_FEATURES = ["ws_lag1", "wd_sin_lag1", "wd_cos_lag1", "ws_lag2", "calm_streak"]

FULL_FEATURES_V6 = (
    ["temp", "humid", "ws", "wd_sin", "wd_cos", "rain",
     "night", "night_ws", "night_humid",
     "hour", "month_sin", "month_cos", "dow"]      # <- block 대신 hour, year 제거
    + LAG_FEATURES
    + SPATIAL_FEATURES
    + PRIOR_FEATURES
)


# 중기예보(D+4~7)용 — 일 단위 기온·강수확률만 제공되므로 바람이 없다.
# 시간 프로파일은 hour·night 이 만든다.
REDUCED_FEATURES_V6 = ["temp", "rain", "hour", "night",
                       "month_sin", "month_cos", "dow"]


def add_lag_features(f: pd.DataFrame) -> pd.DataFrame:
    """시차·무풍 누적. 그룹별로 시간 정렬한 뒤 계산한다.

    * 정렬을 빠뜨리면 shift 가 엉뚱한 행을 가리킨다. 반드시 sort_values 먼저.
    * 그룹이 섞인 채로 shift 하면 A그룹 끝 행이 B그룹 첫 행을 참조한다.
      groupby 안에서 처리해야 한다.
    """
    parts = []
    for g, d in f.sort_values(["group", "dt_h"]).groupby("group", sort=False):
        d = d.copy()
        d["ws_lag1"] = d["ws"].shift(1)
        d["ws_lag2"] = d["ws"].shift(2)
        d["wd_sin_lag1"] = d["wd_sin"].shift(1)
        d["wd_cos_lag1"] = d["wd_cos"].shift(1)

        # 무풍 연속 시간: 풍속 <1.0 이면 +1, 넘으면 0으로 리셋
        calm = (d["ws"] < 1.0).astype(int)
        grp = (calm == 0).cumsum()
        d["calm_streak"] = calm.groupby(grp).cumsum().astype(float)
        parts.append(d)

    out = pd.concat(parts).sort_values(["dt_h", "group"]).reset_index(drop=True)
    # 각 그룹 첫 1~2행은 시차가 없다 → 현재값으로 채운다 (외삽 대신 보수적 대체)
    out["ws_lag1"] = out["ws_lag1"].fillna(out["ws"])
    out["ws_lag2"] = out["ws_lag2"].fillna(out["ws_lag1"])
    out["wd_sin_lag1"] = out["wd_sin_lag1"].fillna(out["wd_sin"])
    out["wd_cos_lag1"] = out["wd_cos_lag1"].fillna(out["wd_cos"])
    return out


def _base_v6(f: pd.DataFrame) -> pd.DataFrame:
    f = f.copy()
    if "hour" not in f.columns:
        f["hour"] = f["dt_h"].dt.hour
    f["calm"] = (f["ws"] < 1.5).astype(int)
    f["humid80"] = (f["humid"] > 80).astype(int)
    # 야간 정의를 시간 단위로 — 기존은 block [0,1,7] = 0~6시, 21~24시
    f["night"] = (~f["hour"].between(7, 17)).astype(int)
    f["night_calm"] = f["night"] * f["calm"]
    f["night_ws"] = f["night"] * f["ws"]
    f["night_humid"] = f["night"] * f["humid"]
    month = f["dt_h"].dt.month
    f["month_sin"] = np.sin(2 * np.pi * month / 12)
    f["month_cos"] = np.cos(2 * np.pi * month / 12)
    f["dow"] = f["dt_h"].dt.dayofweek
    return f


def run_v6(lab: pd.DataFrame) -> pd.DataFrame:
    f = add_lag_features(_base_v6(lab))
    f.to_parquet(MID_DIR / "features_v6.parquet")
    print(f"  피처 테이블 {len(f):,}행 / full_v6 {len(FULL_FEATURES_V6)}개")
    return f


def add_lag_features_region(f: pd.DataFrame) -> pd.DataFrame:
    """v6r — 시차는 지역 경계를 넘으면 안 된다. groupby('region') 으로 계산."""
    parts = []
    for r, d in f.sort_values(["region", "dt_h"]).groupby("region", sort=False):
        d = d.copy()
        d["ws_lag1"] = d["ws"].shift(1)
        d["ws_lag2"] = d["ws"].shift(2)
        d["wd_sin_lag1"] = d["wd_sin"].shift(1)
        d["wd_cos_lag1"] = d["wd_cos"].shift(1)
        calm = (d["ws"] < 1.0).astype(int)
        grp = (calm == 0).cumsum()
        d["calm_streak"] = calm.groupby(grp).cumsum().astype(float)
        parts.append(d)
    out = pd.concat(parts).sort_values(["dt_h", "region"]).reset_index(drop=True)
    out["ws_lag1"] = out["ws_lag1"].fillna(out["ws"])
    out["ws_lag2"] = out["ws_lag2"].fillna(out["ws_lag1"])
    out["wd_sin_lag1"] = out["wd_sin_lag1"].fillna(out["wd_sin"])
    out["wd_cos_lag1"] = out["wd_cos_lag1"].fillna(out["wd_cos"])
    return out


def run_v6r(lab: pd.DataFrame) -> pd.DataFrame:
    f = add_lag_features_region(_base_v6(lab))
    for c in FULL_FEATURES_V6:
        if f[c].dtype == "float64":
            f[c] = f[c].astype("float32")
    f.to_parquet(MID_DIR / "features_v6r.parquet")
    print(f"  피처 테이블 {len(f):,}행 / full_v6 {len(FULL_FEATURES_V6)}개")
    return f


def build_serving_features(b: pd.DataFrame) -> pd.DataFrame:
    """예보 블록 → 학습과 동일한 피처. 학습·서빙 로직 일원화.

    b 는 dt_h, group, temp, humid, ws, wd, wd_sin, wd_cos, rain 을 가져야 한다.
    """
    return add_lag_features(_base_v6(b))
