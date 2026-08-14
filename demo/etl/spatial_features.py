"""S2b — 풍상측 노출 공간 피처.

각 (시각, 그룹|지역)에 대해 그 시각 풍향의 풍상측 부채꼴 안에 있는 축산농가 수를
익산·김제 × 축종별로 센다. 사육두수는 쓰지 않는다 — 좌표만 쓴다.

★ v6.1 변경 — 축종을 돼지만 세던 것을 돼지/소/가금으로 분리했다.
  이유: 민원 라벨이 '가축 분뇨 냄새(닭, 돼지, 소)' 로 3축종 통합인데
        발원을 돼지 160곳(익산 전체의 13.4%)만 세는 것은 라벨과 정의가 어긋난다.
        축종별 악취 원인물질도 다르다(축산악취 관리 지침서: 우사 암모니아·황화수소·
        트리메틸아민 / 계사 암모니아·메틸머캅탄·황화메틸 / 돈사 별도).
  실측: 성능은 유의하게 변하지 않았다(시드 5개, test ROC 0.8414 → 0.8459).
        정의 일치를 위한 수정이며 성능 개선 주장은 하지 않는다.

절대규칙 1 무관: 플룸 출력을 쓰지 않는 순수 기하 계산.
절대규칙 2 준수: 풍향(VEC)은 단기예보 제공 변수.

[구현 편차] 문서의 run() 은 무조건 parquet 를 저장한다. 그러나 문서 13-9 H-1 이
서빙 경로에서도 같은 run() 을 호출하므로, 그대로 두면 서빙이 학습 산출물을
덮어쓴다. → save 인자를 두고 기본 True, 서빙에서는 False 로 부른다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import MID_DIR, DATA_ROOT, GROUP_CENTER, GROUPS

SECTOR_DEG = 30.0     # 풍상측 부채꼴 반각. ±30도.
RADIUS_KM = 15.0      # 시가지 원거리형 발원 거리(5~14km)를 덮는 값.

# 익산 farm_coords_vworld.csv '축종' / 김제 '사육업종' 값에 대한 정규식
SPECIES = {
    "pig": r"돼지|종돈",
    "cattle": r"한우|젖소|육우",
    "poultry": r"육계|산란계|종계|오리|부화용알|메추리",
}

SPATIAL_FEATURES = (
    [f"up_ik_{k}" for k in SPECIES]
    + [f"up_gj_{k}" for k in SPECIES]
    + ["up_nearest_km"]
)
PRIOR_FEATURES = ["prior_rate_1y", "prior_std_1y", "prior_month"]


def load_sources() -> dict:
    """{축종: (익산좌표배열, 김제좌표배열)}"""
    ik = pd.read_csv(DATA_ROOT / "05_지오코딩_결과" / "farm_coords_vworld.csv")
    ik = ik.dropna(subset=["lat", "lon"])
    gj = pd.read_csv(MID_DIR / "gimje_farms_ri.csv").dropna(subset=["lat", "lon"])
    out = {}
    for key, pat in SPECIES.items():
        a = ik[ik["축종"].astype(str).str.contains(pat, na=False)][["lat", "lon"]].to_numpy()
        b = gj[gj["사육업종"].astype(str).str.contains(pat, na=False)][["lat", "lon"]].to_numpy()
        out[key] = (a, b)
    print("  발원 후보 — " + " · ".join(
        f"{k} 익산 {len(v[0])}/김제 {len(v[1])}" for k, v in out.items()))
    return out


def _geo(la, lo, F):
    """(거리 km, 방위각 도) 배열을 한 번에 계산."""
    p = np.pi / 180.0
    LA, LO = F[:, 0], F[:, 1]
    d = 6371.0 * np.arccos(np.clip(
        np.sin(la * p) * np.sin(LA * p) +
        np.cos(la * p) * np.cos(LA * p) * np.cos((LO - lo) * p), -1, 1))
    y = np.sin((LO - lo) * p) * np.cos(LA * p)
    x = (np.cos(la * p) * np.sin(LA * p) -
         np.sin(la * p) * np.cos(LA * p) * np.cos((LO - lo) * p))
    b = np.degrees(np.arctan2(y, x)) % 360
    return d, b


def _ang_diff(a, b):
    return np.abs((a - b + 180) % 360 - 180)


def _fill_one(out, mask, la, lo, src):
    """한 수용점(그룹 또는 지역)의 행들에 축종별 풍상측 농가 수를 채운다."""
    wd = out.loc[mask, "wd"].to_numpy()
    near = np.full(len(wd), np.inf)
    for key, (ik_xy, gj_xy) in src.items():
        for side, F in (("ik", ik_xy), ("gj", gj_xy)):
            col = f"up_{side}_{key}"
            if not len(F):
                out.loc[mask, col] = 0.0
                continue
            d, b = _geo(la, lo, F)
            keep = d <= RADIUS_KM
            d, b = d[keep], b[keep]
            if not len(b):
                out.loc[mask, col] = 0.0
                continue
            inx = _ang_diff(b[None, :], wd[:, None]) <= SECTOR_DEG
            out.loc[mask, col] = inx.sum(axis=1)
            cand = np.where(inx, d[None, :], np.inf).min(axis=1)
            near = np.minimum(near, cand)
    out.loc[mask, "up_nearest_km"] = np.where(np.isfinite(near), near, RADIUS_KM)


def run(lab: pd.DataFrame, save: bool = True) -> pd.DataFrame:
    """v6 — 그룹 중심 기준."""
    src = load_sources()
    out = lab.copy()
    for col in SPATIAL_FEATURES:
        out[col] = np.nan
    for g in GROUPS:
        m = (out["group"] == g).to_numpy()
        if m.any():
            _fill_one(out, m, *GROUP_CENTER[g], src)
    out["up_missing"] = out["wd"].isna().astype(int)
    for col in SPATIAL_FEATURES:
        out[col] = out[col].fillna(0.0)
    if save:
        out.to_parquet(MID_DIR / "label_spatial_v6.parquet")
    print("  공간 피처 %d개 — " % len(SPATIAL_FEATURES) + " · ".join(
        f"{c} 평균 {out[c].mean():.1f}" for c in SPATIAL_FEATURES[:6]))
    return out


def run_regions(lab: pd.DataFrame, save: bool = True) -> pd.DataFrame:
    """v6r — 지역 중심 기준."""
    from config import REGION_CENTER
    src = load_sources()
    out = lab.copy()
    for col in SPATIAL_FEATURES:
        out[col] = np.nan
    for r, (la, lo) in REGION_CENTER.items():
        m = (out["region"] == r).to_numpy()
        if m.any():
            _fill_one(out, m, la, lo, src)
    out["up_missing"] = out["wd"].isna().astype(int)
    for col in SPATIAL_FEATURES:
        out[col] = out[col].fillna(0.0)
    chk = out.groupby(out["wd"].round(0))["up_ik_pig"].nunique()
    print(f"  풍향 1도 단위로 up_ik_pig 이 여러 값을 갖는 각도 수: "
          f"{int((chk > 1).sum())} / {len(chk)}")
    if save:
        out.to_parquet(MID_DIR / "label_spatial_v6r.parquet")
    return out


def add_prior_rate(lab: pd.DataFrame, key: str = "group") -> pd.DataFrame:
    """직전 365일 민원율 3종.  key='group'(v6) 또는 'region'(v6r).

    ★ 반드시 '예측 시점 이전' 데이터로만 계산한다. shift(1) 없이 rolling 하면
      자기 자신이 포함되어 누수가 된다.

    ⚠️ 한계 — 그룹이 2개뿐인 v6 에서 이 컬럼은 '지역 고유 요인'이 아니라
      연도 추세의 대리변수로 작동한다(농촌근거리 2020년 0.0028 → 2025년 0.0369,
      13배 단조 증가). 학습 구간(≤2024) 최대 0.025 를 valid(2025) 가 넘어서
      트리가 외삽하지 못한다. 사용 시 반드시 이 한계를 병기할 것.
    """
    out = lab.sort_values([key, "dt_h"]).copy()
    parts = []
    for _, d in out.groupby(key, sort=False):
        d = d.copy()
        s = d.set_index("dt_h")["y_bin"]
        prev = s.shift(1)                      # ★ 자기 행 제외
        d["prior_rate_1y"] = prev.rolling("365D", min_periods=24 * 30).mean().values
        d["prior_std_1y"] = prev.rolling("365D", min_periods=24 * 30).std().values
        d["prior_month"] = (
            prev.groupby(s.index.month).transform(
                lambda x: x.expanding(min_periods=24 * 7).mean())
        ).values
        parts.append(d)
    out = pd.concat(parts).sort_index()
    out["prior_missing"] = out["prior_rate_1y"].isna().astype(int)
    for c in PRIOR_FEATURES:
        out[c] = out[c].fillna(out[c].mean())
    print(f"  직전1년 민원율 3종 ({key} 단위) — 결측 대체 "
          f"{int(out['prior_missing'].sum()):,}행 ({out['prior_missing'].mean():.1%})")
    return out


def add_prior_rate_region(lab: pd.DataFrame) -> pd.DataFrame:
    return add_prior_rate(lab, key="region")


def run_serving(b: pd.DataFrame) -> pd.DataFrame:
    """서빙용 — 저장하지 않는다 (학습 산출물 덮어쓰기 방지)."""
    return run(b, save=False)


def add_prior_rate_serving(b: pd.DataFrame, asof: pd.Timestamp = None) -> pd.DataFrame:
    """서빙용 직전1년 민원율. 예보 전 구간에 같은 값(상수)을 붙인다.

    ⚠️ 학습에서는 행마다 변하는 값인데 서빙에서는 상수다. 분포가 어긋나므로
      이 컬럼의 기여를 신뢰하지 말 것 (add_prior_rate 의 한계 주석 참조).
    """
    from config import REGION_GROUP
    c = pd.read_parquet(MID_DIR / "complaints_clean.parquet")
    c = c[c["is_iksan"] & c["is_livestock"]].copy()
    c["group"] = c["지역"].map(REGION_GROUP)
    asof = asof or pd.Timestamp.now().floor("h")
    win = c[(c["dt"] > asof - pd.Timedelta("365D")) & (c["dt"] <= asof)]

    out = b.copy()
    for g in out["group"].unique():
        n_pos = win[win["group"] == g]["dt"].dt.floor("h").nunique()
        rate = n_pos / (365 * 24)
        out.loc[out["group"] == g, "prior_rate_1y"] = rate
        out.loc[out["group"] == g, "prior_std_1y"] = np.sqrt(rate * (1 - rate))
        out.loc[out["group"] == g, "prior_month"] = rate   # 근사
    out["prior_missing"] = 0
    return out
