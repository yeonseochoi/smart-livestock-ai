"""S8 — 검증·발표 그래프 6장 (out/*.png).

s8_1 조건 재현 / s8_2 풍향 정합성 / s8_3 거리 감쇠 /
s8_4 플룸 적중률 / s8_5 백테스트 추이 / s8_6 센서 스파이크
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import (ASOS_FULL_CSV, OUT_DIR, PROV, SENSOR_XLSX, SEED,
                    WANGGUNG_LAT, WANGGUNG_LON, DOWNTOWN_LAT, DOWNTOWN_LON,
                    finding)

# legacy import (수정 금지)
from geo import bearing, angle_diff
from plume import pasquill_class, dispersion_factor, plume_half_angle, initial_sigmas

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

R_EARTH = 6371000.0


def _dist_m(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = p2 - p1
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R_EARTH * np.arcsin(np.sqrt(a))


def s8_1_condition(lab: pd.DataFrame) -> dict:
    """야간·무풍·고습 블록 vs 그 외의 민원율."""
    m = lab.assign(cond=((lab["block"].isin([0, 1, 7])) & (lab["ws"] < 1.5)
                         & (lab["humid"] > 80)))
    g = m.groupby("cond")["y_bin"].mean()
    rate_cond = float(g.get(True, np.nan))
    rate_else = float(g.get(False, np.nan))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["야간·무풍·고습", "그 외"], [rate_cond, rate_else],
           color=["#c0392b", "#7f8c8d"])
    ax.set_ylabel("블록 민원 발생률 (y_bin)")
    ax.set_title("S8-1 조건 재현 — 익산시 공식 분석(야간·무풍·고습 최다)과 대조")
    for i, v in enumerate([rate_cond, rate_else]):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom")
    fig.tight_layout(); fig.savefig(OUT_DIR / "s8_1_condition.png", dpi=130); plt.close(fig)
    ratio = rate_cond / rate_else if rate_else else float("nan")
    print(f"  S8-1: 야간·무풍·고습 {rate_cond:.3f} vs 그 외 {rate_else:.3f} (x{ratio:.2f})")
    return {"cond": rate_cond, "else": rate_else, "ratio": ratio}


def s8_2_wind(lab: pd.DataFrame) -> dict:
    """민원 블록의 풍향 분포 ↔ 왕궁→도심 방위각."""
    brg_to_town = bearing(WANGGUNG_LAT, WANGGUNG_LON, DOWNTOWN_LAT, DOWNTOWN_LON)
    vec_expected = (brg_to_town + 180) % 360  # 그 방향으로 불려면 바람은 반대편에서 와야 함
    pos = lab[lab["y_bin"] == 1]
    neg = lab[lab["y_bin"] == 0]
    wd_pos = (np.degrees(np.arctan2(pos["wd_sin"], pos["wd_cos"])) + 360) % 360
    wd_neg = (np.degrees(np.arctan2(neg["wd_sin"], neg["wd_cos"])) + 360) % 360
    fig, ax = plt.subplots(figsize=(8, 4))
    bins = np.arange(0, 361, 15)
    ax.hist(wd_neg, bins=bins, density=True, alpha=0.5, label="민원 없음", color="#7f8c8d")
    ax.hist(wd_pos, bins=bins, density=True, alpha=0.6, label="민원 블록", color="#c0392b")
    ax.axvline(vec_expected, ls="--", color="k",
               label=f"왕궁→도심 확산 시 풍향 {vec_expected:.0f}도 [C좌표]")
    ax.set_xlabel("풍향 (도, 불어오는 방향)"); ax.set_ylabel("밀도")
    ax.set_title("S8-2 풍향 정합성"); ax.legend()
    fig.tight_layout(); fig.savefig(OUT_DIR / "s8_2_wind.png", dpi=130); plt.close(fig)
    # 정합 지표: 민원 블록 풍향과 기대 풍향의 평균 각도차
    d_pos = np.mean([angle_diff(w, vec_expected) for w in wd_pos])
    d_neg = np.mean([angle_diff(w, vec_expected) for w in wd_neg])
    print(f"  S8-2: 기대 풍향과 평균 각도차 — 민원 {d_pos:.1f}도 vs 비민원 {d_neg:.1f}도")
    return {"diff_pos": float(d_pos), "diff_neg": float(d_neg),
            "expected_vec": vec_expected}


def s8_3_distance(complaints: pd.DataFrame) -> dict:
    """축산단지 중심 ↔ 민원 좌표 거리 분포 (문헌 1km/2km/8km 대조)."""
    c = complaints[complaints["is_iksan"] & complaints["is_livestock"]].dropna(
        subset=["위도", "경도"])
    d_km = _dist_m(WANGGUNG_LAT, WANGGUNG_LON, c["위도"].values, c["경도"].values) / 1000
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(d_km, bins=np.arange(0, 25, 0.5), color="#2c3e50")
    for x, lbl in [(1, "일반 1km"), (2, "집단 2km"), (8, "자원화 8km")]:
        ax.axvline(x, ls="--", alpha=0.7, color="#c0392b")
        ax.text(x, ax.get_ylim()[1] * 0.9, lbl, rotation=90, va="top", fontsize=8)
    ax.set_xlabel("왕궁 축산단지 중심으로부터 거리 (km) [C좌표]")
    ax.set_ylabel("가축분뇨 민원 건수")
    ax.set_title("S8-3 거리 감쇠")
    fig.tight_layout(); fig.savefig(OUT_DIR / "s8_3_distance.png", dpi=130); plt.close(fig)
    med = float(np.median(d_km))
    within8 = float((d_km <= 8).mean())
    print(f"  S8-3: 거리 중앙값 {med:.1f}km, 8km 이내 비율 {within8:.1%}")
    return {"median_km": med, "within_8km": within8}


def _load_cloud() -> pd.Series | None:
    """ASOS 원본에서 전운량(10분위) 시계열을 뽑는다 (SKY 근사용)."""
    try:
        head = pd.read_csv(ASOS_FULL_CSV, nrows=1)
        cloud_col = next((c for c in head.columns if "전운량" in c), None)
        dt_col = next((c for c in head.columns if "일시" in c), None)
        if not cloud_col or not dt_col:
            return None
        a = pd.read_csv(ASOS_FULL_CSV, usecols=[dt_col, cloud_col])
        a[dt_col] = pd.to_datetime(a[dt_col])
        s = a.set_index(dt_col)[cloud_col]
        PROV.log("D2 ASOS 전운량", ASOS_FULL_CSV, real=True,
                 note="S8-4 안정도(SKY) 산정용")
        return s
    except Exception as e:
        finding(f"ASOS 전운량 로딩 실패({e!r}) — S8-4 는 맑음 가정으로 계산(보수적)")
        return None


def s8_4_plume_hit(complaints: pd.DataFrame, weather: pd.DataFrame) -> dict:
    """과거 민원 시각의 실측 기상으로 플룸 계산 → 민원 좌표가 플룸 안(유효 반각 이내
    + 8km 이내)인 비율. 플라시보(풍향 90도 회전) 대비로 정합성을 본다."""
    c = complaints[complaints["is_iksan"] & complaints["is_livestock"]].dropna(
        subset=["위도", "경도"]).copy()
    w = weather.set_index("dt")
    cloud = _load_cloud()

    rng = np.random.default_rng(SEED)
    if len(c) > 3000:
        c = c.sample(3000, random_state=SEED)

    hits = placebo_hits = used = 0
    for _, row in c.iterrows():
        t = row["dt"].floor("h")
        if t not in w.index:
            continue
        met = w.loc[t]
        if pd.isna(met["wd"]) or pd.isna(met["ws"]):
            continue
        dist = float(_dist_m(WANGGUNG_LAT, WANGGUNG_LON, row["위도"], row["경도"]))
        if dist < 100 or dist > 8000:
            continue
        brg = bearing(WANGGUNG_LAT, WANGGUNG_LON, row["위도"], row["경도"])
        if cloud is not None and t in cloud.index and not pd.isna(cloud.loc[t]):
            cl = float(cloud.loc[t])
            sky = "4" if cl >= 9 else ("3" if cl >= 6 else "1")
        else:
            sky = "1"
        stability, _ = pasquill_class(float(met["ws"]), sky, t.to_pydatetime(),
                                      WANGGUNG_LAT, WANGGUNG_LON)
        wind_to = (float(met["wd"]) + 180) % 360
        half = plume_half_angle(dist, stability)
        used += 1
        if angle_diff(wind_to, brg) <= half:
            hits += 1
        if angle_diff((wind_to + 90) % 360, brg) <= half:
            placebo_hits += 1

    rate = hits / used if used else float("nan")
    placebo = placebo_hits / used if used else float("nan")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["실제 풍향", "풍향 90도 회전\n(플라시보)"], [rate, placebo],
           color=["#2980b9", "#95a5a6"])
    ax.set_ylabel("민원 좌표가 플룸 유효 반각 안에 든 비율")
    ax.set_title(f"S8-4 플룸 적중률 (n={used:,}, 발원=왕궁 [C좌표])")
    for i, v in enumerate([rate, placebo]):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom")
    fig.tight_layout(); fig.savefig(OUT_DIR / "s8_4_plume_hit.png", dpi=130); plt.close(fig)
    print(f"  S8-4: 플룸 적중 {rate:.3f} vs 플라시보 {placebo:.3f} (n={used:,})")
    return {"hit": rate, "placebo": placebo, "n": used}


def s8_5_backtest() -> dict:
    """주간 랭킹 적중률 추이 (valid 2025 + test 2026)."""
    frames = []
    for split in ["valid", "test"]:
        p = OUT_DIR / f"weekly_hit_{split}.csv"
        if p.exists():
            d = pd.read_csv(p)
            d["split"] = split
            frames.append(d)
    if not frames:
        finding("weekly_hit csv 없음 — S8-5 생략")
        return {}
    d = pd.concat(frames)
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(d))
    colors = d["split"].map({"valid": "#7f8c8d", "test": "#2980b9"})
    ax.bar(x, d["hit"], color=colors)
    ax.axhline(0.2, ls="--", color="#c0392b", label="랜덤 기대 0.20")
    ax.set_xticks(x[::4]); ax.set_xticklabels(d["week"].iloc[::4], rotation=45, fontsize=7)
    ax.set_ylabel("주간 랭킹 적중률"); ax.legend()
    ax.set_title("S8-5 백테스트 — 2025(회색)~2026(파랑) 주간 적중률")
    fig.tight_layout(); fig.savefig(OUT_DIR / "s8_5_backtest.png", dpi=130); plt.close(fig)
    m = float(d[d["split"] == "test"]["hit"].mean())
    print(f"  S8-5: test 평균 주간 적중률 {m:.3f}")
    return {"test_mean": m, "n_weeks": len(d)}


def s8_6_sensor() -> dict:
    """AI Hub NH3 스파이크 전후 농도 그래프 — '작업이 방출 급증을 만든다' 근거.
    서빙 모델 피처로는 쓰지 않는다 (절대 규칙 2)."""
    s = pd.read_excel(SENSOR_XLSX)
    PROV.log("D5 양돈센서(AI Hub)", SENSOR_XLSX, real=True,
             note=f"{len(s):,}행 — S8-6 근거 그래프 전용, 모델 피처 사용 금지")
    s["input_datetime"] = pd.to_datetime(s["input_datetime"])
    # 스파이크 = NH3 30분 '증가'폭 최대 표본 (작업에 의한 방출 급증 근거이므로 상승만)
    idx = s["NH3_0.5__delta_30m"].idxmax()
    spike = s.loc[idx]
    ch, day = spike["chamber"], spike["input_datetime"].normalize()
    win = s[(s["chamber"] == ch)
            & (s["input_datetime"].between(day, day + pd.Timedelta(days=1)))].sort_values(
        "input_datetime")
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(win["input_datetime"], win["NH3_0.5__current"], label="NH3 0.5m", color="#c0392b")
    ax.plot(win["input_datetime"], win["NH3_1.5__current"], label="NH3 1.5m",
            color="#e67e22", alpha=0.7)
    ax.axvline(spike["input_datetime"], ls="--", color="k", label="최대 급증 시점")
    ax.set_ylabel("NH3 (ppm)"); ax.legend()
    ax.set_title(f"S8-6 돈사 NH3 스파이크 ({ch}, {day.date()})")
    fig.tight_layout(); fig.savefig(OUT_DIR / "s8_6_sensor.png", dpi=130); plt.close(fig)
    print(f"  S8-6: 최대 30분 증가폭 {spike['NH3_0.5__delta_30m']:+.2f} ppm ({ch})")
    return {"chamber": str(ch), "max_delta": float(spike["NH3_0.5__delta_30m"])}


def run(lab: pd.DataFrame, complaints: pd.DataFrame, weather: pd.DataFrame) -> dict:
    out = {}
    out["s8_1"] = s8_1_condition(lab)
    out["s8_2"] = s8_2_wind(lab)
    out["s8_3"] = s8_3_distance(complaints)
    out["s8_4"] = s8_4_plume_hit(complaints, weather)
    out["s8_5"] = s8_5_backtest()
    out["s8_6"] = s8_6_sensor()
    return out
