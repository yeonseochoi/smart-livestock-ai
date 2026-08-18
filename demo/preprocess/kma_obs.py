"""기상청 API허브 지상관측 수집 — 포털 수동 다운로드를 대체한다.

왜 만들었나
  기존에는 기상자료개방포털에서 CSV 를 손으로 받아 병합했다. 인코딩이 깨져
  xlsx 로 다시 저장하는 사고가 있었고(2026-08-14), 익산 AWS 2026년분은
  포털 오류로 아예 확보하지 못했다. API 로 받으면 두 문제가 같이 사라진다.

검증 (2026-08-18)
  전주 ASOS 146, 2020-01 한 달(744시각)을 API 와 기존 CSV 로 각각 받아 대조:
  기온·습도·풍속·풍향 **744/744 완전 일치, 최대차 0.00**. 안전하게 대체 가능.

주의 — 풍향 단위 [A]
  API 의 WD 는 0~36 코드다(16방위를 10 으로 나눈 값). **WD x 10 = 도(deg)**.
  실측 확인: WD 14 -> 140도, 5 -> 50도, 11 -> 110도 로 기존 CSV 와 일치.
  그대로 쓰면 풍향이 1/10 로 찌그러지므로 반드시 x10 한다.

주의 — 결측 표기 [A]
  -9 / -9.0 이 결측이다. NaN 으로 바꾼 뒤 기존 파이프라인의 보간에 넘긴다.

주의 — 한 번에 받을 수 있는 양 [A]
  kma_sfctm3 는 요청 기간과 무관하게 **약 745행(≈1개월)** 에서 잘린다.
  1년을 요청해도 745행만 온다. 그래서 월 단위로 쪼개 받는다.
"""
from __future__ import annotations

import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
import requests

# 인증키는 환경변수 우선. 없으면 프로젝트 공용 키로 폴백한다(legacy/kma.py 와 같은 방식).
_FALLBACK_HUB_KEY = "s0zxaCESRKmM8WghEjSppw"
HUB_KEY = os.environ.get("KMA_HUB_KEY", _FALLBACK_HUB_KEY)

ASOS_URL = "https://apihub.kma.go.kr/api/typ01/url/kma_sfctm3.php"
AWS_URL = "https://apihub.kma.go.kr/api/typ01/url/awsh.php"
TIMEOUT = 120

# kma_sfctm3 고정폭 출력의 컬럼 위치 [A] — help=1 헤더로 확인
ASOS_COL = {
    "tm": 0, "stn": 1, "wd": 2, "ws": 3,
    "ta": 11, "td": 12, "hm": 13, "rn": 15, "ca_tot": 25,
}


def _month_ranges(start: str, end: str) -> list[tuple[str, str]]:
    """'YYYYMM' ~ 'YYYYMM' 를 월 단위 (tm1, tm2) 목록으로 쪼갠다."""
    out = []
    cur = pd.Timestamp(f"{start}01")
    last = pd.Timestamp(f"{end}01") + pd.offsets.MonthEnd(0)
    while cur <= last:
        mend = cur + pd.offsets.MonthEnd(0)
        out.append((cur.strftime("%Y%m%d0000"), mend.strftime("%Y%m%d2300")))
        cur = mend + pd.Timedelta(days=1)
    return out


def fetch_asos(stn: int, start: str, end: str, verbose: bool = True) -> pd.DataFrame:
    """전주 ASOS 등 종관관측 시간자료. 컬럼은 기존 CSV 스키마 + 전운량."""
    frames = []
    ranges = _month_ranges(start, end)
    for i, (t1, t2) in enumerate(ranges, 1):
        r = requests.get(ASOS_URL, params={
            "tm1": t1, "tm2": t2, "stn": stn, "help": 0, "authKey": HUB_KEY},
            timeout=TIMEOUT)
        r.raise_for_status()
        rows = [l.split() for l in r.text.splitlines()
                if l and not l.startswith("#")]
        if rows:
            frames.append(pd.DataFrame({
                "지점": stn,
                "일시": [x[ASOS_COL["tm"]] for x in rows],
                "기온C": [float(x[ASOS_COL["ta"]]) for x in rows],
                "강수량mm": [float(x[ASOS_COL["rn"]]) for x in rows],
                "풍속ms": [float(x[ASOS_COL["ws"]]) for x in rows],
                "풍향deg": [float(x[ASOS_COL["wd"]]) * 10 for x in rows],  # WD x10 = deg
                "습도pct": [float(x[ASOS_COL["hm"]]) for x in rows],
                "전운량": [float(x[ASOS_COL["ca_tot"]]) for x in rows],
            }))
        if verbose and i % 12 == 0:
            print(f"    ASOS {stn}: {i}/{len(ranges)}개월", flush=True)
        time.sleep(0.05)  # 예의상 간격

    if not frames:
        raise RuntimeError(f"ASOS {stn} 자료를 한 건도 받지 못했다")
    w = pd.concat(frames, ignore_index=True)
    w["일시"] = pd.to_datetime(w["일시"], format="%Y%m%d%H%M")

    # 결측 -9 / -90(=풍향 -9 x10) 을 NaN 으로
    for c in ["기온C", "강수량mm", "풍속ms", "습도pct", "전운량"]:
        w.loc[w[c] <= -9.0, c] = np.nan
    w.loc[w["풍향deg"] <= -90.0, "풍향deg"] = np.nan
    # 강수량 결측은 무강수(기상청 표기 관행) — 기존 파이프라인과 동일 처리
    w["강수량mm"] = w["강수량mm"].fillna(0.0)

    return w.drop_duplicates("일시").sort_values("일시").reset_index(drop=True)


def fetch_aws(stn: int, start: str, end: str, verbose: bool = True) -> pd.DataFrame:
    """방재기상관측(AWS) 정시자료. 활용신청이 승인돼야 동작한다.

    미승인 상태면 403 이 오므로 호출부에서 기존 CSV 폴백을 쓴다.
    """
    frames = []
    for t1, t2 in _month_ranges(start, end):
        r = requests.get(AWS_URL, params={
            "tm1": t1, "tm2": t2, "stn": stn, "help": 0, "authKey": HUB_KEY},
            timeout=TIMEOUT)
        if "활용신청" in r.text[:300]:
            raise PermissionError("AWS awsh.php 활용신청 미승인 (403)")
        r.raise_for_status()
        rows = [l.split() for l in r.text.splitlines()
                if l and not l.startswith("#")]
        if rows:
            frames.append(pd.DataFrame(rows))
        time.sleep(0.05)
    if not frames:
        raise RuntimeError(f"AWS {stn} 자료 없음")
    return pd.concat(frames, ignore_index=True)


def build_weather_csv(out_path, stn: int = 146,
                      start: str = "201901", end: str | None = None) -> pd.DataFrame:
    """학습용 기상 CSV 를 API 로 새로 만든다. 기존 CSV 와 같은 스키마 + 전운량."""
    if end is None:
        end = datetime.now().strftime("%Y%m")
    print(f"  기상청 API허브 수집: 지점 {stn}, {start} ~ {end}")
    w = fetch_asos(stn, start, end)
    w.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"  저장 완료: {out_path}  ({len(w):,}행, "
          f"{w['일시'].min():%Y-%m-%d} ~ {w['일시'].max():%Y-%m-%d})")
    return w


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "legacy"))
    from console import use_utf8_stdout
    use_utf8_stdout()
    from config import DATA_ROOT
    build_weather_csv(DATA_ROOT / "02_기상데이터" / "asos_146_api_2019_2026.csv")
