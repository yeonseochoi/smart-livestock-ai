"""지시 17 — 중기예보 클라이언트 (getMidLandFcst + getMidTa).

D+4~7 의 일 단위 {tmin, tmax, pop} 를 반환한다. 발표시각은 06시/18시 뿐.

구역코드 (기상청 중기예보 오픈API 활용가이드 코드표 기준):
  [A] 중기육상예보(강수확률) 전라북도 = 11F10000
  [A] 중기기온 대표지점 전주 = 11F10201
  [미확정] 익산 전용 중기기온 코드 — 가이드 코드표에서 익산 항목을 재확인해야
  한다. KMA_KEY 확보 후 `python -m serving.kma_midterm --probe` 를 실행하면 후보
  코드(11F10202~11F10211)를 실호출해 데이터가 오는 코드를 자동 확인한다.
  확정 전까지 기온 대표지점은 전주(11F10201)를 쓴다.
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta

import requests

MID_LAND_REG_ID = "11F10000"   # [A] 전라북도 육상 (강수확률 rnSt)
MID_TA_REG_ID = "11F10201"     # [A] 전주 (기온 taMin/taMax). 익산 코드 확정 시 교체
PROBE_CANDIDATES = [f"11F102{n:02d}" for n in range(2, 12)]  # 익산 후보 탐색용

BASE = "https://apis.data.go.kr/1360000/MidFcstInfoService"
TIMEOUT = 10


def latest_tmfc(now: datetime | None = None) -> str:
    """가장 최근 발표시각 (06시/18시, 발표 후 30분 여유)."""
    now = now or datetime.now()
    slots = [now.replace(hour=6, minute=0), now.replace(hour=18, minute=0)]
    avail = [s for s in slots if now >= s + timedelta(minutes=30)]
    if not avail:
        y = now - timedelta(days=1)
        return y.strftime("%Y%m%d") + "1800"
    return avail[-1].strftime("%Y%m%d") + ("0600" if avail[-1].hour == 6 else "1800")


def _call(endpoint: str, reg_id: str, tmfc: str, key: str) -> dict | None:
    try:
        r = requests.get(f"{BASE}/{endpoint}", params={
            "serviceKey": key, "dataType": "JSON", "numOfRows": 10, "pageNo": 1,
            "regId": reg_id, "tmFc": tmfc}, timeout=TIMEOUT)
        r.raise_for_status()
        items = r.json()["response"]["body"]["items"]["item"]
        return items[0] if items else None
    except Exception:
        return None


def _service_key() -> str | None:
    """환경변수 우선, 없으면 legacy/kma.py 의 폴백 키를 쓴다.

    ★ 수정 — 기존에는 KMA_KEY 미설정 시 무조건 None 을 돌려줘 중기예보가
      항상 mock 으로 빠졌다. legacy/kma.py 에 폴백 서비스키가 이미 있으므로
      그것을 재사용한다. 단, 그 키는 URL 인코딩된 문자열(%2F 등)이라
      requests params= 로 넘기면 이중 인코딩(%25)이 되므로 먼저 unquote 한다.
    """
    k = os.environ.get("KMA_KEY")
    if k:
        return k
    try:
        from urllib.parse import unquote
        import kma  # legacy (수정 금지, import 만)
        return unquote(kma.SERVICE_KEY)
    except Exception:
        return None


def fetch_mid(now: datetime | None = None) -> dict | None:
    """D+4~7 의 {날짜: {tmin, tmax, pop}}. 키 없거나 실패 시 None."""
    key = _service_key()
    if not key:
        return None
    now = now or datetime.now()
    tmfc = latest_tmfc(now)
    land = _call("getMidLandFcst", MID_LAND_REG_ID, tmfc, key)
    ta = _call("getMidTa", MID_TA_REG_ID, tmfc, key)
    if not land or not ta:
        return None

    base_day = datetime.strptime(tmfc[:8], "%Y%m%d")
    out = {}
    for d in range(4, 8):
        date = (base_day + timedelta(days=d)).strftime("%Y-%m-%d")
        # 강수확률: D+4~7 은 오전/오후 분리 제공 → 보수적으로 max
        pops = [land.get(f"rnSt{d}Am"), land.get(f"rnSt{d}Pm"), land.get(f"rnSt{d}")]
        pops = [float(p) for p in pops if p is not None]
        out[date] = {
            "tmin": float(ta.get(f"taMin{d}", "nan")),
            "tmax": float(ta.get(f"taMax{d}", "nan")),
            "pop": max(pops) if pops else None,
        }
    return out


def probe_reg_ids(now: datetime | None = None) -> dict:
    """익산 전용 중기기온 코드 확인 — 후보를 실호출해 응답 유무를 본다."""
    key = _service_key()
    if not key:
        return {"error": "서비스키 없음"}
    tmfc = latest_tmfc(now)
    result = {}
    for reg in PROBE_CANDIDATES:
        item = _call("getMidTa", reg, tmfc, key)
        result[reg] = "응답 있음" if item else "없음"
    return result


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "legacy"))
    from console import use_utf8_stdout
    use_utf8_stdout()
    if "--probe" in sys.argv:
        print(probe_reg_ids())
    else:
        data = fetch_mid()
        print(data if data is not None
              else "KMA_KEY 미설정 또는 호출 실패 — daily_scoring 는 mock 으로 폴백")
