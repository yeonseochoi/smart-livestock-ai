"""기상청 단기예보(getVilageFcst) 조회.

⚠️ 인증키는 Encoding 버전을 URL 문자열에 직접 삽입한다.
   requests.get(url, params={...}) 에 Encoding 키를 넣으면
   %2F 가 %252F 로 다시 인코딩되어 실패하므로 params= 를 쓰지 않는다.
"""

import os
from datetime import datetime, timedelta

import requests

from constants import KMA_BASE_HOURS, KMA_BASE_URL

# 운영 시 환경변수 KMA_KEY 로 분리. 미설정 시에만 아래 Encoding 키로 폴백.
_FALLBACK_SERVICE_KEY = (
    "xnfpZzb%2F76Vll1%2BcnFLcUZePxXdidf2sR4oMm4ZYL3S5c%2Ff3wInDsDQRG2owUlzUfruF8"
    "%2FAn9wLxLA3LImaQYQ%3D%3D"
)
SERVICE_KEY = os.environ.get("KMA_KEY", _FALLBACK_SERVICE_KEY)


def base_datetime(now: datetime | None = None) -> tuple[str, str]:
    """현재 시각 기준으로 조회 가능한 최신 발표시각을 반환."""
    now = now or datetime.now()
    t = now - timedelta(minutes=20)          # 발표 직후 여유
    cand = [h for h in KMA_BASE_HOURS if h <= t.hour]
    if cand:
        return t.strftime("%Y%m%d"), f"{max(cand):02d}00"
    y = t - timedelta(days=1)
    return y.strftime("%Y%m%d"), "2300"


def fetch_forecast(nx: int, ny: int,
                    base_date: str | None = None,
                    base_time: str | None = None) -> dict:
    """
    단기예보 조회 → {'YYYYMMDD HHMM': {'VEC':..,'WSD':..,'TMP':..,'SKY':..}, ...}
    시각 오름차순 정렬됨.
    """
    if base_date is None:
        base_date, base_time = base_datetime()

    url = (f"{KMA_BASE_URL}?serviceKey={SERVICE_KEY}"
           f"&pageNo=1&numOfRows=1000&dataType=JSON"          # 1000: 3일치 확보
           f"&base_date={base_date}&base_time={base_time}"
           f"&nx={nx}&ny={ny}")

    r = requests.get(url, timeout=20)
    r.raise_for_status()

    # 인증 오류는 XML 로 돌아옴 → json() 이 터지기 전에 잡아냄
    if "<OpenAPI_ServiceResponse>" in r.text or "errMsg" in r.text[:200]:
        raise RuntimeError(f"인증/파라미터 오류\n{r.text[:400]}")

    body = r.json()["response"]
    if body["header"]["resultCode"] != "00":
        raise RuntimeError(f"API 오류 {body['header']['resultCode']}: "
                            f"{body['header']['resultMsg']}")

    out: dict[str, dict] = {}
    for it in body["body"]["items"]["item"]:
        key = f"{it['fcstDate']} {it['fcstTime']}"
        out.setdefault(key, {})[it["category"]] = it["fcstValue"]
    return dict(sorted(out.items()))


def fetch_with_fallback(nx: int, ny: int, max_back: int = 3) -> dict:
    """발표시각을 하나씩 되돌리며 재시도 (§7)."""
    if max_back < 1:
        raise ValueError("max_back 은 1 이상이어야 합니다")

    now = datetime.now()
    last_err: Exception | None = None
    for i in range(max_back):
        bd, bt = base_datetime(now - timedelta(hours=3 * i))
        try:
            return fetch_forecast(nx, ny, bd, bt)
        except RuntimeError as e:
            last_err = e
            if i == max_back - 1:
                raise
            continue
    # 이론상 도달하지 않음 (max_back>=1 이면 루프 안에서 return/raise 됨)
    raise last_err  # type: ignore[misc]


if __name__ == "__main__":
    data = fetch_forecast(63, 89)
    print(f"시점 {len(data)}개 수신")
    for k in list(data)[:3]:
        print(" ", k, data[k])
