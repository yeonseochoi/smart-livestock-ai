"""기상청 API 없이 계산 축을 개발하기 위한 가짜 예보.

실제 ``kma.fetch_forecast()``와 같은 시그니처와 반환 계약을 사용하며,
모든 예보값은 실제 API와 동일하게 문자열로 반환한다.
"""

from datetime import datetime, timedelta


_SCENARIO = [
    # (시간 offset, VEC, WSD, TMP, SKY, POP) — VEC=바람이 불어오는 방향
    (0, "270", "3.2", "24", "1", "0"),
    (1, "280", "3.5", "25", "1", "0"),
    (2, "290", "4.0", "26", "1", "0"),
    (3, "300", "3.8", "27", "1", "0"),
    (4, "310", "3.1", "28", "3", "10"),
    (5, "320", "2.5", "28", "3", "10"),
    (6, "200", "1.4", "28", "1", "0"),
    (7, "210", "1.2", "27", "1", "0"),
    (8, "220", "1.1", "26", "1", "0"),
    (9, "230", "1.5", "25", "3", "20"),
    (10, "240", "2.2", "24", "3", "20"),
    (11, "250", "2.8", "23", "4", "30"),
    (12, "260", "3.0", "22", "4", "30"),
    (13, "180", "1.8", "21", "1", "0"),
    (14, "185", "1.5", "20", "1", "0"),
    (15, "190", "1.2", "19", "1", "0"),
    (16, "200", "0.9", "18", "1", "0"),
    (17, "210", "0.8", "18", "1", "0"),
    (18, "215", "0.7", "17", "1", "0"),
    (19, "220", "0.9", "17", "1", "0"),
    (20, "225", "1.1", "18", "1", "0"),
    (21, "230", "1.6", "20", "1", "0"),
    (22, "240", "2.4", "22", "1", "0"),
    (23, "250", "3.0", "24", "1", "0"),
]


def fetch_forecast(
    nx: int,
    ny: int,
    base_date: str | None = None,
    base_time: str | None = None,
) -> dict:
    """3일치 1시간 예보 72개를 시각 오름차순으로 반환한다."""
    start = datetime.now().replace(minute=0, second=0, microsecond=0) + timedelta(
        hours=4
    )
    out: dict[str, dict] = {}

    for day in range(3):
        for offset, vec, wsd, tmp, sky, pop in _SCENARIO:
            forecast_at = start + timedelta(days=day, hours=offset)
            key = forecast_at.strftime("%Y%m%d %H00")
            out[key] = {
                "VEC": vec,
                "WSD": wsd,
                "TMP": tmp,
                "SKY": sky,
                "POP": pop,
            }
    return dict(sorted(out.items()))


def fetch_with_fallback(nx: int, ny: int, max_back: int = 3) -> dict:
    """실패하지 않는 mock 폴백 함수."""
    return fetch_forecast(nx, ny)


if __name__ == "__main__":
    data = fetch_forecast(63, 89)
    print(f"시점 {len(data)}개 생성")
    for timestamp in list(data)[:5]:
        print(" ", timestamp, data[timestamp])
