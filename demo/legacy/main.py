"""실행 진입점.

기본은 실제 API(기상청 + VWorld)를 쓴다. 오프라인으로 붙는지만 보려면:

    python main.py --mock          기상·주거 둘 다 픽스처
    python main.py --mock-weather  주거만 실제 VWorld
    python main.py --mock-home     기상만 실제 기상청

필요한 환경변수
    KMA_KEY      기상청 단기예보 (공공데이터포털 일반 인증키 Encoding)
    VWORLD_KEY   VWorld 개발키
"""

from __future__ import annotations

import argparse

from console import use_utf8_stdout
from recommend import evaluate, format_report
from residence import find_receptors

# ── 농장주가 입력하는 값 ──────────────────────────────────────────
FARM_LAT, FARM_LON = 36.5666, 126.5500     # 지도에서 핀 1개
TONS = 20.0                                # 살포 예정량 (매번 다름)
METHOD = "표면살포"                          # 보유 장비 — 프로필로 1회 설정
TILLAGE = False                            # 살포 후 즉시 경운 여부
# ──────────────────────────────────────────────────────────────


def run(mock_weather: bool = False, mock_home: bool = False,
        tons: float = TONS, method: str = METHOD,
        tillage: bool = TILLAGE) -> dict:
    # ① 주거 수용점
    if mock_home:
        from mock_residence import mock_buildings
        receptors, hist = find_receptors(
            FARM_LAT, FARM_LON,
            buildings=mock_buildings(FARM_LAT, FARM_LON))
    else:
        receptors, hist = find_receptors(FARM_LAT, FARM_LON)

    print(f"주거 수용점 {len(receptors)}동 "
          f"(전체 건물 {sum(hist.values())}동 중)")
    if not receptors:
        print("[!] 주거 건물이 탐지되지 않았습니다. "
              "include_null=True 로 재시도하거나 좌표를 확인하세요.")

    # ② 기상 예보
    if mock_weather:
        from mock_forecast import fetch_forecast as fetch_fn
    else:
        from kma import fetch_forecast as fetch_fn

    # ③ 평가
    res = evaluate(FARM_LAT, FARM_LON, receptors,
                   tons=tons, method=method, tillage=tillage,
                   fetch_forecast_fn=fetch_fn)

    print(format_report(res))
    n_bad = sum(1 for r in res["timeline"] if r["insufficient_forecast"])
    print(f"\ntimeline {len(res['timeline'])}개 "
          f"(6시간 창 불완전 {n_bad}개는 후보에서 제외)")
    return res


def main() -> None:
    use_utf8_stdout()
    p = argparse.ArgumentParser(description="액비 살포 시각 추천")
    p.add_argument("--mock", action="store_true", help="기상·주거 둘 다 픽스처")
    p.add_argument("--mock-weather", action="store_true", help="기상만 픽스처")
    p.add_argument("--mock-home", action="store_true", help="주거만 픽스처")
    p.add_argument("--tons", type=float, default=TONS, help="살포 예정량(톤)")
    p.add_argument("--method", default=METHOD,
                   choices=["표면살포", "밴드살포", "트레일링슈", "주입식"])
    p.add_argument("--tillage", action="store_true", help="살포 후 즉시 경운")
    a = p.parse_args()

    run(mock_weather=a.mock or a.mock_weather,
        mock_home=a.mock or a.mock_home,
        tons=a.tons, method=a.method, tillage=a.tillage)


if __name__ == "__main__":
    main()
