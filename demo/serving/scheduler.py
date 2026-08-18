"""지시 17 — 상시 구동 스케줄러.

매일 06:00 (단기예보 05시 발표 + 여유) 에 daily_scoring(실모델) 실행.
실패해도 스케줄러는 죽지 않고 이전 캘린더를 유지한다 (계획서 S4 요구).

    python -m serving.scheduler          # 상시 구동 (Ctrl+C 종료)
    python -m serving.scheduler --once   # 즉시 1회 실행 후 종료 (데모·검증용)
"""
from __future__ import annotations

import sys
import traceback
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "legacy"))

# [2026-08-18 수정] 6 -> 7. 실측 근거로 바꾼다.
#
#   중기예보는 06시/18시 발표인데 kma_midterm.latest_tmfc() 가 "발표 후 30분" 을
#   기다린다. 그래서 06:00 실행 시점에는 당일 06시 발표를 아직 인정하지 않고
#   **전날 18시 발표본**을 쓴다. 실측 확인:
#       06:00 실행 -> tmFc=전날1800 / 06:29 -> 전날1800 / 06:30 -> 당일0600
#
#   그런데 18시 발표본에는 taMin4 · taMax4 · rnSt4 필드가 **아예 없다**(키 부재).
#   중기예보는 발표시각에 따라 제공 시작일이 달라서 18시본은 D+5 부터만 준다.
#   그 결과 fetch_mid() 가 D+4 하루를 tmin=NaN, tmax=NaN, pop=None 으로 돌려주고,
#   _append_midterm() 이 그 날 24시간을 temp=NaN 으로 채점한다.
#   XGBoost 가 NaN 을 결측으로 처리해 예외는 안 나지만, **하루치가 조용히 열화된다.**
#       실측: fetch_mid(19:00) -> {"2026-08-22": {"tmin": NaN, "tmax": NaN, ...}, ...}
#            fetch_mid(12:00) -> {"2026-08-22": {"tmin": 25.0, "tmax": 33.0, ...}, ...}
#
#   07:00 으로 옮기면 당일 06시 발표본(D+4 포함)을 쓴다.
#   단기예보 발표는 02·05·08·11·14·17·20·23시라 07:00 에도 여전히 05시 발표본이므로
#   **단기 쪽 손해는 없다.**
#
#   참고 [A] — latest_tmfc() 의 "30분" 은 근거 없는 상수였고 실측상 과하다.
#   2026-08-18 18시 발표를 17:55:09 부터 폴링했더니 **이미 제공되고 있었다**
#   (resultCode=00, 발표 5분 전). 다만 폴링을 17:55 에 시작해 진짜 하한은 미측정.
#   RUN_HOUR=7 조합에서는 30분이 무해하므로 상수는 건드리지 않는다.
RUN_HOUR = 7


def job() -> None:
    from serving import daily_scoring
    try:
        res = daily_scoring.run()
        print(f"[{datetime.now():%Y-%m-%d %H:%M}] daily_scoring 성공: "
              f"{res['n_upsert']}행 · 실API={res['real_api']}")
    except Exception:
        # 실패 시 risk_calendar 는 갱신되지 않으므로 이전 캘린더가 자동 유지된다
        print(f"[{datetime.now():%Y-%m-%d %H:%M}] daily_scoring 실패 — 이전 캘린더 유지")
        traceback.print_exc()


def main() -> None:
    from console import use_utf8_stdout  # legacy import (수정 금지)
    use_utf8_stdout()
    if "--once" in sys.argv:
        job()
        return
    from apscheduler.schedulers.blocking import BlockingScheduler
    sched = BlockingScheduler()
    sched.add_job(job, "cron", hour=RUN_HOUR, minute=0, id="daily_scoring",
                  misfire_grace_time=3600)
    print(f"스케줄러 시작 — 매일 {RUN_HOUR:02d}:00 daily_scoring 실행 (Ctrl+C 종료)")
    try:
        sched.start()
    except (KeyboardInterrupt, SystemExit):
        print("스케줄러 종료")


if __name__ == "__main__":
    main()
