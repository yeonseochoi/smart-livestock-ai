"""지시 17 — 상시 구동 스케줄러.

매일 06:00 (단기예보 05시 발표 + 여유) 에 run_daily(실모델) 실행.
실패해도 스케줄러는 죽지 않고 이전 캘린더를 유지한다 (계획서 S4 요구).

    python -m ops.scheduler          # 상시 구동 (Ctrl+C 종료)
    python -m ops.scheduler --once   # 즉시 1회 실행 후 종료 (데모·검증용)
"""
from __future__ import annotations

import sys
import traceback
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "legacy"))

RUN_HOUR = 6


def job() -> None:
    from ops import run_daily
    try:
        res = run_daily.run(dummy=False)
        print(f"[{datetime.now():%Y-%m-%d %H:%M}] run_daily 성공: {res['model']}")
    except Exception:
        # 실패 시 risk_calendar 는 갱신되지 않으므로 이전 캘린더가 자동 유지된다
        print(f"[{datetime.now():%Y-%m-%d %H:%M}] run_daily 실패 — 이전 캘린더 유지")
        traceback.print_exc()


def main() -> None:
    from console import use_utf8_stdout  # legacy import (수정 금지)
    use_utf8_stdout()
    if "--once" in sys.argv:
        job()
        return
    from apscheduler.schedulers.blocking import BlockingScheduler
    sched = BlockingScheduler()
    sched.add_job(job, "cron", hour=RUN_HOUR, minute=0, id="run_daily",
                  misfire_grace_time=3600)
    print(f"스케줄러 시작 — 매일 {RUN_HOUR:02d}:00 run_daily 실행 (Ctrl+C 종료)")
    try:
        sched.start()
    except (KeyboardInterrupt, SystemExit):
        print("스케줄러 종료")


if __name__ == "__main__":
    main()
