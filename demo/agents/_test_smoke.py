"""agents/ 스모크 테스트 — 수동 실행용.

순서:
  1) py run_serve.py   ← risk_hourly / forecast_hourly / farm_config 채움
  2) py agents/_test_smoke.py   ← 이 파일. work_guide/notify_draft 통합 확인

S7 테스트 케이스 ①(경과일 12일 + 후반부 위험 예보 → 더 이른 창으로 앞당김)을
재현하기 위해 DEMO_FARM 의 저장 경과일을 12일로 강제 세팅한다.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datetime import datetime, timedelta

from console import use_utf8_stdout
from config import DEMO_FARM, section
from serving import db
from agents import work_guide, notify_draft

use_utf8_stdout()

# ── 저장 경과일 12일로 세팅 (S7 테스트 케이스 ①) ──────────────────────
DEMO_FARM["last_manure_removal_date"] = (
    datetime.now() - timedelta(days=12)
).strftime("%Y-%m-%d")

con = db.connect()
db.upsert_farm(con, DEMO_FARM)
con.close()
print(f"farm_config 갱신: last_manure_removal_date={DEMO_FARM['last_manure_removal_date']}")

# ── work_guide 4개 작업유형 전부 확인 ──────────────────────────────
for wt in ("분뇨제거", "청소", "환기점검", "저감시설점검"):
    section(f"[work_guide] {wt}")
    try:
        print(work_guide.run(DEMO_FARM["farm_id"], wt))
    except Exception as exc:
        print(f"[에러] {wt}: {exc!r}")

# ── notify_draft 확인 ───────────────────────────────────────────
section("[notify_draft]")
try:
    result = notify_draft.run("청소", datetime.now() + timedelta(hours=6))
    print(result)
except Exception as exc:
    print(f"[에러] notify_draft: {exc!r}")
