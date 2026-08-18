"""기존 out/demo.db(SQLite) 내용을 PostgreSQL 로 1회 복사한다.

쓰는 상황
  run_serve.py 를 다시 돌리면 어차피 같은 값이 다시 적재되므로 필수는 아니다.
  다만 (1) 예보 API 를 다시 부르지 않고 지금 DB 상태를 그대로 옮기고 싶을 때,
  (2) archive 재현용 risk_calendar 처럼 현행 파이프라인이 더는 만들지 않는
  테이블을 살리고 싶을 때 쓴다.

실행
    cd demo
    python -m serving.migrate_sqlite_to_pg           # 미리보기(건수만)
    python -m serving.migrate_sqlite_to_pg --apply   # 실제 복사

동작
  · 대상은 db.TABLES 6개.
  · PK 가 있는 테이블은 ON CONFLICT DO NOTHING 이라 여러 번 돌려도 안전하다.
  · PK 가 없는 receptor 는 --apply 시 해당 farm_id 를 지우고 다시 넣는다
    (그러지 않으면 돌릴 때마다 중복 누적).
  · notification_log 의 id 는 PostgreSQL 이 새로 매긴다. 원본 id 는 버린다
    (참조하는 곳이 없다).
"""
from __future__ import annotations

import sqlite3
import sys

from config import DB_BACKEND, DB_PATH
from serving import db

# 원본 id 를 그대로 옮기지 않는 테이블 -> 복사할 컬럼을 명시한다.
COLUMNS = {
    "notification_log": ["farm_id", "work_type", "sent_at", "message", "approved_yn"],
}


def _sqlite_rows(table: str) -> tuple[list[str], list[tuple]]:
    con = sqlite3.connect(DB_PATH)
    try:
        names = [r[1] for r in con.execute(f"PRAGMA table_info({table})").fetchall()]
        if not names:
            return [], []
        cols = COLUMNS.get(table, names)
        rows = con.execute(f"SELECT {', '.join(cols)} FROM {table}").fetchall()
        return cols, rows
    except sqlite3.OperationalError:
        return [], []          # 그 테이블이 없는 구버전 demo.db
    finally:
        con.close()


def main(apply: bool = False) -> int:
    if DB_BACKEND != "postgres":
        print("DATABASE_URL 이 비어 있어 대상이 SQLite 자신입니다. .env 를 확인하세요.")
        return 1
    if not DB_PATH.exists():
        print(f"원본이 없습니다: {DB_PATH}")
        return 1

    print(f"원본 : SQLite {DB_PATH}")
    print(f"대상 : {db.describe()}")
    print(f"모드 : {'실제 복사' if apply else '미리보기 (--apply 를 붙이면 복사)'}\n")

    con = db.connect()
    total = 0
    for table in db.TABLES:
        cols, rows = _sqlite_rows(table)
        if not rows:
            print(f"  {table:<16} 0행 — 건너뜀")
            continue
        print(f"  {table:<16} {len(rows)}행", end="")
        if apply:
            ph = ",".join("?" * len(cols))
            if table == "receptor":
                # PK 가 없어 중복 판정이 불가능하다. farm_id 단위로 갈아끼운다.
                for fid in {r[cols.index("farm_id")] for r in rows}:
                    con.execute("DELETE FROM receptor WHERE farm_id=?", (fid,))
                sql = f"INSERT INTO receptor({', '.join(cols)}) VALUES({ph})"
            elif table == "notification_log":
                sql = f"INSERT INTO notification_log({', '.join(cols)}) VALUES({ph})"
            else:
                sql = (f"INSERT INTO {table}({', '.join(cols)}) VALUES({ph}) "
                       f"ON CONFLICT DO NOTHING")
            con.executemany(sql, rows)
            con.commit()
            print("  -> 복사 완료")
        else:
            print()
        total += len(rows)

    if apply:
        print("\n적재 결과 (대상 DB 기준)")
        for table in db.TABLES:
            n = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            print(f"  {table:<16} {n}행")
    con.close()
    print(f"\n원본 합계 {total}행")
    return 0


if __name__ == "__main__":
    sys.exit(main(apply="--apply" in sys.argv))
