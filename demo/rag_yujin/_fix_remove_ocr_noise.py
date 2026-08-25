"""EXCLUDE_FILES에 추가한 OCR 노이즈 파일의 기존 청크를 Supabase에서 제거한다.

`_4b_migrate_to_pg.py --drop`은 rag 스키마 전체를 DROP SCHEMA CASCADE로 지우는데,
데이터가 쌓인 상태에서는 Supabase pooler의 statement_timeout에 걸려 실패할 수 있다.
문제 파일 하나만 지우는 건 훨씬 가벼운 쿼리라 타임아웃 없이 끝난다.

실행:
    python rag_yujin/_fix_remove_ocr_noise.py
"""
from __future__ import annotations

import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

import pgvector_store as store  # noqa: E402

# _1_loader.py의 EXCLUDE_FILES와 반드시 같은 파일명이어야 한다.
TARGET_FILE = "축산농가를_위한_냄새_저감_공식_정제텍스트.txt"


def main() -> int:
    con = store.connect()
    try:
        with con.cursor() as cur:
            cur.execute(
                f"DELETE FROM {store.SCHEMA}.chunks WHERE source_file = %s",
                (TARGET_FILE,),
            )
            deleted_chunks = cur.rowcount
            cur.execute(
                f"DELETE FROM {store.SCHEMA}.documents WHERE source_file = %s",
                (TARGET_FILE,),
            )
            deleted_docs = cur.rowcount
        con.commit()
    finally:
        con.close()

    print(f"삭제 완료 — chunks {deleted_chunks}개 / documents {deleted_docs}개 ({TARGET_FILE})")
    if deleted_chunks == 0:
        print("⚠ 지워진 청크가 0개입니다 — 파일명이 실제 DB의 source_file 값과 일치하는지 확인하세요.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
