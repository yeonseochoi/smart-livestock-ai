"""_4_database.py 의 Supabase(pgvector) 판 — 인덱스를 원격 DB에 적재한다.

_4_database.py 와의 차이는 저장 위치 하나뿐이다.
    _4_database.py     로컬 data/chroma_db/   (오프라인·개인 개발용)
    이 파일            Supabase rag 스키마     (배포·GitHub Actions용)

로더(_1) · 청커(_2) · 임베딩(_3) 은 **그대로 재사용**한다. 같은 청킹 규칙과
같은 임베딩 모델을 써야 두 백엔드의 검색 결과가 일치하기 때문이다.

실행:
    python rag_yujin/_4b_migrate_to_pg.py            # 적재
    python rag_yujin/_4b_migrate_to_pg.py --verify   # 적재 없이 현황만
    python rag_yujin/_4b_migrate_to_pg.py --drop     # rag 스키마 통째로 삭제 후 재적재
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

import pgvector_store as store      # noqa: E402


def _print_stats(prefix: str = "") -> None:
    info = store.stats()
    print(f"{prefix}청크 {info['chunks']:,}개 "
          f"(문서 {info['documents']}개 · {info['by_doc_type']}) "
          f"· 질의캐시 {info.get('cached_queries', 0)}개 "
          f"· 용량 {info['size']} · 갱신 {info['indexed_at']}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true", help="적재하지 않고 현황만 출력")
    ap.add_argument("--drop", action="store_true", help="rag 스키마를 지우고 새로 만든다")
    args = ap.parse_args()

    if args.verify:
        _print_stats("현황 — ")
        return 0

    con = store.connect()
    try:
        if args.drop:
            print(f"  rag 스키마를 삭제합니다 (DROP SCHEMA {store.SCHEMA} CASCADE)")
            with con.cursor() as cur:
                cur.execute(f"DROP SCHEMA IF EXISTS {store.SCHEMA} CASCADE")
            con.commit()

        print("[1/5] 스키마 준비")
        store.ensure_schema(con)

        print("[2/5] 문서 로드 + 청킹  (_1_loader → _2_Chunking)")
        from _1_loader import DATA_DIR, _resolve_data_dir, load_law_manual_data
        from _2_Chunking import chunk_documents

        docs = load_law_manual_data(_resolve_data_dir(DATA_DIR))
        chunks = chunk_documents(docs)
        if not chunks:
            print("  ✗ 청크가 0개입니다. _1_loader.py / _2_Chunking.py 를 확인하세요.")
            return 1

        print(f"[3/5] 임베딩  ({len(chunks):,}개 → {store.EMBED_DIM}차원)")
        from _3_embedding import EMBEDDING_MODEL_NAME, get_embedding_function

        print(f"  모델 {EMBEDDING_MODEL_NAME}")
        embed_fn = get_embedding_function()
        started = time.time()
        vectors = embed_fn.embed_documents([c.page_content for c in chunks])
        print(f"  완료 — {len(vectors):,}개 벡터, {time.time() - started:.1f}초")

        if len(vectors[0]) != store.EMBED_DIM:
            print(f"  ✗ 차원 불일치: 모델 {len(vectors[0])} != 테이블 {store.EMBED_DIM}")
            return 1

        print("[4/5] Supabase 적재  (기존 청크 전체 교체)")
        started = time.time()
        n = store.replace_all(con, chunks, vectors)
        print(f"  완료 — {n:,}행, {time.time() - started:.1f}초")

        # 배포 서버에서 torch 를 걷어내기 위해, 대시보드가 실제로 던지는 질의의
        # 벡터를 미리 넣어 둔다. work_guide.py:90 의 문장 템플릿과 정확히 같아야
        # 캐시가 적중한다 — 한 글자라도 다르면 미스가 나고 모델을 로드하게 된다.
        print("[5/5] 질의 벡터 캐시  (배포 서버 torch 제거용)")
        work_types = ("분뇨제거", "청소", "환기점검", "저감시설점검", "액비살포")
        queries = [f"{w} 작업 전후 관리 기준" for w in work_types]
        q_vectors = embed_fn.embed_documents(queries)
        store.cache_queries(con, queries, q_vectors)
        for q in queries:
            print(f"  캐시: {q}")
    finally:
        con.close()

    _print_stats("\n결과 — ")

    print("\n[검색 테스트] '환기팬 점검 방법'")
    vs = store.load_vector_store()
    try:
        manual_r, law_r = store.build_retrievers(vs)
        context = store.get_context_for_query(manual_r, law_r, "환기팬 점검 방법")
        for kind in ("manual", "law"):
            for d in context[kind]:
                m = d.metadata
                preview = d.page_content[:60].replace("\n", " ")
                print(f"  [{m['doc_type']}] {m['source_file'][:38]} / {m['unit']} "
                      f"/ p.{m['page']} · 거리 {m['cosine_distance']:.4f}")
                print(f"      {preview}...")
    finally:
        vs.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
