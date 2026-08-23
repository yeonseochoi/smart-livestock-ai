"""Supabase PostgreSQL(pgvector) 벡터 저장소 — Chroma 대체 백엔드.

왜 만들었나
    로컬 ``data/chroma_db`` 는 .gitignore 대상이라 GitHub Actions 러너나
    Streamlit Cloud 에서는 존재하지 않는다. 잡이 끝나면 컨테이너와 함께
    사라지므로 배포·자동화에서 RAG 를 쓸 수 없었다. A/B 가 SQLite -> Supabase
    로 옮긴 이유와 정확히 같은 문제다.

설계 원칙
    1. ``rag`` 스키마에 격리한다. A/B 의 ``public`` 테이블과 섞지 않는다.
       재구축은 ``DROP SCHEMA rag CASCADE`` 한 방이면 되고, 이는 기존
       ``_4_database.py`` 가 chroma 폴더를 지우고 새로 만드는 패턴과 같다.
    2. LangChain ``Document`` / Retriever 인터페이스를 그대로 흉내낸다.
       그래서 ``agents/rag_adapter.py`` · ``_5_search.format_answer`` ·
       ``build_chain`` 을 한 줄도 고치지 않고 백엔드만 바꿔 끼울 수 있다.
    3. 추가 pip 패키지를 만들지 않는다. 벡터는 '[0.1,0.2,...]' 문자열로
       주고받으므로 ``pgvector`` 파이썬 패키지가 필요 없다. psycopg 하나면 된다.
       (Streamlit Cloud 1GB 제한 때문에 의존성 하나가 아쉽다)

주의 — prepare_threshold=None
    Supabase 트랜잭션 풀러(6543)는 서버측 prepared statement 를 지원하지 않는다.
    psycopg3 는 같은 쿼리를 5회 실행하면 자동으로 prepare 로 바꾸므로, 끄지 않으면
    잘 돌던 잡이 어느 순간 DuplicatePreparedStatement 로 죽는다.
    ``serving/db.py:216`` 이 같은 이유로 같은 설정을 쓴다.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

SCHEMA = "rag"
EMBED_DIM = 768          # jhgan/ko-sroberta-multitask
DEFAULT_K = 3

DDL = f"""
CREATE EXTENSION IF NOT EXISTS vector;
CREATE SCHEMA IF NOT EXISTS {SCHEMA};

CREATE TABLE IF NOT EXISTS {SCHEMA}.documents(
    source_file TEXT PRIMARY KEY,
    doc_type    TEXT NOT NULL,
    n_pages     INTEGER,
    n_chunks    INTEGER,
    indexed_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS {SCHEMA}.chunks(
    chunk_id    TEXT PRIMARY KEY,
    source_file TEXT NOT NULL,
    doc_type    TEXT NOT NULL,
    unit        TEXT,
    page        INTEGER,
    is_annex    BOOLEAN NOT NULL DEFAULT FALSE,
    content     TEXT NOT NULL,
    embedding   vector({EMBED_DIM}) NOT NULL,
    indexed_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS chunks_doc_type_idx ON {SCHEMA}.chunks (doc_type);
CREATE INDEX IF NOT EXISTS chunks_source_idx   ON {SCHEMA}.chunks (source_file);

-- 질의 벡터 캐시 — 배포 서버에서 torch 를 걷어내기 위한 장치.
-- 현재 대시보드의 RAG 질의는 작업유형 5종에서 자동 생성되므로 종류가 유한하다
-- (work_guide.py 의 f"{{work_type}} 작업 전후 관리 기준"). 그 벡터를 미리 넣어 두면
-- 배포 서버는 임베딩 모델(약 450MB + torch 437MB) 없이 검색만 하면 된다.
-- 캐시에 없는 질의가 오면 그때만 모델을 로드한다(자유 질문 기능 대비).
CREATE TABLE IF NOT EXISTS {SCHEMA}.query_cache(
    query      TEXT PRIMARY KEY,
    embedding  vector({EMBED_DIM}) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""

# HNSW 인덱스는 별도로 만든다 — 실패해도 검색은 순차 스캔으로 동작해야 하므로
# DDL 본문과 분리했다. 567개 규모에서는 순차 스캔이 오히려 빠르지만,
# 문서가 늘었을 때를 대비해 걸어 둔다.
DDL_INDEX = (
    f"CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw "
    f"ON {SCHEMA}.chunks USING hnsw (embedding vector_cosine_ops)"
)


def _load_env() -> None:
    """rag_yujin/.env 와 ../.env 에서 DATABASE_URL 등을 읽는다.

    이미 있는 환경변수는 덮어쓰지 않는다 (CI 의 Secrets 가 우선).
    python-dotenv 가 없어도 동작하도록 최소 파서를 직접 둔다.
    """
    for path in (CURRENT_DIR / ".env", CURRENT_DIR.parent / ".env"):
        try:
            if not path.exists():
                continue
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key, value = key.strip(), value.strip().strip("'").strip('"')
                if key and key not in os.environ:
                    os.environ[key] = value
        except Exception:
            continue


def database_url() -> str:
    _load_env()
    url = os.environ.get("DATABASE_URL", "").strip()
    if not url:
        raise RuntimeError(
            "DATABASE_URL 이 없습니다. demo/.env 또는 rag_yujin/.env 에 "
            "Supabase 접속 문자열을 넣거나 환경변수로 설정하세요."
        )
    return url


def connect():
    """psycopg 커넥션. 풀러 안전 설정을 반드시 함께 건다."""
    import psycopg      # requirements: psycopg[binary]

    return psycopg.connect(
        database_url(),
        prepare_threshold=None,      # 풀러 대응 — 모듈 docstring 참조
        connect_timeout=20,
        application_name="slai-rag",
    )


def ensure_schema(con) -> None:
    with con.cursor() as cur:
        cur.execute(DDL)
        try:
            cur.execute(DDL_INDEX)
        except Exception as exc:      # 인덱스 실패는 치명적이지 않다
            print(f"  ⚠ HNSW 인덱스 생성 건너뜀: {type(exc).__name__}: {exc}")
    con.commit()


def to_vector_literal(values: Sequence[float]) -> str:
    """[0.1, 0.2, ...] -> '[0.1,0.2,...]'. pgvector 가 그대로 캐스팅한다."""
    return "[" + ",".join(f"{float(v):.7g}" for v in values) + "]"


# ── 적재 ──────────────────────────────────────────────────────────
def replace_all(con, chunks: Iterable, vectors: Sequence[Sequence[float]]) -> int:
    """청크 전체를 통째로 교체한다.

    부분 갱신이 아니라 전체 교체인 이유는 ``_4_database.py`` 와 같다 —
    청킹 규칙이 바뀌면 chunk_id 가 전부 달라지므로, 남은 옛 청크가
    검색 결과를 오염시킨다. 데이터가 수백 개 규모라 통째로 다시 넣는 게 안전하다.
    """
    chunks = list(chunks)
    if len(chunks) != len(vectors):
        raise ValueError(f"청크 {len(chunks)}개 != 벡터 {len(vectors)}개")

    rows = []
    per_doc: dict[str, dict[str, Any]] = {}
    for chunk, vector in zip(chunks, vectors):
        meta = chunk.metadata
        source = meta["source_file"]
        rows.append((
            meta["chunk_id"], source, meta["doc_type"], meta.get("unit"),
            meta.get("page"), bool(meta.get("is_annex", False)),
            chunk.page_content, to_vector_literal(vector),
        ))
        entry = per_doc.setdefault(
            source, {"doc_type": meta["doc_type"], "n_chunks": 0, "max_page": 0}
        )
        entry["n_chunks"] += 1
        entry["max_page"] = max(entry["max_page"], int(meta.get("page") or 0))

    with con.cursor() as cur:
        cur.execute(f"TRUNCATE {SCHEMA}.chunks")
        cur.execute(f"TRUNCATE {SCHEMA}.documents")
        cur.executemany(
            f"INSERT INTO {SCHEMA}.chunks"
            " (chunk_id, source_file, doc_type, unit, page, is_annex, content, embedding)"
            " VALUES (%s,%s,%s,%s,%s,%s,%s,%s)"
            " ON CONFLICT (chunk_id) DO UPDATE SET"
            "   content=EXCLUDED.content, embedding=EXCLUDED.embedding,"
            "   indexed_at=now()",
            rows,
        )
        cur.executemany(
            f"INSERT INTO {SCHEMA}.documents (source_file, doc_type, n_pages, n_chunks)"
            " VALUES (%s,%s,%s,%s)"
            " ON CONFLICT (source_file) DO UPDATE SET"
            "   n_chunks=EXCLUDED.n_chunks, n_pages=EXCLUDED.n_pages, indexed_at=now()",
            [(s, e["doc_type"], e["max_page"], e["n_chunks"]) for s, e in per_doc.items()],
        )
    con.commit()
    return len(rows)


def stats(con=None) -> dict[str, Any]:
    own = con is None
    con = con or connect()
    try:
        with con.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {SCHEMA}.chunks")
            total = cur.fetchone()[0]
            cur.execute(
                f"SELECT doc_type, COUNT(*) FROM {SCHEMA}.chunks GROUP BY 1 ORDER BY 1"
            )
            by_type = dict(cur.fetchall())
            cur.execute(f"SELECT COUNT(*), MAX(indexed_at) FROM {SCHEMA}.documents")
            n_docs, indexed_at = cur.fetchone()
            try:
                cur.execute(f"SELECT COUNT(*) FROM {SCHEMA}.query_cache")
                n_cached = cur.fetchone()[0]
            except Exception:
                n_cached = 0
            # relkind='r' 로 좁히지 않으면 인덱스가 두 번 계산된다 —
            # pg_total_relation_size(테이블) 이 이미 그 테이블의 인덱스를 포함하는데,
            # 인덱스 자체도 pg_class 행이라 SUM 에 또 들어간다.
            cur.execute(
                f"SELECT pg_size_pretty(SUM(pg_total_relation_size(c.oid))) "
                f"FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace "
                f"WHERE n.nspname=%s AND c.relkind='r'", (SCHEMA,)
            )
            size = cur.fetchone()[0]
        return {"chunks": total, "by_doc_type": by_type, "documents": n_docs,
                "cached_queries": n_cached,
                "indexed_at": str(indexed_at) if indexed_at else None, "size": size}
    finally:
        if own:
            con.close()


# ── 검색 — LangChain 인터페이스 흉내 ───────────────────────────────
def _as_document(row) -> Any:
    """langchain_core.Document 로 감싼다. 없으면 동일 속성의 경량 객체로 대체."""
    chunk_id, source_file, doc_type, unit, page, is_annex, content, distance = row
    metadata = {
        "chunk_id": chunk_id, "source_file": source_file, "doc_type": doc_type,
        "unit": unit, "page": page, "is_annex": is_annex,
        "cosine_distance": float(distance),
    }
    try:
        from langchain_core.documents import Document
        return Document(page_content=content, metadata=metadata)
    except ImportError:
        class _Doc:                       # langchain 없이도 동작해야 한다
            def __init__(self, text, meta):
                self.page_content, self.metadata = text, meta
        return _Doc(content, metadata)


# ── 질의 벡터 캐시 ─────────────────────────────────────────────────
def cache_queries(con, queries: Sequence[str], vectors: Sequence[Sequence[float]]) -> int:
    """자주 쓰는 질의의 벡터를 미리 저장한다. 배포 서버의 torch 를 없애는 장치."""
    rows = [(q, to_vector_literal(v)) for q, v in zip(queries, vectors)]
    with con.cursor() as cur:
        cur.executemany(
            f"INSERT INTO {SCHEMA}.query_cache (query, embedding) VALUES (%s,%s)"
            f" ON CONFLICT (query) DO UPDATE SET"
            f"   embedding=EXCLUDED.embedding, created_at=now()",
            rows,
        )
    con.commit()
    return len(rows)


def lookup_query_vector(con, query: str) -> str | None:
    """캐시된 질의 벡터를 pgvector 리터럴 문자열 그대로 돌려준다(파싱 불필요)."""
    try:
        with con.cursor() as cur:
            cur.execute(
                f"SELECT embedding::text FROM {SCHEMA}.query_cache WHERE query = %s",
                (query,),
            )
            row = cur.fetchone()
            return row[0] if row else None
    except Exception:
        # 캐시 테이블이 없어도 검색 자체는 동작해야 한다. 다만 실패한 문장은
        # 트랜잭션을 abort 상태로 만들어 이후 모든 질의가 죽으므로 반드시 되돌린다.
        try:
            con.rollback()
        except Exception:
            pass
        return None


class PgRetriever:
    """``.invoke(query)`` 로 Document 리스트를 돌려주는 최소 Retriever.

    LangChain BaseRetriever 를 상속하지 않는다 — 상속하면 pydantic 검증과
    langchain-core 버전에 묶이는데, 우리가 쓰는 건 ``.invoke`` 하나뿐이다.

    질의 벡터는 ``rag.query_cache`` 를 먼저 본다. 적중하면 임베딩 모델을
    아예 import 하지 않으므로, 배포 서버에 torch 를 설치하지 않아도 된다.
    """

    def __init__(self, doc_type: str, k: int = DEFAULT_K, embed_fn=None, con=None):
        self.doc_type, self.k, self._embed_fn, self._con = doc_type, k, embed_fn, con

    def _embed(self, query: str) -> Sequence[float]:
        if self._embed_fn is None:
            from _3_embedding import get_embedding_function
            self._embed_fn = get_embedding_function()
        return self._embed_fn.embed_query(query)

    def _query_vector(self, query: str, con) -> str:
        cached = lookup_query_vector(con, query)
        if cached is not None:
            return cached
        # 캐시 미스 — 이때만 임베딩 모델을 로드한다(자유 질문 등).
        return to_vector_literal(self._embed(query))

    def invoke(self, query: str) -> list:
        own = self._con is None
        con = self._con or connect()
        try:
            vector = self._query_vector(query, con)   # 캐시 미스 시 여기서 예외 가능
            with con.cursor() as cur:  # noqa: SIM117
                cur.execute(
                    f"SELECT chunk_id, source_file, doc_type, unit, page, is_annex,"
                    f"       content, embedding <=> %s AS distance"
                    f"  FROM {SCHEMA}.chunks"
                    f" WHERE doc_type = %s"
                    f" ORDER BY embedding <=> %s"
                    f" LIMIT %s",
                    (vector, self.doc_type, vector, int(self.k)),
                )
                return [_as_document(row) for row in cur.fetchall()]
        except Exception:
            # 공유 커넥션을 쓰는 경우, 실패를 되돌리지 않으면 다음 검색까지
            # "current transaction is aborted" 로 연쇄 실패한다.
            try:
                con.rollback()
            except Exception:
                pass
            raise
        finally:
            if own:
                con.close()

    # LangChain 구버전 호환
    def get_relevant_documents(self, query: str) -> list:
        return self.invoke(query)


class PgVectorStore:
    """``load_vector_store()`` 가 돌려주는 핸들. 커넥션 하나를 재사용한다."""

    def __init__(self, con=None, embed_fn=None):
        self.con = con or connect()
        self.embed_fn = embed_fn

    def as_retriever(self, search_kwargs: dict | None = None) -> PgRetriever:
        search_kwargs = search_kwargs or {}
        doc_type = (search_kwargs.get("filter") or {}).get("doc_type", "manual")
        # embed_fn 을 여기서 만들지 않는다 — 만들면 torch 를 import 하게 되어
        # 배포 서버에서 임베딩 모델을 걷어낸 의미가 사라진다. PgRetriever 가
        # 질의캐시에 없을 때만 지연 로드한다.
        return PgRetriever(doc_type, search_kwargs.get("k", DEFAULT_K),
                           self.embed_fn, self.con)

    def similarity_search(self, query: str, k: int = DEFAULT_K,
                          doc_type: str | None = None) -> list:
        if doc_type:
            return PgRetriever(doc_type, k, self.embed_fn, self.con).invoke(query)
        return (PgRetriever("manual", k, self.embed_fn, self.con).invoke(query)
                + PgRetriever("law", k, self.embed_fn, self.con).invoke(query))

    def close(self) -> None:
        global _SHARED
        try:
            self.con.close()
        except Exception:
            pass
        if _SHARED is self:
            _SHARED = None


# ── _5_search.py 와 같은 이름의 진입점 (드롭인 교체용) ──────────────
# 프로세스당 커넥션 하나만 유지한다.
#   Streamlit 은 위젯을 건드릴 때마다 스크립트를 처음부터 다시 실행하고, 그때마다
#   create_provider() -> RagYujinAdapter() -> load_vector_store() 가 새로 불린다.
#   매번 새 커넥션을 열면 Supabase 무료 티어의 동시 커넥션을 금방 소진한다.
#   (GC 가 결국 닫아 주긴 하지만 DB 커넥션을 GC 타이밍에 맡길 수는 없다.)
_SHARED: "PgVectorStore | None" = None


def _is_alive(store: "PgVectorStore | None") -> bool:
    if store is None or store.con.closed:
        return False
    try:
        with store.con.cursor() as cur:
            cur.execute("SELECT 1")
        return True
    except Exception:
        try:
            store.con.close()
        except Exception:
            pass
        return False


def load_vector_store(*, fresh: bool = False) -> PgVectorStore:
    """살아 있는 공유 저장소를 돌려준다. 끊겼으면 새로 연결한다.

    ``fresh=True`` 는 적재 스크립트처럼 독립 커넥션이 필요할 때만 쓴다.
    """
    global _SHARED
    if not fresh and _is_alive(_SHARED):
        return _SHARED       # type: ignore[return-value]

    con = connect()
    try:
        with con.cursor() as cur:
            cur.execute("SELECT to_regclass(%s)", (f"{SCHEMA}.chunks",))
            if cur.fetchone()[0] is None:
                raise FileNotFoundError(
                    f"{SCHEMA}.chunks 테이블이 없습니다.\n"
                    "먼저 python rag_yujin/_4b_migrate_to_pg.py 를 실행하세요."
                )
            cur.execute(f"SELECT COUNT(*) FROM {SCHEMA}.chunks")
            if cur.fetchone()[0] == 0:
                raise FileNotFoundError(
                    f"{SCHEMA}.chunks 가 비어 있습니다.\n"
                    "먼저 python rag_yujin/_4b_migrate_to_pg.py 를 실행하세요."
                )
    except Exception:
        con.close()          # 검증에 실패하면 커넥션을 흘리지 않는다
        raise

    store = PgVectorStore(con)
    if not fresh:
        _SHARED = store
    return store


def build_retrievers(vectorstore: PgVectorStore, k_manual: int = DEFAULT_K,
                     k_law: int = DEFAULT_K):
    """_5_search.build_retrievers 와 같은 시그니처·같은 반환."""
    return (
        vectorstore.as_retriever({"k": k_manual, "filter": {"doc_type": "manual"}}),
        vectorstore.as_retriever({"k": k_law, "filter": {"doc_type": "law"}}),
    )


def get_context_for_query(manual_retriever, law_retriever, query: str) -> dict:
    """_5_search.get_context_for_query 와 같은 반환 구조."""
    return {
        "query": query,
        "manual": manual_retriever.invoke(query),
        "law": law_retriever.invoke(query),
    }


if __name__ == "__main__":
    import json
    print(json.dumps(stats(), ensure_ascii=False, indent=2))
