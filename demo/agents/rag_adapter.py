"""PR #9 ``rag_yujin`` 검색기를 D provider 결과 계약으로 변환한다.

백엔드는 둘 중 하나를 쓴다 (``RAG_BACKEND`` 환경변수).
    pgvector  Supabase ``rag`` 스키마 — 배포·GitHub Actions·팀 공유용
    chroma    로컬 ``rag_yujin/data/chroma_db`` — 오프라인 개발용
    auto      DATABASE_URL 이 있으면 pgvector, 없으면 chroma (기본값)

두 백엔드는 같은 청킹 규칙(_2)과 같은 임베딩 모델(_3)을 쓰므로 검색 결과가
같다. 반환 계약도 동일해서 D 에이전트·대시보드는 어느 쪽인지 몰라도 된다.

연결 실패를 조용히 다른 백엔드로 바꾸지 않는다 — provider 규약과 같다.
실패는 실패로 드러내야 어떤 자료를 보고 있는지 화면에서 알 수 있다.
"""
from __future__ import annotations

import os
from typing import Any


def _resolve_backend(name: str | None = None) -> str:
    """'pgvector' | 'chroma' 로 확정한다."""
    choice = (name or os.environ.get("RAG_BACKEND") or "auto").strip().lower()
    if choice in ("pgvector", "postgres", "supabase"):
        return "pgvector"
    if choice == "chroma":
        return "chroma"
    # auto — .env 를 읽어 본 뒤 DATABASE_URL 유무로 정한다
    try:
        from rag_yujin import pgvector_store
        pgvector_store._load_env()
    except Exception:
        pass
    return "pgvector" if os.environ.get("DATABASE_URL", "").strip() else "chroma"


class RagYujinAdapter:
    """Gemini 답변 체인을 호출하지 않고 검색 결과만 반환한다."""

    # 클래스 기본값. 인스턴스가 만들어질 때 실제 백엔드로 덮어쓴다.
    backend = "rag_yujin-chroma"

    def __init__(self, *, k_manual: int = 3, k_law: int = 3,
                 backend: str | None = None) -> None:
        resolved = _resolve_backend(backend)
        if resolved == "pgvector":
            from rag_yujin import pgvector_store as module
            self.backend = "rag_yujin-pgvector"
        else:
            from rag_yujin import _5_search as module
            self.backend = "rag_yujin-chroma"

        self._module = module
        vectorstore = module.load_vector_store()
        self.manual_retriever, self.law_retriever = module.build_retrievers(
            vectorstore, k_manual=k_manual, k_law=k_law
        )

    def search(
        self,
        question: str,
        query_type: str | None = None,
        *,
        k: int = 3,
        boost: bool = True,
    ) -> dict[str, Any]:
        """기존 ``RagIndex.search``와 호환되는 구조화 결과를 반환한다."""

        del boost  # rag_yujin은 manual/law 분리 검색으로 문서 유형을 보장한다.
        module = getattr(self, "_module", None)
        if module is None:                       # __new__ 로 만든 경우(테스트)
            from rag_yujin import _5_search as module

        context = module.get_context_for_query(
            self.manual_retriever, self.law_retriever, question
        )
        results: list[dict[str, Any]] = []
        for doc_type in ("manual", "law"):
            for document in context[doc_type][: max(1, int(k))]:
                metadata = dict(document.metadata)
                source_file = metadata.get("source_file") or metadata.get("source")
                results.append({
                    "rank": len(results) + 1,
                    "id": metadata.get("chunk_id") or metadata.get("id"),
                    "source_file": source_file,
                    "doc": source_file,
                    "unit": metadata.get("unit"),
                    "page": metadata.get("page"),
                    "doc_type": metadata.get("doc_type", doc_type),
                    "score": None,
                    "score_kind": "순위 기반 검색 결과(신뢰확률 아님)",
                    "snippet": document.page_content.strip(),
                })
        return {
            "refused": False,
            "query_type": query_type,
            "backend": self.backend,
            "results": results,
        }
