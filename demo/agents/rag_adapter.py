"""PR #9 ``rag_yujin`` 검색기를 D provider 결과 계약으로 변환한다."""
from __future__ import annotations

from typing import Any


class RagYujinAdapter:
    """Gemini 답변 체인을 호출하지 않고 Chroma 검색 결과만 반환한다."""

    backend = "rag_yujin-chroma"

    def __init__(self, *, k_manual: int = 3, k_law: int = 3) -> None:
        from rag_yujin._5_search import build_retrievers, load_vector_store

        vectorstore = load_vector_store()
        self.manual_retriever, self.law_retriever = build_retrievers(
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
        from rag_yujin._5_search import get_context_for_query

        context = get_context_for_query(
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
