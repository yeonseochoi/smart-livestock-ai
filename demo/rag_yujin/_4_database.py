"""
벡터 저장 — 양돈장 축산악취 작업유형별 RAG용

이 단계에서 하는 일:
  이전 단계(로더 → 청커 → 임베딩)에서 만든 청크와 임베딩 함수를 그대로 가져다,
  Chroma 벡터DB에 저장(persist)한다. 로더/청커/임베딩 코드는 이 단계 때문에
  수정하지 않는다 — 셋 다 이미 "재사용 가능한 함수"로 만들어뒀으므로 여기서는
  가져다 쓰기만 하면 된다.

재실행 시 정책 — "매번 완전히 새로 만든다":
  Chroma는 같은 persist_directory에 대고 add를 반복하면(예: 이 스크립트를 두 번
  실행하면) 문서가 중복 저장되거나, 같은 ID로 다시 넣을 때 에러가 나거나 조용히
  무시되는 등 chromadb 버전에 따라 동작이 갈린다. 데이터셋이 아직 작고(청크 수백 개)
  자주 바뀌는 단계이므로, 매번 기존 DB 폴더를 통째로 지우고 처음부터 다시 만드는
  쪽이 "지금 DB가 최신 청크와 일치하는가"를 매번 따질 필요가 없어 훨씬 안전하다.
  나중에 데이터가 훨씬 커지면 그때 증분 업데이트 방식으로 바꾸면 된다.
"""
from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path

from langchain_core.documents import Document

CURRENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CURRENT_DIR))

PERSIST_DIR = CURRENT_DIR / "data" / "chroma_db"
# Chroma 컬렉션명은 영문/숫자/_-로 짓는 걸 권장하므로(한글은 버전에 따라 문제될 수 있음) 영문으로 지정.
COLLECTION_NAME = "livestock_odor_kb"


def _check_dependencies() -> None:
    missing = []
    try:
        import langchain_chroma  # noqa: F401
    except ImportError:
        missing.append("langchain-chroma")
    try:
        import chromadb  # noqa: F401
    except ImportError:
        missing.append("chromadb")
    if missing:
        raise ImportError(
            "다음 패키지가 설치되어 있지 않습니다: " + ", ".join(missing) + "\n"
            f"  python -m pip install {' '.join(missing)}"
        )


def build_vector_store(
    chunks: list[Document],
    embedding_fn,
    persist_dir: Path = PERSIST_DIR,
    collection_name: str = COLLECTION_NAME,
):
    """chunks를 embedding_fn으로 임베딩해서 Chroma에 저장하고, 만든 벡터스토어를 반환한다."""
    _check_dependencies()
    from langchain_chroma import Chroma

    if persist_dir.exists():
        print(f"  기존 DB 폴더를 지우고 새로 만듭니다: {persist_dir}")
        shutil.rmtree(persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)

    # chunk_id(청킹 단계에서 만든 SHA1 해시)를 그대로 Chroma 문서 ID로 써서,
    # 같은 내용이 두 번 들어가는 사고를 Chroma 쪽에서도 잡아낼 수 있게 한다.
    ids = [c.metadata["chunk_id"] for c in chunks]
    if len(ids) != len(set(ids)):
        dupes = len(ids) - len(set(ids))
        print(f"  ⚠ chunk_id 중복 {dupes}건 발견 — 청킹 단계 dedup 로직을 다시 확인하세요.")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_fn,
        ids=ids,
        collection_name=collection_name,
        persist_directory=str(persist_dir),
    )
    return vectorstore


if __name__ == "__main__":
    from _1_loader import DATA_DIR, _resolve_data_dir, load_law_manual_data
    from _2_Chunking import chunk_documents
    from _3_embedding import get_embedding_function

    resolved_dir = _resolve_data_dir(DATA_DIR)
    docs = load_law_manual_data(resolved_dir)
    chunks = chunk_documents(docs)
    if not chunks:
        raise SystemExit("청크가 0개입니다. 이전 단계(_1_loader.py / _2_Chunking.py)를 먼저 확인하세요.")

    print(f"\n[임베딩 모델 로드] (첫 실행이면 huggingface에서 내려받느라 시간이 걸릴 수 있습니다)")
    embed_fn = get_embedding_function()

    print(f"\n[벡터 저장] 청크 {len(chunks)}개 → {PERSIST_DIR}")
    start = time.time()
    vectorstore = build_vector_store(chunks, embed_fn)
    elapsed = time.time() - start
    print(f"✅ 완료 — 소요 {elapsed:.1f}초")

    stored_ids = vectorstore.get()["ids"]
    print(f"  저장된 벡터 개수: {len(stored_ids)}개 (청크 개수 {len(chunks)}개와 같아야 정상)")
    if len(stored_ids) != len(chunks):
        print("  ⚠ 개수가 다릅니다 — chunk_id 중복이나 저장 실패 가능성이 있으니 확인이 필요합니다.")

    # 저장만 하고 끝내지 않고, 실제로 검색이 되는지 바로 확인한다.
    print('\n[검색 테스트] "환기팬 점검 방법"')
    results = vectorstore.similarity_search("환기팬 점검 방법", k=3)
    for r in results:
        m = r.metadata
        preview = r.page_content[:60].replace("\n", " ")
        print(f"  [{m['doc_type']}] {m['source_file']} / {m['unit']} — {preview}...")

    print(
        f"\n다음부터는 이 스크립트를 다시 돌리지 않아도, 아래처럼 저장된 DB를 그대로 열어서 쓸 수 있습니다"
        f"(다음 단계 _5_search.py에서 이렇게 씀):\n"
        f"  from langchain_chroma import Chroma\n"
        f"  from _3_embedding import get_embedding_function\n"
        f"  vectorstore = Chroma(\n"
        f"      collection_name={COLLECTION_NAME!r},\n"
        f"      embedding_function=get_embedding_function(),\n"
        f"      persist_directory={str(PERSIST_DIR)!r},\n"
        f"  )"
    )