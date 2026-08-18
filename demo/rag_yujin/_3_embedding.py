"""
임베딩 — 양돈장 축산악취 작업유형별 RAG용
임베딩은 사람이 이해하는 텍스트(문장, 단어 등)를 컴퓨터가 수학적으로 계산할 수 있는 숫자 배열(벡터)로 변환하는 단계로
임베딩 모델을 통과시켜 숫자 벡터로 만들고 벡터 DB에 저장

이 단계에서 하는 일:
  1. 청크(문장/문단)를 벡터로 바꿔주는 "임베딩 함수"를 하나 정의한다.
     (실제로 전체 청크를 벡터로 바꿔서 저장하는 건 다음 단계인 _4_database.py의 몫이다.
      여기서는 그 임베딩 함수가 우리 도메인(축산악취) 문장을 제대로 이해하는지 미리 검증한다.)
  2. 이 파일을 직접 실행하면, 실제 데이터(로더+청커)를 불러와 전부 임베딩해보고,
     작업유형별(분뇨제거/청소/환기점검/저감시설점검) 질의를 넣었을 때 그 작업유형과
     관련된 청크가 실제로 가장 유사하게 나오는지 확인한다("의미상 말이 되는지" 체크).

모델 선택 — jhgan/ko-sroberta-multitask:
  - 한국어 문장 임베딩 벤치마크(KorSTS/KorNLI)에서 검증된 한국어 특화 sentence-transformers 모델.
  - 법령/매뉴얼처럼 딱딱한 문어체 한국어에도 무난하고, 로컬(무료)로 돌릴 수 있어
    OpenAI 임베딩 API 같은 유료 호출 없이 계속 재현 가능하다.
  - 768차원. 문장 하나가 아니라 청크(최대 700자 안팍) 단위로 넣을 것이므로,
    모델 자체의 512 토큰 한계에 걸리지 않도록 청킹 단계(MANUAL_CHUNK_SIZE=700자)를
    이미 보수적으로 잡아두었다.
"""
from __future__ import annotations

import math
import os
import time
from pathlib import Path

# ── 설정 ──────────────────────────────────────────────────────────
EMBEDDING_MODEL_NAME = "jhgan/ko-sroberta-multitask"

# GPU가 있으면 EMBEDDING_DEVICE=cuda 환경변수로 바꿔서 실행하면 된다. 기본은 CPU.
EMBEDDING_DEVICE = os.environ.get("EMBEDDING_DEVICE", "cpu")

# 코사인 유사도로 비교할 것이므로, 임베딩 자체를 단위벡터로 정규화해둔다.
# (정규화해두면 다음 단계 Chroma에서도 cosine 거리 계산이 정확해진다.)
_ENCODE_KWARGS = {"normalize_embeddings": True}

# 작업유형별 검증용 질의. "이 질문을 던졌을 때 관련 있는 청크가 실제로 상위에 뜨는가"를
# 눈으로 확인하기 위한 것이지, 정식 평가셋(_6_test.py에서 다룰 예정)은 아니다.
WORKTYPE_QUERIES = {
    "분뇨제거": "돈사에서 분뇨를 제거하고 반출할 때 악취를 줄이는 방법",
    "청소": "축사를 세척하고 소독할 때 악취를 줄이는 방법",
    "환기점검": "환기팬과 입기구, 배기구를 점검해서 악취를 줄이는 방법",
    "저감시설점검": "탈취시설이나 스크러버 같은 악취저감시설을 점검하고 운영하는 방법",
}

_embedding_fn_cache = None  # 모델을 매번 새로 로드하면 느리므로 한 번만 로드해서 재사용


def _check_dependencies() -> None:
    """필수 패키지가 없으면 모델 로드 시도할 때마다 알 수 없는 에러로 죽지 말고
    시작할 때 한 번에 무엇이 없는지 알려준다 (_1_loader.py와 같은 패턴)."""
    missing = []
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        missing.append("sentence-transformers")
    try:
        import langchain_huggingface  # noqa: F401
    except ImportError:
        missing.append("langchain-huggingface")
    if missing:
        raise ImportError(
            "다음 패키지가 설치되어 있지 않습니다: " + ", ".join(missing) + "\n"
            "지금 이 스크립트를 실행 중인 환경에서 아래 명령을 실행하세요:\n"
            f"  python -m pip install {' '.join(missing)}\n"
            "(처음 실행 시 huggingface에서 모델 파일(약 450MB)을 내려받으므로 인터넷 연결이 필요합니다.\n"
            " 한 번 받으면 로컬 캐시(~/.cache/huggingface)에 남아 다음부터는 안 받습니다.)"
        )


def get_embedding_function():
    """전체 파이프라인(이 파일의 테스트, 그리고 다음 단계인 _4_database.py)이
    똑같은 임베딩 함수를 재사용하도록 이 함수 하나로 통일한다.
    검색할 때(질문 임베딩)와 저장할 때(청크 임베딩) 서로 다른 모델/설정을 쓰면
    벡터 공간이 어긋나서 검색이 안 되므로, 반드시 이 함수를 통해서만 가져다 써야 한다."""
    global _embedding_fn_cache
    if _embedding_fn_cache is not None:
        return _embedding_fn_cache

    _check_dependencies()
    from langchain_huggingface import HuggingFaceEmbeddings

    _embedding_fn_cache = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": EMBEDDING_DEVICE},
        encode_kwargs=_ENCODE_KWARGS,
    )
    return _embedding_fn_cache


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """embed_documents/embed_query 결과(파이썬 list[float])를 그대로 비교하기 위한
    순수 파이썬 구현. normalize_embeddings=True를 이미 켜두었지만, 혹시 꺼진 채로
    쓰이더라도 항상 정확하도록 분모에서 직접 정규화한다(어느 쪽이든 안전)."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _1_loader import DATA_DIR, _resolve_data_dir, load_law_manual_data
    from _2_Chunking import chunk_documents

    _check_dependencies()

    resolved_dir = _resolve_data_dir(DATA_DIR)
    docs = load_law_manual_data(resolved_dir)
    chunks = chunk_documents(docs)
    if not chunks:
        raise SystemExit("청크가 0개입니다. _1_loader.py / _2_Chunking.py를 먼저 확인하세요.")

    print(f"\n[임베딩 모델] {EMBEDDING_MODEL_NAME} (device={EMBEDDING_DEVICE})")
    print("모델을 처음 불러오는 경우 huggingface에서 내려받느라 시간이 걸릴 수 있습니다...")
    embed_fn = get_embedding_function()

    texts = [c.page_content for c in chunks]
    print(f"\n[임베딩 실행] 청크 {len(texts)}개를 벡터로 변환합니다...")
    start = time.time()
    vectors = embed_fn.embed_documents(texts)
    elapsed = time.time() - start
    dim = len(vectors[0]) if vectors else 0
    print(f"✅ 완료 — {len(vectors)}개 벡터, 차원 {dim}, 소요 {elapsed:.1f}초 "
          f"(청크당 평균 {elapsed / max(len(vectors), 1) * 1000:.0f}ms)")

    # ── 작업유형별 의미 검증 ──────────────────────────────────────
    # "환기점검"이라고 물었을 때 진짜로 환기 관련 청크가 상위에 뜨는지 눈으로 확인한다.
    # 법령 조문 vs 매뉴얼이 뒤섞여도 의미상 관련된 게 뽑히는지가 핵심이다.
    print("\n[작업유형별 질의 검증] (정식 평가셋은 아니고, 임베딩이 말이 되는지 눈으로 보는 용도)")
    for worktype, query in WORKTYPE_QUERIES.items():
        query_vec = embed_fn.embed_query(query)
        scored = sorted(
            ((cosine_similarity(query_vec, v), c) for v, c in zip(vectors, chunks)),
            key=lambda x: x[0],
            reverse=True,
        )
        print(f"\n  ▶ [{worktype}] \"{query}\"")
        for score, c in scored[:3]:
            m = c.metadata
            preview = c.page_content[:60].replace("\n", " ")
            print(f"    {score:.3f} | [{m['doc_type']}] {m['source_file']} / {m['unit']} — {preview}...")

    print(
        "\n참고: 위 상위 결과가 해당 작업유형과 관련 없어 보이면(예: 환기점검인데 분뇨 관련 조문만 나옴),\n"
        "      청킹 단위가 너무 크거나(문맥 희석) 모델이 도메인에 안 맞을 수 있습니다 — 다음 단계로\n"
        "      넘어가기 전에 알려주세요. 다음 단계(_4_database.py)에서는 이 임베딩 함수를 그대로 가져다\n"
        "      Chroma.from_documents(chunks, get_embedding_function(), persist_directory=...)로 벡터\n"
        "      DB를 만들 예정입니다(임베딩을 다시 두 번 계산하지 않도록 함수를 재사용)."
    )