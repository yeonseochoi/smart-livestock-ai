"""Small unit tests that do not require PDFs, OCR, or model downloads."""
from rag.ingest import _chunk_law, _finalize, doc_hierarchy
from rag.search import infer_query_type


def test_law_references_do_not_split_articles():
    pages = [{"page": 1, "source": "text", "text":
              "제1조(목적) 이 법의 목적과 적용 범위를 충분히 설명하는 문장이다. "
              "국민의 안전과 환경 보호에 필요한 사항을 정한다.\n"
              "제7조에 따른 기준을 적용하지만 이것은 조문 경계가 아니다.\n"
              "제2조(정의) 정의 문장이다. 충분한 본문 길이를 채우기 위해 "
              "관련 용어와 적용 범위 및 세부 기준을 추가로 설명한다."}]
    chunks = _chunk_law(pages, "시험법(법률)")
    units = [chunk["unit"] for chunk in chunks]
    assert units == ["제1조", "제2조"]
    assert "제7조에 따른" in chunks[0]["text"]


def test_metadata_and_deduplication():
    base = {"doc": "시험 시행령", "unit": "별표 1", "page": 3,
            "page_end": 3, "text": "별표 1 " + "기준 " * 20}
    chunks = _finalize([dict(base), dict(base)])
    assert len(chunks) == 1
    assert chunks[0]["is_annex"] is True
    assert chunks[0]["hier"] == "시행령"
    assert chunks[0]["id"]


def test_query_type_inference():
    assert infer_query_type("과태료 부과기준") == "과태료"
    assert infer_query_type("환기 시설 점검") == "환기점검"
