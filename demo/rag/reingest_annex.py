"""지시 16 — 가축분뇨법 시행령 별표 1~5 재추출 준비 스크립트.

원문(DOC/DOCX/새 PDF)이 도착하면:
    python -m rag.reingest_annex "경로\새파일.docx"
1) 텍스트 추출 (docx 는 zip 내 document.xml 파싱 — 추가 의존성 없음)
2) 별표 단위 재청킹 → rag_chunks.json 의 해당 문서 별표 청크 교체
3) 임베딩 캐시 무효화 (다음 검색 시 자동 재생성)
4) 별표 평가 5문항(ANNEX_QA)을 포함해 재측정

인자 없이 실행하면 현재 별표 청크의 건강 상태만 점검한다 (대기 모드).
"""
from __future__ import annotations

import json
import re
import sys
import zipfile
from pathlib import Path

from config import MID_DIR

TARGET_DOC_HINT = "가축분뇨의 관리 및 이용에 관한 법률 시행령"

# 깨졌던 별표에 대한 추가 평가셋 (재추출 후 QA30 에 합산해 재측정)
ANNEX_QA = [
    ("허가대상 배출시설의 규모 기준은", "과태료" if False else None,
     "가축분뇨", "허가대상"),        # 별표 1
    ("신고대상 배출시설은 어떤 시설인가", None, "가축분뇨", "신고대상"),  # 별표 2
    ("퇴비액비화기준의 부숙도 기준은", None, "가축분뇨", "퇴비액비화"),   # 별표 3
    ("가축분뇨법상 과징금의 산정기준은", None, "가축분뇨", "과징금"),     # 별표 4
    ("가축분뇨관련영업의 허가기준은", None, "가축분뇨", "허가기준"),      # 별표 5
]

ANNEX_SPLIT = re.compile(r"(?=\[?별표\s*\d)")


def extract_docx_text(path: Path) -> str:
    """python-docx 없이 docx 텍스트 추출 (document.xml 의 <w:t> 수집)."""
    with zipfile.ZipFile(path) as z:
        xml = z.read("word/document.xml").decode("utf-8", errors="ignore")
    # 문단 경계 유지: </w:p> → 개행
    xml = re.sub(r"</w:p>", "\n", xml)
    return re.sub(r"<[^>]+>", "", xml)


def health_check() -> list[str]:
    with open(MID_DIR / "rag_chunks.json", encoding="utf-8") as fh:
        chunks = json.load(fh)
    broken = [f"{c['doc']} / {c['unit']} ({len(c['text'])}자)"
              for c in chunks
              if c.get("is_annex") and TARGET_DOC_HINT in c["doc"]
              and len(c["text"]) < 300]
    print(f"  가축분뇨법 시행령 별표 청크 상태: 깨짐 {len(broken)}건")
    for b in broken:
        print(f"    - {b}")
    return broken


def reingest(src_path: Path) -> None:
    if src_path.suffix.lower() == ".docx":
        text = extract_docx_text(src_path)
    elif src_path.suffix.lower() == ".pdf":
        from pypdf import PdfReader
        text = "\n".join((p.extract_text() or "") for p in PdfReader(str(src_path)).pages)
    else:
        raise SystemExit(f"지원 형식: .docx/.pdf (받은 것: {src_path.suffix}) — "
                         ".hwp 는 hwp5txt 등으로 docx/pdf 변환 후 투입")

    annex_parts = [p.strip() for p in ANNEX_SPLIT.split(text)
                   if p.strip().startswith(("별표", "[별표")) and len(p.strip()) >= 300]
    if not annex_parts:
        raise SystemExit("원문에서 300자 이상 별표를 찾지 못했습니다 — 파일 확인 필요")

    with open(MID_DIR / "rag_chunks.json", encoding="utf-8") as fh:
        chunks = json.load(fh)
    doc_name = next((c["doc"] for c in chunks if TARGET_DOC_HINT in c["doc"]),
                    TARGET_DOC_HINT)
    kept = [c for c in chunks
            if not (c.get("is_annex") and TARGET_DOC_HINT in c["doc"])]
    for part in annex_parts:
        unit = part.split("\n")[0][:30]
        kept.append({"doc": doc_name, "unit": unit, "text": part,
                     "hier": "시행령", "is_annex": True})
    with open(MID_DIR / "rag_chunks.json", "w", encoding="utf-8") as fh:
        json.dump(kept, fh, ensure_ascii=False)
    emb = MID_DIR / "rag_emb.npy"
    if emb.exists():
        emb.unlink()  # 청크 수 변경 → 캐시 무효화
    print(f"  별표 {len(annex_parts)}건 재청킹 완료, 임베딩 캐시 무효화")

    # 재측정: QA30 + ANNEX_QA
    from rag.search import RagIndex
    from rag import eval_qa_v2
    idx = RagIndex(backend="sroberta")
    base = eval_qa_v2.run(idx, boost=True)
    hits = 0
    for q, qt, dkw, tkw in ANNEX_QA:
        res = idx.search(q, qt, k=3)
        hits += any(dkw in r["doc"] and tkw in (r["snippet"] + r["unit"])
                    for r in res["results"])
    print(f"  별표 5문항 적중 {hits}/5 / 합산 {base['hits'] + hits}/35 = "
          f"{(base['hits'] + hits) / 35:.0%}")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from console import use_utf8_stdout
    use_utf8_stdout()
    if len(sys.argv) > 1:
        reingest(Path(sys.argv[1]))
    else:
        print("대기 모드 — 원문 도착 시: python -m rag.reingest_annex <파일.docx|pdf>")
        health_check()
