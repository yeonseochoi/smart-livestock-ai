"""S6 — RAG 인제스트: PDF 14종 → 조문/소제목 단위 청크 + 메타데이터.

계획서는 임베딩+Chroma 를 지정했지만 데모 환경에 Chroma·임베딩 모델이 없어
TF-IDF(문자 n-gram) 검색으로 대체한다 — 청킹 원칙(조문 단위, 별표 통청크,
고정 길이 금지)은 그대로 지킨다. 대체 사실은 validation_report 에 명시.
"""
from __future__ import annotations

import json
import re

from pypdf import PdfReader

from config import RAG_PDF_DIR, MID_DIR, PROV, finding

# 법령류(조문 단위 청킹) 판별 키워드
LAW_HINT = ("법률", "시행령", "시행규칙", "조례")
ARTICLE_RE = re.compile(r"(?=\n\s*제\s*\d+\s*조(?:의\s*\d+)?\s*[(\[])")
ANNEX_RE = re.compile(r"(?=\n\s*\[?별표\s*\d*\]?)")


def doc_hierarchy(doc: str) -> str:
    """법령 위계 분류 (v3 지시 12). 시행규칙→시행령→법률 순으로 검사
    (문서명에 '법률'과 '시행령'이 같이 들어가므로 구체적인 것 먼저)."""
    if "시행규칙" in doc:
        return "시행규칙"
    if "시행령" in doc:
        return "시행령"
    if "조례" in doc:
        return "조례"
    if "(법률)" in doc or doc.endswith("법"):
        return "법률"
    return "매뉴얼"


def _extract(pdf_path) -> list[tuple[int, str]]:
    reader = PdfReader(str(pdf_path))
    pages = []
    for i, p in enumerate(reader.pages):
        try:
            pages.append((i + 1, p.extract_text() or ""))
        except Exception:
            pages.append((i + 1, ""))
    return pages


def _chunk_law(full: str, doc: str) -> list[dict]:
    """조문 단위 분리, 별표는 통째로 1청크."""
    # 별표를 먼저 떼어낸다
    parts = ANNEX_RE.split(full)
    chunks = []
    for part in parts:
        head = part.strip()[:20]
        if head.startswith("별표") or head.startswith("[별표"):
            chunks.append({"doc": doc, "unit": head.split("\n")[0][:30],
                           "text": part.strip()})
            continue
        for art in ARTICLE_RE.split(part):
            art = art.strip()
            if len(art) < 30:
                continue
            m = re.match(r"제\s*\d+\s*조(?:의\s*\d+)?", art)
            chunks.append({"doc": doc, "unit": m.group(0) if m else "서문",
                           "text": art})
    return chunks


def _chunk_manual(pages: list[tuple[int, str]], doc: str) -> list[dict]:
    """매뉴얼: 소제목(숫자. / 제N장·절 / 가나다.) 단위, 실패 시 페이지 단위."""
    chunks = []
    buf, unit, start_page = [], "서문", 1
    head_re = re.compile(r"^\s*(제\s*\d+\s*[장절]|[0-9]+(\.[0-9]+)*[.)]\s|[가-하][.)]\s)")
    for pno, text in pages:
        for line in text.split("\n"):
            if head_re.match(line) and len(line.strip()) < 60:
                if buf and len("\n".join(buf)) > 80:
                    chunks.append({"doc": doc, "unit": unit, "page": start_page,
                                   "text": "\n".join(buf)})
                buf, unit, start_page = [], line.strip()[:40], pno
            buf.append(line)
    if buf:
        chunks.append({"doc": doc, "unit": unit, "page": start_page,
                       "text": "\n".join(buf)})
    return chunks


def run() -> list[dict]:
    pdfs = sorted(RAG_PDF_DIR.glob("*.pdf"))
    PROV.log("D6 법령·매뉴얼 PDF", RAG_PDF_DIR, real=True, note=f"{len(pdfs)}종")

    all_chunks: list[dict] = []
    empty_docs = []
    for pdf in pdfs:
        doc = pdf.stem
        pages = _extract(pdf)
        full = "\n".join(t for _, t in pages)
        if len(full.strip()) < 200:
            empty_docs.append(doc)
            continue
        if any(h in doc for h in LAW_HINT):
            cs = _chunk_law(full, doc)
        else:
            cs = _chunk_manual(pages, doc)
        # 너무 긴 청크(추출 실패로 통짜가 된 경우)는 별표가 아니면 3000자에서 분할
        fixed = []
        for c in cs:
            if len(c["text"]) > 3000 and not c["unit"].startswith(("별표", "[별표")):
                for i in range(0, len(c["text"]), 3000):
                    fixed.append({**c, "unit": f"{c['unit']}#{i//3000}",
                                  "text": c["text"][i:i + 3000]})
            else:
                fixed.append(c)
        all_chunks.extend(fixed)

    if empty_docs:
        finding(f"PDF 텍스트 추출 실패(스캔본 추정) {len(empty_docs)}종: "
                + ", ".join(empty_docs) + " — RAG 검색 범위에서 빠짐. OCR 필요")

    # v3 지시 12: 위계 메타데이터 + 별표 여부 필드
    for c in all_chunks:
        c["hier"] = doc_hierarchy(c["doc"])
        c["is_annex"] = c["unit"].startswith(("별표", "[별표"))

    # 별표 추출이 깨진 항목(본문 300자 미만) → "재다운로드 필요 목록"
    broken = [f"{c['doc']} / {c['unit']} ({len(c['text'])}자)"
              for c in all_chunks if c["is_annex"] and len(c["text"]) < 300]
    redownload = broken + [f"{d} (스캔본 — 텍스트 추출 불가, OCR 또는 원문 재확보)"
                           for d in empty_docs]
    from config import OUT_DIR
    (OUT_DIR / "rag_redownload_list.txt").write_text(
        "\n".join(redownload) or "(없음)", encoding="utf-8")
    if redownload:
        print(f"  [DOC 재다운로드 필요] {len(redownload)}건 → out/rag_redownload_list.txt")
        for line in redownload[:5]:
            print(f"    - {line}")

    with open(MID_DIR / "rag_chunks.json", "w", encoding="utf-8") as fh:
        json.dump(all_chunks, fh, ensure_ascii=False)
    print(f"  청크 {len(all_chunks):,}개 생성 (문서 {len(pdfs) - len(empty_docs)}/{len(pdfs)}종)")
    return all_chunks
