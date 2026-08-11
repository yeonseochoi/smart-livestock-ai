"""PDF ingestion for the livestock-odour RAG index.

Text PDFs are read with pypdf. Pages with too little embedded text optionally
fall back to OCR (PyMuPDF + pytesseract). Chunks retain page ranges and stable
IDs so citations and embedding caches can be validated.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from pypdf import PdfReader

from config import MID_DIR, OUT_DIR, PROV, RAG_PDF_DIR, finding

LAW_HINT = ("법률", "시행령", "시행규칙", "조례")
# Anchoring at the beginning of a line is essential: references such as
# "제7조에 따라" inside an article must not start a new chunk.
ARTICLE_RE = re.compile(
    r"(?m)(?=^\s*제\s*\d+\s*조(?:의\s*\d+)?(?![가-힣\d])\s*(?:\([^)]*\))?)"
)
ANNEX_RE = re.compile(r"(?m)(?=^\s*\[?별표\s*\d+(?:의\s*\d+)?\]?)")
HEADING_RE = re.compile(
    r"^\s*(제\s*\d+\s*[장절]|\d+(?:\.\d+)*[.)]\s+|[가-하][.)]\s+|[①-⑳]\s*)"
)
SPACE_RE = re.compile(r"[ \t\u00a0]+")


def doc_hierarchy(doc: str) -> str:
    if "시행규칙" in doc:
        return "시행규칙"
    if "시행령" in doc:
        return "시행령"
    if "조례" in doc:
        return "조례"
    if "(법률)" in doc or doc.endswith("법"):
        return "법률"
    return "매뉴얼"


def _clean_text(text: str) -> str:
    text = text.replace("\x00", "").replace("\r", "\n")
    text = re.sub(r"(?<=\S)-\n(?=[가-힣A-Za-z])", "", text)
    lines = [SPACE_RE.sub(" ", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def _ocr_page(pdf_path: Path, page_index: int, lang: str) -> str:
    """OCR one page when optional OCR packages and Tesseract are available."""
    try:
        import fitz  # PyMuPDF
        import pytesseract
        from PIL import Image
    except ImportError:
        return ""
    try:
        with fitz.open(pdf_path) as doc:
            pix = doc[page_index].get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
            image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        return pytesseract.image_to_string(image, lang=lang, config="--psm 6")
    except Exception as exc:
        print(f"  [OCR 실패] {pdf_path.name} p.{page_index + 1}: {exc}")
        return ""


def extract_pages(pdf_path: Path, ocr: bool = True, ocr_lang: str = "kor+eng") -> tuple[list[dict], int]:
    reader = PdfReader(str(pdf_path))
    pages: list[dict] = []
    ocr_count = 0
    for i, page in enumerate(reader.pages):
        try:
            text = _clean_text(page.extract_text() or "")
        except Exception:
            text = ""
        source = "text"
        if ocr and len(re.sub(r"\s", "", text)) < 80:
            recovered = _clean_text(_ocr_page(pdf_path, i, ocr_lang))
            if len(recovered) > len(text):
                text, source, ocr_count = recovered, "ocr", ocr_count + 1
        pages.append({"page": i + 1, "text": text, "source": source})
    return pages, ocr_count


def _page_for_offset(page_spans: list[tuple[int, int, int]], offset: int) -> int:
    for start, end, page in page_spans:
        if start <= offset < end:
            return page
    return page_spans[-1][2] if page_spans else 1


def _chunk_law(pages: list[dict], doc: str) -> list[dict]:
    full_parts, spans, cursor = [], [], 0
    for p in pages:
        text = p["text"]
        full_parts.append(text)
        spans.append((cursor, cursor + len(text) + 1, p["page"]))
        cursor += len(text) + 1
    full = "\n".join(full_parts)
    starts = sorted(set([0] + [m.start() for m in ARTICLE_RE.finditer(full)]
                        + [m.start() for m in ANNEX_RE.finditer(full)]))
    chunks = []
    for pos, end in zip(starts, starts[1:] + [len(full)]):
        text = full[pos:end].strip()
        if len(text) < 40:
            continue
        first = text.splitlines()[0].strip()[:80]
        am = re.match(r"제\s*\d+\s*조(?:의\s*\d+)?", first)
        xm = re.match(r"\[?별표\s*\d+(?:의\s*\d+)?\]?", first)
        unit = (xm or am).group(0) if (xm or am) else "서문"
        chunks.append({"doc": doc, "unit": unit, "text": text,
                       "page": _page_for_offset(spans, pos),
                       "page_end": _page_for_offset(spans, max(pos, end - 1))})
    return chunks


def _chunk_manual(pages: list[dict], doc: str) -> list[dict]:
    chunks, buf = [], []
    unit, start_page, end_page = "서문", 1, 1

    def flush() -> None:
        nonlocal buf
        text = "\n".join(buf).strip()
        if len(text) >= 80:
            chunks.append({"doc": doc, "unit": unit, "page": start_page,
                           "page_end": end_page, "text": text})
        buf = []

    for p in pages:
        for line in p["text"].splitlines():
            if HEADING_RE.match(line) and len(line) <= 100:
                flush()
                unit, start_page = line[:80], p["page"]
            buf.append(line)
            end_page = p["page"]
            if len("\n".join(buf)) >= 2200:
                flush()
                unit, start_page = f"{unit} (계속)", p["page"]
    flush()
    return chunks


def _finalize(chunks: list[dict]) -> list[dict]:
    result, seen = [], set()
    for chunk in chunks:
        text = _clean_text(chunk["text"])
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if len(text) < 40 or digest in seen:
            continue
        seen.add(digest)
        chunk["text"] = text
        chunk["hier"] = doc_hierarchy(chunk["doc"])
        chunk["is_annex"] = bool(re.match(r"\[?별표", chunk["unit"]))
        chunk["id"] = hashlib.sha1(
            f"{chunk['doc']}|{chunk['unit']}|{chunk.get('page')}|{digest}".encode("utf-8")
        ).hexdigest()[:16]
        result.append(chunk)
    return result


def run(ocr: bool = True, ocr_lang: str = "kor+eng") -> list[dict]:
    MID_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pdfs = sorted(RAG_PDF_DIR.glob("*.pdf"))
    PROV.log("D6 법령·매뉴얼 PDF", RAG_PDF_DIR, real=True, note=f"{len(pdfs)}종")
    all_chunks, docs, empty_docs, total_ocr = [], [], [], 0
    for pdf in pdfs:
        pages, ocr_count = extract_pages(pdf, ocr=ocr, ocr_lang=ocr_lang)
        total_ocr += ocr_count
        char_count = sum(len(p["text"]) for p in pages)
        if char_count < 200:
            empty_docs.append(pdf.stem)
            docs.append({"doc": pdf.stem, "pages": len(pages), "chars": char_count,
                         "chunks": 0, "ocr_pages": ocr_count, "status": "unreadable"})
            continue
        chunks = (_chunk_law(pages, pdf.stem) if any(h in pdf.stem for h in LAW_HINT)
                  else _chunk_manual(pages, pdf.stem))
        chunks = _finalize(chunks)
        all_chunks.extend(chunks)
        docs.append({"doc": pdf.stem, "pages": len(pages), "chars": char_count,
                     "chunks": len(chunks), "ocr_pages": ocr_count, "status": "ok"})

    all_chunks = _finalize(all_chunks)
    index_path = MID_DIR / "rag_chunks.json"
    index_path.write_text(json.dumps(all_chunks, ensure_ascii=False, indent=1), encoding="utf-8")
    fingerprint = hashlib.sha256(index_path.read_bytes()).hexdigest()
    manifest = {"schema_version": 2, "fingerprint": fingerprint,
                "pdf_count": len(pdfs), "chunk_count": len(all_chunks),
                "ocr_pages": total_ocr, "documents": docs}
    (MID_DIR / "rag_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    # Any new chunk set invalidates old embeddings.
    for path in (MID_DIR / "rag_emb.npy", MID_DIR / "rag_emb.meta.json"):
        if path.exists():
            path.unlink()
    broken = [{"doc": c["doc"], "unit": c["unit"]} for c in all_chunks
              if c["is_annex"] and len(c["text"]) < 300]
    (MID_DIR / "broken_annex.json").write_text(
        json.dumps(broken, ensure_ascii=False, indent=1), encoding="utf-8")
    report = [f"읽기 실패 문서: {d}" for d in empty_docs]
    report += [f"짧은 별표: {b['doc']} / {b['unit']}" for b in broken]
    (OUT_DIR / "rag_redownload_list.txt").write_text("\n".join(report) or "(없음)", encoding="utf-8")
    if empty_docs:
        finding("OCR 후에도 읽지 못한 문서: " + ", ".join(empty_docs))
    print(f"  청크 {len(all_chunks):,}개 / 문서 {len(pdfs)-len(empty_docs)}/{len(pdfs)}종 / OCR {total_ocr}쪽")
    return all_chunks


if __name__ == "__main__":
    run()
