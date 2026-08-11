"""지시 20 — 가축분뇨법 시행령 별표 1~9 복구 시도 (DOC 없이).

pdfplumber 표 추출로 재추출을 시도하고, 복구된 별표는 재청킹,
복구 불가 별표는 data/broken_annex.json 에 등록해 search_rag 가
"원문 미확보 — 원문 링크 안내" 폴백을 내보내게 한다.
(DOC 도착 시 rag/reingest_annex.py 로 교체하는 경로는 그대로 유지)
"""
from __future__ import annotations

import json
import re

import pdfplumber

from config import MID_DIR, RAG_PDF_DIR

BROKEN_JSON = MID_DIR / "broken_annex.json"
TARGET_HINT = "가축분뇨의 관리 및 이용에 관한 법률 시행령"


def recover() -> dict:
    pdf_path = next(RAG_PDF_DIR.glob(f"{TARGET_HINT}*.pdf"))
    recovered: dict[str, str] = {}
    with pdfplumber.open(pdf_path) as pdf:
        n_pages = len(pdf.pages)
        # 별표 본문 후보: "[별표 N]" 헤더 이후 이어지는 텍스트/표
        full_parts = []
        n_tables = 0
        for pg in pdf.pages:
            full_parts.append(pg.extract_text() or "")
            for tbl in pg.extract_tables():
                n_tables += 1
                full_parts.append("\n".join(
                    " | ".join(c or "" for c in row) for row in tbl))
        full = "\n".join(full_parts)

    for m in re.finditer(r"\[별표\s*(\d+)\]([^\n]*)", full):
        start = m.end()
        nxt = full.find("[별표", start)
        body = full[start: nxt if nxt > 0 else len(full)].strip()
        # 페이지 꼬리(법제처/국가법령정보센터) 제거 후 실내용 판정
        body = re.sub(r"법제처\s*\d+\s*국가법령정보센터", "", body).strip()
        if len(body) >= 300:
            recovered[f"별표 {m.group(1)}"] = body

    print(f"  시행령 PDF {n_pages}쪽 / 표 객체 {n_tables}개 / "
          f"본문 300자 이상 복구된 별표: {len(recovered)}건")

    # rag_chunks.json 반영 + 미복구 별표 등록
    with open(MID_DIR / "rag_chunks.json", encoding="utf-8") as fh:
        chunks = json.load(fh)
    doc_name = next((c["doc"] for c in chunks if TARGET_HINT in c["doc"]), TARGET_HINT)

    broken = []
    changed = False
    for c in chunks:
        if TARGET_HINT in c["doc"] and c.get("is_annex") and len(c["text"]) < 300:
            no = re.search(r"별표\s*(\d+)", c["unit"])
            key = f"별표 {no.group(1)}" if no else c["unit"]
            if key in recovered:
                c["text"] = recovered[key]
                changed = True
            else:
                broken.append({"doc": doc_name, "unit": c["unit"]})

    if changed:
        with open(MID_DIR / "rag_chunks.json", "w", encoding="utf-8") as fh:
            json.dump(chunks, fh, ensure_ascii=False)
        emb = MID_DIR / "rag_emb.npy"
        if emb.exists():
            emb.unlink()

    BROKEN_JSON.write_text(json.dumps(broken, ensure_ascii=False, indent=1),
                           encoding="utf-8")
    verdict = ("일부 복구" if recovered else
               "복구 불가 — PDF 에 별표 내용 자체가 없음(법령정보센터 '본문' PDF 는 "
               "별표 제목 목록만 포함). 추출 라이브러리의 문제가 아님")
    print(f"  판정: {verdict} / 폴백 등록 {len(broken)}건 → {BROKEN_JSON.name}")
    return {"n_recovered": len(recovered), "n_broken": len(broken),
            "verdict": verdict}
