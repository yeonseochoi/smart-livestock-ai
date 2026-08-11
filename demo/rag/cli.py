"""Command line interface: python -m rag.cli <command>."""
from __future__ import annotations

import argparse
import json
import shutil
import sys


def main() -> int:
    # Windows PowerShell may inherit CP949; project logs contain Unicode punctuation.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="축산 악취 법령·매뉴얼 RAG 도구")
    sub = parser.add_subparsers(dest="command", required=True)
    ingest = sub.add_parser("ingest", help="PDF 추출·OCR·청킹")
    ingest.add_argument("--no-ocr", action="store_true")
    ingest.add_argument("--ocr-lang", default="kor+eng")
    search = sub.add_parser("search", help="질문 검색")
    search.add_argument("question")
    search.add_argument("--backend", choices=["auto", "hybrid", "sroberta", "tfidf"], default="auto")
    search.add_argument("-k", type=int, default=3)
    evaluate = sub.add_parser("eval", help="30문항 검색 평가")
    evaluate.add_argument("--backend", choices=["auto", "hybrid", "sroberta", "tfidf"], default="tfidf")
    holdout = sub.add_parser("holdout", help="독립 20문항 홀드아웃 평가")
    holdout.add_argument("--backend", choices=["auto", "hybrid", "sroberta", "tfidf"], default="tfidf")
    sub.add_parser("doctor", help="환경·인덱스 상태 진단")
    args = parser.parse_args()

    if args.command == "ingest":
        from rag.ingest import run
        run(ocr=not args.no_ocr, ocr_lang=args.ocr_lang)
    elif args.command == "search":
        from rag.search import RagIndex
        print(json.dumps(RagIndex(args.backend).search(args.question, k=args.k), ensure_ascii=False, indent=2))
    elif args.command == "eval":
        from rag.eval_qa_v2 import run
        from rag.search import RagIndex
        result = run(RagIndex(args.backend))
        misses = [d for d in result["details"] if not d["hit"]]
        print(json.dumps({**result, "details": misses}, ensure_ascii=False, indent=2))
    elif args.command == "holdout":
        from rag.eval_holdout import run
        from rag.search import RagIndex
        result = run(RagIndex(args.backend))
        misses = [d for d in result["details"] if not d["hit"]]
        print(json.dumps({**result, "details": misses}, ensure_ascii=False, indent=2))
    else:
        import importlib.util
        from config import MID_DIR, RAG_PDF_DIR
        status = {"python": sys.version.split()[0], "pdfs": len(list(RAG_PDF_DIR.glob('*.pdf'))),
                  "tesseract": shutil.which("tesseract"),
                  "packages": {n: bool(importlib.util.find_spec(n)) for n in
                      ["pypdf", "sklearn", "sentence_transformers", "fitz", "pytesseract", "PIL"]},
                  "chunks": (MID_DIR / "rag_chunks.json").exists(),
                  "embeddings": (MID_DIR / "rag_emb.npy").exists()}
        print(json.dumps(status, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
