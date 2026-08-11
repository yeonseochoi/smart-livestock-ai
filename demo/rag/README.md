# RAG module

Run commands from the `demo` directory.

```powershell
python -m rag.cli doctor
python -m rag.cli ingest
python -m rag.cli eval --backend tfidf
python -m rag.cli holdout --backend tfidf
python -m rag.cli search "악취저감시설은 어떻게 관리하나요?" --backend tfidf
```

Semantic or hybrid retrieval downloads `jhgan/ko-sroberta-multitask` on first use:

```powershell
python -m rag.cli eval --backend hybrid
```

OCR is optional. Install the Python adapters with
`python -m pip install -r rag/requirements-rag.txt` and install Tesseract with
Korean language data. If OCR is unavailable, text PDFs are still indexed and
unreadable scanned documents are listed in `out/rag_redownload_list.txt`.

Generated index files remain under `demo/data` for compatibility with the rest
of the project. Source code and RAG-specific documentation live only here.
