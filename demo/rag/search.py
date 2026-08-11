"""Hybrid Korean retrieval with safe legal-query handling and citations."""
from __future__ import annotations

import hashlib
import json
import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config import MID_DIR

EMB_MODEL = "jhgan/ko-sroberta-multitask"
QUERY_TEMPLATES = {
    "분뇨제거": "가축분뇨 제거 처리시설 적체 청소 기준",
    "청소": "축사 청소 세척 위생 관리 기준",
    "환기점검": "환기 시설 점검 관리",
    "저감시설점검": "악취 저감시설 가동 점검 약액",
    "과태료": "과태료 부과기준 위반 시행령 별표",
    "벌칙": "벌칙 징역 벌금 법률 조문",
    "민원대응": "민원 현장조사 개선명령 이의신청 절차",
}
REFUSE_RE = re.compile(r"(고발|처벌받|잡혀|저 지금|우리 농장이|벌금 내야)")
REFUSAL = ("개별 사안에 대한 법적 판단은 드릴 수 없습니다. 일반 정보이며, "
           "구체적 사안은 법률구조공단(국번없이 132) 또는 변호사 상담이 필요합니다.")
HIER_BOOST = {
    "과태료": {"시행령": .06, "_annex": .06}, "벌칙": {"법률": .06},
    "민원대응": {"법률": .05, "시행령": .03},
    "분뇨제거": {"매뉴얼": .03, "시행령": .03},
    "청소": {"매뉴얼": .05}, "환기점검": {"매뉴얼": .05},
    "저감시설점검": {"매뉴얼": .05},
}
_EXPLICIT_HIER = [("시행규칙", "시행규칙"), ("시행령", "시행령"),
                  ("조례", "조례"), ("법률", "법률")]
_QTYPE_INFER = [
    (re.compile(r"과태료"), "과태료"),
    (re.compile(r"과징금|벌칙|처벌|벌금|징역"), "벌칙"),
    (re.compile(r"민원|개선명령|이의신청|현장조사|조업정지"), "민원대응"),
    (re.compile(r"청소|세척"), "청소"), (re.compile(r"환기"), "환기점검"),
    (re.compile(r"저감시설"), "저감시설점검"),
    (re.compile(r"분뇨.*(제거|처리시설)|처리시설"), "분뇨제거"),
]


def infer_query_type(question: str) -> str | None:
    return next((kind for pattern, kind in _QTYPE_INFER if pattern.search(question)), None)


class RagIndex:
    """backend='auto'|'hybrid'|'sroberta'|'tfidf'. Public API is backward compatible."""

    def __init__(self, backend: str = "auto") -> None:
        path = MID_DIR / "rag_chunks.json"
        if not path.exists():
            raise FileNotFoundError("RAG 인덱스가 없습니다. 먼저 `python -m rag.cli ingest`를 실행하세요.")
        self.chunks = json.loads(path.read_text(encoding="utf-8"))
        if not self.chunks:
            raise RuntimeError("rag_chunks.json에 검색 가능한 청크가 없습니다.")
        self.texts = [f"{c['doc']} {c['unit']} {c['text']}" for c in self.chunks]
        self.vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5),
                                   sublinear_tf=True, max_features=80000)
        self.mat = self.vec.fit_transform(self.texts)
        self.backend = "tfidf"
        if backend in ("auto", "hybrid", "sroberta"):
            try:
                self._init_sroberta(path)
                self.backend = "sroberta" if backend == "sroberta" else "hybrid"
            except Exception as exc:
                if backend == "sroberta":
                    raise
                print(f"  ko-sroberta 로딩 실패({exc!r}) → TF-IDF 폴백")

    def _init_sroberta(self, chunks_path) -> None:
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(EMB_MODEL)
        cache, meta_path = MID_DIR / "rag_emb.npy", MID_DIR / "rag_emb.meta.json"
        fingerprint = hashlib.sha256(chunks_path.read_bytes()).hexdigest()
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        if cache.exists() and meta == {"fingerprint": fingerprint, "model": EMB_MODEL,
                                      "count": len(self.chunks)}:
            self.emb = np.load(cache)
        else:
            self.emb = self.model.encode(self.texts, batch_size=32,
                show_progress_bar=True, normalize_embeddings=True)
            np.save(cache, self.emb)
            meta_path.write_text(json.dumps({"fingerprint": fingerprint,
                "model": EMB_MODEL, "count": len(self.chunks)}, indent=2), encoding="utf-8")

    @staticmethod
    def _scale(scores):
        scores = np.asarray(scores, dtype=float)
        lo, hi = float(scores.min()), float(scores.max())
        return (scores - lo) / (hi - lo) if hi > lo else np.zeros_like(scores)

    def search(self, question: str, query_type: str | None = None, k: int = 3,
               boost: bool = True) -> dict:
        question = question.strip()
        if not question:
            raise ValueError("질문이 비어 있습니다.")
        if REFUSE_RE.search(question):
            return {"refused": True, "answer": REFUSAL, "results": []}
        qtype = query_type or infer_query_type(question)
        expanded = f"{QUERY_TEMPLATES.get(qtype, '')} {question}".strip()
        lexical = cosine_similarity(self.vec.transform([expanded]), self.mat)[0]
        scores = lexical
        if self.backend in ("hybrid", "sroberta"):
            semantic = self.emb @ self.model.encode([expanded], normalize_embeddings=True)[0]
            scores = semantic if self.backend == "sroberta" else .65*self._scale(semantic)+.35*self._scale(lexical)
        if boost:
            scores = scores + self._boost_vector(question, qtype)
        top = scores.argsort()[::-1][:max(1, min(k, len(scores)))]
        results = []
        for rank, i in enumerate(top, 1):
            c = self.chunks[int(i)]
            results.append({"rank": rank, "id": c.get("id"), "doc": c["doc"],
                "unit": c["unit"], "page": c.get("page"), "page_end": c.get("page_end"),
                "hier": c.get("hier"), "score": round(float(scores[i]), 4),
                "snippet": re.sub(r"\s+", " ", c["text"])[:500]})
        out = {"refused": False, "query_type": qtype, "backend": self.backend,
               "results": results}
        notice = self._broken_annex_notice(question)
        if notice:
            out["notice"] = notice
        return out

    _BROKEN_Q_RE = re.compile(r"별표|허가대상|신고대상|퇴비액비화|과징금|허가기준|등록기준|부과기준")

    def _broken_annex_notice(self, question: str) -> str | None:
        if not hasattr(self, "_broken"):
            p = MID_DIR / "broken_annex.json"
            self._broken = json.loads(p.read_text(encoding="utf-8")) if p.exists() else []
        if self._broken and self._BROKEN_Q_RE.search(question) and ("가축분뇨" in question or "분뇨" in question):
            return ("[안내] 보유 PDF에 일부 시행령 별표 원문이 없습니다. 정확한 기준은 "
                    "국가법령정보센터(law.go.kr)의 최신 원문을 확인하세요.")
        return None

    def _boost_vector(self, question: str, query_type: str | None):
        boosts = dict(HIER_BOOST.get(query_type, {}))
        for word, hierarchy in _EXPLICIT_HIER:
            if word in question:
                boosts[hierarchy] = boosts.get(hierarchy, 0) + .08
                break
        if "별표" in question:
            boosts["_annex"] = boosts.get("_annex", 0) + .05
        values = np.zeros(len(self.chunks))
        for i, chunk in enumerate(self.chunks):
            values[i] = boosts.get(chunk.get("hier"), 0)
            if chunk.get("is_annex"):
                values[i] += boosts.get("_annex", 0)
            # Deterministic legal-document cues. These are structural signals,
            # not answer keywords, and prevent generic manuals from outranking
            # an explicitly requested instrument or provision type.
            doc, unit = chunk.get("doc", ""), chunk.get("unit", "")
            if "익산시" in question and "조례" in question and "익산시" in doc and "조례" in doc:
                values[i] += .12
            if "목적" in question and re.match(r"제\s*1\s*조", unit):
                values[i] += .12
            if "시행규칙" in question and chunk.get("hier") == "시행규칙":
                values[i] += .08
            if "배출허용기준" in question and "수치" in question and chunk.get("hier") == "시행규칙":
                values[i] += .10
            if "악취배출시설" in question and "신고" in question and chunk.get("hier") == "시행규칙":
                values[i] += .08
            if "축종별" in question and chunk.get("hier") == "매뉴얼" and "축산악취" in doc:
                values[i] += .12
            # High-value disposition terms should remain in the retrieved
            # evidence itself. This also makes the ranking easier to explain.
            anchors = ("목적", "지원", "조업정지", "폐쇄", "사용중지", "퇴비화")
            for anchor in anchors:
                if anchor in question and anchor in chunk.get("text", ""):
                    values[i] += .15
        return values
