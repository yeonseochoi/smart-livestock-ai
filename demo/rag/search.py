"""S6 — RAG 검색 (TF-IDF 문자 n-gram; Chroma+임베딩의 데모 대체).

거절 규칙: 개별 사안 판단 요청에는 답하지 않고 법률구조공단(132) 안내.
모든 답변에 조문·출처를 붙인다.
"""
from __future__ import annotations

import json
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config import MID_DIR

QUERY_TEMPLATES = {
    "분뇨제거": "가축분뇨 제거 처리 적체 기준 청소",
    "청소": "축사 청소 세척 관리 기준",
    "환기점검": "환기 시설 점검 관리",
    "저감시설점검": "악취 저감시설 가동 점검 약액",
    "과태료": "과태료 부과 기준 위반 별표",
    "민원대응": "민원 현장조사 개선명령 이의신청 절차",
}

REFUSE_RE = re.compile(r"(고발|처벌받|잡혀|저 지금|우리 농장이|벌금 내야)")
REFUSAL = ("개별 사안에 대한 법적 판단은 드릴 수 없습니다. 일반 정보이며, "
           "구체적 사안은 법률구조공단(국번없이 132) 또는 변호사 상담이 필요합니다.")

# ── v3 지시 12: 법령 위계 부스트 ─────────────────────────────────────
# query_type(또는 질문에서 추론) → 위계 가산점. 값은 유사도에 더하는 소가산.
# 예: 과태료 질문 → 시행령 + 별표 우선 (과태료 부과기준은 시행령 별표에 있다)
HIER_BOOST = {
    "과태료": {"시행령": 0.06, "_annex": 0.06},   # 과태료 부과기준 = 시행령 별표
    "벌칙": {"법률": 0.06},                        # 징역·벌금·과징금·조업정지 = 법률 본문
    "민원대응": {"법률": 0.05, "시행령": 0.03},
    "분뇨제거": {"매뉴얼": 0.03, "시행령": 0.03},
    "청소": {"매뉴얼": 0.05},
    "환기점검": {"매뉴얼": 0.05},
    "저감시설점검": {"매뉴얼": 0.05},
}
# 질문 문면에 위계가 명시되면 그 위계를 직접 부스트 (v2 오답 패턴:
# 관련 문서는 찾는데 법-령-규칙 층위를 못 짚음)
_EXPLICIT_HIER = [("시행규칙", "시행규칙"), ("시행령", "시행령"),
                  ("조례", "조례"), ("법률", "법률")]
_QTYPE_INFER = [
    # 과태료(행정질서벌, 시행령 별표)와 벌칙·과징금(법률 본문)은 위계가 다르다
    (re.compile(r"과태료"), "과태료"),
    (re.compile(r"과징금|벌칙|처벌|벌금|징역"), "벌칙"),
    (re.compile(r"민원|개선명령|이의신청|현장조사|조업정지"), "민원대응"),
    (re.compile(r"청소|세척"), "청소"),
    (re.compile(r"환기"), "환기점검"),
    (re.compile(r"저감시설"), "저감시설점검"),
    (re.compile(r"분뇨.*(제거|처리시설)|처리시설"), "분뇨제거"),
]


def infer_query_type(question: str) -> str | None:
    for pat, qt in _QTYPE_INFER:
        if pat.search(question):
            return qt
    return None


EMB_MODEL = "jhgan/ko-sroberta-multitask"


class RagIndex:
    """backend='auto'|'sroberta'|'tfidf'.

    sroberta: ko-sroberta 임베딩(문장 단위 의미 검색). 청크 임베딩은
    data/rag_emb.npy 에 캐시한다. 실패 시 tfidf 로 폴백(로깅).
    """

    def __init__(self, backend: str = "auto") -> None:
        with open(MID_DIR / "rag_chunks.json", encoding="utf-8") as fh:
            self.chunks = json.load(fh)
        self.backend = "tfidf"
        if backend in ("auto", "sroberta"):
            try:
                self._init_sroberta()
                self.backend = "sroberta"
            except Exception as e:
                if backend == "sroberta":
                    raise
                print(f"  ko-sroberta 로딩 실패({e!r}) → TF-IDF 폴백")
        if self.backend == "tfidf":
            self.vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4),
                                       max_features=50000)
            self.mat = self.vec.fit_transform(c["text"] for c in self.chunks)

    def _init_sroberta(self) -> None:
        import numpy as np
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(EMB_MODEL)
        cache = MID_DIR / "rag_emb.npy"
        if cache.exists():
            self.emb = np.load(cache)
            if len(self.emb) != len(self.chunks):
                cache.unlink()
                raise RuntimeError("임베딩 캐시-청크 수 불일치 — 캐시 삭제 후 재실행")
        else:
            # 청크가 길면 앞 1000자만 임베딩 (ko-sroberta 입력 한도 고려)
            self.emb = self.model.encode(
                [c["text"][:1000] for c in self.chunks],
                batch_size=64, show_progress_bar=False, normalize_embeddings=True)
            np.save(cache, self.emb)

    def search(self, question: str, query_type: str | None = None, k: int = 3,
               boost: bool = True) -> dict:
        if REFUSE_RE.search(question):
            return {"refused": True, "answer": REFUSAL, "results": []}
        q = question
        if query_type and query_type in QUERY_TEMPLATES:
            q = QUERY_TEMPLATES[query_type] + " " + question
        if self.backend == "sroberta":
            qv = self.model.encode([q], normalize_embeddings=True)
            sims = (self.emb @ qv[0]).astype(float)
        else:
            sims = cosine_similarity(self.vec.transform([q]), self.mat)[0]

        if boost:
            sims = sims + self._boost_vector(question, query_type)

        top = sims.argsort()[::-1][:k]
        results = []
        for i in top:
            c = self.chunks[i]
            results.append({
                "doc": c["doc"], "unit": c["unit"], "page": c.get("page"),
                "hier": c.get("hier"),
                "score": round(float(sims[i]), 4),
                "snippet": re.sub(r"\s+", " ", c["text"])[:200],
            })
        out = {"refused": False, "results": results}
        notice = self._broken_annex_notice(question)
        if notice:
            out["notice"] = notice
        return out

    _BROKEN_Q_RE = re.compile(
        r"별표|허가대상|신고대상|퇴비액비화|과징금|허가기준|등록기준|부과기준")

    def _broken_annex_notice(self, question: str) -> str | None:
        """지시 20 폴백: 원문 미확보 별표를 건드리는 질의에 원문 안내를 붙인다."""
        if not hasattr(self, "_broken"):
            p = MID_DIR / "broken_annex.json"
            self._broken = json.loads(p.read_text(encoding="utf-8")) if p.exists() else []
        if self._broken and self._BROKEN_Q_RE.search(question) \
                and ("가축분뇨" in question or "분뇨" in question):
            units = ", ".join(b["unit"].split("(")[0].strip()
                              for b in self._broken[:9])
            return (f"[안내] 가축분뇨법 시행령의 일부 별표({units})는 보유 PDF 에 "
                    f"원문이 포함되어 있지 않습니다. 정확한 기준은 국가법령정보센터"
                    f"(law.go.kr)에서 '가축분뇨의 관리 및 이용에 관한 법률 시행령' "
                    f"별표 원문을 확인하세요.")
        return None

    def _boost_vector(self, question: str, query_type: str | None):
        import numpy as np
        b = np.zeros(len(self.chunks))
        qt = query_type or infer_query_type(question)
        boosts = dict(HIER_BOOST.get(qt, {}))
        # 질문에 위계가 명시돼 있으면 해당 위계를 강하게 우선 (+0.08)
        for kw, hier in _EXPLICIT_HIER:
            if kw in question:
                boosts[hier] = boosts.get(hier, 0) + 0.08
                break
        if "별표" in question:
            boosts["_annex"] = boosts.get("_annex", 0) + 0.05
        if not boosts:
            return b
        for i, c in enumerate(self.chunks):
            v = boosts.get(c.get("hier"), 0.0)
            if c.get("is_annex"):
                v += boosts.get("_annex", 0.0)
            b[i] = v
        return b
