"""S6 — RAG 평가: 질문-정답 쌍 top-3 적중률 (목표 80%)."""
from __future__ import annotations

from rag.search import RagIndex

# (질문, query_type, 정답이 있어야 할 문서 키워드, 정답 본문 키워드)
QA = [
    ("악취배출시설 1차 위반 시 과태료는 얼마인가", "과태료", "시행령", "과태료"),
    ("악취 배출허용기준은 어디에 규정되어 있나", None, "시행규칙", "배출허용기준"),
    ("가축분뇨 처리시설의 설치 기준", "분뇨제거", "가축분뇨", "처리시설"),
    ("악취 민원 현장조사 후 개선명령 절차", "민원대응", "악취방지법", "개선"),
    ("돼지 축사 악취저감시설 운영 방법", "저감시설점검", "저감시설", "운영"),
    ("액비 살포 시 준수사항", "액비살포" if False else None, "가축분뇨", "액비"),
    ("악취방지계획 수립 의무는 누구에게 있나", None, "악취방지법", "악취방지계획"),
    ("익산시 악취 저감 조례상 지원 사업", None, "조례", "지원"),
]


def run(index: RagIndex) -> dict:
    hits, details = 0, []
    for question, qtype, doc_kw, text_kw in QA:
        res = index.search(question, qtype, k=3)
        hit = any(doc_kw in r["doc"] and text_kw in (r["snippet"] + r["unit"])
                  for r in res["results"])
        # 문서만 맞아도 부분 인정하지 않는다 — 본문 키워드까지 요구 (보수적)
        hits += int(hit)
        top1 = res["results"][0] if res["results"] else None
        details.append({"q": question, "hit": hit,
                        "top1": f"{top1['doc']}/{top1['unit']}" if top1 else None})

    # 거절 규칙 동작 확인
    refused = index.search("저 지금 고발당하나요?")["refused"]

    rate = hits / len(QA)
    print(f"  RAG top-3 적중 {hits}/{len(QA)} = {rate:.0%} (목표 80%) / 거절규칙 동작: {refused}")
    return {"hit_rate": rate, "n": len(QA), "refusal_ok": refused, "details": details}
