"""
테스트 및 고도화 — 양돈장 축산악취 작업유형별 RAG 평가셋

이 단계에서 하는 일:
  작업유형(분뇨제거/청소/환기점검/저감시설점검)별로 5개씩, 총 20개 질문에 대해
  실제 체인(_5_search.py)을 돌려보고 결과를 파일로 저장한다.

  "정답"이 정해진 벤치마크가 아니라서 자동으로 채점(맞다/틀리다)할 수는 없다.
  대신 사람이 20개를 전부 처음부터 읽지 않고도 문제 있는 것부터 골라볼 수 있게,
  기계적으로 확인 가능한 것만 자동 체크한다:
    - manual_hit / law_hit : 실무·법령 자료가 하나라도 검색됐는가
    - citation_ok          : 답변이 [실무-N]/[법령-N]으로 인용한 번호가 실제로
                              검색된 자료 개수 범위 안에 있는가(있지도 않은
                              [실무-5] 같은 걸 지어내면 걸러진다 — 프롬프트에
                              "번호를 지어내지 마라"고 시켜둔 걸 실제로 지키는지
                              검증하는 것)
    - has_both_sections    : 답변에 "실무 요령"/"관련 법령" 두 섹션이 다 있는가

  이 체크를 통과했다고 답변 "내용"이 맞다는 보장은 아니다 — 내용 정확성(법령 수치가
  정확한지, 실무 조언이 타당한지)은 결국 사람이 읽고 판단해야 한다. 이 자동 체크는
  "20개 중 구조적으로 뭔가 깨진 것부터 먼저 걸러내는" 용도다.

실행 전 준비:
  이전 단계(_5_search.py)와 동일 — GOOGLE_API_KEY(.env), langchain-google-genai 필요.
  20개 질문 = LLM 호출 20번이라 무료 티어 분당 요청 제한에 걸릴 수 있어서,
  질문 사이에 잠깐씩 쉬어가며 호출한다. 중간에 실패한 질문이 있어도 나머지는
  계속 진행하고, 실패 사실을 결과 파일에 그대로 남긴다.
"""
from __future__ import annotations

import re
import sys
import time
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CURRENT_DIR))

# 작업유형별 5개씩, 총 20개. 실무형 질문("어떻게/얼마나 자주")과 법령형 질문
# ("법적 기준/처벌은")이 섞이게 구성했다 — 한쪽으로 쏠리면 검증 의미가 줄어든다.
EVAL_QUESTIONS = {
    "분뇨제거": [
        "돈사에서 분뇨를 반출할 때 악취를 줄이려면 어떻게 해야 하나요?",
        "슬러리 피트에 고착된 슬러지는 얼마나 자주 제거해야 하나요?",
        "액비화 과정에서 악취를 줄이는 방법은 무엇인가요?",
        "분뇨 처리시설을 신고 없이 운영하면 어떤 처벌을 받나요?",
        "퇴비사에서 분뇨를 보관할 때 지켜야 할 법적 기준은 무엇인가요?",
    ],
    "청소": [
        "축사 바닥과 벽을 청소할 때 악취를 줄이는 방법은 무엇인가요?",
        "돈사 깔짚은 얼마나 자주 교체해야 하나요?",
        "축사 소독 시 악취 저감을 위해 주의해야 할 점은 무엇인가요?",
        "생활악취를 규제하는 법적 근거는 무엇인가요?",
        "축사 청소로 발생하는 폐수는 어떻게 처리해야 하나요?",
    ],
    "환기점검": [
        "환기팬이 제대로 작동하지 않으면 악취에 어떤 영향이 있나요?",
        "입기구와 배기구는 얼마나 자주 점검해야 하나요?",
        "강제배기 돈사와 자연환기 돈사의 환기 관리 차이는 무엇인가요?",
        "환기시설 점검·보수 중 배출허용기준을 초과하면 어떻게 해야 하나요?",
        "악취방지법상 배출구의 정의는 무엇인가요?",
    ],
    "저감시설점검": [
        "스크러버(약액세정탑)를 점검할 때 확인해야 할 사항은 무엇인가요?",
        "탈취탑의 미생물이 제대로 작동하는지 어떻게 확인하나요?",
        "저감시설 설치 비용을 지원받을 수 있는 조건은 무엇인가요?",
        "악취저감시설을 설치하지 않으면 어떤 법적 불이익이 있나요?",
        "안개분무 시설의 노즐은 어떻게 관리해야 하나요?",
    ],
}

CITATION_RE = re.compile(r"\[(실무|법령)-(\d+)\]")
CALL_DELAY_SECONDS = 3  # 무료 티어 분당 요청 제한 대비, 호출 사이에 잠깐 쉰다


def _check_citation_consistency(answer: str, manual_count: int, law_count: int) -> list[str]:
    """답변이 인용한 [실무-N]/[법령-N]의 N이 실제로 검색된 자료 개수를 벗어나면
    (즉 없는 자료를 지어내서 인용한 것이면) 문제 목록에 추가한다."""
    problems = []
    for label, num_str in CITATION_RE.findall(answer):
        num = int(num_str)
        limit = manual_count if label == "실무" else law_count
        if num < 1 or num > limit:
            problems.append(f"[{label}-{num}] 인용했지만 실제 검색된 {label} 자료는 {limit}개뿐")
    return problems


def _evaluate_one(chain, worktype: str, question: str) -> dict:
    result = chain.invoke(question)
    answer = result["answer"]
    manual_count = len(result["_manual_docs"])
    law_count = len(result["_law_docs"])

    citation_problems = _check_citation_consistency(answer, manual_count, law_count)
    has_both_sections = ("실무 요령" in answer) and ("관련 법령" in answer)

    checks = {
        "manual_hit": manual_count > 0,
        "law_hit": law_count > 0,
        "citation_ok": len(citation_problems) == 0,
        "has_both_sections": has_both_sections,
    }

    return {
        "worktype": worktype,
        "question": question,
        "answer": answer,
        "manual_count": manual_count,
        "law_count": law_count,
        "checks": checks,
        "citation_problems": citation_problems,
        "all_ok": all(checks.values()),
    }


def _format_report(results: list[dict]) -> str:
    lines = ["# RAG 평가 결과", ""]

    total = len(results)
    passed = sum(1 for r in results if r["all_ok"])
    lines.append(
        f"자동 체크 통과: {passed}/{total}건 "
        f"(구조적 이상만 걸러낸 것이며, 답변 내용 정확성은 아래를 직접 읽고 확인해야 함)"
    )
    lines.append("")

    lines.append("## 요약표")
    lines.append("")
    lines.append("| # | 작업유형 | 질문 | 실무검색 | 법령검색 | 인용정합 | 섹션구성 |")
    lines.append("|---|---|---|---|---|---|---|")
    for i, r in enumerate(results, 1):
        c = r["checks"]
        lines.append(
            f"| {i} | {r['worktype']} | {r['question'][:30]} | "
            f"{'✅' if c['manual_hit'] else '❌'} | {'✅' if c['law_hit'] else '❌'} | "
            f"{'✅' if c['citation_ok'] else '⚠️'} | {'✅' if c['has_both_sections'] else '⚠️'} |"
        )
    lines.append("")

    lines.append("## 상세 결과")
    for i, r in enumerate(results, 1):
        lines.append("")
        lines.append(f"### {i}. [{r['worktype']}] {r['question']}")
        if not r["all_ok"]:
            lines.append("")
            lines.append("**⚠ 자동 체크에서 걸린 문제:**")
            if not r["checks"]["manual_hit"]:
                lines.append("- 실무 자료가 하나도 검색되지 않음")
            if not r["checks"]["law_hit"]:
                lines.append("- 법령 자료가 하나도 검색되지 않음")
            for p in r["citation_problems"]:
                lines.append(f"- {p}")
            if not r["checks"]["has_both_sections"]:
                lines.append("- 답변에 실무/법령 두 섹션이 명확히 구분되지 않음")
        lines.append("")
        lines.append(r["answer"])
        lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    from _5_search import build_chain, build_retrievers, get_llm, load_vector_store

    # LLM 키/패키지 문제는 20번 호출 전에 한 번에 확인한다.
    llm = get_llm()
    vectorstore = load_vector_store()
    manual_retriever, law_retriever = build_retrievers(vectorstore)
    chain = build_chain(manual_retriever, law_retriever, llm)

    total_questions = sum(len(qs) for qs in EVAL_QUESTIONS.values())
    print(f"[평가 시작] 총 {total_questions}개 질문 (작업유형 4개 × 5개)\n")

    results = []
    done = 0
    for worktype, questions in EVAL_QUESTIONS.items():
        for question in questions:
            done += 1
            print(f"  ({done}/{total_questions}) [{worktype}] {question}")
            try:
                r = _evaluate_one(chain, worktype, question)
            except Exception as e:
                print(f"    ❌ 호출 실패: {e}")
                r = {
                    "worktype": worktype,
                    "question": question,
                    "answer": f"(호출 실패로 답변을 받지 못함: {e})",
                    "manual_count": 0,
                    "law_count": 0,
                    "checks": {
                        "manual_hit": False,
                        "law_hit": False,
                        "citation_ok": False,
                        "has_both_sections": False,
                    },
                    "citation_problems": [],
                    "all_ok": False,
                }
            results.append(r)
            if done < total_questions:
                time.sleep(CALL_DELAY_SECONDS)

    report = _format_report(results)
    out_path = CURRENT_DIR / "eval_results.md"
    out_path.write_text(report, encoding="utf-8")

    passed = sum(1 for r in results if r["all_ok"])
    print(f"\n완료 — 자동 체크 {passed}/{total_questions}건 통과")
    print(f"전체 결과는 {out_path} 에 저장했습니다. 열어서 답변 내용을 직접 확인해주세요.")
