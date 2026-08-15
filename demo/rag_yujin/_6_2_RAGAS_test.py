"""
RAGAS 자동 평가 파이프라인 — 최소 단위 증명(PoC)

RAGAS는 사람이 직접 만든 정답 없이도 RAG 시스템의 성능을 자동으로 평가할 수 있는 오픈소스 프레임워크.
문서를 얼마나 잘 찾아왔는지(검색)와 답변을 얼마나 잘 만들었는지(생성)를 나누어서 세밀하게 채점 가능.

RAGAS의 4가지 핵심 평가 지표
-> 사실성, 답변 관련성, 문맥 정밀도, 문맥 재현율

이 스크립트가 증명하는 것 (전체 커버리지가 아니라 "파이프라인이 실제로 맞물려 돌아가는가"를 증명하는 게 목적):
  문서(청크) 투입 → RAGAS가 질문·정답 자동 생성 → 우리 체인(_5_search.py)으로
  실제 답변 생성 → RAGAS가 Faithfulness/Answer Relevancy/Context Precision/Context Recall로 자동 채점 → 리포트 저장

왜 전체 데이터(15개 파일·559개 청크)를 안 넣고 10~20개만 쓰나:
  RAGAS의 자동 테스트셋 생성은 문서로 지식그래프를 만드는 과정에서 LLM을 문서 단위로 여러 번 호출한다.
  전체를 넣으면 무료 티어 요청 제한에 금방 걸리고 시간도 오래 걸린다.
  지금 목표는 파이프라인의 뼈대가 실제로 동작하는지 증명하는 것이므로, 이미 검증된 좋은 결과가 나왔던 작업유형(분뇨제거/환기점검) 관련 청크만
  우리 검색기로 추려서 최소 규모로 돌린다.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CURRENT_DIR))

# ── 최소 규모 설정 — "전체 커버리지"가 아니라 "파이프라인 증명"이 목적 ──────
SAMPLE_WORKTYPES = ["분뇨제거", "환기점검"]  # 이미 결과가 좋았던 두 작업유형만
SAMPLE_K_PER_QUERY = 5  # 작업유형당 manual/law 검색기에서 각각 최대 5개 → 최대 20개, dedup 후엔 보통 더 적음
TESTSET_SIZE = 12  # RAGAS가 자동 생성할 질문 개수 (요청하신 10~15개 범위)
CALL_DELAY_SECONDS = 4  # 우리 체인 호출 사이 대기 (무료 티어 분당 제한 대비)

# RAGAS의 default_transforms()는 입력 문서의 25% 이상이 500토큰을 넘으면
# HeadlinesExtractor + HeadlineSplitter 경로를 켠다. 근데 이 경로의 splitter는
# (extractor와 달리) 500토큰 이하 문서에도 필터 없이 그대로 적용되면서
# "'headlines' property not found in this node" 에러로 죽는다 — ragas 0.4.3
# 소스(testset/transforms/default.py)에서 직접 확인한 라이브러리 버그다.
# 우리가 넣는 문서는 이미 _2_Chunking.py가 적절히 쪼갠 조각이라 RAGAS가 또
# 나눌 필요가 없으므로, 테스트셋 "생성"에만 쓰는 사본을 이 상한선 아래로 잘라서
# 아예 그 버그 경로를 안 타게 만든다(실제 답변 생성·채점에 쓰는 원본 청크는
# 그대로 둔다 — 이 잘림은 _run_chain_on_testset 이후 단계엔 영향 없음).
TESTSET_GEN_MAX_TOKENS = 450

TESTSET_CACHE_PATH = CURRENT_DIR / "ragas_testset_cache.json"
RAW_RESULTS_CACHE_PATH = CURRENT_DIR / "ragas_raw_results_cache.json"
REPORT_PATH = CURRENT_DIR / "ragas_eval_report.md"


def _check_dependencies() -> None:
    missing = []
    try:
        import ragas  # noqa: F401
    except ImportError:
        missing.append("ragas")
    if missing:
        raise ImportError(
            "다음 패키지가 설치되어 있지 않습니다: " + ", ".join(missing) + "\n"
            "아래 순서로 설치하세요(순서 중요 — langchain-community 충돌 회피):\n"
            "  python -m pip install ragas\n"
            '  python -m pip install "langchain-community<0.4"'
        )
    try:
        import langchain_community
        major, minor = (int(x) for x in langchain_community.__version__.split(".")[:2])
        if (major, minor) >= (0, 4):
            raise ImportError(
                f"langchain-community가 {langchain_community.__version__}로 설치되어 있는데, "
                "0.4.x 이상에서는 ragas가 import 단계에서 바로 깨집니다"
                "(langchain_community.chat_models.vertexai 모듈이 삭제됨).\n"
                '  python -m pip install "langchain-community<0.4" 로 낮춰주세요.'
            )
    except ImportError:
        pass  # langchain_community 자체가 없는 경우는 _1_loader.py 쪽에서 이미 체크함


def _sample_relevant_chunks(vectorstore):
    """전체 559개 청크가 아니라, 이미 결과가 좋았던 작업유형(분뇨제거/환기점검)
    쿼리로 검색되는 청크만 우리 검색기로 추려서 RAGAS 지식그래프 생성 규모를
    최소화한다."""
    from _3_embedding import WORKTYPE_QUERIES
    from _5_search import build_retrievers

    manual_retriever, law_retriever = build_retrievers(
        vectorstore, k_manual=SAMPLE_K_PER_QUERY, k_law=SAMPLE_K_PER_QUERY
    )

    seen_ids = set()
    sampled = []
    for worktype in SAMPLE_WORKTYPES:
        query = WORKTYPE_QUERIES[worktype]
        for doc in manual_retriever.invoke(query) + law_retriever.invoke(query):
            chunk_id = doc.metadata.get("chunk_id")
            if chunk_id in seen_ids:
                continue
            seen_ids.add(chunk_id)
            sampled.append(doc)
    return sampled


def _trim_docs_for_testset_generation(docs, max_tokens: int = TESTSET_GEN_MAX_TOKENS):
    """RAGAS 테스트셋 "생성" 단계에만 쓸 사본을 만든다 — 원본 청크가 max_tokens를
    넘으면 잘라서, default_transforms()가 (버그 있는) 헤드라인 분할 경로를 타지
    않도록 한다. 실제 토큰 수는 ragas가 쓰는 것과 동일한 tokenizer
    (num_tokens_from_string)로 직접 재서 확인하므로 글자 수 어림짐작이 아니다."""
    from langchain_core.documents import Document as LCDocument
    from ragas.utils import num_tokens_from_string

    trimmed = []
    for d in docs:
        text = d.page_content
        while num_tokens_from_string(text) > max_tokens and len(text) > 50:
            text = text[: int(len(text) * 0.85)]
        trimmed.append(LCDocument(page_content=text, metadata=dict(d.metadata)))
    return trimmed


def _generate_or_load_testset(sampled_docs):
    """지식그래프 생성은 이 스크립트에서 가장 비싼(LLM 호출 많은) 단계라, 이미
    생성해둔 결과가 파일로 있으면 재사용한다 — 뒤 단계(채점)에서 실패해도
    이 단계부터 처음부터 다시 돌릴 필요가 없게 하기 위함."""
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.run_config import RunConfig
    from ragas.testset import Testset, TestsetGenerator

    if TESTSET_CACHE_PATH.is_file():
        print(f"  기존에 생성된 테스트셋을 재사용합니다: {TESTSET_CACHE_PATH}")
        with open(TESTSET_CACHE_PATH, encoding="utf-8") as f:
            return Testset.from_list(json.load(f))

    from _3_embedding import get_embedding_function
    from _5_search import get_llm

    # 생성용 LLM/임베딩도 우리가 이미 쓰고 있는 것(Gemini + ko-sroberta)을 그대로 재사용 —
    # 별도 API 키나 모델을 새로 준비할 필요가 없게 하기 위함.
    generator_llm = LangchainLLMWrapper(get_llm())
    generator_embeddings = LangchainEmbeddingsWrapper(get_embedding_function())

    # max_workers=1로 완전히 직렬화 — 무료 티어 분당 요청 제한에 병렬 호출로
    # 한 번에 걸리는 걸 막기 위함. 대신 넉넉한 재시도/대기로 일시적 제한 초과를 버틴다.
    run_config = RunConfig(timeout=300, max_retries=15, max_wait=90, max_workers=1)

    generator = TestsetGenerator(llm=generator_llm, embedding_model=generator_embeddings)
    gen_docs = _trim_docs_for_testset_generation(sampled_docs)
    print(
        f"  청크 {len(gen_docs)}개로 지식그래프 생성 중... (시간이 꽤 걸립니다, "
        f"'headlines' 버그 회피를 위해 문서당 {TESTSET_GEN_MAX_TOKENS}토큰 이하로 잘라서 투입)"
    )
    testset = generator.generate_with_langchain_docs(
        gen_docs, testset_size=TESTSET_SIZE, run_config=run_config
    )

    with open(TESTSET_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(testset.to_list(), f, ensure_ascii=False, indent=2)
    print(f"  생성된 테스트셋을 저장했습니다: {TESTSET_CACHE_PATH}")
    return testset


def _run_chain_on_testset(testset, chain):
    """RAGAS가 자동 생성한 질문 각각을 우리 체인에 실제로 태워서, 우리 파이프라인의
    실제 답변(response)과 실제 검색 결과(retrieved_contexts)를 얻는다. RAGAS가
    자체적으로 만든 reference(정답)와 비교해 채점하는 건 다음 단계다.
    질문마다 진행 상황을 파일에 누적 저장해서, 중간에 실패해도 이미 처리한
    질문은 다시 호출하지 않고 이어서 할 수 있게 한다."""
    raw_results = []
    if RAW_RESULTS_CACHE_PATH.is_file():
        with open(RAW_RESULTS_CACHE_PATH, encoding="utf-8") as f:
            raw_results = json.load(f)
    done_questions = {r["user_input"] for r in raw_results}

    samples = testset.to_list()
    for i, sample in enumerate(samples, 1):
        question = sample["user_input"]
        if question in done_questions:
            print(f"  ({i}/{len(samples)}) 이미 처리됨, 건너뜀: {question[:40]}")
            continue

        print(f"  ({i}/{len(samples)}) 체인 호출: {question[:40]}")
        try:
            result = chain.invoke(question)
            raw_results.append({
                "user_input": question,
                "reference": sample.get("reference", ""),
                "response": result["answer"],
                "retrieved_contexts": [d.page_content for d in result["_manual_docs"] + result["_law_docs"]],
            })
        except Exception as e:
            print(f"    ❌ 실패: {e}")
            raw_results.append({
                "user_input": question,
                "reference": sample.get("reference", ""),
                "response": f"(호출 실패: {e})",
                "retrieved_contexts": [],
            })

        with open(RAW_RESULTS_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(raw_results, f, ensure_ascii=False, indent=2)

        if i < len(samples):
            time.sleep(CALL_DELAY_SECONDS)

    return raw_results


def _score_with_ragas(raw_results):
    from ragas import EvaluationDataset, SingleTurnSample, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (
        Faithfulness,
        LLMContextPrecisionWithReference,
        LLMContextRecall,
        ResponseRelevancy,
    )
    from ragas.run_config import RunConfig

    from _3_embedding import get_embedding_function
    from _5_search import get_llm

    evaluator_llm = LangchainLLMWrapper(get_llm())
    evaluator_embeddings = LangchainEmbeddingsWrapper(get_embedding_function())
    run_config = RunConfig(timeout=300, max_retries=15, max_wait=90, max_workers=1)

    samples = [
        SingleTurnSample(
            user_input=r["user_input"],
            response=r["response"],
            retrieved_contexts=r["retrieved_contexts"] or ["(검색 결과 없음)"],
            reference=r["reference"],
        )
        for r in raw_results
    ]
    dataset = EvaluationDataset(samples=samples)

    metrics = [
        Faithfulness(llm=evaluator_llm),
        ResponseRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings),
        LLMContextPrecisionWithReference(llm=evaluator_llm),
        LLMContextRecall(llm=evaluator_llm),
    ]

    print(f"  RAGAS로 {len(samples)}개 채점 중... (지표 4개 x 질문 수만큼 LLM 호출)")
    result = evaluate(dataset=dataset, metrics=metrics, run_config=run_config)
    return result


def _format_report(raw_results, ragas_result) -> str:
    df = ragas_result.to_pandas()
    lines = ["# RAGAS 평가 결과 (최소 단위 파이프라인 증명)", ""]
    lines.append(
        f"샘플 청크(분뇨제거·환기점검 관련) 기반으로 RAGAS가 자동 생성한 "
        f"{len(raw_results)}개 질문에 대한 평가입니다. 전체 데이터셋 커버리지가 "
        f"아니라 파이프라인 동작 증명이 목적입니다."
    )
    lines.append("")

    lines.append("## 평균 점수")
    lines.append("")
    for col in df.columns:
        if df[col].dtype.kind in "fc":
            lines.append(f"- **{col}**: {df[col].mean():.3f}")
    lines.append("")

    lines.append("## 질문별 상세")
    lines.append("")
    for i, r in enumerate(raw_results, 1):
        lines.append(f"### {i}. {r['user_input']}")
        lines.append("")
        lines.append(f"**우리 체인 답변:** {r['response']}")
        lines.append("")
        lines.append(f"**RAGAS 참조 정답(자동 생성):** {r['reference']}")
        if i <= len(df):
            row = df.iloc[i - 1]
            scores = ", ".join(
                f"{col}={row[col]:.3f}" for col in df.columns if df[col].dtype.kind in "fc"
            )
            lines.append("")
            lines.append(f"**점수:** {scores}")
        lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    _check_dependencies()

    from _4_database import PERSIST_DIR
    from _5_search import build_chain, build_retrievers, get_llm, load_vector_store

    if not PERSIST_DIR.is_dir():
        raise SystemExit("벡터DB가 없습니다. 먼저 python _4_database.py 를 실행하세요.")

    llm = get_llm()
    vectorstore = load_vector_store()

    print("[1/4] 관련 청크 샘플링 (분뇨제거·환기점검)")
    sampled_docs = _sample_relevant_chunks(vectorstore)
    print(f"  {len(sampled_docs)}개 청크 샘플링 완료")

    print("\n[2/4] RAGAS 테스트셋 생성(또는 캐시 재사용)")
    testset = _generate_or_load_testset(sampled_docs)

    print("\n[3/4] 생성된 질문을 우리 체인에 태워서 실제 답변 얻기")
    manual_retriever, law_retriever = build_retrievers(vectorstore)
    chain = build_chain(manual_retriever, law_retriever, llm)
    raw_results = _run_chain_on_testset(testset, chain)

    print("\n[4/4] RAGAS로 채점")
    ragas_result = _score_with_ragas(raw_results)

    report = _format_report(raw_results, ragas_result)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"\n완료 — 결과를 {REPORT_PATH} 에 저장했습니다.")
