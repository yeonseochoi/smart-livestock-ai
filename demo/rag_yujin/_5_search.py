"""
검색 및 답변 — 양돈장 축산악취 작업유형별 RAG용

이 단계에서 하는 일 (검색기 + 체인, 한 세트):
  1. Chroma 벡터스토어를 "검색기(Retriever)" 2개로 감싼다 — 실무 요령용(manual),
     법령용(law). doc_type으로 필터링해서, 한쪽 문서 유형이 우연히 유사도 점수가
     높아 다른 한쪽이 결과에서 아예 빠지는 일이 없게 각각 최소 K개씩 보장한다.
  2. 그 검색기 2개 + Gemini LLM을 LCEL로 엮어 "체인(Chain)"을 만든다.
     질문 → (실무 검색 + 법령 검색) → 프롬프트 구성 → Gemini 호출 → 답변 생성,
     이 전체 흐름이 chain.invoke(질문) 한 번으로 실행된다.

LLM은 Gemini API 무료 티어를 쓴다(유진님 선택). 답변은 자료에 있는 내용만 근거로
생성하도록 프롬프트에 명시했고, 실제로 어떤 청크를 근거로 썼는지도 항상 같이
보여줘서(답변 생성 대신 검색된 원문을 안 보여주면 나중에 잘못된 답이어도 확인할
방법이 없다) LLM이 자료에 없는 걸 지어냈는지 사람이 바로 확인할 수 있게 했다.

실행 전 준비:
  1) python -m pip install langchain-google-genai python-dotenv
  2) https://aistudio.google.com/apikey 에서 무료로 API 키 발급
  3) rag_yujin 폴더에 ".env" 파일을 만들고 아래 한 줄만 적기 (매번 터미널에서
     환경변수를 다시 설정할 필요 없이, 이 파일에서 자동으로 읽어온다)
       GOOGLE_API_KEY=발급받은키
     ⚠ .env는 절대 git에 커밋하지 말 것 — 저장소 .gitignore에 ".env"가 없다면
       꼭 추가해야 한다(안 그러면 키가 깃 기록에 그대로 남는다).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CURRENT_DIR))


def _load_env_file() -> None:
    """rag_yujin/.env 파일이 있으면 거기서 GOOGLE_API_KEY 등을 읽어온다.
    python-dotenv가 없으면 조용히 건너뛰되, .env 파일이 실제로 있는데 못 읽는
    상황(패키지 안 깔림)만 콕 집어 알려준다 — .env가 아예 없으면(환경변수를 직접
    설정해서 쓰는 경우) 안내가 필요 없으므로 아무 말도 안 한다."""
    env_path = CURRENT_DIR / ".env"
    try:
        from dotenv import load_dotenv
    except ImportError:
        if env_path.is_file():
            print(
                "⚠ .env 파일은 있는데 python-dotenv가 설치되어 있지 않아 못 읽었습니다.\n"
                "  python -m pip install python-dotenv 로 설치하세요."
            )
        return
    load_dotenv(env_path)


_load_env_file()

# gemini-2.5-flash는 "신규 사용자에게는 더 이상 제공되지 않는 모델(404)"이라는 걸
# 유진님 실행 결과로 확인함. 공식 문서(ai.google.dev/gemini-api/docs/models,
# .../docs/pricing) 기준 gemini-3.5-flash로 교체 — 다만 이것도 구글이 무료 티어
# 구성을 바꾸면 언제든 다시 막힐 수 있다. 그때는 코드를 고칠 필요 없이 .env에
# GEMINI_MODEL_NAME=원하는모델명 한 줄만 추가하면 된다(위 두 문서에서 현재 사용
# 가능한 모델명을 확인할 수 있음).
GEMINI_MODEL_NAME = os.environ.get("GEMINI_MODEL_NAME", "gemini-3.5-flash")


def _check_dependencies() -> None:
    missing = []
    try:
        import langchain_chroma  # noqa: F401
    except ImportError:
        missing.append("langchain-chroma")
    if missing:
        raise ImportError(
            "다음 패키지가 설치되어 있지 않습니다: " + ", ".join(missing) + "\n"
            f"  python -m pip install {' '.join(missing)}"
        )


def _check_llm_dependencies() -> None:
    missing = []
    try:
        import langchain_google_genai  # noqa: F401
    except ImportError:
        missing.append("langchain-google-genai")
    if missing:
        raise ImportError(
            "다음 패키지가 설치되어 있지 않습니다: " + ", ".join(missing) + "\n"
            f"  python -m pip install {' '.join(missing)}"
        )


def load_vector_store():
    """_4_database.py가 만들어둔 Chroma DB를 그대로 연다(재임베딩하지 않음).
    같은 collection_name/persist_directory/embedding_function을 써야 벡터 공간이
    맞으므로, 값을 여기 새로 적지 않고 _3_embedding.py / _4_database.py의
    상수·함수를 그대로 가져다 쓴다."""
    _check_dependencies()
    from langchain_chroma import Chroma

    from _3_embedding import get_embedding_function
    from _4_database import COLLECTION_NAME, PERSIST_DIR

    if not PERSIST_DIR.is_dir():
        raise FileNotFoundError(
            f"벡터DB가 없습니다: {PERSIST_DIR}\n"
            "먼저 python _4_database.py 를 실행해서 DB를 만드세요."
        )

    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embedding_function(),
        persist_directory=str(PERSIST_DIR),
    )


def build_retrievers(vectorstore, k_manual: int = 3, k_law: int = 3):
    """doc_type으로 필터링된 검색기(Retriever) 2개를 만들어 반환한다: (manual_retriever, law_retriever).
    LangChain 표준 Retriever(BaseRetriever)라서 .invoke(질문)으로 바로 검색하거나,
    나중에 LCEL 체인에 그대로 끼워 넣을 수 있다."""
    manual_retriever = vectorstore.as_retriever(
        search_kwargs={"k": k_manual, "filter": {"doc_type": "manual"}}
    )
    law_retriever = vectorstore.as_retriever(
        search_kwargs={"k": k_law, "filter": {"doc_type": "law"}}
    )
    return manual_retriever, law_retriever


def get_context_for_query(manual_retriever, law_retriever, query: str) -> dict:
    """실무 요령(manual)과 관련 법령(law)을 각각 검색해서 딕셔너리로 반환한다.
    나중에 이 반환값을 그대로 LLM 프롬프트의 context로 넘기면 된다."""
    return {
        "query": query,
        "manual": manual_retriever.invoke(query),
        "law": law_retriever.invoke(query),
    }


def format_answer(context: dict) -> str:
    """검색 결과를 사람이 읽기 좋은 텍스트로 정리한다(추출형 — LLM 없이 원문 그대로 보여줌)."""
    lines = [f"질문: {context['query']}", ""]

    lines.append("[관련 실무 요령 — 이렇게 하면 악취가 덜 납니다]")
    if context["manual"]:
        for i, d in enumerate(context["manual"], 1):
            m = d.metadata
            preview = d.page_content.strip().replace("\n", " ")[:150]
            lines.append(f"  {i}. ({m['source_file']} / {m['unit']} / p.{m['page']})")
            lines.append(f"     {preview}...")
    else:
        lines.append("  (관련 실무 자료를 찾지 못했습니다)")

    lines.append("")
    lines.append("[지켜야 하는 관련 법령]")
    if context["law"]:
        for i, d in enumerate(context["law"], 1):
            m = d.metadata
            preview = d.page_content.strip().replace("\n", " ")[:150]
            lines.append(f"  {i}. ({m['source_file']} / {m['unit']})")
            lines.append(f"     {preview}...")
    else:
        lines.append("  (관련 법령을 찾지 못했습니다)")

    return "\n".join(lines)


def search(query: str, manual_retriever=None, law_retriever=None, k_manual: int = 3, k_law: int = 3) -> str:
    """스크립트 밖에서(예: 파이썬 인터프리터) 질문 하나만 넣고 바로 정리된 답을 받고 싶을 때 쓰는 편의 함수.
    LLM 없이 검색된 원문만 보여준다(체인을 쓰려면 ask()를 대신 쓸 것)."""
    if manual_retriever is None or law_retriever is None:
        vectorstore = load_vector_store()
        manual_retriever, law_retriever = build_retrievers(vectorstore, k_manual=k_manual, k_law=k_law)
    context = get_context_for_query(manual_retriever, law_retriever, query)
    return format_answer(context)


# ── 체인(Chain) — 검색기 2개 + Gemini LLM ─────────────────────────
def get_llm(temperature: float = 0.2):
    """Gemini LLM 객체를 만든다. API 키가 없으면 (반복 재시도로 에러를 도배하지 않고)
    여기서 한 번에 무엇이 없는지, 어떻게 해결하는지 알려주고 바로 멈춘다."""
    _check_llm_dependencies()
    from langchain_google_genai import ChatGoogleGenerativeAI

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GOOGLE_API_KEY가 설정되어 있지 않습니다.\n"
            "1) https://aistudio.google.com/apikey 에서 무료로 키를 발급받으세요.\n"
            "2) rag_yujin 폴더에 .env 파일을 만들고 아래 한 줄을 적으세요(권장 — 한 번만 하면 됨):\n"
            "     GOOGLE_API_KEY=발급받은키\n"
            "   (python -m pip install python-dotenv 가 먼저 설치되어 있어야 합니다)\n"
            "   .env 대신 터미널에서 그때그때 설정해도 됩니다(터미널을 새로 열 때마다 다시 설정해야 함):\n"
            '     PowerShell: $env:GOOGLE_API_KEY="발급받은키"\n'
            "     cmd:        set GOOGLE_API_KEY=발급받은키"
        )
    return ChatGoogleGenerativeAI(model=GEMINI_MODEL_NAME, temperature=temperature, google_api_key=api_key)


def _format_docs_for_prompt(docs, label: str) -> str:
    """검색된 청크를 LLM 프롬프트에 넣을 형태로 정리한다. [실무-1]처럼 번호를 매겨서
    LLM이 답변에 그 번호로 근거를 표시하게 하고, 나중에 사람이 검증할 때도 같은
    번호로 원문을 대조할 수 있게 한다."""
    if not docs:
        return f"({label} 관련 자료를 찾지 못함)"
    parts = []
    for i, d in enumerate(docs, 1):
        m = d.metadata
        parts.append(
            f"[{label}-{i}] 출처: {m['source_file']} / {m['unit']} (p.{m.get('page', '?')})\n"
            f"{d.page_content.strip()}"
        )
    return "\n\n".join(parts)


def _build_prompt():
    from langchain_core.prompts import ChatPromptTemplate

    # 섹션(원칙/형식/유의사항)으로 나눠 쓴 이유: LLM에게 "근거 기반 답변 + 정해진 형식"을
    # 한 문단으로 뭉쳐 지시하면 형식 지시가 묻혀서 무시되는 경우가 있다. 라벨을 나눠두면
    # 지시를 따르는 정확도가 올라간다(실제 프롬프트 엔지니어링에서 자주 쓰는 방식).
    system_prompt = (
        "당신은 양돈장(돼지 축사) 현장 관리자가 악취를 줄이도록 돕는 실무 도우미입니다. "
        "답변을 볼 사람은 전문 연구자가 아니라 현장에서 바로 참고할 실무자이므로, "
        "불필요한 전문 용어 설명 없이 구체적이고 실행 가능한 조언을 주세요.\n\n"
        "[답변 원칙]\n"
        "- 아래 제공된 '실무 자료'와 '법령 자료'에 있는 내용만 근거로 답변하세요. 자료에 없는 "
        "내용은 절대 지어내지 마세요.\n"
        "- 실무 자료나 법령 자료가 부족하면(예: '관련 자료를 찾지 못함'으로 표시된 경우) 그 사실을 "
        "그대로 알리고, 억지로 답을 만들어내지 마세요.\n"
        "- 법령 조문의 수치(배출허용기준, 과태료 금액, 기한 등)는 원문 그대로 정확히 인용하세요. "
        "어림잡거나 바꿔 쓰지 마세요.\n"
        "- 질문과 직접 관련 없는 내용은 제공된 자료에 있어도 답변에 넣지 마세요.\n\n"
        "[답변 형식]\n"
        "반드시 다음 두 섹션으로 나눠 한국어로, 실행 가능한 항목(불릿) 단위로 답변하세요:\n"
        "1) 실무 요령 — 악취를 줄이는 구체적인 작업 방법\n"
        "2) 관련 법령 — 지켜야 하는 법적 기준·의무(위반 시 불이익이 자료에 있으면 함께 명시)\n"
        "각 항목 뒤에는 근거 자료를 [실무-1]처럼 표시하세요. 제공된 자료 번호 외의 번호는 "
        "절대 만들어내지 마세요.\n\n"
        "[유의사항]\n"
        "이 답변은 참고용 정보이며, 실제 법적 의무 이행이나 시설 투자 전에는 관할 지자체나 "
        "전문가 확인을 권장한다는 점을 답변 마지막에 한 줄로 덧붙이세요."
    )

    return ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            (
                "human",
                "질문: {query}\n\n[실무 자료]\n{manual_context}\n\n[법령 자료]\n{law_context}",
            ),
        ]
    )


def build_chain(manual_retriever, law_retriever, llm):
    """검색기 2개 + LLM을 LCEL로 엮은 RAG 체인.
    chain.invoke(질문)을 하면 검색 → 프롬프트 구성 → LLM 호출까지 한 번에 실행되고,
    결과 딕셔너리에는 답변 텍스트("answer")뿐 아니라 실제로 근거로 쓴 원문 청크
    ("_manual_docs", "_law_docs")도 같이 담겨서, 답변을 그대로 믿지 않고 원문과
    대조해볼 수 있다."""
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnableLambda, RunnablePassthrough

    def retrieve_context(query: str) -> dict:
        manual_docs = manual_retriever.invoke(query)
        law_docs = law_retriever.invoke(query)
        return {
            "query": query,
            "manual_context": _format_docs_for_prompt(manual_docs, "실무"),
            "law_context": _format_docs_for_prompt(law_docs, "법령"),
            "_manual_docs": manual_docs,
            "_law_docs": law_docs,
        }

    generate_answer = _build_prompt() | llm | StrOutputParser()

    return RunnableLambda(retrieve_context) | RunnablePassthrough.assign(answer=generate_answer)


def _preview_content(doc, max_len: int = 120) -> str:
    """청크 원문 앞부분을 미리보기로 잘라서 반환한다. 페이지 번호만으로는 검색된
    자료가 질문과 실제로 관련 있는지 바로 판단하기 어려워서, 출처 표시에 내용
    일부를 같이 보여주기 위해 쓴다."""
    text = doc.page_content.strip().replace("\n", " ")
    return text[:max_len] + ("..." if len(text) > max_len else "")


def format_final_answer(result: dict) -> str:
    """체인 결과(LLM이 생성한 답변 + 근거 청크)를 사람이 읽기 좋게 정리한다.
    출처 표시는 페이지 번호 대신 실제 청크 내용의 앞부분(미리보기)을 보여주고,
    어느 매뉴얼/법령 파일·조문인지는 괄호로 표시한다 — 페이지 번호만으론 검색된
    자료가 질문과 실제로 관련 있는지 한눈에 판단하기 어렵기 때문이다."""
    lines = [f"질문: {result['query']}", "", result["answer"].strip(), "", "[근거로 쓰인 원문 자료]"]

    lines.append(" 실무:")
    if result["_manual_docs"]:
        for i, d in enumerate(result["_manual_docs"], 1):
            m = d.metadata
            lines.append(f'  [실무-{i}] "{_preview_content(d)}" ({m["source_file"]} / {m["unit"]})')
    else:
        lines.append("  (없음)")

    lines.append(" 법령:")
    if result["_law_docs"]:
        for i, d in enumerate(result["_law_docs"], 1):
            m = d.metadata
            lines.append(f'  [법령-{i}] "{_preview_content(d)}" ({m["source_file"]} / {m["unit"]})')
    else:
        lines.append("  (없음)")

    return "\n".join(lines)


def ask(query: str, manual_retriever=None, law_retriever=None, llm=None, k_manual: int = 3, k_law: int = 3) -> str:
    """스크립트 밖에서 질문 하나만 넣고 Gemini가 생성한 답변(+근거 자료)을 바로 받고 싶을 때 쓰는 편의 함수."""
    if manual_retriever is None or law_retriever is None:
        vectorstore = load_vector_store()
        manual_retriever, law_retriever = build_retrievers(vectorstore, k_manual=k_manual, k_law=k_law)
    if llm is None:
        llm = get_llm()
    chain = build_chain(manual_retriever, law_retriever, llm)
    result = chain.invoke(query)
    return format_final_answer(result)


if __name__ == "__main__":
    from _3_embedding import WORKTYPE_QUERIES

    # LLM 키/패키지 문제는 반복 호출 전에 한 번에 확인한다(4번 도는 루프 안에서
    # 매번 같은 에러가 반복 출력되는 걸 방지 — _1_loader.py에서 이미 겪었던 문제).
    llm = get_llm()

    vectorstore = load_vector_store()
    manual_retriever, law_retriever = build_retrievers(vectorstore)
    chain = build_chain(manual_retriever, law_retriever, llm)

    print(f"[작업유형별 RAG 답변 테스트] (모델: {GEMINI_MODEL_NAME})\n")
    for worktype, query in WORKTYPE_QUERIES.items():
        print("=" * 60)
        print(f"작업유형: {worktype}")
        result = chain.invoke(query)
        print(format_final_answer(result))
        print()
