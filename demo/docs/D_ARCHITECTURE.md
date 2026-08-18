# D 작업 가이드 에이전트 구조

## 1. 현재 단계의 책임

D는 `serving.db`의 위험도와 PR #9 RAG 근거를 결합해 작업 가이드를 만든다.

1. 애플리케이션 코드가 provider 도구를 호출한다.
2. `work_guide.py`가 추천·회피 시간, 점수와 등급을 확정한다.
3. Gemini는 선택적으로 확정 결과를 현장 작업자가 읽기 쉽게 설명한다.
4. Gemini API가 없거나 실패해도 확정 작업 가이드는 동일하게 출력한다.

LLM은 추천 시간, 점수와 등급의 결정권자가 아니다.

## 2. 파일 책임

```text
demo/agents/
├─ contracts.py          # GuideCard, WorkWindow, 출처·센서 계약
├─ provider.py           # DecisionProvider와 provider factory
├─ scoring_policy.py     # 시간·작업유형·저장일 가중치
├─ fixture_provider.py   # 명시적 고정 시연 데이터
├─ legacy_provider.py    # serving.db와 RAG adapter 연결
├─ rag_adapter.py        # rag_yujin Document를 구조화 근거로 변환
├─ tools_schema.py       # 내부 도구 4종과 안전한 dispatch
├─ work_guide.py         # 6시간 추천·회피 계획 확정
└─ gemini_explainer.py   # 확정 계획의 선택적 설명
```

## 3. 위험도 계산

`risk_hourly`에는 농촌근거리와 시가지원거리 위험도가 따로 저장된다.
`LegacyProvider`는 같은 시각의 두 값 중 큰 값을 선택한다. 확률끼리 곱하지 않는다.

단기 구간은 1시간 위험값을 그대로 반환한다. 작업 가이드는 연속된 6개 값에
`TIME_WEIGHTS = (0.30, 0.22, 0.17, 0.13, 0.10, 0.08)`을 직접 적용한다.
3시간 평균을 먼저 만들지 않는다. reduced 중기 구간은 일 단위로만 표시하고
시간 추천 후보에는 사용하지 않는다.

## 4. RAG 연결

`rag_adapter.py`는 PR #9의 다음 검색 흐름을 재사용한다.

```text
load_vector_store
→ build_retrievers(manual/law)
→ get_context_for_query
→ source_file, unit, page, snippet 구조로 변환
```

Gemini 답변 체인인 `rag_yujin.ask()`는 호출하지 않는다. RAG는 근거 검색만 담당하고,
작업 가이드 설명은 `gemini_explainer.py` 한 곳에서 생성한다. Chroma DB가 없으면
RAG 상태를 `unavailable`로 표시하며 작업 시간 계산은 계속할 수 있다.

## 5. 내부 도구와 향후 확장

현재 내부 도구는 다음 4개다.

- `get_risk_calendar`
- `get_storage_days`
- `search_rag`
- `get_farm_config`

현재는 애플리케이션 코드가 이 도구를 직접 호출한다. `TOOL_SPECS`와
`dispatch_tool()`은 공급자 중립 형태로 보존하므로, 추후 Gemini function calling을
도입할 때 같은 provider 메서드를 연결할 수 있다.

## 6. 실행 설정

- 기본 current main DB/RAG 연결: `D_PROVIDER_MODE=legacy`
- 고정 구조 검증: `D_PROVIDER_MODE=fixture`
- Gemini 설명: `GOOGLE_API_KEY` 설정
- Gemini 모델 교체: `GEMINI_MODEL_NAME` 설정

실제 provider 생성에 실패했을 때 fixture로 조용히 전환하지 않는다. 데이터 출처를
오인하지 않도록 오류와 `connected`, `fixture`, `unavailable` 상태를 그대로 노출한다.
