# D 파트 구조 설계와 선택 근거

작성 목적: 공식 익산시 정보공개 자료가 도착하기 전에 D의 책임과 A/B/C 연결 계약을
고정하고, 자료가 오면 화면을 다시 만들지 않고 provider만 교체하기 위함이다.

## 1. 결론

D의 중심은 새 `app/`이 아니라 기존 `demo/agents/`다.

- `agents/`: 도구 호출, 작업 계획, 근거 결합, 알림 초안, 선택적 LLM 설명
- `app/`: `agents/`를 호출해 결과를 보여주고 농장주 승인 상태를 관리하는 얇은 UI
- `ops/`, `rag/`, 향후 센서 adapter: 데이터를 제공하되 D의 화면을 직접 알지 않음

따라서 A/B/C 내부 파일 형식이 바뀌어도 `DecisionProvider` 계약만 맞추면 D의 에이전트와
화면 흐름은 유지된다.

## 2. 왜 기존 `agents/`를 발전시키는가

### 유지할 가치가 있는 기존 자산

1. `demo/README.md`가 이미 D의 공식 진입점을 `agents/`로 지정한다.
2. `demo/demo.py::step6_agents`가 `agents.work_guide`와 `agents.notify_draft`를 호출한다.
3. `agents/tools_schema.py`에는 계획서의 도구 6종 이름이 이미 정의되어 있다.
4. `scoring/s5_recommend.py`에는 기존 6시간 창 계산식, 작업 가중치, 저장 가중치가 있다.
5. `rag/search.py::RagIndex.search`에는 C 담당자의 거절 규칙과 출처 메타데이터가 있다.

이 자산을 두고 별도 서비스 로직을 `app/`에 다시 만들면 진입점과 계산식이 두 벌이 된다.
따라서 기존 공개 함수 `work_guide.run`, `notify_draft.run`, `LocalTools`는 호환 진입점으로
남기고 내부 구현을 계약 기반으로 교체했다.

### 그대로 둘 수 없었던 문제와 조치

| 기존 코드 근거 | 문제 | 이번 조치 |
| --- | --- | --- |
| `work_guide.py::run` | `ANTHROPIC_API_KEY`가 있어도 실제 도구 호출 루프가 구현되지 않음 | 기본 계획은 코드가 확정하고, 선택적 OpenAI 설명을 `openai_explainer.py`로 분리 |
| `work_guide.py::_offline_guide` | 공식 자료 도착 전인데 “익산 6년 실측”을 고정 문구로 주장 | provider 출처와 한계만 표시하고 해당 주장을 제거 |
| `tools_schema.py::get_forecast`, `get_plume_assessment` | 전달된 `farm_id` 대신 전역 `DEMO_FARM`과 mock 사용 | `LegacyProvider`가 농가 설정을 먼저 조회하고 mock 사용 여부를 source에 표시 |
| `notify_draft.py::run` | 농장주 승인 전에 receptor 삭제·삽입과 notification_log 기록 | `create_draft`는 순수 초안만 반환하고 승인은 Streamlit 세션에만 기록 |
| 기존 자유 문자열·dict | 화면이 출처, 오류, 갱신시각, fixture 여부를 안정적으로 구분하기 어려움 | `contracts.py`와 `{status,data,source,error}` 응답 규격 추가 |
| 기존 도구 6종 | 익산악취24 측정소 지도 입력이 없음 | `DecisionProvider.get_sensor_snapshot`을 별도 관측 포트로 추가 |

## 3. 최종 폴더 책임

```text
demo/
├─ agents/                       # D의 핵심
│  ├─ contracts.py               # GuideCard, NotificationDraft, 센서 관측 계약
│  ├─ provider.py                # DecisionProvider + factory
│  ├─ scoring_policy.py          # 기존 S5의 작업·저장일 가중치만 분리한 순수 정책
│  ├─ fixture_provider.py        # 공식 자료 대기 중인 명시적 fixture
│  ├─ legacy_provider.py         # 기존 B/C/legacy adapter
│  ├─ tools_schema.py            # 도구 6종 + 허용 목록 dispatch
│  ├─ work_guide.py              # 6시간 추천/회피와 근거 결합
│  ├─ notify_draft.py            # 승인 전 초안, 실발송 없음
│  └─ openai_explainer.py        # 선택적 설명만 담당
├─ app/
│  └─ dashboard.py               # 지도·캘린더·계획·근거·승인 UI
└─ docs/D_ARCHITECTURE.md        # 이 문서
```

`app/`을 완전히 없애지 않은 이유는 Streamlit 실행 파일과 에이전트 도메인 로직을
분리하기 위해서다. 화면은 반복 실행과 세션 상태라는 UI 특성이 있지만, 작업 추천은
일반 Python 함수로도 테스트·재사용되어야 한다.

## 4. 데이터 흐름과 계산 책임

1. 센서 지도: `provider.get_sensor_snapshot()` → 관측지도·상세표
2. 미래 위험: B `risk_calendar` → `provider.get_risk_calendar()`
3. 작업 계획: `agents.work_guide.plan_work()` → 추천 Top 3·회피 Top 3
4. 행동 근거: C `RagIndex.search()` → 문서명·단원·쪽수·snippet
5. 알림 초안: 작업 창 확정 → 플룸 후보를 참고로만 조회 → 편집 가능한 문구
6. 사람 승인: Streamlit 세션에 승인 로그만 기록, 외부 발송 없음

추천 순위와 등급은 LLM이 만들지 않는다. B 위험값과 기존 S5 가중식을 코드가 계산한다.
플룸은 `affects_risk_grade=False`로 유지하며 추천 점수에 곱하지 않는다.

## 5. provider 교체 규칙

`DecisionProvider`는 Python `typing.Protocol`로 정의했다. 구현체가 특정 부모 클래스를
상속하지 않아도 동일 메서드 계약을 구현하면 fixture·legacy·실데이터 adapter를 바꿀 수
있다. 화면은 구현체를 직접 import하지 않고 factory만 호출한다.

- 기본: `D_PROVIDER_MODE=fixture`
- 기존 팀 데모 연결: `D_PROVIDER_MODE=legacy`
- 실제 adapter: `D_PROVIDER_FACTORY=package.module:create_provider`

외부 factory는 `storage_days`와 `rag_index` 키워드 인자를 받고 `DecisionProvider`
구현을 반환한다.

실제 factory 로딩이나 호출이 실패하면 fixture로 자동 전환하지 않는다. 공식 결과처럼
보이는 가짜 값이 섞이지 않도록 화면에 오류를 노출한다.

## 6. 익산시 자료 도착 후 처리

“파일만 복사”하는 것이 아니라 아래 검증 후 adapter를 연결한다.

1. 원본 보존: 파일명, 기간, 시작·종료일, 행 수, 해시 기록
2. 코드북 확인: 측정주기, 시간대(KST), 결측·점검 코드, 검출한계, 복합악취 의미·단위
3. 측정소 기준정보: 공식 ID·명칭·좌표계·이전 이력 매핑
4. canonical 변환: `SensorObservation`으로 열 이름과 단위 변환
5. A/B 재검증: 민원 라벨 결합 가능성, 학습/검증 분할, 모델·등급 기준 재산출
6. provider 연결: 센서 → A/B → C 순서로 붙이고 source를 `connected`로 변경
7. D 회귀검증: 같은 화면에서 빈 자료, 좌표 결측, 지연 자료, C 검색 거절을 확인

관측시각(`observed_at`), 수집시각(`ingested_at`), 예보 발표시각
(`forecast_issued_at`), 예측 유효시각(`valid_at`)은 서로 대체하지 않는다.

## 7. 외부 설계 근거

- Python 공식 `typing.Protocol`: 명시적 상속 없이 동일 메서드 구조를 가진 구현을
  교체할 수 있는 구조적 서브타이핑의 근거.
  <https://docs.python.org/3/library/typing.html#typing.Protocol>
- Streamlit 공식 실행 모델: 위젯 상호작용마다 스크립트가 위에서 아래로 재실행되므로,
  계획·선택 창·승인 초안은 `st.session_state`로 관리하고 입력/데이터 버전이 바뀌면
  무효화해야 한다.
  <https://docs.streamlit.io/develop/concepts/architecture/session-state>
- OpenAI 공식 API 문서: Responses API에 함수 도구를 연결할 수 있으나, 이 프로젝트는
  모델을 결정권자로 쓰지 않고 확정된 계산 결과의 설명기로만 제한한다.
  <https://platform.openai.com/docs/guides/function-calling>

외부 문서는 구현 수단을 선택한 근거이며, 민원 위험 가중치나 모델 성능의 근거가 아니다.
가중치·성능·복합악취 단위는 익산시 원자료와 팀 검증으로 별도 확정해야 한다.
