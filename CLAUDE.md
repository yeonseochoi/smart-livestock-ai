# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

익산시 축산 악취 민원 저감 의사결정 시스템. 기상 예보로 "냄새 나는 작업을 언제 하면
민원이 덜 날까"를 시각 단위로 답한다.

**이 저장소에서는 한국어로 쓴다** — 답변, 코드 주석, 커밋 메시지, 문서 전부.

## 절대 규칙

위반하면 조용히 틀린 결과가 나온다. 전부 실제로 사고가 난 뒤에 생긴 규칙이다.

1. **ML(XGBoost)과 플룸(가우시안)을 곱하지 않는다.** 바람과 주야간이 양쪽에 다 들어가
   있어 곱하면 이중 계산이다. 플룸은 등급을 올리거나 내리지 않고, ML이 매긴 점수 중
   **어느 수용점 유형을 볼지 고르는 데만** 쓴다. `config.PLUME_GRADE_BUMP = False` 는
   영구 OFF이며 켜지 않는다.
2. **서빙 피처는 기상청 예보 API가 주는 변수만 쓴다.** NH3·CO2 를 넣는 순간 "학습은
   되는데 예측은 못 하는" 모델이 된다. 암모니아는 예보가 존재하지 않는다.
3. **`demo/legacy/` 는 수정하지 않는다.** import 만 한다 — 아래 "legacy 는 죽은 코드가
   아니다" 참조. `run_check.py` 가 원본과 SHA-256 을 대조한다.
4. **법령 하드필터 → ML 랭킹 → 플룸 선택** 순서를 뒤집지 않는다. 법이 먼저 자른다.
   반대로 하면 "법적으로 불가능한 시각이 A등급"이 된다.
5. **민원 데이터는 익산 필터 후 사용한다.** 원본에 완주군이 섞여 있다. 기대 행 수는
   `config.EXPECT_*` 상수에 있고 다르면 필터가 누락된 것이다.
6. **근거 없이 정한 상수엔 `[C]` 주석을 붙인다.** 좌표 상수 하나가 3.35km 어긋나
   검증이 통째로 뒤집힐 뻔한 적이 있다. 이 규약 덕에 잡았다.
7. **판단은 코드, 설명은 RAG·LLM.** Gemini 는 추천 시각과 점수를 바꾸지 않는다.
   LLM 이 틀려도 등급은 안 바뀌어야 한다.

## 현행 / 옛날 구분

`archive/` 와 `rag/` 는 과거 산출물이다. **`legacy/` 는 아니다.**

| 구분 | 대상 |
| --- | --- |
| **현행** | `run_train.py` · `run_serve.py` · `run_check.py`, `preprocess/` `model/` `serving/` `advisor/` `analysis/` `agents/` `app/` `rag_yujin/` (`pgvector_store.py` 포함) |
| **옛날 — 건드리지 않음** | `archive/demo_v2~v5.py` 등 v2~v5 검증 라운드, **`rag/`** (구 RAG 구현) |

- **현행 RAG 는 `rag_yujin/` 이다.** `rag/` 를 import 하는 것은 `archive/demo*.py` 뿐이며,
  현역 코드에서는 `agents/rag_adapter.py` 가 `rag_yujin._5_search` 를 쓴다.
  RAG 작업은 `rag_yujin/` 에서 한다.
- `archive/` 는 발표·보고서 수치가 나온 코드라 보존한다. 파일명(`v2`,`v3`…)도 바꾸지
  않는다 — 바꾸면 보고서와 대조가 안 된다.

### legacy/ 는 죽은 코드가 아니다

이름과 달리 **현역 코드 20곳 이상이 import 중이다** — `advisor/recommend.py`,
`serving/daily_scoring.py`, `serving/kma_midterm.py`, `analysis/plume_select.py`,
`analysis/figures.py`, `agents/notify_draft.py`, `run_*.py` 전부.

평면 import 구조(`from geo import ...`)라 `config.py` 가 `sys.path` 에 등록해 준다.
그래서 `from legacy.geo import ...` 가 아니라 `from geo import ...` 로 쓴다.
삭제·이동·리팩터링 대상이 아니며, 고칠 것이 있으면 호출부에서 감싼다.

## 실행

전부 `demo/` 디렉터리에서 실행한다. 데이터 경로를 `demo/` 의 부모에서 찾는다.

```powershell
pip install -r demo/requirements.txt
cd demo

python run_train.py     # 학습 (약 3분) — data/*.parquet, *.pkl 생성
python run_serve.py     # 서빙 — 예보 받아 위험도 계산 + DB 적재
python run_check.py     # 회귀 점검 (8섹션 39항목)
```

**`run_train.py` 가 선행 조건이다.** `data/` 는 gitignore 대상이라 클론 직후 비어 있고,
그 상태로 `run_serve.py` / `run_check.py` 를 돌리면 parquet·pkl 이 없어 실패한다.
"파일이 없다"는 에러가 나면 먼저 `run_train.py` 를 돌렸는지 확인한다.

D 대시보드:
```powershell
pip install -r requirements-d.txt
streamlit run app/dashboard.py
```

## 테스트 — 파트별로 갈라져 있다

| 파트 | 명령 | 전제 조건 |
| --- | --- | --- |
| A/B (ML·플룸) | `python run_check.py` | `run_train.py` 선행 필수 |
| A/B 플룸 검증 | `python -m analysis.plume_validation <좌표.csv>` | 농가 좌표 CSV 인자 |
| C/D (RAG·Agent) | `python tests/test_d_agents.py` | 없음 — 네트워크·DB 불필요, 11 tests |
| D 스모크 | `python agents/_test_smoke.py` | `run_serve.py` 로 DB 를 채운 뒤 |
| C RAG 평가 | `rag_yujin/_6_1_Rule-based_test.py` · `_6_2_RAGAS_test.py` | `GOOGLE_API_KEY` + Chroma DB. 단위 테스트가 아니라 **LLM 호출 평가 스크립트**다 |

빠르게 돌릴 수 있는 것은 `python tests/test_d_agents.py` 하나뿐이다 (0.1초 미만).

**러너 관련 함정:**
- **`pytest` 는 설치돼 있지 않다** — 어느 requirements 에도 없다. `rag/test_rag.py` 는
  pytest 스타일이지만 구 `rag/` 모듈용이라 현행에서 돌릴 일이 없다.
  pytest 를 쓰려면 requirements 에 추가부터 해야 한다.
- **`python -m unittest discover` 는 실패한다** — `tests/` 에 `__init__.py` 가 없어
  `Start directory is not importable` 이 난다. 파일을 직접 실행한다.

## 환경 변수 / 키

`demo/.env` 에 둔다 (`.env.example` 복사). **`.env` 는 절대 커밋하지 않는다.**
키가 없으면 전부 mock 폴백으로 돌고, **폴백을 탔다는 사실이 콘솔에 반드시 찍힌다**
(`config.PROV`). 이 로깅을 없애지 않는다.

- **`KMA_KEY` 는 설정하지 말 것.** `serving/kma_midterm.py` 의 `_service_key()` 가
  환경변수 경로에만 `unquote()` 를 빠뜨려서, 설정하면 이중 인코딩으로 중기예보(D+4~7)가
  **조용히** 깨진다. 설정하지 않으면 파일 안의 폴백 키로 정상 동작한다.
  GitHub Actions Secret 에도 넣지 않는다.
- `DATABASE_URL` 있으면 PostgreSQL(Supabase), 없으면 `out/demo.db` SQLite 폴백.
  Supabase 는 반드시 **Session/Transaction pooler** 문자열을 쓴다 — Direct connection 은
  무료 티어에서 IPv6 전용이라 Actions 러너(IPv4)에서 붙지 않는다.
- `GOOGLE_API_KEY` 없으면 Gemini 설명 없이 동일한 확정 계획을 출력한다.
- `D_PROVIDER_MODE` — 기본 `legacy`(실 DB + RAG), `fixture`(고정 시연 데이터).
  연결 실패를 fixture 로 조용히 바꾸지 않는다.
- `RAG_BACKEND` — `pgvector`(Supabase `rag` 스키마) / `chroma`(로컬) / `auto`.
  두 백엔드는 같은 청킹·같은 임베딩이라 검색 결과가 같다. 적재는 각각
  `_4b_migrate_to_pg.py` · `_4_database.py`.

> ⚠️ **미해결**: `demo/legacy/kma.py` 에 기상청 폴백 서비스키가 하드코딩돼 있다.
> 키 폐기·재발급이 필요하다. (README 는 "저장소가 공개 상태"라고 적었지만
> `gh repo view` 확인 결과 **PRIVATE** 이다. 긴급도는 문서보다 낮다.) 절대 규칙 3 때문에 파일을 직접
> 고칠 수 없으므로 배포 사본에서 치환하거나 별도 조치를 취한다.

## Windows 환경

- **콘솔이 cp949 다.** 이모지나 em dash(—)를 그냥 `print` 하면 `UnicodeEncodeError` 로
  죽는다. 진입점에서 `from console import use_utf8_stdout; use_utf8_stdout()` 를
  호출한다 (기존 스크립트가 다 그렇게 한다).
- **저장소 경로에 한글과 공백이 있다.** 셸에서 경로는 반드시 인용하고, CI 에서는
  `hashFiles` 로 캐시 키를 잡을 수 없어 `DATA_VERSION` 수동 버전을 쓴다
  (`.github/workflows/daily-serve.yml`).

## 코드 규약

- Python 3.10 (CI 도 3.10). `requirements.txt` 는 검증된 버전으로 **핀 고정**이며
  임의로 올리지 않는다. `shap`-`xgboost` 조합과 `langchain-community<0.4` 는 버전 민감이다.
- 랜덤 시드는 `config.SEED` (42) 로 고정한다.
- 경로 상수는 `demo/config.py` 에 모은다. 파일 안에서 경로를 새로 만들지 않는다.
- 중간 산출물(`data/`), `*.db` `*.pkl` `*.parquet`, `rag_yujin/data/chroma_db/` 는
  커밋하지 않는다. 전부 위 세 명령으로 재생성된다.
- `app/dashboard.py` 에는 모델·추천·RAG 로직을 두지 않는다. Streamlit 화면 표시와
  세션 상태만 담당한다. 자료가 바뀌면 `agents/` 의 provider 만 교체한다.

## 수치의 출처

**문서와 파일이 다르면 파일이 옳다.** 성능표를 손으로 옮겨 적다 틀린 전례가 있어
지표를 코드로 옮겼다. `run_train.py` 가 `demo/out/training_results.json` 에 기록하며,
README 표는 구 데이터셋 기준이라 갱신되지 않은 상태다.

## Git

- 브랜치: `feat/` · `fix/` · `refactor/` · `docs/` · `chore/` · `agent/` 접두사.
- 커밋: Conventional Commits 한국어 — `feat(db): 서빙 DB를 PostgreSQL로 이관`.
- `main` 직접 푸시하지 않고 PR 로 머지한다.

## 더 읽을 것

필요할 때만 연다. CLAUDE.md 에 옮겨 적지 않는다.

- @README.md — 프로젝트 개요, 알려진 한계 5가지, 남은 작업
- @demo/README.md — 설치·실행·파일 구조·성능표·파이프라인 흐름도·파트별 진입점
- @docs/PIPELINE.md — 단계별 설계 근거 ("왜 1시간 격자인가")
- @demo/docs/D_ARCHITECTURE.md — D 에이전트 provider 계약 설계
- @demo/archive/README.md — v2~v5 검증 라운드가 무엇이었나
