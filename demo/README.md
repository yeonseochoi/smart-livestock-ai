# 악취·분뇨 프로젝트 데모 (팀 공유용)

> ⚠️ **공개 저장소 업로드 금지** — `legacy/kma.py` 안에 기상청 API **폴백 키가
> 하드코딩**되어 있다 (원본 액비 프로젝트에서 복사된 사본, 수정 금지 원칙 때문에
> 그대로 유지 중). GitHub 등 공개 저장소에 올리기 전에 반드시 해당 폴백 키를
> 제거하고 환경변수(`KMA_KEY`)로 전환할 것. 팀 내부 공유(드라이브·압축파일)만 허용.

`구현 내용.md`(구현 정리_0809)의 구현 순서 ①~⑦을 실데이터로 실행하고,
검증 라운드 v1~v5 를 수행한 코드다. 검증 결과는 `out/validation_report_final.md`
(및 라운드별 validation_report_*.md)에 있다.

## 실행법 — "프로젝트 데이터" 폴더를 옆에 두고

이 데모는 데이터 경로를 **demo 폴더의 부모에서** 찾는다. 폴더 배치가 이렇게
되어 있으면 바로 실행된다:

```
아무 폴더/
├── demo/                  ← 이 폴더 (압축 해제)
├── 프로젝트 데이터/          ← 실데이터 통합 폴더 (팀 드라이브에서 복사)
│   ├── 01_민원데이터/익산시 악취 민원 데이터.xlsx
│   ├── 02_기상데이터/weather_hourly_2020_202607.csv, aws_702_..., asos_146_...
│   ├── 03_RAG_법령매뉴얼/*.pdf (14종)
│   ├── 04_양돈센서_AIHub/validation_matched_sensor_30m.xlsx
│   └── 전북특별자치도 익산시_축산농가 현황_*.csv
└── 작업폴더/최종구현 py파일/   ← (선택) legacy 원본 — demo.py 가 무결성 해시 검증에 사용
```

```powershell
pip install -r requirements.txt
cd demo
python demo.py        # ①~⑦ 전체 실행 — data/·out/ 산출물이 여기서 재생성된다
```

배포본에서는 중간 산출물(data/)과 재생성 가능한 그래프·DB(out/ 일부)를 지워
두었다. **`python demo.py` 한 번이면 전부 다시 만들어진다** (약 3~5분).
그 다음 `python demo_v2.py` ~ `demo_v5.py` 로 검증 라운드도 재현 가능하다.

API 키는 전부 선택사항이다. 기존 파이프라인은 `KMA_KEY`(단기·중기예보)와
`VWORLD_KEY`(주거건물)를 사용한다. D의 설명 생성은 `OPENAI_API_KEY`와
`OPENAI_MODEL`이 모두 있을 때만 활성화되며, 없어도 규칙 기반 설명으로 끝까지 실행된다.

## 파트별 진입점

| 파트 | 담당 영역 | 코드 | 실행/재현 |
| --- | --- | --- | --- |
| **A** | 데이터·모델 | `etl/`(s0 정제→s1 라벨→s2 피처) → `model/s3_train.py` | `python demo.py` ①·④ 구간. 모델 실험은 `model/s3_train.py` 의 하이퍼파라미터·피처 리스트(`etl/s2_features.py`) 수정 후 demo.py 재실행 |
| **B** | 운영 배관 | `ops/` — db.py(SQLite 4테이블), run_daily.py(예보→확률→등급→upsert), kma_mid.py(중기예보), scheduler.py | `python -m ops.scheduler --once` (1회) / `python -m ops.scheduler` (상시, 매일 06:00). KMA_KEY 확보 시 `python -m ops.kma_mid --probe` 로 익산 기온 구역코드 확정 |
| **C** | RAG | `rag/` — ingest.py(조문 청킹+위계 메타), search.py(ko-sroberta+위계 부스트), eval_qa_v2.py(30문항) | 평가 재실행: `python demo_v3.py` 의 v3-12 구간. 시행령 별표 원문(DOC) 도착 시 `python -m rag.reingest_annex <파일.docx>` |
| **D** | 에이전트·의사결정 서비스 | `agents/` — 도구 계약, provider adapter, 6시간 작업 계획, 근거 결합, 알림 초안, 선택적 OpenAI 설명. `app/`은 얇은 Streamlit 화면만 담당 | 기존 콘솔: `python demo.py` ⑥. 화면: `streamlit run app/dashboard.py` |

### D 파트 구조와 실행

D의 중심은 `agents/`다. 기존 `work_guide.py`, `notify_draft.py`,
`tools_schema.py` 진입점을 유지하면서 다음 책임을 추가했다.

- `provider.py`: A/B/C와 익산 측정소 자료가 지켜야 할 교체 가능한 인터페이스
- `scoring_policy.py`: 기존 S5에서 D가 사용하는 작업·저장일 가중치만 분리한 순수 정책
- `fixture_provider.py`: 공식 자료 도착 전임을 명시하는 고정 시연 데이터
- `legacy_provider.py`: 기존 B SQLite, C `RagIndex`, legacy 플룸을 새 계약으로 연결
- `work_guide.py`: B 위험값으로 추천 Top 3·회피 Top 3를 결정론적으로 계산
- `notify_draft.py`: DB 변경이나 실발송 없이 승인 전 알림 초안만 생성
- `openai_explainer.py`: 확정된 결과를 바꾸지 않고 선택적으로 문장만 생성

`app/dashboard.py`에는 모델·추천·RAG 로직을 두지 않는다. Streamlit의 화면 표시와
세션 단위 승인 기록만 담당하므로, 실제 자료가 도착해도 `agents`의 provider만
교체하면 같은 화면을 유지할 수 있다.

```powershell
cd demo
pip install -r requirements-d.txt
streamlit run app/dashboard.py
```

기본값은 `D_PROVIDER_MODE=fixture`다. 기존 B/C 데모 산출물을 연결하려면
`D_PROVIDER_MODE=legacy`를 명시한다. 실제 adapter는
`D_PROVIDER_FACTORY=package.module:create_provider`로 주입하며, 연결 실패를
fixture로 조용히 바꾸지 않는다. 외부 factory는 `storage_days`와 `rag_index`
키워드 인자를 받고 `DecisionProvider` 구현을 반환해야 한다. 상세 설계 근거는
`docs/D_ARCHITECTURE.md`에 있다.

공용 분석: `analysis/` (S8 그래프, v2~v5 실험, plume_multi 다중 발원 하네스).
발표 그래프 6종은 `out/figs/` (captions.md 에 한 줄 캡션).

## 절대 규칙 3개 (전원 준수)

1. **ML 과 플룸은 곱하지 않는다** — 이산 보정만. 현재 플룸 등급 상향은
   `config.PLUME_GRADE_BUMP = False` (검증 미통과로 OFF — 근거는 final 보고서 2절)
2. **서빙 피처는 예보 API 제공 변수만** — NH3·CO2·방향성 피처 금지
3. **legacy/ 수정 금지, import 만** — demo.py 가 실행 때마다 원본과 해시 대조

## 대기 중인 데이터 (도착 시 명령 한 줄)

| 데이터 | 명령 |
| --- | --- |
| 농가 허가 대장 실좌표 | `python -m analysis.plume_multi <좌표.csv>` → 플룸 재판정 |
| 가축분뇨법 시행령 원문(DOC) | `python -m rag.reingest_annex <파일.docx>` |
| KMA_KEY | `setx KMA_KEY <키>` 후 `python -m ops.kma_mid --probe` |
| 과거 단기예보 자료 | 예보 열화 백테스트 (validation_report_final 4절 — 추가 옵션) |
