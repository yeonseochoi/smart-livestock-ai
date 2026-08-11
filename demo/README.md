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

기존 `demo.py` 콘솔 검증 하네스의 API 키는 전부 선택사항이다:
`KMA_KEY`(단기·중기예보), `VWORLD_KEY`(주거건물), `ANTHROPIC_API_KEY`(과거 D 검증
경로). 새 D Streamlit 경로는 아래 설명처럼 `OPENAI_API_KEY`를 선택적으로 사용한다.

## 파트별 진입점

| 파트 | 담당 영역 | 코드 | 실행/재현 |
| --- | --- | --- | --- |
| **A** | 데이터·모델 | `etl/`(s0 정제→s1 라벨→s2 피처) → `model/s3_train.py` | `python demo.py` ①·④ 구간. 모델 실험은 `model/s3_train.py` 의 하이퍼파라미터·피처 리스트(`etl/s2_features.py`) 수정 후 demo.py 재실행 |
| **B** | 운영 배관 | `ops/` — db.py(SQLite 4테이블), run_daily.py(예보→확률→등급→upsert), kma_mid.py(중기예보), scheduler.py | `python -m ops.scheduler --once` (1회) / `python -m ops.scheduler` (상시, 매일 06:00). KMA_KEY 확보 시 `python -m ops.kma_mid --probe` 로 익산 기온 구역코드 확정 |
| **C** | RAG | `rag/` — ingest.py(조문 청킹+위계 메타), search.py(ko-sroberta+위계 부스트), eval_qa_v2.py(30문항) | 평가 재실행: `python demo_v3.py` 의 v3-12 구간. 시행령 별표 원문(DOC) 도착 시 `python -m rag.reingest_annex <파일.docx>` |
| **D** | 의사결정 서비스 | `app/` — dashboard.py(Streamlit), guide_service.py(결정론적 추천), openai_guide.py(선택적 설명), backend.py(도구 6종 계약) | `cd demo` 후 `streamlit run app/dashboard.py`. fixture 기본, `OPENAI_API_KEY`+`OPENAI_MODEL`은 선택 |

### D 파트 팀 공유용 한눈에 보기

**무엇을 만드는가?** 농장주가 측정소 관측을 확인하고, 상대적으로 민원 위험이 낮은
작업 시간을 선택한 뒤, 관련 관리 근거와 주민 알림 초안까지 한 흐름에서 확인하는
공모전 시연용 서비스다.

**현재 어디까지 되었는가?** 익산시 정보공개 자료가 오기 전이므로 가상 측정소와 고정
점수로 지도·위험 캘린더·작업 추천·근거 카드·알림 승인 흐름을 먼저 구현했다. 화면의
`FIXTURE` 표시는 실제 익산시 관측이나 모델 결과가 아니라는 뜻이다.

**시연 순서**

1. 측정소 지도에서 현재·과거 관측 형태를 확인한다.
2. 7일 민원 위험 캘린더에서 미래의 상대 위험을 확인한다.
3. 6시간 작업 창 추천 Top 3와 회피 Top 3를 비교한다.
4. 법령·매뉴얼 근거를 확인한다.
5. 작업 시간을 확정하고 주민 알림 초안을 편집한 뒤 승인만 기록한다.

센서 지도는 **현재·과거 관측**, 위험 캘린더는 **미래 민원 위험 예측**이다. 두 값을
같은 위험도로 해석하지 않으며, 플룸은 미검증 참고 정보라 추천 등급에 반영하지 않는다.

**실제 자료가 오면** 원본을 보존하고 열·단위·시간대·좌표·결측·품질코드를 확인한 뒤
`sensor-observation-v1` 계약으로 매핑한다. 이후 A/B 모델을 다시 학습·검증하고 D의
provider를 교체한다. 이때 Streamlit 화면과 사용자 흐름은 그대로 유지하는 것이 목표다.

### D 회의용 Streamlit 구조 초안

정보공개청구 자료가 도착하기 전에는 `app/`의 고정 fixture로 화면·계약·승인 흐름을
검증한다. 기존 `agents/` 코드는 검증 이력으로 유지하며 새 화면은 직접 의존하지 않는다.
첫 탭의 측정소 지도는 익산악취24 화면에서 예상한 센서·기상 필드의 canonical 계약을
사용한다. 현재 위치·명칭·수치는 가상이며, 실제 자료가 오면 센서 adapter만 교체한다.
`dashboard.py`는 provider 구현을 직접 import하지 않는다. 실연결 factory가 준비되면
`D_BACKEND_FACTORY=package.module:create_backend`를 설정하고 같은 화면을 사용한다.
factory는 `storage_days` 키워드 인자를 받아 `DecisionBackend` 구현을 반환해야 하며,
연결 오류를 fixture로 조용히 대체하지 않는다.

```powershell
cd demo
pip install -r requirements-d.txt
streamlit run app/dashboard.py
```

OpenAI 설명은 선택사항이다. 사용할 때만 `OPENAI_API_KEY`와 팀이 선택한
`OPENAI_MODEL`을 환경변수로 설정한다. 키가 없으면 규칙 기반 설명으로 동일 흐름을
끝까지 시연한다. 상세 설계와 회의 결정사항은 `docs/D_PROTOTYPE_PLAN.md`에 있다.

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
