# 익산시 축산 악취 민원 저감 의사결정 시스템

> ⚠️ **API 키 노출 주의** — `legacy/kma.py` 에 기상청 폴백 서비스키가 하드코딩되어 있고
> 이 저장소는 공개 상태다. **키 폐기 후 재발급이 필요하다.** 이후 환경변수(`KMA_KEY`)
> 전용으로 전환할 것. `legacy/` 는 수정 금지 원칙(절대규칙 3)이 걸려 있으므로
> 배포용 사본에서만 치환하거나 별도 조치가 필요하다.

**"냄새 나는 작업을 언제 하면 민원이 덜 날까"를 시각 단위로 알려주는 시스템.**
농가가 액비를 뿌리거나 분뇨를 반출할 때, 앞으로 3~7일 중 어느 시각이 가장 안전한지
순위를 매겨 준다.

```
   입력   기상청 단기예보(D+1~3, 1시간) + 중기예보(D+4~7, 일 단위)
   출력   시각별 위험 등급 + 추천/회피 시각 + 저감 조치 조언
   성능   시가지 test ROC-AUC 0.899 · A등급 시각의 민원율이 전체 평균의 1/17
```

**대상은 익산시 전 축종이다.** 민원 라벨이 "가축 분뇨 냄새(닭, 돼지, 소)" 한 종류라
축종 구분이 없다. 돼지·한우·가금 모두 서비스 대상이다.

> **이 문서는 코드 사용 설명서다.** 설치 · 실행 · 파일 구조 · 성능을 다룬다.
> 설계 근거("왜 1시간 격자인가", "왜 플룸을 곱하지 않는가")는
> [`../docs/PIPELINE.md`](../docs/PIPELINE.md) 에 있다.
> 프로젝트 전체 개요는 [`../README.md`](../README.md).

---

## 빠른 시작

폴더 배치가 이렇게 되어 있어야 한다 (데이터 경로를 `demo/` 의 부모에서 찾는다).

```
저장소 루트/
├── demo/                    ← 이 폴더
├── 프로젝트 데이터/
│   ├── 01_민원데이터/익산시 악취 민원 데이터_20190528-20260818.xlsx   ← 현행(2019.05~)
│   ├── 02_기상데이터/asos_146_api_2019_2026.csv · aws_702_… · asos_146_…
│   ├── 03_RAG_법령매뉴얼/*.pdf
│   ├── 04_양돈센서_AIHub/validation_matched_sensor_30m.xlsx
│   ├── 05_지오코딩_결과/farm_coords_vworld.csv        ← 익산 농가 좌표
│   ├── 전북특별자치도 김제시_축산현황_20250515_geocoded.csv
│   └── 전북특별자치도 익산시_축산농가 현황_*.csv
└── 작업폴더/최종구현 py파일/   ← (선택) legacy 원본. run_check.py 가 해시 검증에 사용
```

```powershell
pip install -r requirements.txt
cd demo

python run_train.py     # 학습    — 데이터 정제부터 모델 저장까지 (약 3분)
python run_serve.py     # 서빙·추천 — 예보 받아 위험도 계산 + 작업시각 추천
python run_check.py     # 검증    — 회귀 점검 39항목
```

중간 산출물(`data/`)과 재생성 가능한 결과(`out/` 일부)는 커밋하지 않는다.
위 세 명령이면 전부 다시 만들어진다.

**v5 트랙(과거 버전)** 도 그대로 보존되어 있다.
```powershell
python archive/demo.py              # v5 전체 실행 (과거 버전 재현)
python archive/demo_v2.py ~ demo_v5.py   # 과거 검증 라운드 재현
python run_train_region.py          # 지역 격자 병행 트랙 (실험)
```

API 키는 전부 선택사항: `KMA_KEY`(단기·중기예보), `VWORLD_KEY`(주거건물),
`ANTHROPIC_API_KEY`(에이전트 LLM)가 없으면 mock 폴백으로 돌고 콘솔에 로깅된다.

> `KMA_KEY` 는 **설정하지 말 것.** `serving/kma_midterm.py` 의 `_service_key()` 가
> 환경변수 경로에만 `unquote()` 가 없어서, 설정하면 이중 인코딩으로 중기예보가
> 조용히 깨진다. 설정하지 않으면 파일 안의 폴백 키로 정상 동작한다.

### 서빙 DB — PostgreSQL(Supabase) / SQLite

`DATABASE_URL` 이 있으면 PostgreSQL, 없으면 기존 `out/demo.db` 로 자동 폴백한다.
백엔드가 무엇이든 호출부 코드는 동일하다 (`serving/db.py` 가 자리표시자를 번역한다).

```powershell
copy demo\.env.example demo\.env      # DATABASE_URL 을 채운다 (.env 는 커밋 안 됨)
python -m serving.migrate_sqlite_to_pg          # 미리보기
python -m serving.migrate_sqlite_to_pg --apply  # 기존 demo.db 내용 복사 (선택)
```

PostgreSQL 로 옮긴 이유는 시각화가 아니라 **무인 운영**이다. GitHub Actions 는 잡이
끝나면 컨테이너가 사라져 `demo.db` 가 증발한다 (`.github/workflows/daily-serve.yml`).
Supabase 접속 문자열은 반드시 **Session/Transaction pooler** 쪽을 쓴다 —
Direct connection 은 무료 티어에서 IPv6 전용이라 Actions 러너에서 붙지 않는다.

---

## 파트별 진입점

| 파트 | 담당 영역 | 코드 | 실행/재현 |
| --- | --- | --- | --- |
| **A** | 데이터·모델 | `preprocess/`(clean_data → build_grid → build_features/spatial_features) → `model/train_model.py` | `python run_train.py`. 모델 실험은 `model/train_model.py` 하이퍼파라미터 + `preprocess/build_features.py` 피처 리스트 수정 |
| **B** | 운영 배관 | `serving/` — db.py(PostgreSQL/SQLite 이중 백엔드), daily_scoring.py(예보→확률→등급→upsert), kma_midterm.py(중기예보), scheduler.py | `python run_serve.py` / `python -m serving.scheduler --once` |
| **C** | RAG | `rag/` — ingest.py(조문 청킹+위계 메타), search.py(ko-sroberta+위계 부스트), eval_qa_v2.py(30문항) | 평가 재실행: `python archive/demo_v3.py` 의 v3-12 구간. 시행령 별표 원문(DOC) 도착 시 `python -m rag.reingest_annex <파일.docx>` |
| **D** | 에이전트 | `agents/` — work_guide.py(작업 가이드), notify_draft.py(주민 알림), tools_schema.py(도구 6종 스키마) | `archive/demo.py` ⑥ 구간. ANTHROPIC_API_KEY 설정 시 Claude tool use 로 전환되는 구조 |

공용 분석: `analysis/` (figures 그래프 6종, v2~v5 실험, plume_validation 다중 발원 하네스).
발표 그래프는 `out/figs/` (captions.md 에 한 줄 캡션).

---

## 파이프라인 흐름

```
╔══════════════════════════════════════════════════════════════════════════════╗
║  [A] 오프라인 학습 — run_train.py                                             ║
╚══════════════════════════════════════════════════════════════════════════════╝

   민원 xlsx 13,039행        기상 CSV 57,695행       농가 CSV
        │                         │                      │
        ▼                         ▼                      ▼
   ┌──────────────┐        ┌──────────────┐    ┌────────────────────┐
   │ clean_data   │        │ clean_data   │    │ geocode_gimje      │
   │ 익산·101 필터 │        │ 순환 보간     │    │ 리(里) 중앙값       │
   │ 중복 10분 제거│        │ (sin/cos)    │    │ 12.2% → 95.9%      │
   └──────┬───────┘        └──────┬───────┘    └─────────┬──────────┘
          └───────────┬───────────┘                      │
                      ▼                                  │
             ┌──────────────────┐                        │
             │  build_grid      │                        │
             │  (1시간 × 유형)   │  115,352행             │
             │  + 정답 y_bin     │                        │
             │  + 가중치 y_sev   │                        │
             │  + 중심좌표 저장   │ → group_center.json    │
             └────────┬─────────┘                        │
                      ▼                                  ▼
             ┌───────────────────────────────────────────────┐
             │  spatial_features    ★ 순수 기하 — 플룸 아님    │
             │  풍상측 ±30° · 15km 안 농가 수                 │
             │    up_ik_pig / cattle / poultry               │
             │    up_gj_pig / cattle / poultry               │
             │  + 직전 1년 민원율 (shift(1) 후 rolling)        │
             └────────┬──────────────────────────────────────┘
                      ▼
             ┌──────────────────┐
             │ build_features   │  시차(1·2h 전 바람) · 무풍연속
             │ 28개 피처 완성    │  야간 상호작용 · 계절 sin/cos
             └────────┬─────────┘
                      ▼
             ┌────────────────────────────────────────┐
             │  train_model      ★★ XGBoost 여기 하나뿐 │
             │  train ≤2023 / valid 2024 / test 2025  │
             │  full 2개(28피처) + reduced 2개(7피처)   │
             │  sample_weight = 악취강도 1~5           │
             │  월별 등급컷 12개 산출                   │
             └────────┬───────────────────────────────┘
                      ▼
             model_{유형}_full.pkl · model_{유형}_reduced.pkl
             grade_cuts_{유형}.json · group_center.json
                      │
╔═════════════════════│════════════════════════════════════════════════════════╗
║  [B] 매일 서빙 — serving/daily_scoring.py                                        ║
╚═════════════════════│════════════════════════════════════════════════════════╝
                      │
   기상청 단기예보 API (D+1~3, 1시간, 약 83시점)
   TMP · REH · WSD · VEC · PTY · SKY · POP
        │             │
        ▼             │
   ┌────────────────────────────┐
   │ 관측 마지막 48시간 이어붙이기 │  ← 없으면 ws_lag1 · calm_streak 이
   │ (시차·무풍 계산용)           │     0에서 시작해 학습 분포와 어긋남
   └────────┬───────────────────┘
            ▼
   build_serving_features() → spatial_features.run_serving() → prior_serving()
            │                  ★ 학습과 같은 함수·같은 중심좌표
            ▼
   ┌────────────────────────────┐
   │  유형별 full 모델 predict   │
   │  + reduced 모델 (D+4~7)    │
   └────────┬───────────────────┘
            ▼
   ┌────────────────────────────────────┐
   │  serving/db.py                         │
   │   risk_hourly    PK(date,hour,grp)  ← grp 없으면 한 유형이 덮어써짐
   │   forecast_hourly  예보 원값 (플룸 선택용)
   └────────┬───────────────────────────┘
            │
╔═══════════│══════════════════════════════════════════════════════════════════╗
║  [C] 추천 — advisor/recommend.py                                          ║
╚═══════════│══════════════════════════════════════════════════════════════════╝
            │
   농가 입력 (작업유형 · 축종 · 살포량 · 공법 · 경운 · 저장경과일)
            ▼
   ┌─────────────────────────────────────┐
   │  ① 법령 하드필터   [미구현]           │
   │     rules/spread_rules.py           │
   │     강우 · 결빙 · 부숙도 · 질소상한    │
   └──────────────┬──────────────────────┘
                  │  ★ 반드시 ML 보다 먼저.
                  │    반대로 하면 "법적으로 불가능한 시각이 A등급"
                  ▼
   ┌─────────────────────────────────────┐
   │  ② ML 랭킹                           │
   │     window_risk()                │
   │     TIME_WEIGHTS 6칸 = 살포 후 0~5h  │
   │     [0.30 0.22 0.17 0.13 0.10 0.08] │
   └──────────────┬──────────────────────┘
                  │  ★ ML 이 먼저 나와야 플룸이 고를 대상이 생긴다
                  ▼
   ┌─────────────────────────────────────┐
   │  ③ 플룸 — 수용점 유형 '선택'만         │
   │     analysis/plume_select.py        │
   │       ≤3km  → 플룸 유효 반각          │
   │       >3km  → 섹터 ±30° 로 대체       │
   │  ★★ 절대규칙 1 — 곱하지 않는다        │
   │     PLUME_GRADE_BUMP = False (영구)  │
   └──────────────┬──────────────────────┘
                  ▼
   ┌─────────────────────────────────────┐
   │  ④ 조합  combine(max/mean/min/first) │
   │     × WORK_WEIGHT [C] × storage [C]  │
   └──────────────┬──────────────────────┘
                  ▼
   ┌─────────────────────────────────────┐
   │  ⑤ 등급 — 두 층                       │
   │   위험/주의/낮음 : 월별 컷 (계절 보정)  │
   │   A / B / C     : 후보 내 상대 순위    │
   └──────────────┬──────────────────────┘
                  ▼
            추천 시각 · 회피 시각 · 등급 · 저감 조언
                  │
╔═════════════════│════════════════════════════════════════════════════════════╗
║  [D] 설명 계층 — 판단하지 않는다                                              ║
╚═════════════════│════════════════════════════════════════════════════════════╝
                  ▼
   legacy/emission.py  advice_lines()   "주입식 57% 감소" · "즉시 경운 38% 감소"
                                        ★ kg 절대값은 출력하지 않는다
   rag/                근거 조문 검색     (인덱스 ① 살포판정 · ② 저감팁 미구축)
   agents/             LLM 서술          ★ 판단하지 않는다. 틀려도 등급이 안 바뀐다
```

---

## 선후관계 3가지 절대 규칙

```
   ① 법령 필터 → ML 랭킹        법이 먼저 자른다. 모델은 법을 모른다.
   ② ML 예측  → 플룸           고를 대상이 먼저 존재해야 한다. 곱하지 않고 선택만.
   ③ 판단 = 코드 / 설명 = RAG·LLM   RAG·LLM 이 틀려도 등급이 안 바뀐다.
```

### 물리 정보가 들어가는 자리 — 둘까지만

```
   ① 피처 (풍상측 노출)   수용점 기준 "어디서 오나"   학습 단계   ✅ 사용
   ② 조합 (유형 선택)     발원 기준   "어디로 가나"   서빙 단계   ✅ 사용
   ③ 등급 보정 (BUMP)    최종 등급 가감              추천 단계   ❌ 영구 OFF

   ①과 ②는 방향도 단계도 다르므로 이중 계산이 아니다. ③까지 켜면 삼중이며 규칙 1 위반.
```

---

## 파일 구조

```
   흐름:  preprocess  →  model  →  serving  →  advisor
          (전처리)      (학습)     (매일 채점)   (추천)
```

```
demo/
  run_train.py          학습 실행          ← 진입점
  run_train_region.py   지역격자 병행 트랙  ← 진입점
  run_serve.py          서빙 + 추천        ← 진입점
  run_check.py          회귀 점검 39항목    ← 진입점
  config.py             경로 · 그룹 정의 · 기상 지점 · 분할 연도

  preprocess/           ── 데이터 준비
    kma_obs.py          기상청 API허브 지상관측 수집 (포털 수동 CSV 대체)
    clean_data.py       민원 xlsx · 기상 csv 정제
    geocode_gimje.py    김제 농가 리(里) 단위 재지오코딩
    build_grid.py       격자 생성 + 정답 라벨
    build_features.py   시차 · 야간 · 계절 피처
    spatial_features.py 풍상측 노출 + 직전1년 민원율

  model/                ── 학습
    train_model.py      XGBoost 학습 · 평가 · 등급컷

  serving/              ── 매일 도는 채점 (예보 → 위험도 → DB)
    db.py               DB 스키마 (PostgreSQL/SQLite 이중 백엔드)
    migrate_sqlite_to_pg.py  기존 demo.db -> PostgreSQL 1회 복사
    daily_scoring.py    예보 → 모델 → 위험도 → DB
    kma_midterm.py      중기예보 API
    scheduler.py        정기 실행

  advisor/              ── 농가 요청 시 추천
    recommend.py        6시간 창 · 플룸 선택 · 등급 · 조언

  analysis/             ── 분석 · 검증
    plume_select.py       플룸 수용점 유형 선택
    plume_validation.py   플룸 적중률 검증 (lift 1.61 산출)
    figures.py            발표 그래프 6종 생성

  archive/   과거 검증 라운드 v2~v5 (현역 파이프라인 밖 — archive/README.md 참조)
    demo_v2~v5.py · v2_experiments · v3_aws · v4_diag14 · v4_metrics · v5_farms

  legacy/    ★ 수정 금지 — run_check.py 가 원본과 SHA-256 대조
    plume.py · emission.py · diffusion.py · geo.py · kma.py · constants.py
    residence.py · console.py · main.py · recommend.py · mock_*.py

  rag/       법령 검색       agents/    LLM 설명
  docs/      RENAME_2026-08-15.md — 파일명 변경 이력
```

---

## 모델 입출력

```
   한 줄 = (1시간, 수용점 유형)     57,696시각 × 2 = 115,352행
   정답 = 그 시각 그 유형에 민원이 났는가 (0/1)
   가중치 = 악취 강도 1~5
```

| 묶음 | 피처 |
| --- | --- |
| 기상 6 | temp · humid · ws · wd_sin · wd_cos · rain |
| 야간 3 | night · night_ws · night_humid |
| 시각 4 | hour · month_sin · month_cos · dow |
| 시차·무풍 5 | ws_lag1 · ws_lag2 · wd_sin_lag1 · wd_cos_lag1 · calm_streak |
| 공간 6 | up_ik_pig/cattle/poultry · up_gj_pig/cattle/poultry (+ up_nearest_km) |
| 과거율 3 | prior_rate_1y · prior_std_1y · prior_month |

기상 6개는 **모두 기상청 단기예보가 주는 변수**(TMP·REH·WSD·VEC·PTY)와 1:1 대응한다 — 절대규칙 2.

---

## 성능

| 그룹 | 분할 | ROC-AUC | PR-AUC | lift | 상위20% 재현율 | A등급 민원율 |
| --- | --- | --- | --- | --- | --- | --- |
| 시가지원거리 | valid 2024 | 0.884 | 0.195 | 10.5배 | 81.1% | 평균의 1/16.4 |
| **시가지원거리** | **test 2025** | **0.899** | **0.221** | **7.7배** | **85.3%** | **평균의 1/16.7** |
| 농촌근거리 | valid 2024 | 0.785 | 0.141 | 3.4배 | 62.0% | 평균의 1/12.0 |
| 농촌근거리 | test 2025 | 0.724 | 0.074 | 2.5배 | 48.8% | 평균의 1/7.3 |

중기예보용 reduced 모델(7피처, 바람 없음): 시가지 test PR-AUC 0.185(lift 6.4배) / 농촌 0.082(2.8배).

> **[2026-08-18]** 위 표는 구 데이터셋(민원 13,039행 · 포털 기상 CSV) 기준 값이다.
> 민원 재크롤링(2019.05~, +33%)과 기상 API 전환 이후 재학습한 값은
> `out/training_results.json` 을 볼 것. 시가지 test ROC-AUC 0.8977 · 농촌 0.7442 로
> 사실상 재현됐으나, 발표용 수치 갱신 여부는 미정이라 표는 그대로 둔다.

**이 표의 값은 `run_train.py` 가 `out/training_results.json` 에 기록한다.**
문서와 파일이 다르면 파일이 옳다. 손으로 옮겨 적다 틀린 전례가 있어 지표를 코드로 옮겼다.

**농촌근거리가 낮은 이유는 파악돼 있다** — 왕궁면 흥암리 민원이 2020~2023 총 1건에서
2024년 517건으로 급증해 학습 구간과 평가 구간의 지역 구성이 다르다.
원인(앱 보급 / 지역명 표기 변경 / 실제 사건) 확인이 필요하다.

---

## 대기 중인 데이터 (도착 시 명령 한 줄)

| 데이터 | 명령 |
| --- | --- |
| 농가 허가 대장 실좌표 | `python -m analysis.plume_validation <좌표.csv>` → 플룸 재판정 |
| 가축분뇨법 시행령 원문(DOC) | `python -m rag.reingest_annex <파일.docx>` |
| KMA_KEY | `setx KMA_KEY <키>` 후 `python -m serving.kma_midterm --probe` |
| 과거 단기예보 자료 | 예보 열화 백테스트 |
