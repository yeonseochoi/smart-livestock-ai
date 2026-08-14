"""python demo.py 한 번으로 구현 순서 ①~⑦ 전체를 실행하는 데모.

①  S0~S1  정제 → 라벨 테이블            (실데이터 D1, D2)
②  S4     배관 더미 관통                (mock 예보 + random 확률)
③  S6     RAG                          (실데이터 D6 PDF 14종)
④  S2~S3  피처 → 학습·평가              (XGBoost full/reduced + 로지스틱)
⑤  S8-1~3 분석 그래프
⑥  S7     에이전트 2종 + S8-4 플룸 적중률
⑦  S5     통합 + 실모델 교체 + S8-5 백테스트  (+ S8-6 센서 근거 그래프)

마지막에 out/validation_report.md 를 생성한다.
"""
from __future__ import annotations

import json
import sys
import traceback
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # archive/ -> demo/

import config
from config import DEMO_FARM, OUT_DIR, PROV, FINDINGS, finding, section

# legacy 콘솔 유틸 (cp949 대응) — import 만, 수정 금지
from console import use_utf8_stdout

RESULTS: dict = {}


def step1_etl():
    section("① S0~S1 — 데이터 정제 → 라벨 테이블 (실데이터)")
    from preprocess import clean_data, build_grid
    c, w = clean_data.run()
    lab = build_grid.run_legacy(c, w)
    RESULTS["s0_s1"] = {"complaints": len(c), "weather": len(w),
                        "label_rows": len(lab),
                        "pos_rate": round(float(lab["y_bin"].mean()), 4)}
    return c, w, lab


def step2_pipe():
    section("② S4 — 운영 배관 더미 관통 (모델 없이 전 구간 확인)")
    from serving import daily_scoring
    RESULTS["s4_dummy"] = daily_scoring.run_legacy(dummy=True)


def step3_rag():
    section("③ S6 — RAG 지식베이스 (실데이터 PDF 14종)")
    from rag import ingest, eval_qa
    from rag.search import RagIndex
    ingest.run()
    idx = RagIndex()
    RESULTS["s6_rag"] = eval_qa.run(idx)
    return idx


def step4_model(lab):
    section("④ S2~S3 — 피처 엔지니어링 → 모델 학습·평가")
    from preprocess import build_features
    from model import train_model
    feat = build_features.run_legacy(lab)
    res = train_model.run_legacy(feat)
    res.pop("_valid_proba", None)
    RESULTS["s3_model"] = res
    return feat


def step5_analysis_123(lab, c, w):
    section("⑤ S8-1~3 — 조건 재현·풍향 정합·거리 감쇠 (실데이터)")
    from analysis import figures
    RESULTS["s8_1"] = figures.s8_1_condition(lab)
    RESULTS["s8_2"] = figures.s8_2_wind(lab)
    RESULTS["s8_3"] = figures.s8_3_distance(c)


def step6_agents(rag_index, c, w):
    section("⑥ S7 — 에이전트 2종 + S8-4 플룸 적중률")
    from agents import work_guide, notify_draft
    from analysis import figures

    print("\n[작업 가이드 에이전트 — 테스트 케이스 ①: 경과일 12일 + 위험 예보]")
    guide = work_guide.run(DEMO_FARM["farm_id"], "분뇨제거", rag_index)
    print(guide)
    RESULTS["s7_guide"] = guide

    print("\n[주민 알림 초안 에이전트]")
    RESULTS["s7_notify"] = notify_draft.run(
        "분뇨제거", datetime.now().replace(minute=0, second=0, microsecond=0)
        + timedelta(hours=6))
    RESULTS["s7_notify"].pop("message", None) or None

    print("\n[S8-4 플룸 적중률 (실측 기상 x 실제 민원 좌표)]")
    RESULTS["s8_4"] = figures.s8_4_plume_hit(c, w)


def step7_integrate(c=None):
    section("⑦ S5 — 통합: 실모델 교체 → 작업 창 추천 + S8-5 백테스트")
    from serving import daily_scoring
    from advisor import recommend
    from analysis import figures

    RESULTS["s4_real"] = daily_scoring.run_legacy(dummy=False)

    rec = recommend.recommend_legacy("액비살포", storage_days=12, tons=20.0,
                                 method="표면살포")
    print(f"  [추천] {rec['recommended']['t']} final {rec['recommended']['final']}"
          f" 등급 {rec['recommended']['grade']}"
          + (" (플룸 보정 상향)" if rec["recommended"]["plume_bumped"] else ""))
    if rec["recommended"]["plume_note"]:
        print(f"        {rec['recommended']['plume_note']}")
    print(f"  [회피] {rec['avoid']['t']} final {rec['avoid']['final']}")
    if "emission_kg" in rec:
        print(f"  [배출] {rec['emission_kg']} kg NH3 — {rec['emission_note']}")
    RESULTS["recommend"] = rec

    RESULTS["s8_5"] = figures.s8_5_backtest()
    RESULTS["s8_6"] = figures.s8_6_sensor()


def write_report():
    section("보고서 생성 — out/validation_report.md")
    m = RESULTS.get("s3_model", {})
    r = RESULTS

    def g(*keys, default="N/A"):
        cur = r
        for k in keys:
            if not isinstance(cur, dict) or k not in cur:
                return default
            cur = cur[k]
        return cur

    runtime_findings = "\n".join(f"- {f}" for f in FINDINGS) or "- (없음)"

    report = f"""# 데모 구현 검증 보고서 (validation_report)

작성: {datetime.now():%Y-%m-%d %H:%M} · `python demo.py` 단일 실행 결과 · 시드 42

이 보고서의 목적은 "다 잘됐다"가 아니라, **계획서(구현 내용.md)를 실데이터로
한 바퀴 돌렸을 때 드러난 논리적 허점·모호한 지점·데이터 문제를 기록**하는 것이다.

## 0. 전제부터 문제: 지시서 파일이 없다

요청받은 **데모.md 가 로컬 디스크(C: 전체 검색)와 Notion 어디에도 존재하지 않았다.**
가장 최신 계획 문서인 `구현 정리_0809/구현 내용.md` 를 지시서로 삼아 구현했고,
"절대 규칙 3개"는 그 문서에서 강조된 다음 3개로 해석했다:
1. ML과 플룸은 **곱하지 않는다** (이산 보정만)
2. 서빙 피처는 **예보 API가 주는 변수만** (NH3·CO2 금지)
3. **legacy 수정 금지**, import 만 (원본 `작업폴더\\최종구현 py파일` → `demo/legacy/` 사본)

→ 만약 실제 데모.md 가 따로 있다면 이 해석부터 대조 검증이 필요하다.

## 1. 실행 결과 요약 (①~⑦ 전 구간 통과 여부)

| 단계 | 내용 | 결과 |
| --- | --- | --- |
| ① S0~S1 | 정제→라벨 {g('s0_s1', 'label_rows')}행, 양성률 {g('s0_s1', 'pos_rate')} | 실행됨 |
| ② S4 | 더미 배관 관통 {g('s4_dummy', 'n_upsert')}블록 upsert | 실행됨 |
| ③ S6 | RAG top-3 적중률 {g('s6_rag', 'hit_rate')} (목표 0.80) | 실행됨 |
| ④ S2~S3 | full test PR-AUC {g('s3_model', 'full_test', 'pr_auc')}, 주간 적중률 {g('s3_model', 'full_test', 'weekly_hit')} | 실행됨 |
| ⑤ S8-1~3 | 야간·무풍·고습 배율 x{g('s8_1', 'ratio', default=0):.2f} 등 그래프 3장 | 실행됨 |
| ⑥ S7+S8-4 | 에이전트 2종 + 플룸 적중 {g('s8_4', 'hit', default=0):.3f} (플라시보 {g('s8_4', 'placebo', default=0):.3f}) | 실행됨 |
| ⑦ S5+S8-5 | 실모델 교체 + 추천 {g('recommend', 'recommended', 't')} | 실행됨 |

### 모델 성능 (시계열 분할: train 20~24 / valid 25 / test 26.1~7)

| 모델 | valid PR-AUC | valid 주간적중 | test PR-AUC | test 주간적중 |
| --- | --- | --- | --- | --- |
| XGB full | {g('s3_model', 'full_valid', 'pr_auc')} | {g('s3_model', 'full_valid', 'weekly_hit')} | {g('s3_model', 'full_test', 'pr_auc')} | {g('s3_model', 'full_test', 'weekly_hit')} |
| XGB reduced | {g('s3_model', 'reduced_valid', 'pr_auc')} | {g('s3_model', 'reduced_valid', 'weekly_hit')} | {g('s3_model', 'reduced_test', 'pr_auc')} | {g('s3_model', 'reduced_test', 'weekly_hit')} |
| 로지스틱(베이스라인) | {g('s3_model', 'logit_valid', 'pr_auc')} | {g('s3_model', 'logit_valid', 'weekly_hit')} | {g('s3_model', 'logit_test', 'pr_auc')} | {g('s3_model', 'logit_test', 'weekly_hit')} |

랜덤 기대치 = 0.20. SHAP 상위 피처: {g('s3_model', 'shap_top5')}

## 2. 데이터 출처 (실데이터 vs mock — 콘솔 로그와 동일)

{PROV.table_md()}

## 3. 실행 중 자동 감지된 문제 (finding 로그)

{runtime_findings}

## 4. 계획서의 논리적 허점·모호한 지점 (구현하며 발견)

### 설계 허점

1. **학습-서빙 기상 불일치가 통째로 빠져 있다.** 모델은 ASOS *실측* 기상으로
   학습하지만 서빙 입력은 *예보*다. 예보 오차(특히 풍향·풍속)가 성능을 얼마나
   깎는지에 대한 검증 계획이 계획서에 없다. 과거 예보-실측 페어로 열화량을
   측정하는 백테스트가 반드시 추가되어야 한다.
2. **reduced 모델의 해상도 모순.** 중기예보는 일 단위(기온·강수확률)인데 라벨과
   reduced 피처셋에는 `block` 이 있다. D+4~7 서빙에서 블록 간 차이를 만드는 입력은
   block 하나뿐이므로 이 모델은 사실상 "일 위험도 + 학습된 블록 프로파일"이다.
   또 중기기온은 최저/최고 2값인데 어느 값을 `기온` 피처에 넣는지 미규정.
3. **등급 컷과 final(t)의 스케일 불일치.** 등급 컷은 예측 *확률*의 분위수인데
   S5의 final = 확률 x work_weight x storage_factor 는 1을 넘을 수 있는 다른
   스케일이다. "플룸이면 등급 1단계 상향"의 대상이 어느 스케일의 등급인지 계획서가
   정의하지 않아, 데모에서는 블록 등급(확률 기반)에 보정을 걸었다.
4. **work_weight 값이 계획서에 없다.** 수식에는 등장하지만 5개 작업유형의 값과
   근거가 미정의 → 데모는 임의값 [C]. 근거등급 규약을 계획서 스스로 못 지킨 부분.
5. **왕궁 축산단지 좌표 미정의.** S8-2·3·4 세 그래프가 전부 이 좌표에 의존하는데
   계획서 어디에도 좌표가 없다. 데모는 근사 좌표 [C]를 썼고, 좌표가 수백 m만
   틀어져도 방위각·거리 분석이 함께 틀어진다.
6. **TIME_WEIGHTS(1시간 6칸)와 3시간 블록 캘린더의 정합 미규정.** window_risk 의
   `risk_prob(t+j시간)` 에서 j시간 단위 확률이 존재하지 않는다(캘린더는 블록 단위).
   데모는 't+j가 속한 블록의 확률'로 해석했다 — 계획서가 명시해야 할 부분.
7. **주간 적중률의 k=11 고정 가정.** 기상 결측 블록 drop 으로 주당 블록이 56개가
   아닌 주가 존재한다. k = ceil(0.2 x 실제 블록 수)로 일반화해 계산했다.
8. **year 피처의 외삽 문제.** test(2026)는 학습 범위(≤2024) 밖 연도라 트리 모델에서
   경계값으로 뭉개진다. "플랫폼 이용률 변화 보정"은 미래 시점에는 작동하지 않는다.
9. **구현 순서표(①~⑦)에 S8-6 이 없다.** 그래프 6장이 발표 산출물인데 순서표는
   S8-5 까지만 배정한다. 데모에서는 ⑦ 뒤에 붙여 실행했다.

### mock/인프라의 계약 불일치

10. **legacy mock_forecast 에 습도(REH)가 없다.** full 모델 피처에 습도가 필수인데
    mock 은 TMP/VEC/WSD/SKY/POP 만 준다. 데모는 SKY→습도 근사 [C]로 때웠지만,
    "더미로 배관을 먼저 관통"하는 계획의 전제(mock=실 API 계약 동일)가 깨져 있다.
11. **Chroma+임베딩 대신 TF-IDF 로 대체.** 데모 환경에 Chroma·임베딩 모델이 없어
    문자 n-gram TF-IDF 검색으로 구현했다(청킹 원칙은 준수). 계획서의 top-3 80%
    목표는 임베딩 기준이므로 아래 수치와 직접 비교하면 안 된다.
12. **kma_midterm.py 는 여전히 미구현.** 구역코드 확정(익산 코드 존재 여부)과 서비스키가
    필요해 데모에서는 mock 일집계로 대체했다 — 계획서의 미해결 항목 그대로.

### 실데이터에서 드러난 것

13. RAG top-3 적중률 **{g('s6_rag', 'hit_rate')}** (목표 0.80) — 상세는 out/results.json.
    법령 PDF 는 조문 청킹이 대체로 동작하나, 표(별표) 추출 품질이 낮은 문서가 있다.
14. 플룸 적중률 {g('s8_4', 'hit', default=0):.3f} vs 플라시보(풍향 90도 회전)
    {g('s8_4', 'placebo', default=0):.3f} — 실제 풍향이 플라시보보다 높긴 하나
    **절대 수준 자체가 매우 낮다.** 원인은 ① 플룸 유효 반각이 5~13도로 좁아
    '단일 발원 x 반각 이내' 정의로는 적중이 희박하고, ② 민원 전부가 왕궁 단일
    발원이라는 가정이 비현실적이며(익산 전역에 개별 농가 존재), ③ 발원 좌표가
    [C] 근사값이기 때문이다. 계획서의 S8-4 는 "민원 좌표가 플룸 안에 든 비율"의
    분모·발원 정의를 명시하지 않았다 — 농가별 발원 목록 없이는 이 그래프가
    물리 모델의 실증 근거가 되기 어렵다.
15. 조건 재현(S8-1): 야간·무풍·고습 블록 민원율이 그 외의
    **x{g('s8_1', 'ratio', default=0):.2f}** — 익산시 공식 분석과 방향 일치 여부 확인용.
16. "복잡한 모델이 정말 필요했나": 로지스틱 대비 XGB full 의 test PR-AUC 차이가
    {g('s3_model', 'full_test', 'pr_auc')} vs {g('s3_model', 'logit_test', 'pr_auc')}.
    차이가 작다면 발표에서 단순 모델 채택도 정당하다.

## 5. 절대 규칙 준수 확인

- **규칙 1 (곱셈 금지)**: S5 는 플룸 factor 를 final 에 곱하지 않는다.
  `recommend.py` 의 보정은 등급 1단계 상향(이산)뿐이다.
- **규칙 2 (서빙 피처 철칙)**: FULL/REDUCED_FEATURES 에 NH3·CO2·방향성 피처 없음.
  센서 데이터는 S8-6 그래프에만 사용.
- **규칙 3 (legacy 수정 금지)**: `demo/legacy/` 는 원본 사본이며 import 만 했다.
  실행 후 원본↔사본 SHA-256 해시 검증 결과:
  **{"전 파일 일치 (수정 없음)" if r.get('legacy_hash_ok') else "불일치 발견 — 즉시 확인 필요"}**

## 6. 산출물 목록

- `demo/data/` : complaints_clean / weather_hourly / label_table / features (parquet),
  model_full.pkl / model_reduced.pkl / grade_cuts.json / rag_chunks.json
- `demo/out/` : demo.db (risk_calendar 등 4테이블), s3_shap_top5.png,
  s8_1~s8_6 그래프 6장, weekly_hit_valid/test.csv, results.json, 이 보고서
"""
    (OUT_DIR / "validation_report.md").write_text(report, encoding="utf-8")

    slim = {k: v for k, v in RESULTS.items() if not k.startswith("_")}
    slim.pop("s7_guide", None)
    with open(OUT_DIR / "results.json", "w", encoding="utf-8") as fh:
        json.dump(slim, fh, ensure_ascii=False, indent=2, default=str)
    print(f"  저장: {OUT_DIR / 'validation_report.md'}")


def verify_legacy_untouched():
    """절대 규칙 3 검증: legacy 사본이 원본과 바이트 단위로 같은지 확인."""
    import hashlib
    src = config.PROJECT_DIR / "작업폴더" / "최종구현 py파일"
    ok = True
    for f in sorted(config.LEGACY_DIR.glob("*.py")):
        orig = src / f.name
        if not orig.exists():
            continue
        h1 = hashlib.sha256(f.read_bytes()).hexdigest()
        h2 = hashlib.sha256(orig.read_bytes()).hexdigest()
        if h1 != h2:
            ok = False
            finding(f"legacy/{f.name} 이 원본과 다름 — 절대 규칙 3 위반 가능성!")
    print(f"  legacy 원본 무결성: {'일치 (수정 없음 확인)' if ok else '불일치 발견!'}")
    RESULTS["legacy_hash_ok"] = ok


def main():
    use_utf8_stdout()
    print("악취·분뇨 프로젝트 데모 — 구현 내용.md ①~⑦ 단일 실행")
    print(f"프로젝트: {config.PROJECT_DIR}")

    # S7 테스트 케이스 ①: 분뇨 저장 경과 12일 세팅
    DEMO_FARM["last_manure_removal_date"] = (
        datetime.now() - timedelta(days=12)).strftime("%Y-%m-%d")

    c, w, lab = step1_etl()
    step2_pipe()
    rag_index = step3_rag()
    step4_model(lab)
    step5_analysis_123(lab, c, w)
    step6_agents(rag_index, c, w)
    step7_integrate()

    section("절대 규칙 3 — legacy 무결성 검증")
    verify_legacy_untouched()

    write_report()
    section("완료")
    print(f"발견된 문제 {len(FINDINGS)}건 — out/validation_report.md 참조")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
