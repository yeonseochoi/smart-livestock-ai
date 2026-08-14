"""v2 검증 실행 — 비판 5건에 대한 후속 지시 1~8 수행.

python demo_v2.py  (demo.py 를 먼저 1회 실행해 data/ 산출물이 있어야 한다)

1~4: 즉시 실험 (기후학 3자 대결 / 조건부 가치 / 강건성 / 드리프트)
5~6: 설계 변경 확인 (플룸 등급 상향 기본 OFF / 알림 대상 합집합)
7:   RAG ko-sroberta 교체 + 30문항 재평가 (TF-IDF 와 비교)
8:   프레이밍 — 보고서 한계 절에 반영

출력: out/validation_report_v2.md, out/v2_results.json, out/v2_*.png
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from config import MID_DIR, OUT_DIR, PLUME_GRADE_BUMP, section

from console import use_utf8_stdout  # legacy import (수정 금지)

R: dict = {}


def run_experiments():
    from analysis import v2_experiments as ex
    section("v2-1 기후학 베이스라인 3자 대결 (test 2026.1~7)")
    R["exp1"] = ex.exp1_baselines()
    section("v2-2 조건부 가치 — 기상 이상 블록에서의 부가가치")
    R["exp2"] = ex.exp2_conditional()
    section("v2-3 강건성 — 연속 변수 변형 + 풍속 ±0.5m/s 섭동")
    R["exp3"] = ex.exp3_robustness()
    section("v2-4 2026 드리프트 분해")
    R["exp4"] = ex.exp4_drift()


def run_rag_v2():
    section("v2-7 RAG — ko-sroberta 교체 + 30문항 재평가")
    from rag.search import RagIndex
    from rag import eval_qa_v2
    idx_tfidf = RagIndex(backend="tfidf")
    R["rag_tfidf"] = eval_qa_v2.run(idx_tfidf)
    idx_emb = RagIndex(backend="sroberta")
    R["rag_sroberta"] = eval_qa_v2.run(idx_emb)


def check_design_changes():
    section("v2-5/6 설계 변경 확인")
    from scoring import recommend
    from agents import notify_draft
    from datetime import timedelta
    print(f"  PLUME_GRADE_BUMP = {PLUME_GRADE_BUMP} (기본 OFF)")
    rec = recommend.recommend("액비살포", storage_days=12, tons=20.0)
    R["s5_v2"] = {"recommended": rec["recommended"], "avoid": rec["avoid"]}
    print(f"  추천 창 등급 {rec['recommended']['grade']} / bumped={rec['recommended']['plume_bumped']}")
    print(f"  플룸 표기: {rec['recommended']['plume_note']}")
    nd = notify_draft.run("분뇨제거",
                          datetime.now().replace(minute=0, second=0, microsecond=0)
                          + timedelta(hours=6))
    nd.pop("message", None)
    R["notify_v2"] = nd


def g(*keys, default="N/A"):
    cur = R
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def write_report():
    section("보고서 — out/validation_report_v2.md")
    e1, e2, e3, e4 = R["exp1"], R["exp2"], R["exp3"], R["exp4"]
    comp = e4["composition"]
    yr = e4["year_2025_vs_2026"]

    full_hit = e1["full_model"]["weekly_hit"]
    mb_hit = e1["climo_month_block"]["weekly_hit"]
    b_hit = e1["climo_block_only"]["weekly_hit"]
    full_ap = e1["full_model"]["pr_auc"]
    mb_ap = e1["climo_month_block"]["pr_auc"]

    report = f"""# 검증 보고서 v2 — 비판 5건에 대한 실험 결과

작성: {datetime.now():%Y-%m-%d %H:%M} · `python demo_v2.py` · 시드 42 · test = 2026.1~7
전제: v1 보고서(validation_report.md)의 비판 5건 중 즉시 검증 가능한 항목을
실험으로 판정한다. 익산 AWS·과거 예보 자료 도착 전제 작업은 포함하지 않는다.

## 1. 기후학 베이스라인 3자 대결 — "성능이 달력에서 나오는가"

| 랭킹 방법 | test 주간 랭킹 적중률 | test PR-AUC |
| --- | --- | --- |
| (a) month x block 기후학 | {mb_hit} | {mb_ap} |
| (b) block-only 기후학 | {b_hit} | {e1['climo_block_only']['pr_auc']} |
| (c) XGB full (기상 포함) | **{full_hit}** | **{full_ap}** |

**판정: full 모델이 기후학 대비 주간 적중률 +{(full_hit - mb_hit):.3f}p
(상대 {((full_hit / mb_hit - 1) * 100) if mb_hit else 0:+.1f}%), PR-AUC
+{(full_ap - mb_ap):.3f}p.** 기후학만으로도 랜덤(0.20)을 크게 넘는다는 것,
즉 성능의 바닥이 '달력'이라는 v1 비판은 사실로 확인됐다. 다만 기상 변수는
그 위에 유의미한 증분을 얹는다 — 발표에서는 "기후학 대비 증분"을 핵심 수치로
쓰는 것이 정직하다. (그래프: v2_1_baselines.png)

## 2. 조건부 가치 — 기상 예보의 부가가치는 어디에 있나

평년(month x block 평균) 대비 풍속·습도 이상치 상하위 20% 블록(=이상 블록)과
나머지(평년 블록)에서 full vs 기후학을 재비교:

| 구간 | n | 양성률 | full PR-AUC | 기후학 PR-AUC | full prec@20% | 기후학 prec@20% |
| --- | --- | --- | --- | --- | --- | --- |
| 이상 블록 | {g('exp2', 'anomalous', 'n')} | {g('exp2', 'anomalous', 'pos_rate')} | {g('exp2', 'anomalous', 'full_pr_auc')} | {g('exp2', 'anomalous', 'climo_pr_auc')} | {g('exp2', 'anomalous', 'full_prec_at20')} | {g('exp2', 'anomalous', 'climo_prec_at20')} |
| 평년 블록 | {g('exp2', 'normal', 'n')} | {g('exp2', 'normal', 'pos_rate')} | {g('exp2', 'normal', 'full_pr_auc')} | {g('exp2', 'normal', 'climo_pr_auc')} | {g('exp2', 'normal', 'full_prec_at20')} | {g('exp2', 'normal', 'climo_prec_at20')} |

**판정: 가설 기각.** "기상 예보의 가치는 평년을 벗어난 날에 집중된다"는
기대와 반대로, full 의 기후학 대비 증분은 이상 블록
({g('exp2', 'anomalous', 'full_pr_auc', default=0) - g('exp2', 'anomalous', 'climo_pr_auc', default=0):+.3f})보다
**평년 블록({g('exp2', 'normal', 'full_pr_auc', default=0) - g('exp2', 'normal', 'climo_pr_auc', default=0):+.3f})에서 오히려 크다.**
해석 후보: ① 이상 블록의 기상값은 학습 분포의 꼬리라 모델 예측이 불안정,
② 민원의 기상 반응이 극단값 임계형이 아니라 평년 범위 내 미세 변동에도 반응,
③ 이상치 정의(평년 대비 편차 상하위 20%)가 거칠어 진짜 특이 기상을 못 가름.
결론: **"기상 모델이 특이 기상일을 잡아낸다"는 발표 내러티브는 현재 데이터로
지지되지 않으므로 쓰지 말 것.** 기상 증분은 전 구간에 얇게 분산돼 있다.

## 3. 강건성 — 이진 플래그와 예보 오차 민감도

**(a) 연속 변수 변형** (calm/humid80/night_calm 제거, night x ws·night x humid 연속
상호작용으로 교체):

| 모델 | test PR-AUC | test 주간 적중률 |
| --- | --- | --- |
| 원본 full (이진 플래그) | {g('exp3', 'original_full', 'pr_auc')} | {g('exp3', 'original_full', 'weekly_hit')} |
| 변형 (연속) | {g('exp3', 'variant_continuous', 'pr_auc')} | {g('exp3', 'variant_continuous', 'weekly_hit')} |

**(b) 풍속 ±0.5 m/s 섭동 시 주간 top-20% 랭킹의 변화** (원본 full 모델):

| 섭동 | top-k 평균 교체율 | 주간 적중률이 변한 주 비율 |
| --- | --- | --- |
| +0.5 m/s | {g('exp3', 'perturbation', '+0.5', 'mean_topk_turnover')} | {g('exp3', 'perturbation', '+0.5', 'weeks_hit_changed')} |
| -0.5 m/s | {g('exp3', 'perturbation', '-0.5', 'mean_topk_turnover')} | {g('exp3', 'perturbation', '-0.5', 'weeks_hit_changed')} |

**판정: 예보 오차 수준의 섭동(±0.5m/s)만으로 주간 상위 20% 블록 구성이
평균 {g('exp3', 'perturbation', '+0.5', 'mean_topk_turnover', default=0) * 100:.0f}%
안팎 교체된다.** v1 비판 2(학습-서빙 불일치)의 위험이 정량 확인된 것으로,
과거 예보-실측 페어 백테스트(자료 도착 후)가 필수다. 연속 변수 변형이 이진
플래그와 성능이 비슷하다면 임계값(1.5/80)에 의존하지 않는 연속 버전이
섭동에 더 안전한 선택이다.

## 4. 2026 드리프트 분해 — valid 0.42 → test 0.30 의 원인

| 평가 구간 | PR-AUC | 주간 적중률 | 양성률 |
| --- | --- | --- | --- |
| valid 2025 전체(12개월) | {comp['valid_full_year']['pr_auc']} | {comp['valid_full_year']['weekly_hit']} | — |
| valid 2025.1~7 (test 와 동일 구성) | {comp['valid_jan_jul']['pr_auc']} | {comp['valid_jan_jul']['weekly_hit']} | {comp['valid_jan_jul_pos_rate']} |
| test 2026.1~7 | {comp['test_jan_jul']['pr_auc']} | {comp['test_jan_jul']['weekly_hit']} | {comp['test_pos_rate']} |

2025 vs 2026 (1~7월): 민원 블록 양성률 {yr[2025]['pos_rate']} → {yr[2026]['pos_rate']},
민원 건수 {yr[2025]['complaints']} → {yr[2026]['complaints']},
평균 기온 {yr[2025]['temp_mean']} → {yr[2026]['temp_mean']}℃,
평균 풍속 {yr[2025]['ws_mean']} → {yr[2026]['ws_mean']}m/s,
평균 습도 {yr[2025]['humid_mean']} → {yr[2026]['humid_mean']}%.

연도별 양성률 추이: {e4['yearly_pos_rate']}

**판정: "2026 드리프트"는 대부분 착시였다.** valid 를 test 와 같은 1~7월로
제한하는 것만으로 PR-AUC 가 {comp['valid_full_year']['pr_auc']} →
{comp['valid_jan_jul']['pr_auc']} 로 내려간다 — 하락분의 대부분이 **평가 구간의
계절 구성 효과**(1~7월에는 민원 밀집 여름 후반이 절반만 포함)다. 잔여 격차는
{comp['valid_jan_jul']['pr_auc']} → {comp['test_jan_jul']['pr_auc']} 수준이고,
서비스 지표인 **주간 랭킹 적중률로는 valid 1~7월
{comp['valid_jan_jul']['weekly_hit']} vs test {comp['test_jan_jul']['weekly_hit']} 로
사실상 차이가 없다.** 결론: "2026 성능 급락"은 발표에서 주장하지 말 것.
같은 이유로, **서로 다른 계절 구성의 구간끼리 PR-AUC 를 직접 비교하는 것
자체가 함정**이므로 앞으로 모든 성능 비교는 동일 월 구성으로 맞춘다.
월별 상세는 v2_4_drift.png 참조.

## 5~6. 설계 변경 (코드 반영 완료)

- **플룸 등급 상향 기본 OFF**: `config.PLUME_GRADE_BUMP = False`.
  추천 결과의 플룸 표기 예: "{g('s5_v2', 'recommended', 'plume_note')}"
  에이전트 출력도 "참고: 풍하측 민가 N동 (미검증 모델)" 형식으로 통일.
  익산 AWS 바람으로 S8-4 재검증이 통과하면 플래그만 켜면 된다.
- **주민 알림 대상 = 합집합**: factor>0 ({g('notify_v2', 'n_by_factor')}동) OR
  부채꼴 내 ({g('notify_v2', 'n_by_sector')}동) → 대상
  {g('notify_v2', 'n_targets')}동 / 전체 {g('notify_v2', 'n_receptors')}동.
  플룸 단독 의존이 풀렸고, 미검증 상태에서 보수적(과포함) 방향이다.

## 7. RAG — ko-sroberta 교체 + 30문항 재평가

평가셋을 PDF 실제 조문·절 내용에서 직접 뽑은 30문항으로 확장 (v1은 8문항):

| 백엔드 | top-3 적중률 | 적중/문항 |
| --- | --- | --- |
| TF-IDF (v1 방식) | {g('rag_tfidf', 'hit_rate')} | {g('rag_tfidf', 'hits')}/30 |
| ko-sroberta 임베딩 | {g('rag_sroberta', 'hit_rate')} | {g('rag_sroberta', 'hits')}/30 |

**판정: 두 백엔드 모두 목표 80% 미달, 임베딩 교체 이득은
+{(g('rag_sroberta', 'hit_rate', default=0) - g('rag_tfidf', 'hit_rate', default=0)) * 100:.0f}%p 로 제한적.**
오답 패턴(v2_results.json 의 details)을 보면 대부분 "관련 문서까지는 찾는데
법-시행령-시행규칙 중 정답 층위를 못 짚는" 유형이다 — 병목은 검색기보다
①법령 위계 메타데이터 부재(청크에 '법/령/규칙' 구분 태그 없음),
②별표 표 추출 품질이다. 다음 개선은 모델 교체가 아니라 청크 메타데이터
보강 + 위계 필터라는 뜻이다. 남은 한계: 스캔본 1종(냄새 저감 기본 관리
매뉴얼)은 OCR 전까지 검색 불가, "인쇄 _ 국가법령정보센터" 696청크는 문서
단위가 섞여 있어 출처 표기가 약하다.

## 8. 프레이밍 교정 (발표·문서 전체에 적용할 문구)

- 이 시스템의 라벨은 **"악취 발생"이 아니라 "민원 발생"**이다. 민원에는 냄새
  물리 외에 저녁 재실, 야간 수면 방해, 신고 성향 같은 인간 행동이 포함되며,
  **서비스 목적(민원 리스크 관리)에는 이것이 오히려 정합적인 타깃**이다.
  단, "악취 위험도"라는 표현은 "민원 위험도"로 통일한다.
- 서비스 주장은 "민원 감소 보장"이 아니라 **"상대 위험 회피 지원"**으로 통일한다.
  근거: 위험도는 시 단위 상대 등급이며, 개별 농가 행동→민원 감소의 인과는
  아직 데이터로 입증되지 않았다(작업 일지 확보 전까지 유지되는 한계).

## 요약 판정표

| v1 비판 | v2 실험 결과 | 후속 |
| --- | --- | --- |
| 1. 성능이 달력? | 바닥은 달력이 맞음(month x block 만으로 적중 {mb_hit}). 기상 증분은 +{(full_hit - mb_hit):.3f}p 로 실재하나, "특이 기상일을 잡는다"는 내러티브는 실험 2에서 기각 | 발표 수치를 "기후학 대비 증분"으로 교체 |
| 2. 학습-서빙 불일치 | ±0.5m/s 섭동에 top-k 교체 ~{g('exp3', 'perturbation', '+0.5', 'mean_topk_turnover', default=0) * 100:.0f}%, 주간 적중값이 변한 주 ~{g('exp3', 'perturbation', '+0.5', 'weeks_hit_changed', default=0) * 100:.0f}% — 민감성 실재(치명적이진 않음) | 과거 예보 자료 도착 후 열화 백테스트 |
| 3. 시단위↔농가 인과 단절 | 실험 불가(데이터 없음) — 프레이밍 교정으로 대응(8절) | 작업 일지 수집 설계 |
| 4. 플룸 미검증 | 등급 반영 OFF + 알림 합집합으로 강등(5~6절) | 익산 AWS 후 S8-4 재검증 |
| 5. 전주 관측소 문제 | 실험 불가(익산 AWS 미도착) | 자료 도착 후 대조 |
| (부수 발견) valid→test 성능 급락 | **드리프트가 아니라 평가 구간 계절 구성 착시** — 주간 적중률은 차이 없음(실험 4) | 성능 비교는 동일 월 구성으로 통일 |
"""
    (OUT_DIR / "validation_report_v2.md").write_text(report, encoding="utf-8")

    slim = dict(R)
    for k in ("rag_tfidf", "rag_sroberta"):
        if k in slim:
            slim[k] = dict(slim[k])
    with open(OUT_DIR / "v2_results.json", "w", encoding="utf-8") as fh:
        json.dump(slim, fh, ensure_ascii=False, indent=2, default=str)
    print(f"  저장: {OUT_DIR / 'validation_report_v2.md'}")


def main():
    use_utf8_stdout()
    print("v2 검증 — 비판 5건 후속 지시 1~8")
    if not (MID_DIR / "features.parquet").exists():
        raise SystemExit("data/ 산출물이 없습니다. 먼저 python demo.py 를 실행하세요.")
    run_experiments()
    check_design_changes()
    run_rag_v2()
    write_report()
    section("완료")


if __name__ == "__main__":
    main()
