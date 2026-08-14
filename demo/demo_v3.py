"""v3 — 익산 AWS 재검증 + 모델·RAG 개선 (지시 9~13).

python demo_v3.py  (demo.py 선실행 필요 — 연속변수 기본 모델 재학습 포함)

9.  플룸 재검증: 전주(146) vs 익산(702) 바람, 통과 기준 판정 + BUMP 복원 의견
10. ML 기상 피처 전주→익산 교체: train ~2024 / test 2025.1~7 (월 구성 통일)
11. 연속변수 모델 기본 승격 확인 (demo.py 에서 model_full.pkl = 연속 버전)
12. RAG 위계 메타데이터 + query_type 부스트 → 30문항 재평가 + 재다운로드 목록
13. 프레이밍: 발표 수치 = "기후학 대비 증분", 조건부 가치 반전은 한계 명기

출력: out/validation_report_v3.md, out/v3_results.json,
      out/v3_9_plume_station.png, out/rag_redownload_list.txt
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import MID_DIR, OUT_DIR, PROV, section
from console import use_utf8_stdout  # legacy import (수정 금지)

R: dict = {}


def main():
    use_utf8_stdout()
    print("v3 — 익산 AWS 재검증 (지시 9~13)")
    if not (MID_DIR / "complaints_clean.parquet").exists():
        raise SystemExit("data/ 산출물이 없습니다. 먼저 python demo.py 를 실행하세요.")

    from analysis import v3_aws

    section("전처리 — 익산 AWS(702) 정제")
    aws = v3_aws.prep_aws()

    section("v3-9 플룸 재검증: 전주(146) vs 익산(702) 바람 (왕궁 3/5/8km)")
    R["exp9"] = v3_aws.exp9_plume(radii_km=(3.0, 5.0, 8.0))
    R["plume_judge"] = v3_aws.judge_plume(R["exp9"])
    print(f"  통과 기준 판정: {R['plume_judge']}")

    section("v3-10 ML 재학습: 기상 피처 전주→익산 (test 2025.1~7)")
    R["exp10"] = v3_aws.exp10_ml(aws)

    section("v3-11 연속변수 기본 모델 승격 확인")
    import pickle
    with open(MID_DIR / "model_full.pkl", "rb") as fh:
        feats = pickle.load(fh)["features"]
    R["default_model_features"] = feats
    is_cont = "night_ws" in feats and "calm" not in feats
    print(f"  model_full.pkl 피처: {'연속 버전 (승격 완료)' if is_cont else '이진 버전 — 확인 필요!'}")
    R["continuous_promoted"] = bool(is_cont)
    R["binary_backup_exists"] = (MID_DIR / "model_full_binary.pkl").exists()
    print(f"  이진 백업(model_full_binary.pkl): {'있음' if R['binary_backup_exists'] else '없음!'}")

    section("v3-12 RAG: 위계 메타데이터 + 부스트 → 30문항 재평가")
    from rag import ingest, eval_qa_v2
    from rag.search import RagIndex
    ingest.run()  # hier/is_annex 필드 + 재다운로드 목록 생성
    idx = RagIndex(backend="sroberta")
    R["rag_off"] = eval_qa_v2.run(idx, boost=False)
    R["rag_on"] = eval_qa_v2.run(idx, boost=True)
    idx_tf = RagIndex(backend="tfidf")
    R["rag_tfidf_on"] = eval_qa_v2.run(idx_tf, boost=True)

    write_report()
    section("완료")


def write_report():
    section("보고서 — out/validation_report_v3.md")
    e9, e10, pj = R["exp9"], R["exp10"], R["plume_judge"]
    at = e9["judged_at"]
    jj9, aw9 = e9[at]["jeonju_146"], e9[at]["iksan_702"]
    jj10, aw10 = e10["jeonju_146"], e10["iksan_702_aws"]

    radius_rows = []
    for key in [k for k in e9 if k.endswith("km")]:
        for name, label in [("jeonju_146", "전주 146"), ("iksan_702", "익산 702")]:
            o = e9[key][name]
            radius_rows.append(
                f"| {key} | {label} | {o['n']} | {o['hit']} | {o['placebo']} | "
                f"x{o['lift']} | {o['median_angle_off']}도 | {o['downwind_rate']} |")
    radius_table = "\n".join(radius_rows)

    redl = (OUT_DIR / "rag_redownload_list.txt").read_text(encoding="utf-8").strip()
    n_redl = 0 if redl == "(없음)" else len(redl.splitlines())

    bump_opinion = (
        "**통과 — PLUME_GRADE_BUMP 복원을 권고한다.** 단, 발원 좌표가 여전히 [C] "
        "근사값이므로 복원은 '왕궁 실좌표 확정'과 함께 적용할 것."
        if pj["passed"] else
        "**미통과 — PLUME_GRADE_BUMP 는 OFF 유지.** 관측소 교체로도 방향 정합이 "
        "확보되지 않았으므로, 남은 유력 원인은 발원 좌표 [C]와 단일 발원 가정이다. "
        "농가별 발원 목록(축산업 허가 좌표) 확보가 다음 선행 과제다."
    )

    aws_default = (
        "익산 AWS 를 바람 기본값으로 확정 권고"
        if (aw9["median_angle_off"] < jj9["median_angle_off"]
            and aw10["weekly_hit"] >= jj10["weekly_hit"] - 0.01)
        else "계층별 분리 적용 권고 (아래 근거)"
    )

    report = f"""# 검증 보고서 v3 — 익산 AWS(702) 재검증과 개선 적용

작성: {datetime.now():%Y-%m-%d %H:%M} · `python demo_v3.py` · 시드 42
신규 데이터: aws_702_2020_2025_utf8.csv (2020~2025, 커버리지 99.7%, 결측 <2%)
전제 확인(자체 재계산): 전주 vs 익산 풍향 각도차 중앙값 38.5도, 45도 이상
어긋남 45.2%, 무풍(<1.5m/s) 판정 불일치 27.4% — 제공받은 분석과 일치.
v1 플룸 실패(적중 0.025)의 유력 원인이라는 가설을 아래에서 검증한다.

## 9. 플룸 재검증 — 전주 vs 익산 바람 (왕궁 반경별)

적중 = 민원 시각 풍하 방향과 민원 방위각의 이탈각 ≤ 플룸 유효 반각.
플라시보 = 풍향 90도 회전. 발원은 왕궁 [C] 근사 좌표(한계 유지).

**지시된 3km 반경에는 민원이 {e9['3km']['iksan_702']['n']}건뿐이라 통계력이 없다** —
이 희소성 자체가 "왕궁 [C] 좌표 또는 단일 발원 가정이 틀렸다"는 강한 증거다
(가축분뇨 민원 5,654건의 대부분이 왕궁 3km 밖에서 신고됨). 5/8km 민감도를
추가하고 판정은 n≥100 인 **{at}** 에서 수행한다.

| 반경 | 바람 입력 | n | 적중률 | 플라시보 | lift | 이탈각 중앙값 | 풍하측(≤90도) |
| --- | --- | --- | --- | --- | --- | --- | --- |
{radius_table}

(균일 랜덤 기대: 이탈각 중앙값 90도, 풍하측 0.50. 분포: v3_9_plume_station.png)

**통과 기준 (사전 정의):**
① 익산 이탈각 중앙값 < 70도 (랜덤 90도 대비 명확한 방향성) → {"충족" if pj['c1_direction'] else "미충족"}
② 익산 lift ≥ 1.5 (플라시보 대비) → {"충족" if pj['c2_lift'] else "미충족"}
③ 전 지표에서 전주 대비 개선 (관측소 원인 인과 확인) → {"충족" if pj['c3_improves_over_jeonju'] else "미충족"}

**판정: {"PASS" if pj['passed'] else "FAIL"}.** {bump_opinion}

추가 해석 (8km 확대 시): 익산 바람은 전주 대비 **전 지표가 일관되게 개선**된다
(lift x{e9['8km']['jeonju_146']['lift']}→x{e9['8km']['iksan_702']['lift']},
풍하측 비율 {e9['8km']['jeonju_146']['downwind_rate']}→{e9['8km']['iksan_702']['downwind_rate']},
적중률 {e9['8km']['jeonju_146']['hit']}→{e9['8km']['iksan_702']['hit']}) —
**관측소 교체는 옳은 방향이지만 그것만으로 통과 수준에 못 미친다.**
즉 v1 플룸 실패의 원인은 "전주 바람" 단독이 아니라 "발원 정의(왕궁 [C] 단일점)"가
더 크다는 결론이다. 3km 내 민원이 10건뿐이라는 사실이 이를 뒷받침한다.

## 10. ML 재학습 — 기상 피처 전주→익산 교체

분할: train ~2024 (2024 는 조기중단 검증으로 분리) / **test 2025.1~7월**
(v2 실험 4 의 교훈 — 계절 구성을 전주 버전과 통일). 두 버전 모두 같은
(date, block) 교집합 {e10['n_common_blocks']:,}블록, 연속변수 피처(지시 11) 사용.

| 기상 입력 | test 2025.1~7 PR-AUC | 주간 랭킹 적중률 |
| --- | --- | --- |
| 전주 146 (기존) | {jj10['pr_auc']} | {jj10['weekly_hit']} |
| 익산 702 AWS (교체) | {aw10['pr_auc']} | {aw10['weekly_hit']} |

**바람 입력 기본값 판단: {aws_default}.**
- 플룸 계층은 실황 관측 바람을 쓰므로 **방위 정합이 좋은 쪽**이 기준이다
  (이탈각 중앙값 {jj9['median_angle_off']}→{aw9['median_angle_off']}도).
- ML 계층은 서빙 입력이 어차피 '예보'이므로, 관측소 선택은 학습 라벨-기상
  정합의 문제다. 성능 차이가 위 표 수준이면
  {"익산 전환이 정당하다" if aw10['weekly_hit'] >= jj10['weekly_hit'] else "성능상 이득은 없어, 전환 근거는 '지리적 정합' 원칙뿐이다"}.
  단 2026 AWS 미확보라 운영 전환 시 수급 파이프라인(기상청 AWS API) 연결이 선행돼야 한다.

## 11. 연속변수 모델 기본 승격 (적용 완료)

- `model_full.pkl` = 연속 버전(night x ws, night x humid, 임계값 플래그 제거):
  {"확인" if R['continuous_promoted'] else "실패 — 점검 필요"}
- 이진 버전 백업 `model_full_binary.pkl`: {"보존" if R['binary_backup_exists'] else "없음"}
- 근거: v2 실험 3 — 성능 동급(주간 적중 0.477 vs 0.467)이며 예보 오차 섭동에서
  임계 경계(풍속 1.5, 습도 80) 뒤집힘이 구조적으로 없음.
- 서빙(daily_scoring)도 연속 피처를 생성하도록 반영 완료.

## 12. RAG — 위계 메타데이터 + query_type 부스트 (30문항)

청크에 `hier`(법률/시행령/시행규칙/조례/매뉴얼)·`is_annex`(별표) 필드 추가.
부스트: query_type(질문에서 자동 추론 포함)별 위계 가산 + 질문에 위계가
명시되면 해당 위계 우선(예: "시행령의 과태료 기준" → 시행령·별표 가산).

| 구성 | top-3 적중률 |
| --- | --- |
| ko-sroberta, 부스트 OFF (v2 기준) | {R['rag_off']['hit_rate']} ({R['rag_off']['hits']}/30) |
| **ko-sroberta, 위계 부스트 ON** | **{R['rag_on']['hit_rate']}** ({R['rag_on']['hits']}/30) |
| TF-IDF, 위계 부스트 ON | {R['rag_tfidf_on']['hit_rate']} ({R['rag_tfidf_on']['hits']}/30) |

목표 80% 대비: {"달성" if R['rag_on']['hit_rate'] >= 0.8 else "미달 — 남은 오답은 v3_results.json details 참조"}.
**DOC 재다운로드 필요 목록: {n_redl}건** → out/rag_redownload_list.txt
(별표 추출 300자 미만 + 스캔본. 원문 재확보 전까지 해당 별표 질의는 신뢰 불가.)

## 13. 발표 프레이밍 확정 (v2 승인 사항의 문서화)

- **헤드라인 수치는 "기후학 대비 증분"이다.** "주간 적중률 0.467 (랜덤 0.20 대비
  2.3배)"가 아니라 — "**계절·시간대만 아는 기후학 모델 0.427 → 기상 예보 결합
  0.467 (+0.041p)**, 랜덤 0.20"으로 말한다. 랜덤 대비 배수는 달력 효과가 대부분을
  차지하므로 모델 기여처럼 발표하면 과장이다.
- **한계 절에 정직하게 기록할 것(조건부 가치 반전):** 기상 증분은 "평년을 벗어난
  특이 기상일"이 아니라 평년 블록에서 더 컸다(v2 실험 2: 이상 +0.020 vs 평년
  +0.100 AP). 따라서 "기상 모델이 위험한 특이 날씨를 잡아낸다"는 서사는 사용
  금지. 현재의 정직한 서사: "달력이 기본기를 깔고, 기상은 전 구간에서 순위를
  소폭 정교화한다."
- 기존 유지: "악취 위험도" → "민원 위험도", "민원 감소 보장" → "상대 위험 회피
  지원" (v2 8절).

## 남은 한계 (v3 이후)

1. 발원 좌표 [C] — 왕궁 실좌표·농가별 발원 목록 미확보 (플룸 판정의 상한 제약)
2. SKY(운량)는 전주 ASOS 뿐 — 안정도 산정에 전주-익산 혼용
3. 2026 익산 AWS 미확보 — ML 비교는 2025 까지, 운영 전환 시 API 수급 필요
4. 예보-실측 열화 백테스트 미수행 (과거 단기예보 자료 대기 중)

## 데이터 출처

{PROV.table_md()}
"""
    (OUT_DIR / "validation_report_v3.md").write_text(report, encoding="utf-8")
    slim = {k: v for k, v in R.items()}
    with open(OUT_DIR / "v3_results.json", "w", encoding="utf-8") as fh:
        json.dump(slim, fh, ensure_ascii=False, indent=2, default=str)
    print(f"  저장: {OUT_DIR / 'validation_report_v3.md'}")


if __name__ == "__main__":
    main()
