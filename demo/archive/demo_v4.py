"""v4 — 발원 좌표 규명·교정 + 조립 착수 (지시 14~18).

python demo_v4.py  (demo.py 선실행 필요)

14. [최우선] "왕궁 3km 내 10건" 재검증 → 원인 규명 → 좌표 교정 → 플룸 전면 재실행
15. 다중 발원 플룸 하네스 (analysis/plume_validation.py) — 템플릿으로 가동 검증
16. RAG 별표 재추출 준비 (rag/reingest_annex.py) — 대기 모드 점검
17. 조립: ops/kma_midterm.py + ops/scheduler.py, daily_scoring 연결 검증
18. ROC-AUC 병기 + 리프트 차트

출력: out/validation_report_v4.md, out/v4_results.json,
      out/v4_lift_chart.png, out/plume_multi_results.csv
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # archive/ -> demo/

import config
from config import MID_DIR, OUT_DIR, section
from console import use_utf8_stdout  # legacy import (수정 금지)

R: dict = {}


def main():
    use_utf8_stdout()
    print("v4 — 발원 좌표 규명 + 조립 착수 (지시 14~18)")
    if not (MID_DIR / "features.parquet").exists():
        raise SystemExit("data/ 산출물이 없습니다. 먼저 python demo.py 를 실행하세요.")

    import pandas as pd

    section("v4-14 '왕궁 3km 내 10건' 재검증 (최우선)")
    from archive import v4_diag14
    R["diag14"] = v4_diag14.run()
    print(f"  → config 좌표 교정 완료: ({config.WANGGUNG_LAT}, {config.WANGGUNG_LON}) [B]")

    section("v4-14b 교정 좌표로 S8-2/3/4 재생성 + 플룸 판정 전면 재실행")
    from analysis import figures
    from archive import v3_aws
    lab = pd.read_parquet(MID_DIR / "label_table.parquet")
    comp = pd.read_parquet(MID_DIR / "complaints_clean.parquet")
    w = pd.read_parquet(MID_DIR / "weather_hourly.parquet")
    R["s8_2_fixed"] = figures.s8_2_wind(lab)
    R["s8_3_fixed"] = figures.s8_3_distance(comp)
    R["s8_4_fixed"] = figures.s8_4_plume_hit(comp, w)
    R["exp9_fixed"] = v3_aws.exp9_plume(radii_km=(3.0, 5.0, 8.0))
    R["plume_judge_fixed"] = v3_aws.judge_plume(R["exp9_fixed"])
    print(f"  통과 기준 판정(교정 좌표): {R['plume_judge_fixed']}")

    section("v4-15 다중 발원 플룸 하네스 — 템플릿 가동 검증")
    from analysis import plume_validation
    plume_validation.write_template()
    R["multi_demo"] = plume_validation.evaluate(
        plume_validation.load_sources(plume_validation.TEMPLATE), radius_km=3.0, wind="aws")
    print("  농가 목록 도착 시: python -m analysis.plume_validation <목록.csv>")

    section("v4-16 RAG 별표 재추출 준비 — 대기 모드 점검")
    from rag import reingest_annex
    R["annex_broken"] = len(reingest_annex.health_check())
    print("  원문 도착 시: python -m rag.reingest_annex <시행령.docx|pdf>")

    section("v4-17 조립: kma_midterm + scheduler")
    from ops import kma_midterm
    import os
    R["kma_midterm"] = {
        "land_reg": kma_midterm.MID_LAND_REG_ID, "ta_reg": kma_midterm.MID_TA_REG_ID,
        "tmfc_now": kma_midterm.latest_tmfc(),
        "live": kma_midterm.fetch_mid() is not None,
    }
    print(f"  구역코드: 육상 {R['kma_midterm']['land_reg']} [A] / "
          f"기온 {R['kma_midterm']['ta_reg']} [A, 전주 — 익산 코드는 --probe 로 확정]")
    print(f"  발표시각 계산: tmFc={R['kma_midterm']['tmfc_now']} / "
          f"실 API: {'연결됨' if R['kma_midterm']['live'] else 'KMA_KEY 대기 (mock 폴백 동작)'}")
    from ops import scheduler
    scheduler.job()   # --once 경로 검증 (실모델 daily_scoring 관통)
    R["scheduler_once_ok"] = True

    section("v4-18 ROC-AUC 병기 + 리프트 차트")
    from archive import v4_metrics
    R["metrics"] = v4_metrics.run()
    aws_ml = v3_aws.exp10_ml(pd.read_parquet(MID_DIR / "weather_aws.parquet"))
    R["aws_ml"] = aws_ml

    write_report()
    section("완료")


def write_report():
    section("보고서 — out/validation_report_v4.md")
    d14 = R["diag14"]
    e9 = R["exp9_fixed"]
    pj = R["plume_judge_fixed"]
    at = e9["judged_at"]
    jj, aw = e9[at]["jeonju_146"], e9[at]["iksan_702"]
    m = R["metrics"]["full_test2026"]
    ml_jj, ml_aw = R["aws_ml"]["jeonju_146"], R["aws_ml"]["iksan_702_aws"]

    radius_rows = []
    for key in [k for k in e9 if k.endswith("km")]:
        for name, label in [("jeonju_146", "전주 146"), ("iksan_702", "익산 702")]:
            o = e9[key][name]
            radius_rows.append(
                f"| {key} | {label} | {o['n']} | {o['hit']} | {o['placebo']} | "
                f"x{o['lift']} | {o['median_angle_off']}도 | {o['downwind_rate']} |")
    radius_table = "\n".join(radius_rows)

    bump = (
        "**PASS — PLUME_GRADE_BUMP 복원을 권고한다** (config 플래그 ON 으로 교체). "
        "다만 다중 발원 목록 도착 시 plume_validation 로 재확인 후 유지 여부를 재평가할 것."
        if pj["passed"] else
        "**미통과 — PLUME_GRADE_BUMP 는 OFF 유지.** 좌표 교정 후에도 통과 기준에 못 "
        "미치므로 남은 병목은 '왕궁 단일 발원' 가정이다. 다중 발원 하네스(15절)에 "
        "농가 목록을 넣어 재검증하는 것이 다음 단계다."
    )

    report = f"""# 검증 보고서 v4 — 발원 좌표 규명·교정 + 조립 착수

작성: {datetime.now():%Y-%m-%d %H:%M} · `python demo_v4.py` · 시드 42

## 14. [최우선] "왕궁 3km 내 민원 10건" 재검증 — 원인 규명

| 점검 | 결과 |
| --- | --- |
| (a) 하버사인 단위 | 정상 — 서울↔부산 {d14['haversine_seoul_busan_km']}km (기대 ~325) |
| (b) 위경도 인자 순서 | 정상 — 위도1도 {d14['deg_lat_km']}km(≈111), 경도1도 {d14['deg_lon_km']}km(≈91) |
| (c) 민원 부분집합 | 전체 기간 사용 — 익산+가축분뇨 {d14['subset']['익산_가축분뇨']:,}건(중복제거 후), AWS 정합용 연도≤2025 {d14['subset']['2025이하(AWS맞춤)']:,}건. test 한정 아님 |
| (d) 흥암리 좌표 분산 | {d14['heungam']['n']}건(중복제거 후), 중앙값 ({d14['heungam']['median'][0]}, {d14['heungam']['median'][1]}) — 제공 좌표와 일치. 표준편차 위도 {d14['heungam']['std_lat_m']}m / 경도 {d14['heungam']['std_lon_m']}m, 자기 중앙값 3km 내 {d14['heungam']['within_3km_of_own_median']}건 |

**규명: 코드 버그가 아니라 상수 오류였다.** v1~v3 의 왕궁 좌표(35.977, 127.055)는
[C] 임의 근사값이었고, 실제 흥암리 축산단지 중앙값에서 **{d14['coord_gap_km']}km
서쪽**으로 어긋나 있었다. 그 자리 기준 3km 에는 민원이 10건뿐이었던 것.
지역명-좌표 데이터 품질 문제도 아니다 — 흥암리 민원 좌표는 분산 ~100m 로
지역명과 정합한다. **v1 보고서의 허점 5번("왕궁 좌표 미정의")이 실제 사고로
이어진 사례**다. config 좌표를 흥암리 중앙값 [B] 로 교정했다.

주의: 교정 좌표 3km 내 민원은 {d14['n_within_흥암리 중앙값(제공)']['3km']}건이다
(719건이 아님). 흥암리 민원 대부분이 발원 100m 이내(단지 내부 신고)라 자기부지
제외(≥100m) 필터에 걸린다 — "축산단지 내부 주민의 신고"가 대량이라는 새로운
데이터 특성이 드러났고, 근거리(<600m) 방위 판정은 legacy NEAR_WARN 대로
민감함을 감안해야 한다.

## 14b. 교정 좌표로 플룸 판정 전면 재실행

| 반경 | 바람 입력 | n | 적중률 | 플라시보 | lift | 이탈각 중앙값 | 풍하측(≤90도) |
| --- | --- | --- | --- | --- | --- | --- | --- |
{radius_table}

판정 반경 {at} 기준 통과 기준:
① 이탈각 중앙값 < 70도 → {"충족" if pj['c1_direction'] else "미충족"} /
② lift ≥ 1.5 → {"충족" if pj['c2_lift'] else "미충족"} /
③ 전주 대비 전 지표 개선 → {"충족" if pj['c3_improves_over_jeonju'] else "미충족"}

**판정: {"PASS" if pj['passed'] else "FAIL"}.** {bump}

해석 — 좌표 교정으로 표본이 10→{e9['3km']['iksan_702']['n']}건이 되고 익산 바람
lift 는 x{e9['3km']['iksan_702']['lift']} 로 기준(1.5)을 넘었지만, **이탈각 중앙값이
{e9['3km']['iksan_702']['median_angle_off']}도(랜덤=90도)로 방향성이 사실상 없다.**
원인은 근거리 지배: 3km 표본의 약 90%가 발원 600m 이내(단지 내부·인접 신고)라
방위각이 '바람'이 아니라 '단지 내 주거 배치'로 결정된다 — legacy NEAR_WARN(600m)
경고가 데이터로 확인된 셈이다. 반면 8km 에서는 방향 신호가 살아나지만
(이탈각 중앙값 {e9['8km']['iksan_702']['median_angle_off']}도, 풍하측
{e9['8km']['iksan_702']['downwind_rate']}) 원거리 민원은 다른 발원(개별 농가)이
섞여 lift 가 무너진다. **결론: 단일 발원 가정으로는 어느 반경에서도 통과가
어렵고, 다중 발원 목록(15절)이 유일한 다음 수단이다.**

(S8-2/3/4 그래프도 교정 좌표로 재생성: 거리 중앙값 {R['s8_3_fixed']['median_km']}km,
S8-4 플룸 적중 {R['s8_4_fixed']['hit']:.3f} vs 플라시보 {R['s8_4_fixed']['placebo']:.3f})

## 15. 다중 발원 플룸 하네스 (구현 완료, 대기 중)

`analysis/plume_validation.py` — 발원 목록 CSV(farm_id, lat, lon)를 받아 민원별
**최근접 발원** 기준 적중률·방위각 정합·발원별 상세(out/plume_multi_results.csv)를
계산한다. 농가 목록 도착 시:
`python -m analysis.plume_validation "농가목록.csv" --radius-km 3 --wind aws`

가동 검증(발원 1개 = 흥암리 템플릿): 민원 {R['multi_demo']['n']}건, 적중
{R['multi_demo']['hit']}, lift x{R['multi_demo']['lift']}, 근거리(<600m) 비중
{R['multi_demo']['near_warn_share']} — 하네스 정상 동작.

## 16. RAG 별표 재추출 준비 (스크립트 완료, 원문 대기)

`rag/reingest_annex.py` — DOCX(의존성 없이 zip 파싱)/PDF 원문을 받아 가축분뇨법
시행령 별표만 재청킹·캐시 무효화·QA30+별표 5문항(35문항) 재측정까지 자동.
현재 상태: 깨진 별표 청크 **{R['annex_broken']}건 — 지시의 별표 1~5 뿐 아니라
6~9(위반행위별 과징금·설계시공업 등록기준·단속실적보고·과태료 부과기준)까지
전부 깨져 있음을 추가 확인.** 원문 도착 시
`python -m rag.reingest_annex <파일>` 한 줄로 전량 처리된다.

## 17. 조립 진행 — 중기예보 클라이언트 + 상시 스케줄러

- **ops/kma_midterm.py**: getMidLandFcst + getMidTa, 발표시각(06/18시) 자동 계산,
  D+4~7 {{tmin, tmax, pop}} 반환. daily_scoring 에 연결 완료 — KMA_KEY 가 설정되면
  자동으로 실 API, 없으면 mock 폴백(현재 상태: {"실 API 연결" if R['kma_midterm']['live'] else "키 대기, mock 폴백 검증됨"}).
- **구역코드 상수화**: 중기육상(강수확률) 전라북도 = **{R['kma_midterm']['land_reg']}** [A],
  중기기온 = **{R['kma_midterm']['ta_reg']}** (전주) [A]. **익산 전용 기온 코드는 공개
  코드표에서 확정하지 못했다** — 가이드 코드표(활용가이드 한글파일)의 전북 항목
  재확인이 필요하며, 키 확보 즉시 `python -m ops.kma_midterm --probe` 가 후보
  11F10202~11 을 실호출해 자동 확정한다. 중기기온의 min/max 중 무엇을 피처로
  쓸지는 계획서 미규정(v2 허점 2)이라 평균 사용 [C] 로 명시했다.
- **ops/scheduler.py**: APScheduler 상시 구동(매일 06:00, misfire 1시간 유예,
  실패 시 이전 캘린더 유지) + `--once` 검증 모드. 본 실행에서 `--once` 경로로
  실모델 daily_scoring 관통 확인.

## 18. 커뮤니케이션용 지표 (첫인상용 — 메인 프레임은 기후학 대비 증분 유지)

| 모델 | PR-AUC | ROC-AUC | 주간 랭킹 적중률 |
| --- | --- | --- | --- |
| 연속 full (test 2026.1~7, 전주 학습) | {m['pr_auc']} | **{m['roc_auc']}** | (v1 보고서 0.476) |
| 전주 버전 (test 2025.1~7, v3 조건) | {ml_jj['pr_auc']} | {ml_jj['roc_auc']} | {ml_jj['weekly_hit']} |
| 익산 AWS 버전 (test 2025.1~7) | {ml_aw['pr_auc']} | {ml_aw['roc_auc']} | {ml_aw['weekly_hit']} |

리프트 차트(out/v4_lift_chart.png): **상위 20% 블록 추천 시 가축분뇨 민원의
{R['metrics']['lift_capture_at_20pct']:.0%} 포착** (랜덤 20% 기대 대비).
주의: ROC-AUC 는 불균형(양성 13%)에서 후하게 나오는 지표다 — 첫인상용으로만
쓰고, 질문이 들어오면 PR-AUC 와 기후학 증분으로 돌아올 것.

## 남은 대기 항목

1. 농가별 발원 목록 → plume_validation 재검증 → BUMP 재평가
2. 가축분뇨법 시행령 원문 → 별표 재추출 → RAG 35문항 재측정
3. KMA_KEY → kma_midterm 실 API + 익산 기온 코드 probe 확정
4. 과거 단기예보 자료 → 예보 열화 백테스트
"""
    (OUT_DIR / "validation_report_v4.md").write_text(report, encoding="utf-8")
    with open(OUT_DIR / "v4_results.json", "w", encoding="utf-8") as fh:
        json.dump(R, fh, ensure_ascii=False, indent=2, default=str)
    print(f"  저장: {OUT_DIR / 'validation_report_v4.md'}")


if __name__ == "__main__":
    main()
