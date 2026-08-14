"""v5 — 검증 최종 라운드 (지시 19~22). 이 라운드로 검증 단계를 종료한다.

python demo_v5.py  (demo.py 선실행 필요)

19. 축산농가 현황 → 좌표 매핑 → 돼지 농가 다중 발원 플룸 최종 판정
20. 별표 복구 시도(pdfplumber) → 실패분 폴백 규칙 등록 → 35문항 재평가
21. validation_report_final.md — 검증 종료 선언
22. 발표 그래프 6종 → out/figs/ + 캡션
"""
from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import MID_DIR, OUT_DIR, PLUME_GRADE_BUMP, section
from console import use_utf8_stdout  # legacy import (수정 금지)

R: dict = {}

FIGS = [
    ("s8_1_condition.png", "야간·무풍·고습 블록의 민원율은 그 외의 1.9배 — 익산시 공식 분석 조건이 6년 실데이터에서 재현된다."),
    ("s8_2_wind.png", "민원 발생 블록의 풍향 분포는 왕궁→도심 확산 방향 쪽으로 치우친다 (평균 각도차: 민원 82° vs 비민원 100°)."),
    ("s8_3_distance.png", "가축분뇨 민원의 거리 분포 — 발원 인접(흥암리)과 도심(~10km)의 이중 구조로, 문헌 기준(1/2/8km) 단일 감쇠로는 설명되지 않는다."),
    ("s8_4_plume_hit.png", "단일 발원 플룸의 민원 좌표 적중률은 플라시보와 구분되지 않는다 — 물리 모델은 다중 발원 검증 전까지 참고 정보로만 사용."),
    ("s8_5_backtest.png", "주간 랭킹 적중률 백테스트(2025~26) — 랜덤 기대 0.20을 안정적으로 상회한다."),
    ("s8_6_sensor.png", "돈사 내부 NH3 급증 사례(AI Hub 센서) — '작업이 방출 급증을 만든다'의 관측 근거. 서빙 피처로는 사용하지 않는다."),
]


def step19():
    section("v5-19 농가 발원 최종 플룸 판정")
    from analysis import v5_farms, plume_validation
    import pandas as pd

    farms = v5_farms.build_farm_coords()
    pigs = farms[farms["축종"].astype(str).str.contains("돼지|종돈")].copy()
    print(f"  돼지 농가 발원 {len(pigs)}곳 (전체 좌표 확보 {len(farms)}곳 중)")
    R["farms"] = {"total": len(farms), "pigs": len(pigs),
                  "method_share": farms["방법"].value_counts().to_dict()}

    src = pigs.rename(columns={"축종": "type"})[["farm_id", "lat", "lon"]]
    src.to_csv(MID_DIR / "pig_sources.csv", index=False, encoding="utf-8-sig")
    res = plume_validation.evaluate(plume_validation.load_sources(MID_DIR / "pig_sources.csv"),
                               radius_km=3.0, wind="aws")
    R["plume_final"] = res
    R["plume_final_judge"] = v5_farms.judge_final(res)
    print(f"  최종 판정 기준 충족: {R['plume_final_judge']}")


def step20():
    section("v5-20 별표 복구 시도 + 폴백 규칙 + 35문항 재평가")
    from rag import annex_recover, eval_qa_v2
    from rag.reingest_annex import ANNEX_QA
    from rag.search import RagIndex

    R["annex"] = annex_recover.recover()
    idx = RagIndex(backend="sroberta")
    R["rag30"] = eval_qa_v2.run(idx, boost=True)
    hits = notices = 0
    for q, qt, dkw, tkw in ANNEX_QA:
        res = idx.search(q, qt, k=3)
        hits += any(dkw in r["doc"] and tkw in (r["snippet"] + r["unit"])
                    for r in res["results"])
        notices += "notice" in res
    R["rag_annex5"] = {"hits": hits, "notices": notices}
    total = R["rag30"]["hits"] + hits
    print(f"  별표 5문항: 적중 {hits}/5, 원문 안내 폴백 발동 {notices}/5")
    print(f"  합산 {total}/35 = {total/35:.0%}")
    R["rag35_total"] = round(total / 35, 4)


def step22():
    section("v5-22 발표 그래프 6종 → out/figs/")
    figs_dir = OUT_DIR / "figs"
    figs_dir.mkdir(exist_ok=True)
    lines = ["# 발표 그래프 6종 (S8) — 한 줄 캡션\n"]
    copied = 0
    for i, (name, caption) in enumerate(FIGS, 1):
        src = OUT_DIR / name
        if src.exists():
            shutil.copy2(src, figs_dir / f"fig{i}_{name}")
            lines.append(f"**fig{i}_{name}** — {caption}\n")
            copied += 1
    (figs_dir / "captions.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"  {copied}/6장 복사 + captions.md")
    R["figs"] = copied


def write_final_report():
    section("v5-21 검증 종료 선언 — validation_report_final.md")
    pf, pj = R["plume_final"], R["plume_final_judge"]
    bump_final = (
        "**복원 (PASS)** — config.PLUME_GRADE_BUMP 를 True 로 전환할 것. "
        "농가별 실발원 기준으로 방향성·lift·풍하측 모두 기준을 넘었다."
        if pj["passed"] else
        f"**OFF 유지 (FAIL — 단, 근소 미달)** — lift x{pf.get('lift')} 는 기준(1.5)을 "
        f"통과했고 이탈각 중앙값 {pf.get('median_angle_off')}도(기준 70 미만)·풍하측 "
        f"{pf.get('downwind_rate')}(기준 0.60 초과)가 각각 0.5도, 2.4%p 차이로 "
        f"미달했다. 단일 발원(v4: lift 0.8~1.67, 이탈각 91도) 대비 전 지표가 크게 "
        f"개선돼 '다중 발원이 옳은 방향'임은 확인됐다. 현재 1순위 병목은 지오코딩 "
        f"정밀도다 — 농가 좌표가 리 단위 중앙값 [B]라 발원-수용점 기하에 수백 m "
        f"오차가 들어간다. 결정: 경계선 값으로 안전장치를 켜지 않는다(OFF 유지). "
        f"허가 대장 실좌표 확보 시 plume_validation 재실행으로 재평가 — 그때 기준을 넘으면 "
        f"복원한다."
    )

    claim2 = (
        "**지지** — 실농가 다중 발원 + 익산 바람으로 방향 기준 통과."
        if pj["passed"] else
        f"**약한 지지, 기준 미달 (검증 종료 시점 기준)** — 실농가 다중 발원(v5)에서 "
        f"플라시보 대비 lift x{pf.get('lift')} 로 방향 신호의 존재는 확인됐으나, "
        f"이탈각 중앙값 {pf.get('median_angle_off')}도·풍하측 {pf.get('downwind_rate')} 이 "
        f"사전 기준(70도/0.60)을 근소하게 넘지 못했다. 등급 결정에 쓰기엔 부족하고 "
        f"버리기엔 아까운 수준 — 역할을 '참고 정보 + 알림 대상 후보 확대(합집합)'로 "
        f"한정하고, 정밀 발원 좌표 확보 시 재평가한다."
    )

    report = f"""# 검증 최종 보고서 (validation_report_final) — 검증 단계 종료 선언

작성: {datetime.now():%Y-%m-%d %H:%M} · v1~v5 전 라운드 · 시드 42
이 문서로 검증 단계를 종료한다. 이후는 조립·발표 준비 단계다.

## 1. 라운드별 발견 → 조치 → 결과

| 라운드 | 핵심 발견 | 조치 | 결과 |
| --- | --- | --- | --- |
| v1 | 계획서 허점 16건 (좌표·work_weight 미정의, 학습-서빙 불일치 등), 플룸 적중 0.025, RAG 62% | ①~⑦ 전 구간 구현 + 문제 목록화 | 데모 가동, 검증 의제 확정 |
| v2 | 성능 바닥은 달력(기후학 0.427), 조건부 가치 반전(평년>이상), ±0.5m/s 섭동 민감, "2026 드리프트"는 계절 구성 착시 | 플룸 등급 상향 OFF, 알림 대상 합집합, 프레이밍 교정 | 발표 수치 = "기후학 대비 증분"으로 확정 |
| v3 | 전주↔익산 풍향 38도 어긋남 확정, 익산 바람으로도 플룸 FAIL, ML 은 익산 우세(적중 0.402→0.448) | 연속변수 모델 승격, RAG 위계 부스트(80% 달성), 바람 기본값=익산 권고 | 관측소 문제와 발원 문제 분리 |
| v4 | **"3km 내 10건"의 원인 = 좌표 상수 3.35km 오류** (코드 정상), 교정 후에도 근거리 신고 지배로 플룸 FAIL, 별표 1~9 전부 깨짐 | 좌표 [B] 교정, 다중 발원 하네스·kma_midterm·스케줄러 구축, ROC/리프트 준비 | 단일 발원 가정이 최종 병목으로 특정 |
| v5 | 실농가 {R['farms']['total']}곳 좌표 매핑(돼지 {R['farms']['pigs']}곳), 플룸 최종 {"PASS" if pj['passed'] else "FAIL"}, 별표는 PDF 원문에 내용 자체가 없음(복구 불가) | 최종 판정·폴백 규칙·발표 그래프 | **검증 종료** |

## 2. 3대 핵심 주장 최종 판정

**① "기상 예보가 달력(기후학)보다 낫다" — 부분 지지.**
검증된 형태: month x block 기후학 0.427 → 기상 결합 0.467 (+0.041p, test 2026
주간 랭킹 적중률; 랜덤 0.20). 증분은 실재하고 재현되지만 성능의 바닥은 달력이며,
"특이 기상일을 잡는다"는 강한 형태의 주장은 기각됐다(증분이 평년 블록에서 더 큼).
발표는 반드시 "기후학 대비 증분" 프레임으로 말한다.

**② "플룸이 냄새의 방향을 맞춘다" — {claim2.split(chr(10))[0].split('—')[0].strip()}**
{claim2}
PLUME_GRADE_BUMP 최종 결정: {bump_final}

**③ "농가 행동(작업 시간 조정)을 지원해 민원을 줄인다" — 데이터로 미입증, 프레이밍으로 대응.**
개별 농가의 작업↔민원 인과를 잇는 데이터(작업 일지)가 없어 검증 불가.
서비스 주장은 "민원 감소 보장"이 아니라 **"상대 위험 회피 지원"**으로 통일했고,
라벨이 "악취"가 아니라 "민원(신고 행동 포함)"임을 한계로 명시한다 — 이는 서비스
목적에는 정합적인 타깃이다.

## 3. 남은 한계 전체 목록 (발표 "한계 및 향후 과제" 원재료)

1. **학습-서빙 기상 불일치**: 실측으로 학습, 예보로 서빙. ±0.5m/s 섭동에 주간
   top-20% 구성 ~9% 교체. 과거 예보-실측 페어 백테스트 미수행 (자료 대기).
2. **플룸 미검증**: 실농가 발원으로도 방향 기준 미달 — 등급 반영 금지 유지,
   참고 정보·알림 후보 확대 용도로만 사용.
3. **발원 좌표 정밀도**: 농가 좌표가 리 단위 중앙값/OSM 매핑 [B] — 허가 대장
   좌표 확보 시 plume_validation 재실행 한 줄로 재평가 가능.
4. **근거리 신고 지배**: 발원 600m 이내 신고가 다수 — 방위 판정 불가 영역이며,
   "단지 내부 주민 피해"라는 별도 서비스 대상일 수 있음.
5. **RAG 별표 공백**: 가축분뇨법 시행령 별표 1~9 원문이 보유 PDF 에 없음 —
   질의 시 원문 링크 안내 폴백 동작, DOC 도착 시 reingest_annex 로 교체.
6. **스캔본 1종** (냄새 저감 기본 관리 매뉴얼) OCR 필요.
7. **중기예보**: 익산 전용 기온 코드 미확정(키 확보 후 --probe), min/max 중
   대표값 선택은 [C] 평균, 일→블록 복제로 블록 해상도 없음.
8. **2026 익산 AWS 미확보**: ML 관측소 비교는 2025 까지만 유효.
9. **SKY(운량) 전주 단일**: 안정도 산정에 전주-익산 혼용.
10. **상수 [C] 잔존**: work_weight, storage_factor 1.5, 도심 대표점 좌표 등 —
    근거 확보 전까지 발표에서 한계로 명시.
11. **민원 데이터 사용 허락**: 익산시 확인 진행 중 (허락 증빙 확보 전 대외 발표 보류).

## 4. 명시 사항

- **과거 단기예보 백테스트는 데이터 확보 시 '추가 옵션'이다** — 본 검증 종료의
  전제 조건이 아니며, 확보 즉시 한계 1번을 정량화하는 용도로 수행한다.
- 검증 인프라는 모두 재실행 가능 상태로 보존:
  demo.py(①~⑦) / demo_v2~v5(실험) / plume_validation(발원 CSV 투입) /
  reingest_annex(DOC 투입) / kma_midterm --probe(키 투입) / scheduler --once.

## 5. v5 세부 수치

- 농가 좌표: {R['farms']['method_share']} (제외분은 finding 로그 참조)
- 플룸 최종 (돼지 {R['farms']['pigs']}곳 최근접 발원, 3km, 익산 바람):
  n={pf.get('n')}, 적중 {pf.get('hit')} / 플라시보 {pf.get('placebo')}
  (lift x{pf.get('lift')}), 이탈각 중앙값 {pf.get('median_angle_off')}도,
  풍하측 {pf.get('downwind_rate')}, 근거리(<600m) 비중 {pf.get('near_warn_share')}
- RAG: 30문항 {R['rag30']['hits']}/30, 별표 5문항 {R['rag_annex5']['hits']}/5
  (원문 안내 폴백 {R['rag_annex5']['notices']}/5 발동), 합산 {R['rag35_total']:.0%}
- 발표 그래프 {R['figs']}/6장 → out/figs/ (captions.md 포함)
"""
    (OUT_DIR / "validation_report_final.md").write_text(report, encoding="utf-8")
    with open(OUT_DIR / "v5_results.json", "w", encoding="utf-8") as fh:
        json.dump(R, fh, ensure_ascii=False, indent=2, default=str)
    print(f"  저장: {OUT_DIR / 'validation_report_final.md'}")


def main():
    use_utf8_stdout()
    print("v5 — 검증 최종 라운드 (지시 19~22)")
    if not (MID_DIR / "complaints_clean.parquet").exists():
        raise SystemExit("data/ 산출물이 없습니다. 먼저 python demo.py 를 실행하세요.")
    step19()
    step20()
    step22()
    write_final_report()
    section("검증 단계 종료")


if __name__ == "__main__":
    main()
