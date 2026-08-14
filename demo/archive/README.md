# archive/ — 과거 검증 라운드 (현역 파이프라인 밖)

여기 있는 파일은 **현재 파이프라인이 import 하지 않는다.**
2026-08-10 ~ 08-12 에 수행한 v2~v5 검증 라운드의 러너와 일회성 실험 스크립트다.

지우지 않고 남긴 이유는 **발표·보고서의 수치가 이 코드에서 나왔기 때문**이다.
결과는 `../out/v2_results.json` ~ `v5_results.json` 과
`../out/validation_report_v2.md` ~ `validation_report_v4.md` 에 있다.

## 구성

| 파일 | 무엇을 했나 |
| --- | --- |
| `demo_v2.py` | v2 라운드 — 연속변수 피처 vs 이진 플래그 비교 등 |
| `demo_v3.py` | v3 라운드 — 익산 AWS(702) 도착 후 실험, RAG 하이브리드 |
| `demo_v4.py` | v4 라운드 — 왕궁 좌표 교정(지시 14), 중기예보, 스케줄러 |
| `demo_v5.py` | v5 라운드 — 농가 좌표 투입, 다중 발원 플룸 최종 판정 |
| `v2_experiments.py` | v2 실험 본체 |
| `v3_aws.py` | 전주 ASOS vs 익산 AWS 비교 (주간적중 0.4015 → 0.4477) |
| `v4_diag14.py` | 왕궁 좌표 진단 |
| `v4_metrics.py` | v4 지표 산출 |
| `v5_farms.py` | 농가 좌표 분석 |

## 실행

`demo/` 를 기준으로 경로가 잡히도록 sys.path 를 한 단계 올려두었다.

```powershell
cd demo
python archive/demo_v2.py
python archive/demo_v5.py
```

## 주의

- 이 파일들은 **`analysis/figures.py` · `analysis/plume_validation.py` 를 여전히 참조**한다.
  그 둘은 발표 그래프와 플룸 검증 수치를 만드는 현역 산출 도구라 `analysis/` 에 남아 있다.
- 파일명은 옛 규칙(`v2`, `v3`…)을 유지했다. 과거 라운드의 식별자이므로 바꾸면
  보고서와 대조가 안 된다.
