# Other/meta 센서 기반 30분 후 NH3 데이터셋

## 정의

- 시계열 키: `chamber + pig_classification + datetime`
- 정답: 입력 마지막 시각 `t`에서 정확히 30분 뒤의 `(NH3_0.5 + NH3_1.5) / 2`
- 분할: 전체 캘린더 날짜를 시간순 70% / 15% / 15%로 먼저 분할한 뒤, 각 split 내부에서만 표본 생성
- 독립성 해석: 행이 아니라 `chamber + pig_classification + scenario_date` 블록을 반복 시계열 단위로 취급

## 시퀀스 NPZ

- 파일: `train_sensor_30m.npz`, `validation_sensor_30m.npz`, `test_sensor_30m.npz`
- `X` shape: `(n_samples, 12, 8)`; dtype `float32`
- 시간축: `t-55, t-50, ..., t-5, t` (5분 간격 12시점)
- 센서축 순서: `T_0.5, RH_0.5, CO2_0.5, NH3_0.5, T_1.5, RH_1.5, CO2_1.5, NH3_1.5`
- `y`: `target_NH3_mean_30m`
- 메타데이터: `sample_id`, `chamber`, `pig_classification`, `scenario_date`, `input_datetime`, `target_datetime`, `source_file`
- lag 표의 `60분 전` 값을 엄밀히 계산하기 위해 표본 생성 시 `t-60` 원시행도 추가로 요구합니다. NPZ `X`에는 이 행을 넣지 않습니다.

## lag-feature CSV

- 파일: `train_sensor_30m.csv`, `validation_sensor_30m.csv`, `test_sensor_30m.csv`
- 각 센서별 현재, 5/10/15/30/60분 전, 최근 15/30/60분 평균, 최근 30분 최소/최대/모표준편차, 30분 변화량을 포함합니다.
- 구간 통계는 끝점을 포함합니다. 예: 최근 30분은 `t-30..t`의 7개 관측값입니다.
- `chamber`, `pig_classification`은 범주형 입력입니다.
- 실제 학습에는 `train_sensor_30m_model.csv`, `validation_sensor_30m_model.csv`, `test_sensor_30m_model.csv`를 사용합니다. 이 파일들은 `source_file` 열을 제외합니다.
- 추적용 원본 CSV에는 검증을 위해 `source_file`을 보존합니다.
- `scenario_date`는 블록 평가와 split 확인을 위한 메타데이터이며 모델 입력에서는 제외합니다.
- 원본 추적용 `source_file`은 시퀀스 NPZ 메타데이터에 보존됩니다.

## 날짜 분할 및 표본

| split | 관측 표본 시작 | 관측 표본 종료 | 시나리오 날짜 블록 | 표본 수 |
|---|---:|---:|---:|---:|
| train | 2023-08-03 | 2023-09-15 | 124 | 33,480 |
| validation | 2023-09-16 | 2023-09-24 | 81 | 21,870 |
| test | 2023-09-25 | 2023-10-04 | 100 | 27,000 |

캘린더 경계는 다음과 같습니다: [{'split': 'train', 'calendar_start': '2023-08-03', 'calendar_end': '2023-09-15', 'calendar_days': 44, 'observed_dates': 43}, {'split': 'validation', 'calendar_start': '2023-09-16', 'calendar_end': '2023-09-24', 'calendar_days': 9, 'observed_dates': 9}, {'split': 'test', 'calendar_start': '2023-09-25', 'calendar_end': '2023-10-04', 'calendar_days': 10, 'observed_dates': 10}].

## 관리정보 보조 실험

- 각 input 시각에서 같은 `chamber + pig_classification + scenario_date` 안의 가장 최근 과거 관리값만 사용합니다.
- 허용 간격은 0~60분이며 미래값과 날짜 간 전방 채움은 금지합니다.
- `*_matched_sensor_30m.csv`와 `*_sensor_management_30m.csv`는 동일 sample_id 부분집합입니다.

## 기준모델

- 현재 NH3 평균 유지, Train 전체 평균, Train chamber+pig별 평균, Ridge Regression을 평가합니다.
- Ridge alpha: `1.0`. 수치 lag feature 표준화 + Train 범주 one-hot, 절편은 패널티에서 제외했습니다.
- 고농도는 Train target의 90분위 경계 `8.895000` 이상으로 고정합니다.
- 블록 bootstrap: `2000`회, chamber+pig+scenario_date 블록 MAE를 복원추출하여 평균 MAE 95% 신뢰구간을 계산합니다.

## 주요 결과 파일

- `baseline_metrics.csv`: 행 단위 기준모델 지표
- `block_mae.csv`, `block_mae_summary.csv`: 블록별 및 블록 요약 평가
- `split_summary.csv`, `sample_distribution.csv`, `management_coverage.csv`: 데이터 통계
- `validation_checks.csv`: 무결성 검사
- `report.json`: 기계 판독용 요약
