"""데모 공통 설정 — 경로, 실데이터 탐색, 데이터 출처(실데이터/mock) 로깅.

절대 규칙 3개 (구현 내용.md, 어떤 경우에도 위반 금지):
  1. ML(통계)과 플룸(물리)은 곱하지 않는다 — 이산 보정만 허용
  2. 서빙 피처는 예보 API가 주는 변수만 — NH3·CO2 금지
  3. legacy/ 는 수정 금지, import 만 한다 (원본: 작업폴더\최종구현 py파일)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

DEMO_DIR = Path(__file__).resolve().parent
PROJECT_DIR = DEMO_DIR.parent
DATA_ROOT = PROJECT_DIR / "프로젝트 데이터"

LEGACY_DIR = DEMO_DIR / "legacy"
OUT_DIR = DEMO_DIR / "out"
MID_DIR = DEMO_DIR / "data"          # 중간 산출물 (parquet)
DB_PATH = OUT_DIR / "demo.db"

# legacy 모듈은 평면 import(from geo import ...) 구조라 sys.path 에 등록한다.
# 파일은 절대 수정하지 않는다 (절대 규칙 3).
if str(LEGACY_DIR) not in sys.path:
    sys.path.insert(0, str(LEGACY_DIR))

# ── 실데이터 경로 (프로젝트 데이터 통합 폴더, README.md 기준) ─────────
COMPLAINTS_XLSX = DATA_ROOT / "01_민원데이터" / "익산시 악취 민원 데이터.xlsx"
WEATHER_CSV = DATA_ROOT / "02_기상데이터" / "weather_hourly_2020_202607.csv"
ASOS_FULL_CSV = DATA_ROOT / "02_기상데이터" / "asos_146_2020_2025_utf8.csv"  # 전운량 포함 27요소
RAG_PDF_DIR = DATA_ROOT / "03_RAG_법령매뉴얼"
SENSOR_XLSX = DATA_ROOT / "04_양돈센서_AIHub" / "validation_matched_sensor_30m.xlsx"

SEED = 42  # 공통 규약 4: 랜덤 시드 고정

# 플룸 기반 등급 1단계 상향 보정 스위치.
# S8-4 실증 실패(적중 0.025 vs 플라시보 0.018)로 기본 OFF — 익산 AWS 바람 확보 후
# 재검증이 통과하면 켠다. OFF 여도 "참고: 풍하측 민가 N동(미검증 모델)"은 표시한다.
PLUME_GRADE_BUMP = False

# ── 데모 시나리오 상수 ─────────────────────────────────────────────
# [B] 왕궁 축산단지 중심 = 지역=='왕궁면 흥암리' 가축분뇨 민원 727건(원본 기준)의
#     좌표 중앙값. 2026-08-10 검증: 중복제거 후 719건, 좌표 표준편차 ~100m 로
#     지역명-좌표 정합 확인. (v1~v3 의 구좌표 35.977, 127.055 [C] 는 실제에서
#     3.35km 서쪽으로 어긋나 있었음 — v4 지시 14 로 교정)
WANGGUNG_LAT, WANGGUNG_LON = 35.968937, 127.090910
# [C] 도심(부송·어양·영등동) 대표점 근사값. 위와 같은 한계.
DOWNTOWN_LAT, DOWNTOWN_LON = 35.945, 126.970

# 데모 농장 = 왕궁 축산단지 내 가상 농가 1곳
DEMO_FARM = {
    "farm_id": "F001",
    "name": "왕궁 데모농가",
    "lat": WANGGUNG_LAT,
    "lon": WANGGUNG_LON,
    "facility_type": "normal",
    "last_manure_removal_date": None,   # run 시점에 '12일 전'으로 세팅 (S7 테스트 케이스 ①)
}

# S0 필터 검증 기준치 (구현 내용.md D1 주의사항 — 다르면 필터 누락)
EXPECT_RAW_ROWS = 13039
EXPECT_IKSAN_ROWS = 11955
EXPECT_LIVESTOCK_ROWS = 5654
EXPECT_POS_RATE = 0.132

LIVESTOCK_CODE = 101


# ── 데이터 출처 로깅 ───────────────────────────────────────────────
class Provenance:
    """어떤 데이터를 썼는지(실데이터 vs mock/폴백) 기록하고 콘솔에 남긴다."""

    def __init__(self) -> None:
        self.entries: list[dict] = []

    def log(self, name: str, path, real: bool, note: str = "") -> None:
        self.entries.append(
            {"name": name, "path": str(path), "real": real, "note": note}
        )
        tag = "[실데이터]" if real else "[MOCK/폴백]"
        print(f"  {tag} {name}: {path}" + (f" — {note}" if note else ""))

    def table_md(self) -> str:
        lines = ["| 데이터 | 출처 | 경로/비고 |", "| --- | --- | --- |"]
        for e in self.entries:
            src = "실데이터" if e["real"] else "mock/폴백"
            note = f" — {e['note']}" if e["note"] else ""
            lines.append(f"| {e['name']} | {src} | {e['path']}{note} |")
        return "\n".join(lines)


PROV = Provenance()

# 구현하며 발견한 계획서의 허점·모호점·실데이터 문제를 모은다 (validation_report 용)
FINDINGS: list[str] = []


def finding(msg: str) -> None:
    FINDINGS.append(msg)
    print(f"  [발견] {msg}")


def section(title: str) -> None:
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)
