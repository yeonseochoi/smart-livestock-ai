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

# ── 서빙 DB 백엔드 ────────────────────────────────────────────────
# [2026-08-18] SQLite -> PostgreSQL(Supabase) 이관.
#   이유는 시각화가 아니라 무인 운영이다. GitHub Actions 는 잡이 끝나면
#   컨테이너가 사라져 demo.db 가 증발한다. 매일 자동 적재를 하려면 DB 가
#   저장소 밖에 있어야 한다.
#   단, 인터넷·계정 없이도 로컬 개발과 run_check 가 돌아가야 하므로
#   SQLite 경로를 지우지 않고 폴백으로 남겼다.
#     DATABASE_URL 있음 -> PostgreSQL   /   없음 -> 기존 out/demo.db
#   접속 문자열은 .env 에만 두고 절대 커밋하지 않는다 (.gitignore 참조).
def _load_dotenv(*paths) -> None:
    """python-dotenv 없이도 동작하는 최소 .env 로더.

    이미 있는 환경변수는 덮어쓰지 않는다 (CI 의 Secrets 가 우선).
    """
    for path in paths:
        try:
            if not path.exists():
                continue
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                k, v = k.strip(), v.strip().strip("'").strip('"')
                if k and k not in os.environ:
                    os.environ[k] = v
        except Exception:
            continue        # .env 가 깨져도 import 자체는 살아 있어야 한다


_load_dotenv(DEMO_DIR / ".env", PROJECT_DIR / ".env")
DATABASE_URL = os.environ.get("DATABASE_URL", "").strip()
DB_BACKEND = "postgres" if DATABASE_URL else "sqlite"

# legacy 모듈은 평면 import(from geo import ...) 구조라 sys.path 에 등록한다.
# 파일은 절대 수정하지 않는다 (절대 규칙 3).
if str(LEGACY_DIR) not in sys.path:
    sys.path.insert(0, str(LEGACY_DIR))

# ── 실데이터 경로 (프로젝트 데이터 통합 폴더, README.md 기준) ─────────
# [2026-08-18] 민원 재크롤링본으로 교체 — 2019-05-28 부터 2026-08-17 까지 확장.
#   구본 익산시 악취 민원 데이터.xlsx : 13,039행 / 2020-01-01 ~ 2026-07-30
#   신본 _20190528-20260818.xlsx      : 16,113행 / 2019-05-28 ~ 2026-08-17  (+3,074행)
#   가축분뇨(101) 익산 기준 5,654 -> 7,547건 (+1,893, +33.5%)
COMPLAINTS_XLSX = DATA_ROOT / "01_민원데이터" / "익산시 악취 민원 데이터_20190528-20260818.xlsx"
COMPLAINTS_XLSX_LEGACY = DATA_ROOT / "01_민원데이터" / "익산시 악취 민원 데이터.xlsx"  # 구본 보존

# [2026-08-18] 기상 관측을 기상청 API허브 수집본으로 교체 (preprocess/kma_obs.py).
#   포털 수동 CSV : 57,695행 / 2020-01 ~ 2026-07 / 전운량 없음
#   API허브 수집본: 66,882행 / 2019-01 ~ 2026-08 / 전운량(CA_TOT) 포함
#   검증: 2020-01 한 달 744시각을 양쪽으로 받아 대조 -> 기온·습도·풍속·풍향
#         744/744 완전 일치, 최대차 0.00. 재수집으로 값이 바뀌지 않음을 확인했다.
WEATHER_CSV = DATA_ROOT / "02_기상데이터" / "asos_146_api_2019_2026.csv"        # 전주 ASOS 146 (API)
WEATHER_CSV_LEGACY = DATA_ROOT / "02_기상데이터" / "weather_hourly_2020_202607.csv"  # 포털 수동본 보존
IKSAN_AWS_CSV = DATA_ROOT / "02_기상데이터" / "aws_702_2020_2025_utf8.csv"     # 익산 AWS 702
ASOS_FULL_CSV = DATA_ROOT / "02_기상데이터" / "asos_146_2020_2025_utf8.csv"  # 전운량 포함 27요소

# ── 기상 관측 지점 선택 ────────────────────────────────────────────
# "jeonju_asos" : 전주 ASOS 146.  2020~2026.07, 풍향 16방위(20도 격자), 무풍 7.1%
# "iksan_aws"   : 익산 AWS 702.   2020~2025,   풍향 연속값(3,601종),   무풍 16.5%
#
# 실측 비교 (분할 동일 <=2023/2024/2025, 시드 5개, test)
#                    시가지 AP     시가지 ROC   시가지 hit
#   전주 ASOS 146      0.2197        0.8971       0.6376   ← 채택
#   익산 AWS  702      0.2138        0.8824       0.6093
#
# 익산이 더 가까운데도 진 이유 (진단)
#   · 방향 신호 자체는 익산이 오히려 낫다 — 민원시각 풍향 lift 최대/최소
#     익산 16.6배 vs 전주 16.3배. 즉 익산 풍향이 틀린 게 아니다.
#   · 그러나 무풍(<0.5m/s) 비율이 16.5% 대 7.1% 로 2.3배다. 이 구간에서
#     풍향은 사실상 난수이고, 공간 피처가 그 난수로 부채꼴을 그린다.
#   · 결측도 408시각 대 34시각으로 12배 많다 (무인 관측).
#   · calm 플래그 추가·무풍 마스킹을 해봐도 뒤집히지 않았다 (cmp_calm.py).
#   · 남는 설명 — 전주의 20도 양자화가 국지 난류를 걸러내는 스무딩으로 작동해
#     '냄새가 퍼지는 광역 기상 패턴'의 대리변수로 더 안정적이다.
#
# 역할 분담: 예측(ML)은 전주, 방향 판정(플룸 plume_validation)은 익산 AWS 를 쓴다.
# 전운량(대기안정도)은 AWS 에 없어 전주 ASOS 만 가능하며, ML 피처에는 없다.
WEATHER_SOURCE = "jeonju_asos"

# 시계열 분할 연도. 익산 AWS 커버리지(~2025)에 맞춰 한 칸씩 당겼다.
# 문서 5장 원안과 동일하며, test(2026)가 7개월·양성 90개뿐이던 문제도 함께 해소된다.
SPLIT_TRAIN_END = 2023      # train: <= 이 연도
SPLIT_VALID_YEAR = 2024
SPLIT_TEST_YEAR = 2025
RAG_PDF_DIR = DATA_ROOT / "03_RAG_법령매뉴얼"
SENSOR_XLSX = DATA_ROOT / "04_양돈센서_AIHub" / "validation_matched_sensor_30m.xlsx"

SEED = 42  # 공통 규약 4: 랜덤 시드 고정

# 액비 살포 시즌 필터. 근거 미확보 상태이므로 기본은 '연중'(None).
# 익산시 축산과·축협 확인 후 [3,4,5,9,10,11] 등으로 바꾼다.
# 근거 없이 좁히면 표본이 급감(주간 양성 블록 약 214개)해 검정력이 무너진다.
SPREAD_SEASON_MONTHS = None      # None = 연중
SPREAD_HOURS = None              # None = 전 시간대, 예: range(8, 18)

# 플룸 기반 등급 1단계 상향 보정 스위치.
# v5 최종: lift 1.61(PASS) / 이탈각 70.5도(FAIL, 기준 70) / 풍하측 0.5759(FAIL, 기준 0.60)
# 현행 결정: 영구 OFF. 플룸은 등급을 보정하지 않고 '조합 시 지역 선택'에만 쓴다.
#          근거 — 피처(풍상측 노출)와 조합(지역 선택) 두 곳에서 이미 물리를 쓰므로
#          여기까지 켜면 삼중 계산이 되어 절대규칙 1 위반.
PLUME_GRADE_BUMP = False

# ── 데모 시나리오 상수 ─────────────────────────────────────────────
# [B] 왕궁 축산단지 중심 = 지역=='왕궁면 흥암리' 가축분뇨 민원 727건(원본 기준)의
#     좌표 중앙값. 2026-08-10 검증: 중복제거 후 719건, 좌표 표준편차 ~100m 로
#     지역명-좌표 정합 확인. (v1~v3 의 구좌표 35.977, 127.055 [C] 는 실제에서
#     3.35km 서쪽으로 어긋나 있었음 — v4 지시 14 로 교정)
WANGGUNG_LAT, WANGGUNG_LON = 35.968937, 127.090910
# [C] 도심(부송·어양·영등동) 대표점 근사값. 위와 같은 한계.
DOWNTOWN_LAT, DOWNTOWN_LON = 35.945, 126.970

# ── 익산시 경계 상자 — 사용자 위치 입력 1차 게이트 ─────────────────
# [C] 익산시 행정경계의 외접 사각형 근사값이다. 실제 경계 폴리곤이 아니다.
#     쓰임은 "명백히 익산 밖인 좌표"를 알아채는 데 한정한다.
#
#     ★ 막는 용도가 아니라 알리는 용도다.
#       범위를 벗어나도 화면은 그대로 돌아가고, 대신 "지금 보고 있는 위험도는
#       익산 기준"이라는 사실을 배지로 표시한다. 막아 버리면 발표 중 심사위원이
#       자기 위치를 찍어 보다가 화면이 죽는다. 폴백을 탔다는 사실을 반드시
#       표시한다는 이 프로젝트의 규약과 같은 태도다.
#
#     정식 경계가 필요해지면 VWORLD 행정구역 경계 API 로 교체한다.
IKSAN_BBOX = (35.85, 126.78, 36.15, 127.20)      # (남위, 서경, 북위, 동경)

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
# [2026-08-18] 민원 재크롤링본(2019-05~2026-08) 기준으로 갱신. 괄호는 구본 값.
EXPECT_RAW_ROWS = 16113          # (구본 13039)
EXPECT_IKSAN_ROWS = 14994        # (구본 11955)
EXPECT_LIVESTOCK_ROWS = 7547     # (구본  5654)
EXPECT_POS_RATE = 0.132          # 3시간블록 양성률 — v6 는 1시간격자라 참고값

LIVESTOCK_CODE = 101

# ── 수용점 그룹 정의 ──────────────────────────────────────────────
# complaints_clean.parquet 의 '지역' 컬럼 값과 정확히 일치해야 한다.
# 근거: 물리 노출 지수 대비 민원 수 상관 0.099 → 지역 차이는 거리·구조가 가름.
#       도달 시간이 다르면 지배 물리가 다르므로 모델을 분리한다.
GROUP_RURAL = "농촌근거리"
GROUP_URBAN = "시가지원거리"

REGION_GROUP = {
    # 농촌 근거리형 — 축산단지 내부 또는 인접, 발원 거리 0.2~2km
    "왕궁면 흥암리": GROUP_RURAL,
    "오산면 신지리": GROUP_RURAL,
    "낭산면 용기리": GROUP_RURAL,
    "춘포면 쌍정리": GROUP_RURAL,
    "춘포면 신동리": GROUP_RURAL,
    # 시가지 원거리형 — 발원 거리 5~14km
    "부송동": GROUP_URBAN,
    "어양동": GROUP_URBAN,
    "영등동": GROUP_URBAN,
    "동산동": GROUP_URBAN,
    "마동": GROUP_URBAN,
    "평화동": GROUP_URBAN,
    "모현동1가": GROUP_URBAN,
    "팔봉동": GROUP_URBAN,
    "송학동": GROUP_URBAN,
}
GROUPS = (GROUP_RURAL, GROUP_URBAN)

# ── 지역 격자용 확장 매핑 ─────────────────────────────────────────
# 민원 50건 이상 지역 21개. 커버율 93.4% (5,195/5,563).
# 기존 14개는 REGION_GROUP 을 그대로 승계하고, 신규 7개만 아래 규칙으로 배정했다.
#   규칙: 지역 민원 좌표 중앙값 → 최근접 익산 돼지농가 거리 < 3km 이면 근거리형.
#   근거: 문서 3장의 그룹 정의(근거리 0.2~2km / 원거리 5~14km) 그대로.
# [미해결] 이 규칙을 기존 14개에 적용하면 부송동(2.81km)·팔봉동(2.26km)이
#          근거리형으로 뒤집힌다. 문서 매핑을 존중해 그대로 두었으나 확인 필요.
REGION_GROUP_R = dict(REGION_GROUP)
REGION_GROUP_R.update({
    "왕궁면 쌍제리": GROUP_RURAL,    # 최근접 0.31km
    "용동면 용성리": GROUP_RURAL,    # 최근접 0.19km
    "삼기면 오룡리": GROUP_RURAL,    # 최근접 1.22km
    "신흥동": GROUP_URBAN,          # 최근접 3.13km
    "황등면 황등리": GROUP_URBAN,    # 최근접 3.95km
    "신동": GROUP_URBAN,            # 최근접 5.44km
    "인화동2가": GROUP_URBAN,        # 최근접 6.22km
})
REGIONS = tuple(REGION_GROUP_R)

# 지역별 중심 좌표는 build_grid.run_regional() 이 민원 좌표 중앙값으로 산출해 채운다.
REGION_CENTER: dict = {}

# 그룹별 대표 좌표 = 해당 그룹 민원 좌표의 중앙값. s1 실행 시 자동 산출해 덮어쓴다.
# 여기 값은 폴백이며 [C] 등급이다.
GROUP_CENTER = {
    GROUP_RURAL: (35.968937, 127.090910),   # WANGGUNG 좌표와 동일 [B]
    GROUP_URBAN: (35.945, 126.970),         # DOWNTOWN 근사 [C]
}

# ★ 학습(build_grid.run)이 산출한 실제 중심을 파일에서 되읽는다.
#   이게 없으면 학습은 민원 중앙값 중심으로 공간 피처를 만들고, 서빙은 위 폴백값
#   중심으로 만들어 두 분포가 어긋난다 (시가지 기준 약 0.9km 차이).
_GC_PATH = MID_DIR / "group_center.json"
if _GC_PATH.exists():
    try:
        import json as _json
        _saved = _json.loads(_GC_PATH.read_text(encoding="utf-8"))
        for _k, _v in _saved.items():
            if _k in GROUP_CENTER:
                GROUP_CENTER[_k] = tuple(_v)
    except Exception:
        pass

_RC_PATH = MID_DIR / "region_center.json"
if _RC_PATH.exists():
    try:
        import json as _json
        REGION_CENTER.update({k: tuple(v) for k, v in
                              _json.loads(_RC_PATH.read_text(encoding="utf-8")).items()})
    except Exception:
        pass


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
