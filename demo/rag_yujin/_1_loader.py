"""
문서 로더 — 양돈장 축산악취 작업유형별 RAG용
데이터를 rag_yujin 코드와 같은 폴더 안(data/핵심자료)에 두어, 팀 공용 폴더인
"프로젝트 데이터"의 구조가 바뀌어도(실제로 이미 두 번 바뀌었었다) 영향을 받지 않게 했다.
"""
from __future__ import annotations

import re
from pathlib import Path

from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_core.documents import Document

# ── 경로 설정 ─────────────────────────────────────────────────────
# 실제 폴더 구조: smart-livestock-ai-main/demo/rag_yujin/data/핵심자료
# 코드와 데이터가 같은 폴더 트리 안에 있어 상위로 몇 단계 올라가야 하는지 셀 필요가 없다.
CURRENT_DIR = Path(__file__).resolve().parent
DATA_DIR = CURRENT_DIR / "data" / "핵심자료"

# 예전 위치(팀 공용 "프로젝트 데이터" 폴더 아래)에서 아직 안 옮겼을 경우를 대비한 대체 경로.
_LEGACY_CANDIDATES = [
    CURRENT_DIR.parent.parent / "프로젝트 데이터" / "03_RAG_법령매뉴얼" / "핵심자료",
    CURRENT_DIR.parent.parent / "프로젝트 데이터" / "03_RAG_법령매뉴얼" / "핵심자료_12개",
]


def _resolve_data_dir(preferred: Path) -> Path:
    """기본 경로(data/핵심자료)가 없으면 예전 위치(팀 공용 폴더)를 대신 찾아본다."""
    if preferred.is_dir():
        return preferred
    for candidate in _LEGACY_CANDIDATES:
        if candidate.is_dir():
            print(f"⚠ 기본 경로(data/핵심자료)가 없어 예전 위치를 대신 사용합니다: {candidate}")
            print("  → rag_yujin/data/핵심자료로 옮겨두면 다음부터 이 경고가 안 뜹니다.")
            return candidate
    tried = [preferred, *_LEGACY_CANDIDATES]
    tried_str = "\n  - ".join(str(p) for p in tried)
    raise FileNotFoundError(f"데이터 폴더를 찾지 못했습니다. 시도한 경로:\n  - {tried_str}")

# 법령류 판별 키워드 — 청킹 단계(법령=조문 단위 / 매뉴얼=섹션 단위)에서 그대로 재사용한다.
LAW_KEYWORDS = ("법률", "시행령", "시행규칙", "조례")

# 정제 텍스트(_정제텍스트.txt)로 대체된 원본 PDF는 로더에서 제외한다.
# 2026-08-15 기준: 두 원본 PDF(기본 관리 매뉴얼, 냄새 저감 공식) 모두 폴더에서
# 직접 삭제되어 지금은 비어 있다. 나중에 원본 PDF를 다시 넣게 되면 파일명을
# 여기 추가하면 정제 텍스트와 중복 로드되는 걸 막을 수 있다.
#
# [C] 2026-08-19 추가: "축산농가를_위한_냄새_저감_공식_정제텍스트.txt"는 파일 자체
# 주석에 "재OCR, OCR 노이즈 있음"이라 명시돼 있고, 실제로 그림·아이콘이 많은 페이지가
# 깨진 텍스트로 RAG 근거카드에 그대로 노출되는 게 확인돼 제외한다. 같은 내용을 다루는
# 더 깨끗한 자료(축산악취 관리 지침서.pdf 등)가 이미 있어 근거 품질 손실은 적다.
# 원본 PDF를 다시 OCR해서 정제하면 이 줄을 지우고 다시 포함시킬 수 있다.
EXCLUDE_FILES: set[str] = {"축산농가를_위한_냄새_저감_공식_정제텍스트.txt"}

# 확장자별 로더 매핑. PyPDFLoader 대신 PyMuPDFLoader를 쓰는 이유는 추출 품질이
# 더 안정적이기 때문(재OCR 검증 때 썼던 라이브러리와 동일 계열).
LOADER_MAPPING = {
    ".pdf": (PyMuPDFLoader, {}),
    ".txt": (TextLoader, {"encoding": "utf-8"}),
}

# 페이지당 평균 글자 수가 이 값 미만이면 "텍스트 추출이 부실할 수 있음" 경고를 낸다.
# (별표 스텁처럼 제목만 있고 본문이 없는 문서를 조기에 잡아내기 위함)
WEAK_TEXT_CHARS_PER_PAGE = 50

# ── 표/체크리스트/목차가 깨져서 저장되는 문제 — 파일명이 아니라 내용으로 잡는다 ──
# PyMuPDFLoader는 PDF의 시각적 레이아웃(표 칸, 체크박스 열, 목차 들여쓰기)을 모르고
# 페이지 안의 글자를 위치 순서대로 이어 붙인다. 그 결과 "예/아니오" 체크박스 표나
# 목차 페이지가 사람이 못 읽는 한 덩어리 텍스트로 추출되고, 그게 그대로 RAG
# 근거카드에 노출된다(2026-08-25, "붙임1. 축산농장 악취저감시설 운영 매뉴얼" 1·3쪽
# 확인). 이런 파일이 앞으로 또 추가될 수 있으므로, 파일명을 EXCLUDE_FILES에 하나씩
# 추가하는 대신 내용 패턴으로 감지해 해당 페이지만 자동으로 제외한다.
_CHECKLIST_TOKEN_RE = re.compile(r"(?<![가-힣])(예|아니오)(?![가-힣])")
_QUESTION_MARK_RE = re.compile(r"\?")
_TOC_LINE_RE = re.compile(r"^\d{1,3}\s+\S")

# 아래 임계값은 근거가 있는 게 아니라, 실제로 깨진 걸로 확인된 두 사례(체크리스트
# 표 1건, 목차 1건)에서 역산한 경험치다. 오탐(정상 문단인데 제외됨)이나 미탐(깨진
# 페이지가 안 걸러짐) 사례가 나오면 이 값들부터 조정한다. [C]
_CHECKLIST_MIN_HITS = 6      # "예"/"아니오" 단어가 이만큼 나오면 체크박스 표로 의심
_CHECKLIST_MIN_QMARKS = 5    # 물음표가 이만큼 나오면 "~했는가?" 식 체크항목 나열로 의심
_TOC_MIN_LINES = 5           # 최소 이만큼의 줄이 있어야 목차 판정을 시도
_TOC_MIN_RATIO = 0.5         # 그중 "숫자 + 제목" 형태인 줄의 비율이 이 이상이면 목차로 의심


def _garbled_reason(text: str) -> str | None:
    """표/체크리스트/목차가 레이아웃 없이 풀어져 못 읽는 텍스트가 됐는지 감지한다.

    걸리면 이유 문자열을, 정상으로 보이면 None을 반환한다.
    """
    stripped = text.strip()
    if not stripped:
        return None

    checklist_hits = len(_CHECKLIST_TOKEN_RE.findall(stripped))
    qmarks = len(_QUESTION_MARK_RE.findall(stripped))
    if checklist_hits >= _CHECKLIST_MIN_HITS and qmarks >= _CHECKLIST_MIN_QMARKS:
        return f"체크리스트/표 의심 ('예'/'아니오' {checklist_hits}회, '?' {qmarks}회)"

    lines = [ln.strip() for ln in stripped.split("\n") if ln.strip()]
    if len(lines) >= _TOC_MIN_LINES:
        toc_lines = sum(1 for ln in lines if _TOC_LINE_RE.match(ln))
        if toc_lines >= _TOC_MIN_LINES and toc_lines / len(lines) >= _TOC_MIN_RATIO:
            return f"목차/색인 의심 (번호 나열 줄 {toc_lines}/{len(lines)})"

    return None

# 2026-08-18: 조례 3건(구례군/완주군/정읍시) 제외 + 익산시 조례 신규 추가 결정에 따라
# 15 → 12로 조정. (법령 9 - 3 + 1(익산시) + 매뉴얼 PDF 4 + 정제텍스트 2 = 12)
# 2026-08-19: OCR 노이즈로 정제텍스트 1건을 EXCLUDE_FILES에 추가하며 12 → 11로 조정.
# (정제텍스트 2 - 1 = 1)
# 폴더를 또 건드렸을 때 조용히 개수가 달라지는 걸 바로 알아채기 위한 참고값이며,
# 강제 검증은 아니다(달라도 에러 내지 않고 안내만 한다).
EXPECTED_FILE_COUNT = 11


def classify_doc_type(filename: str) -> str:
    """법령류인지 매뉴얼류인지 파일명으로 1차 분류. 청킹 단계에서 그대로 씀."""
    return "law" if any(kw in filename for kw in LAW_KEYWORDS) else "manual"


def _check_dependencies() -> None:
    """필수 패키지가 없으면 파일마다 반복해서 실패하지 말고 시작할 때 한 번에 알린다."""
    missing = []
    try:
        import fitz  # noqa: F401  (pymupdf)
    except ImportError:
        missing.append("pymupdf")
    if missing:
        raise ImportError(
            "다음 패키지가 설치되어 있지 않습니다: " + ", ".join(missing) + "\n"
            "지금 이 스크립트를 실행 중인 환경에서 아래 명령을 실행하세요:\n"
            f"  python -m pip install {' '.join(missing)}\n"
            "(pip install만 하면 다른 파이썬 환경에 설치될 수 있어 python -m pip install을 권장합니다)"
        )


def load_law_manual_data(directory_path: Path | str) -> list[Document]:
    _check_dependencies()

    directory_path = Path(directory_path)
    if not directory_path.is_dir():
        raise FileNotFoundError(
            f"데이터 폴더가 없습니다: {directory_path}\n"
            f"경로가 맞는지, 폴더명이 '핵심자료'가 맞는지 확인하세요."
        )

    print(f"[{directory_path}] 경로에서 데이터 로드를 시작합니다...")

    documents: list[Document] = []
    failed: list[str] = []
    skipped: list[str] = []
    weak: list[str] = []
    garbled: list[str] = []

    for path in sorted(directory_path.iterdir()):
        if not path.is_file():
            continue
        if path.name.startswith(("~$", ".", "_")):
            # 임시파일(~$...), 숨김파일, 매니페스트(_데이터셋_매니페스트.md) 등은 문서가 아니므로 제외
            skipped.append(path.name)
            continue
        if path.name in EXCLUDE_FILES:
            skipped.append(path.name)
            continue

        ext = path.suffix.lower()
        if ext not in LOADER_MAPPING:
            skipped.append(path.name)
            continue

        loader_cls, loader_kwargs = LOADER_MAPPING[ext]
        try:
            docs = loader_cls(str(path), **loader_kwargs).load()
        except Exception as e:
            failed.append(path.name)
            print(f"  ❌ {path.name} 로드 실패: {e}")
            continue

        kept_docs = []
        garbled_pages = []
        for d in docs:
            reason = _garbled_reason(d.page_content)
            if reason:
                garbled_pages.append(f"p.{d.metadata.get('page', '?')}({reason})")
            else:
                kept_docs.append(d)
        docs = kept_docs
        if garbled_pages:
            garbled.append(f"{path.name}: {', '.join(garbled_pages)}")
        if not docs:
            # 이 파일의 모든 페이지가 걸러졌으면 문서 자체가 없는 것과 같으니 건너뛴다.
            skipped.append(path.name)
            continue

        doc_type = classify_doc_type(path.name)
        total_chars = sum(len(d.page_content) for d in docs)
        avg_chars = total_chars / max(len(docs), 1)
        if avg_chars < WEAK_TEXT_CHARS_PER_PAGE:
            weak.append(path.name)

        for d in docs:
            d.metadata["source_file"] = path.name
            d.metadata["doc_type"] = doc_type  # 다음 단계(청킹)에서 law/manual 분기에 사용

        documents.extend(docs)
        print(f"  ✅ {path.name} — {len(docs)}개 페이지/문서, 평균 {avg_chars:.0f}자/페이지 ({doc_type})")

    loaded_file_count = len({d.metadata["source_file"] for d in documents})
    print(f"\n✅ 총 {len(documents)}개 문서(페이지) 로드 완료 (파일 {loaded_file_count}개)")
    if skipped:
        print(f"  제외됨({len(skipped)}개, 정제 텍스트로 대체됨/비문서 파일): {skipped}")
    if failed:
        print(f"  ❌ 로드 실패({len(failed)}개): {failed}")
    if weak:
        print(f"  ⚠ 텍스트가 부실해 보이는 파일({len(weak)}개, 별표 스텁이거나 OCR 필요 가능성): {weak}")
    if garbled:
        print(f"  ⚠ 표/체크리스트/목차로 보여 제외된 페이지({len(garbled)}개 파일에서 발생): {garbled}")
    if loaded_file_count != EXPECTED_FILE_COUNT:
        print(f"  ℹ 참고: 기준 파일 개수({EXPECTED_FILE_COUNT}개)와 다릅니다. "
              f"폴더 구성이 바뀌었다면 정상이니 위 목록으로 의도한 변경이 맞는지만 확인하세요.")

    return documents


if __name__ == "__main__":
    resolved_dir = _resolve_data_dir(DATA_DIR)
    print(f"[사용 경로] {resolved_dir}")
    docs = load_law_manual_data(resolved_dir)

    if docs:
        law_docs = [d for d in docs if d.metadata["doc_type"] == "law"]
        manual_docs = [d for d in docs if d.metadata["doc_type"] == "manual"]
        print(f"\n[유형별 집계] 법령 {len(law_docs)}개 페이지 / 매뉴얼 {len(manual_docs)}개 페이지")

        print("\n[첫 번째 문서 내용 미리보기]")
        print("-" * 50)
        print(f"출처: {docs[0].metadata.get('source_file')} ({docs[0].metadata.get('doc_type')})")
        print(docs[0].page_content[:300])
        print("-" * 50)