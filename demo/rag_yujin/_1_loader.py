"""
문서 로더 — 양돈장 축산악취 작업유형별 RAG용
데이터를 rag_yujin 코드와 같은 폴더 안(data/핵심자료)에 두어, 팀 공용 폴더인
"프로젝트 데이터"의 구조가 바뀌어도(실제로 이미 두 번 바뀌었었다) 영향을 받지 않게 했다.
"""
from __future__ import annotations

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
EXCLUDE_FILES: set[str] = set()

# 확장자별 로더 매핑. PyPDFLoader 대신 PyMuPDFLoader를 쓰는 이유는 추출 품질이
# 더 안정적이기 때문(재OCR 검증 때 썼던 라이브러리와 동일 계열).
LOADER_MAPPING = {
    ".pdf": (PyMuPDFLoader, {}),
    ".txt": (TextLoader, {"encoding": "utf-8"}),
}

# 페이지당 평균 글자 수가 이 값 미만이면 "텍스트 추출이 부실할 수 있음" 경고를 낸다.
# (별표 스텁처럼 제목만 있고 본문이 없는 문서를 조기에 잡아내기 위함)
WEAK_TEXT_CHARS_PER_PAGE = 50

# 2026-08-18: 조례 3건(구례군/완주군/정읍시) 제외 + 익산시 조례 신규 추가 결정에 따라
# 15 → 12로 조정. (법령 9 - 3 + 1(익산시) + 매뉴얼 PDF 4 + 정제텍스트 2 = 12)
# 폴더를 또 건드렸을 때 조용히 개수가 달라지는 걸 바로 알아채기 위한 참고값이며,
# 강제 검증은 아니다(달라도 에러 내지 않고 안내만 한다).
EXPECTED_FILE_COUNT = 12


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