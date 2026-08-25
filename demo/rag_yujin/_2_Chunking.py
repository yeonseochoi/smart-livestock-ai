"""
텍스트 분할(청킹) — 양돈장 축산악취 작업유형별 RAG용

전략:
  - doc_type == "law"    → 정규식 기반 "제N조/별표N" 경계로 우선 분할 (글자수로 자르지 않음),
                           그 결과가 너무 길 때만 RecursiveCharacterTextSplitter로 2차 보조 분할
  - doc_type == "manual" → 소제목(장/절/번호) 경계로 우선 분할, 섹션이 너무 길 때만
                           RecursiveCharacterTextSplitter로 2차 보조 분할

CharacterTextSplitter/RecursiveCharacterTextSplitter를 그대로 쓰지 않는 이유:
  둘 다 글자 수 기준이라 "제7조에 따라..." 같은 본문 내 조문 참조와 실제 조문 시작을
  구분하지 못해 법령 조문이 중간에 잘릴 수 있다. 법령 답변 정확도가 중요하므로
  법령은 반드시 구조(조문 경계) 기준으로 먼저 자르고, 그 안에서 너무 길 때만 보조로
  글자수 기준을 쓴다(매뉴얼과 동일한 원칙).
"""
from __future__ import annotations

import hashlib
import re
from collections import defaultdict

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ── 법령용 정규식 (조문/별표 경계) ─────────────────────────────────
# 줄 시작(^)에서만 매치해야 한다 — "제7조에 따라" 같은 본문 내 참조가
# 새 청크 경계로 오인되면 조문이 쪼개진다.
ARTICLE_RE = re.compile(r"(?m)(?=^\s*제\s*\d+\s*조(?:의\s*\d+)?(?![가-힣\d])\s*(?:\([^)]*\))?)")
ANNEX_RE = re.compile(r"(?m)(?=^\s*\[?별표\s*\d+(?:의\s*\d+)?\]?)")

# ── 매뉴얼용 소제목 패턴 ────────────────────────────────────────
# "■ " 항목은 "붙임1. 축산농장 악취저감시설 운영 매뉴얼(돼지)" 실제 원문 확인 후 추가함
# (예: "■ 자연환기 돈사" / "■ 강제배기 돈사" — 같은 설비를 환기방식별로 나누는 실제 절 경계).
# "•"는 추가하지 않았다 — 같은 문서에서 "•"는 개요 항목의 본문 불릿(예: "• 안개분무 시스템의
# 정상작동 여부")으로 쓰이고 있어서, 헤딩으로 인식하면 문장 하나하나가 쪼개져 버린다.
HEADING_RE = re.compile(r"^\s*(제\s*\d+\s*[장절]|\d+(?:\.\d+)*[.)]\s+|[가-하][.)]\s+|[①-⑳]\s*|■\s*)")

# 재OCR 정제 텍스트(_정제텍스트.txt)에 넣어둔 "## N쪽" 표시. TextLoader는 페이지
# 메타데이터를 안 주기 때문에, 이 표시가 없으면 그 문서의 모든 청크가 page=1로
# 뭉개진다 — 실제로 이 버그가 있었어서 여기서 페이지 메타데이터로 복원한다.
PAGE_MARKER_RE = re.compile(r"(?m)^##\s*(\d+)쪽\s*$\n?")

# PyMuPDF가 페이지 여백의 쪽번호를 본문 첫 줄로 그대로 뽑아오는 경우가 있다
# (예: "3\n축산환경관리원\n개요\n..."의 맨 앞 "3"). 쪽번호는 이미 metadata['page']로
# 따로 갖고 있으므로, 본문에 숫자만 있는 줄이 남으면 청크 텍스트만 지저분해진다.
_LEADING_PAGE_NUMBER_RE = re.compile(r"^\s*\d{1,4}\s*\n")

# [C] 2026-08-26: PDF에서 뽑은 줄바꿈은 대부분 "문단 구분"이 아니라 그냥 화면
# 폭에 맞춘 "시각적 줄바꿈"이라 문장 중간 아무 데서나 나온다. 2차 글자수 분할
# (_manual_fallback_splitter/_law_fallback_splitter)의 구분자 우선순위가
# ["\n\n", "\n", ". ", "다. ", ...] 순서인데, "\n"이 "다. "(문장 종결)보다
# 먼저 시도되다 보니 PDF가 문장 중간에서 줄을 바꾼 지점을 그대로 청크 경계로
# 써버린다. 실제로 "축산악취 관리 지침서.pdf 제2장 (2)" 청크가 "수 있다. ·..."로
# 시작하는 문제로 확인됨 — 원래 문장 "...예방할 수 있다."이 줄바꿈에서 반으로
# 잘려 뒷부분만 다음 청크 맨 앞에 남은 것이다. 진짜 문단 구분("\n\n", 빈 줄)은
# 그대로 두고, 홑 줄바꿈만 공백으로 펴서 "\n"이 더는 문장 중간을 끊는 분리자로
# 안 쓰이게 한다.
_SINGLE_NEWLINE_RE = re.compile(r"(?<!\n)\n(?!\n)")

# [C] 2026-08-26 추가: 위 처리로 문장 중간 줄바꿈은 없앴지만, 그러면서 "▪ 항목1
# ∙ 세부내용 ▪ 항목2 ∙ 세부내용"처럼 원래 줄이 나뉘어 있던 목록 항목까지 전부
# 한 줄로 뭉쳐져 근거카드가 읽기 어려운 한 덩어리 글이 됐다("수 있다"류 버그를
# 고치다 보니 이번엔 가독성이 나빠짐). 줄바꿈 바로 다음이 목록 기호로 시작하면
# 그건 문장 중간이 아니라 진짜 항목 구분이므로, 그 줄바꿈만은 살려서 화면에
# 항목별로 줄이 나뉘어 보이게 한다("축산악취 관리 지침서.pdf"에서 실제 쓰인
# 기호 ▪·∙ 기준으로 확인함, 법령의 ①~⑳·가./나.·(1)/(2)·1./2.도 같이 잡는다).
_BULLET_LINE_RE = re.compile(
    r"^(▪|∙|•|◦|－|\(\d{1,2}\)|\d{1,2}[.)]\s|[가-하][.)]\s|[①-⑳])"
)


def _flatten_wraps(text: str) -> str:
    """PDF의 시각적 줄바꿈을 정리해 문장은 안 끊기고 목록 구조는 남긴다.

    문단 구분("\\n\\n")은 그대로 둔다. 홑 줄바꿈은 원칙적으로 공백으로 펴서
    문장 중간에서 청크가 잘리지 않게 하되, 줄바꿈 다음이 목록 기호로 시작하면
    그 줄바꿈은 항목 구분이므로 살려 둔다(화면 표시는 st.text가 줄바꿈을
    그대로 보여준다). 이 함수는 소제목/조문 경계로 이미 나뉜 섹션 텍스트에만
    쓴다 — 원문 전체(full_text)에 쓰면 HEADING_RE가 줄 단위로 소제목을 찾는
    로직이 깨진다.
    """

    def _replace(match: re.Match) -> str:
        rest = text[match.end():]
        return "\n" if _BULLET_LINE_RE.match(rest) else " "

    return _SINGLE_NEWLINE_RE.sub(_replace, text).strip()


# [C] 2026-08-26 추가: 위 두 수정 이후에도 2차 글자수 분할의 overlap(겹침) 구간이
# 조각 경계에서 앞 조각 문장의 마침표·불릿만 살짝 걸쳐 남기는 경우가 있었다
# (실제 확인: "축산악취 관리 지침서.pdf 제2장 (2)"가 ". · 축사 외부로..."로 시작,
# "제1장 (2)"가 ". · 또한, 축사..."로 시작 — 둘 다 앞 조각 끝의 "다. ·" 잔재만
# 남고 실제 내용은 그 다음부터 시작함). 진짜 문장·목록 항목은 마침표·가운뎃점만
# 으로 시작하지 않으므로 안전하게 지울 수 있다.
_LEADING_DEBRIS_RE = re.compile(r"^[\s\.,·•\-–—]+")


def _strip_split_debris(text: str) -> str:
    """2차 분할의 overlap 경계에 남는 자투리 문장부호를 앞에서 지운다."""
    return _LEADING_DEBRIS_RE.sub("", text)


MIN_CHUNK_LEN = 40
MANUAL_CHUNK_SIZE = 700       # ko-sroberta 등 임베딩 모델 토큰 한계를 고려한 보수적 값
MANUAL_CHUNK_OVERLAP = 80
MANUAL_SPLIT_THRESHOLD = int(MANUAL_CHUNK_SIZE * 1.3)  # 이 길이를 넘는 섹션만 2차 분할

# 법령도 매뉴얼과 같은 원칙: 조문/별표 경계 분할이 항상 우선이고, 그 결과가 너무 길
# 때만 보조로 글자수 기준 분할을 쓴다. 이 상한이 원래 없었는데, 실제 데이터로 확인해보니
# "제16조의6"이 10,051자, "별표3"이 6,424자짜리 단일 청크로 남아있었다(_peek_chunk_sizes.py
# 로 실측 확인) — 이렇게 길면 임베딩 하나가 서로 다른 여러 항·호 내용을 뭉뚱그리게 되어
# 검색 정밀도/재현율을 둘 다 깎아먹는다. 매뉴얼과 같은 700자 기준을 그대로 적용한다.
LAW_CHUNK_SIZE = 700
LAW_CHUNK_OVERLAP = 100
LAW_SPLIT_THRESHOLD = int(LAW_CHUNK_SIZE * 1.3)  # 이 길이를 넘는 조문/별표만 2차 분할

# HEADING_RE는 "2) 관행수준의 조단백질 함량：비육전기(50~80kg) 19%, ..." 처럼 번호로
# 시작하는 "긴 문장(본문)"도 소제목으로 오인해서 매칭할 때가 있다. 이런 걸 그대로 라벨로
# 쓰면, 그 섹션이 나중에 글자수 기준으로 여러 조각(‑1, ‑2, ‑3...)나 뉘고 뒤쪽 조각은 원래
# 문장과 전혀 다른 내용(예: 탈취법 설명)을 담고 있는데도 그 라벨을 그대로 물려받아서
# 출처가 엉뚱해 보인다(실제로 "축산악취 관리 지침서.pdf"에서 이 현상이 확인됨).
# 진짜 소제목은 보통 짧은 명사구라서, 길이로 걸러낸다 — 파일마다 정규식을 계속 추가하는
# 것보다 이렇게 일반적으로 막는 쪽이 다른 파일에도 안전하게 적용된다.
MAX_HEADING_LABEL_LEN = 30

_manual_fallback_splitter = RecursiveCharacterTextSplitter(
    chunk_size=MANUAL_CHUNK_SIZE,
    chunk_overlap=MANUAL_CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", "다. ", " ", ""],  # 한국어 문장 종결에 맞춘 구분자 우선순위
)

# 법령 조문은 보통 "①②③..." 항 번호나 "1. / 가." 호 번호가 줄 단위로 이어지므로,
# 매뉴얼과 동일한 구분자 우선순위를 쓰면 항·호 경계에서 자연스럽게 갈라진다
# (원문 추출 단계에서 이미 각 항이 줄바꿈으로 구분되어 있음).
_law_fallback_splitter = RecursiveCharacterTextSplitter(
    chunk_size=LAW_CHUNK_SIZE,
    chunk_overlap=LAW_CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", "다. ", " ", ""],
)


# ── 공통 유틸 ──────────────────────────────────────────────────────
def _group_pages_by_source(documents: list[Document]) -> dict[str, list[Document]]:
    grouped: dict[str, list[Document]] = defaultdict(list)
    for d in documents:
        grouped[d.metadata["source_file"]].append(d)
    for docs in grouped.values():
        docs.sort(key=lambda d: d.metadata.get("page", 0))
    return grouped


def _join_with_page_spans(page_docs: list[Document]) -> tuple[str, list[tuple[int, int, int]]]:
    """페이지(또는 TXT 전체)를 이어붙이며 (시작오프셋, 끝오프셋, 페이지번호)를 기록한다.
    조문이 페이지 경계에 걸쳐 있어도 정확한 페이지 번호를 되찾기 위함."""
    parts, spans, cursor = [], [], 0
    for d in page_docs:
        text = _LEADING_PAGE_NUMBER_RE.sub("", d.page_content, count=1)
        page_no = d.metadata.get("page", 0) + 1  # PyMuPDFLoader는 0-base, TXT는 page 없음(=1)
        parts.append(text)
        spans.append((cursor, cursor + len(text) + 1, page_no))
        cursor += len(text) + 1
    return "\n".join(parts), spans


def _page_for_offset(spans: list[tuple[int, int, int]], offset: int) -> int:
    for start, end, page in spans:
        if start <= offset < end:
            return page
    return spans[-1][2] if spans else 1


def _make_chunk_id(source: str, unit: str, page, text: str) -> str:
    digest = hashlib.sha1(f"{source}|{unit}|{page}|{text}".encode("utf-8")).hexdigest()
    return digest[:16]


def _clean_heading_label(raw_heading: str) -> str:
    """HEADING_RE에 매칭됐다고 해서 다 진짜 소제목은 아니다 — 번호로 시작하는 긴 문장을
    소제목으로 오인한 경우, 그걸 그대로 citation 라벨로 쓰면 나중에 사람이 답변의 출처를
    검증하려 할 때 엉뚱한 내용을 가리키게 된다. 짧은 것만 라벨로 인정하고, 길면
    "본문"이라는 정직한 일반 라벨로 대체한다(거짓으로 구체적인 척하는 라벨보다 낫다)."""
    heading = raw_heading.strip()
    if len(heading) <= MAX_HEADING_LABEL_LEN:
        return heading
    return "본문"


def _expand_page_markers(documents: list[Document]) -> list[Document]:
    """재OCR 정제 텍스트(.txt)에 심어둔 "## N쪽" 표시를 페이지 단위 Document로 쪼갠다.

    TextLoader는 파일 전체를 하나의 Document로 반환하고 page 메타데이터를 주지 않는다.
    그 상태로 두면 _join_with_page_spans가 이 문서의 모든 청크를 page=1로 뭉갠다.
    PDF에서 온 Document(페이지 마커가 없음)는 그대로 통과시킨다.
    """
    expanded: list[Document] = []
    for d in documents:
        matches = list(PAGE_MARKER_RE.finditer(d.page_content))
        if not matches:
            expanded.append(d)
            continue

        # 첫 마커 이전에 내용이 있으면(마커 없이 시작하는 서두) page=1로 보존한다.
        first_start = matches[0].start()
        if d.page_content[:first_start].strip():
            piece_meta = dict(d.metadata)
            piece_meta["page"] = 0  # _join_with_page_spans에서 +1 되어 최종 1쪽
            expanded.append(Document(page_content=d.page_content[:first_start], metadata=piece_meta))

        for i, m in enumerate(matches):
            page_num = int(m.group(1))
            content_start = m.end()
            content_end = matches[i + 1].start() if i + 1 < len(matches) else len(d.page_content)
            piece_text = d.page_content[content_start:content_end]
            if not piece_text.strip():
                continue
            piece_meta = dict(d.metadata)
            piece_meta["page"] = page_num - 1  # _join_with_page_spans에서 +1 되어 최종 page_num
            expanded.append(Document(page_content=piece_text, metadata=piece_meta))

    return expanded


# ── 법령 청킹 ──────────────────────────────────────────────────────
def chunk_law_document(source_file: str, page_docs: list[Document]) -> list[Document]:
    """조문/별표 단위로 우선 분할(글자수 기준으로 절대 자르지 않음). 그 결과가
    LAW_SPLIT_THRESHOLD보다 길 때만 보조로 글자수 기준 2차 분할을 한다(매뉴얼과 동일한
    원칙) — "제16조의6"처럼 10,000자가 넘는 조문이 통째로 한 청크로 남아있던 문제 수정."""
    full_text, spans = _join_with_page_spans(page_docs)
    starts = sorted(set(
        [0]
        + [m.start() for m in ARTICLE_RE.finditer(full_text)]
        + [m.start() for m in ANNEX_RE.finditer(full_text)]
    ))

    chunks = []
    for pos, end in zip(starts, starts[1:] + [len(full_text)]):
        raw_text = full_text[pos:end].strip()
        if len(raw_text) < MIN_CHUNK_LEN:
            continue
        first_line = raw_text.splitlines()[0].strip()[:80]
        article_m = re.match(r"제\s*\d+\s*조(?:의\s*\d+)?", first_line)
        annex_m = re.match(r"\[?별표\s*\d+(?:의\s*\d+)?\]?", first_line)
        unit = (annex_m or article_m).group(0) if (annex_m or article_m) else "서문"
        page = _page_for_offset(spans, pos)
        is_annex = bool(annex_m)
        text = _flatten_wraps(raw_text)

        if len(text) <= LAW_SPLIT_THRESHOLD:
            chunks.append(Document(
                page_content=text,
                metadata={
                    "source_file": source_file,
                    "doc_type": "law",
                    "unit": unit,
                    "page": page,
                    "is_annex": is_annex,
                    "chunk_id": _make_chunk_id(source_file, unit, page, text),
                },
            ))
        else:
            # 조문/별표 경계로는 이미 다 나눴는데도 여전히 긴 경우(항·호가 많은 조문,
            # 표 형태의 별표 등)만 여기서 글자수 기준으로 보조 분할한다. label에
            # (i+1)을 붙여 어느 조문의 몇 번째 조각인지 출처에서 바로 알 수 있게 한다.
            for i, piece in enumerate(_law_fallback_splitter.split_text(text)):
                if i > 0:
                    piece = _strip_split_debris(piece)
                chunks.append(Document(
                    page_content=piece,
                    metadata={
                        "source_file": source_file,
                        "doc_type": "law",
                        "unit": f"{unit} ({i + 1})",
                        "page": page,
                        "is_annex": is_annex,
                        "chunk_id": _make_chunk_id(source_file, f"{unit}-{i}", page, piece),
                    },
                ))
    return chunks


# ── 매뉴얼 청킹 ────────────────────────────────────────────────────
def chunk_manual_document(source_file: str, page_docs: list[Document]) -> list[Document]:
    """소제목 단위로 우선 분할하고, 섹션이 너무 길 때만
    RecursiveCharacterTextSplitter로 2차 분할(구조 분할이 항상 우선)."""
    full_text, spans = _join_with_page_spans(page_docs)

    sections: list[tuple[str, int, str]] = []  # (제목, 시작오프셋, 본문)
    heading, start, cursor, buf = "서문", 0, 0, []
    for line in full_text.splitlines(keepends=True):
        if HEADING_RE.match(line) and len(line.strip()) <= 100:
            # 첫 줄부터 소제목인 경우(buf가 비어 있음)에도 그 제목이 "서문"으로
            # 잘못 남지 않도록, flush와 heading 갱신을 분리한다.
            if buf:
                sections.append((heading, start, "".join(buf)))
                buf = []
            heading, start = line.strip()[:80], cursor
        buf.append(line)
        cursor += len(line)
    if buf:
        sections.append((heading, start, "".join(buf)))

    chunks = []
    for heading, start_offset, raw_text in sections:
        raw_text = raw_text.strip()
        if len(raw_text) < MIN_CHUNK_LEN:
            continue
        page = _page_for_offset(spans, start_offset)

        label = _clean_heading_label(heading)
        text = _flatten_wraps(raw_text)

        if len(text) <= MANUAL_SPLIT_THRESHOLD:
            chunks.append(Document(
                page_content=text,
                metadata={
                    "source_file": source_file, "doc_type": "manual",
                    "unit": label, "page": page,
                    "chunk_id": _make_chunk_id(source_file, label, page, text),
                },
            ))
        else:
            # 구조로 나눈 섹션 자체가 여전히 길 때만 여기서 글자수 기준으로 보조 분할.
            # 뒤쪽 조각(i>=1)일수록 원래 헤딩과 실제 내용이 멀어질 위험이 커지므로
            # (섹션 앞부분 헤딩을 그대로 물려받지만 그 사이 다른 내용으로 흘러갔을 수 있음),
            # label을 이미 정직하게 정리해둔 값으로 통일해서 쓴다.
            for i, piece in enumerate(_manual_fallback_splitter.split_text(text)):
                if i > 0:
                    piece = _strip_split_debris(piece)
                chunks.append(Document(
                    page_content=piece,
                    metadata={
                        "source_file": source_file, "doc_type": "manual",
                        "unit": f"{label} ({i + 1})", "page": page,
                        "chunk_id": _make_chunk_id(source_file, f"{label}-{i}", page, piece),
                    },
                ))
    return chunks


# ── 진입점 ────────────────────────────────────────────────────────
def chunk_documents(documents: list[Document]) -> list[Document]:
    documents = _expand_page_markers(documents)
    grouped = _group_pages_by_source(documents)
    all_chunks: list[Document] = []

    for source_file, page_docs in grouped.items():
        doc_type = page_docs[0].metadata.get("doc_type", "manual")
        chunker = chunk_law_document if doc_type == "law" else chunk_manual_document
        chunks = chunker(source_file, page_docs)

        # 중복 제거는 "같은 문서 안"에서만 한다(문서별로 seen을 새로 시작).
        # 전역으로 하면 서로 다른 문서(예: 구례군/완주군/정읍시 조례처럼 조문이
        # 거의 같은 별개 조례)가 우연히 같은 문장을 가질 때, 뒤에 처리된 문서의
        # 정당한 내용이 "중복"으로 오인되어 조용히 통째로 사라질 수 있다.
        seen_in_doc: set[str] = set()
        deduped = []
        for c in chunks:
            h = hashlib.sha256(c.page_content.encode("utf-8")).hexdigest()
            if h in seen_in_doc:
                continue
            seen_in_doc.add(h)
            deduped.append(c)

        all_chunks.extend(deduped)
        print(f"  {source_file} ({doc_type}) → {len(deduped)}개 청크")

    print(f"\n✅ 총 {len(all_chunks)}개 청크 생성 (원본 문서 {len(grouped)}개)")
    return all_chunks


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _1_loader import DATA_DIR, _resolve_data_dir, load_law_manual_data

    resolved_dir = _resolve_data_dir(DATA_DIR)
    docs = load_law_manual_data(resolved_dir)
    chunks = chunk_documents(docs)

    if chunks:
        print("\n[청크 예시 3개]")
        for c in chunks[:3]:
            m = c.metadata
            print(f"- [{m['doc_type']}] {m['source_file']} / {m['unit']} / p.{m['page']}")
            print(f"  {c.page_content[:120]}")
