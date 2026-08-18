"""
청크 길이 분포 진단 — 청킹/검색 고도화 작업 전에 실제 데이터로 확인하기 위한 스크립트.

API 호출 없음(LLM/임베딩 안 씀), 로컬에서 즉시 끝남. _2_Chunking.py를 손대기 전에
"진짜 문제가 뭔지" 숫자로 먼저 확인하는 게 목적 — 특히 법령 청킹은 크기 상한이
아예 없어서(제N조/별표N 경계로만 자름), 긴 조문·별표가 그대로 거대한 단일 청크가
되고 있는지를 본다.
"""
from __future__ import annotations

import statistics
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CURRENT_DIR))

from _1_loader import DATA_DIR, _resolve_data_dir, load_law_manual_data  # noqa: E402
from _2_Chunking import chunk_documents  # noqa: E402


def _percentile(sorted_vals: list[int], p: float) -> int:
    if not sorted_vals:
        return 0
    idx = min(len(sorted_vals) - 1, int(len(sorted_vals) * p))
    return sorted_vals[idx]


def _report_for(label: str, chunks: list) -> None:
    lengths = sorted(len(c.page_content) for c in chunks)
    if not lengths:
        print(f"[{label}] 청크 없음")
        return
    print(f"\n[{label}] 청크 {len(lengths)}개")
    print(
        f"  최소 {lengths[0]}자 / 중앙값 {statistics.median(lengths):.0f}자 / "
        f"평균 {statistics.mean(lengths):.0f}자 / 최대 {lengths[-1]}자"
    )
    print(
        f"  90퍼센타일 {_percentile(lengths, 0.90)}자 / "
        f"95퍼센타일 {_percentile(lengths, 0.95)}자 / "
        f"99퍼센타일 {_percentile(lengths, 0.99)}자"
    )
    for threshold in (700, 1000, 1500, 2000, 3000):
        over = sum(1 for l in lengths if l > threshold)
        if over:
            print(f"  {threshold}자 초과: {over}개 ({over / len(lengths) * 100:.1f}%)")


if __name__ == "__main__":
    resolved_dir = _resolve_data_dir(DATA_DIR)
    docs = load_law_manual_data(resolved_dir)
    chunks = chunk_documents(docs)

    manual_chunks = [c for c in chunks if c.metadata.get("doc_type") == "manual"]
    law_chunks = [c for c in chunks if c.metadata.get("doc_type") == "law"]

    _report_for("전체", chunks)
    _report_for("매뉴얼(manual)", manual_chunks)
    _report_for("법령(law)", law_chunks)

    print("\n[법령 청크 중 가장 긴 10개]")
    for c in sorted(law_chunks, key=lambda c: len(c.page_content), reverse=True)[:10]:
        m = c.metadata
        print(f"  {len(c.page_content):5d}자  {m['source_file'][:40]} / {m['unit']}")

    print("\n[매뉴얼 청크 중 가장 긴 10개]")
    for c in sorted(manual_chunks, key=lambda c: len(c.page_content), reverse=True)[:10]:
        m = c.metadata
        print(f"  {len(c.page_content):5d}자  {m['source_file'][:40]} / {m['unit']}")
