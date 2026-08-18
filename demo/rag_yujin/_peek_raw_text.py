"""
1회용 진단 스크립트 — "붙임1. 축산농장 악취저감시설 운영 매뉴얼(돼지) (1).pdf"의
PyMuPDFLoader 추출 결과 앞부분을 그대로 찍어본다. 이 파일이 소제목을 어떤 식으로
표기하는지(번호? 기호? 굵은 글씨라 텍스트로는 구분이 안 되는지) 눈으로 확인하기 위함.
확인 끝나면 지워도 되는 파일이다 (_1_loader.py / _2_Chunking.py와는 무관).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _1_loader import DATA_DIR, _resolve_data_dir

from langchain_community.document_loaders import PyMuPDFLoader

TARGET_FILENAME = "붙임1. 축산농장 악취저감시설 운영 매뉴얼(돼지) (1).pdf"

resolved_dir = _resolve_data_dir(DATA_DIR)
target_path = resolved_dir / TARGET_FILENAME
if not target_path.is_file():
    raise SystemExit(f"파일을 못 찾음: {target_path}")

docs = PyMuPDFLoader(str(target_path)).load()
print(f"총 {len(docs)}페이지\n")

# 앞쪽 5페이지 정도만 원문 그대로(줄바꿈 유지) 출력 — 소제목이 어떻게 생겼는지 보기 위함
for i, d in enumerate(docs[:5]):
    print(f"===== {i + 1}페이지 =====")
    print(d.page_content)
    print()
