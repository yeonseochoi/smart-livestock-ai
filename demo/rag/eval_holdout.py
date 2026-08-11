"""Independent 20-question retrieval holdout.

Unlike eval_qa_v2, every case records a short reference answer and its expected
source location. Do not tune retrieval rules against this file; add future
development questions to a separate set and preserve this score.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Case:
    id: str
    question: str
    expected_doc: str
    expected_unit: str
    evidence_any: tuple[str, ...]
    reference_answer: str


CASES = [
    Case("H01", "엄격한 악취 배출허용기준의 설정 범위는 어디에서 확인하나요?",
         "악취방지법 시행규칙", "제8조", ("별표 3", "엄격한 배출허용기준"),
         "악취방지법 시행규칙 제8조에 따라 설정 범위는 별표 3에서 확인한다."),
    Case("H02", "악취 기술진단의 내용과 방법은 어느 별표에 있나요?",
         "악취방지법 시행규칙", "제13조의2", ("별표 5", "기술진단"),
         "시행규칙 제13조의2는 기술진단의 내용·방법을 별표 5로 연결한다."),
    Case("H03", "기술진단 결과를 받은 뒤 저감조치 이행계획은 며칠 안에 세워야 하나요?",
         "악취방지법 시행규칙", "제13조의2", ("30일", "이행하기 위한 계획"),
         "관할 행정기관은 기술진단 결과를 받은 날부터 30일 이내에 이행계획을 수립한다."),
    Case("H04", "악취 개선권고를 내릴 때 조치기간은 최대 얼마나 정할 수 있나요?",
         "악취방지법 시행령", "제6조", ("6개월", "조치기간"),
         "시행령 제6조에 따라 원칙적으로 6개월 범위에서 조치기간을 정한다."),
    Case("H05", "익산시 악취시설 보조금은 어떤 사업에 지원할 수 있나요?",
         "익산시 악취방지 및 저감 조례", "제12조", ("설치사업", "개선사업"),
         "악취방지시설 설치·개선사업과 시장이 필요하다고 인정하는 저감사업에 지원할 수 있다."),
    Case("H06", "익산시 악취시설 보조금은 누구에게 우선 지원하나요?",
         "익산시 악취방지 및 저감 조례", "제12조", ("영세업자",),
         "조례 제12조는 영세업자에게 우선 지원하도록 정한다."),
    Case("H07", "익산시 악취방지 추진계획은 얼마나 자주 수립하나요?",
         "익산시 악취방지 및 저감 조례", "제4조", ("매년", "추진계획"),
         "시장은 악취방지 추진계획을 매년 수립한다."),
    Case("H08", "익산시는 악취배출사업장 관련 정보를 언제까지 공개할 수 있나요?",
         "익산시 악취방지 및 저감 조례", "제7조", ("3월", "홈페이지"),
         "조례 제7조에 따라 관련 정보를 매년 3월까지 홈페이지에 공개할 수 있다."),
    Case("H09", "가축분뇨 공공처리시설 사용을 시작할 때 무엇을 공고해야 하나요?",
         "가축분뇨의 관리 및 이용에 관한 법률", "제25조", ("처리대상 배출시설", "지역"),
         "처리대상 배출시설의 범위와 지역을 공고해야 한다."),
    Case("H10", "가축분뇨 배출시설 설치허가 신청서는 누구에게 제출하나요?",
         "가축분뇨의 관리 및 이용에 관한 법률 시행령", "제7조", ("시장", "군수", "구청장"),
         "허가신청서와 첨부서류를 시장·군수·구청장에게 제출한다."),
    Case("H11", "퇴비나 액비를 살포하는 사람도 가축분뇨 관리 의무의 대상인가요?",
         "가축분뇨의 관리 및 이용에 관한 법률", "제17조", ("퇴비ㆍ액비를 살포하는 자", "살포"),
         "법 제17조는 퇴비·액비를 살포하는 자도 관리 의무 대상으로 규정한다."),
    Case("H12", "밀폐공간 작업 특별교육은 최초 작업 전에 몇 시간 이상 해야 하나요?",
         "자원화시설 안전관리 매뉴얼", "특별교육", ("16시간", "최초"),
         "매뉴얼은 밀폐공간 최초 작업 전 특별교육 16시간 이상을 제시한다."),
    Case("H13", "밀폐공간에 들어가기 전에 준비할 환기와 호흡보호 장비는 무엇인가요?",
         "자원화시설 안전관리 매뉴얼", "밀폐공간작업 기본 작업절차", ("환기팬", "공기호흡기", "송기마스크"),
         "환기팬과 공기호흡기 또는 송기마스크, 가스농도 측정기 등을 준비한다."),
    Case("H14", "자원화시설에서 출입통제 안전표지를 붙일 장소의 예시는 무엇인가요?",
         "자원화시설 안전관리 매뉴얼", "안전보건표지", ("분뇨저장조", "가스 발생 구역"),
         "분뇨저장조, 가스 발생 구역, 기계실, 전기실 등이 예시다."),
    Case("H15", "자원화시설 안전교육에는 위험성 평가 내용도 포함되나요?",
         "자원화시설 안전관리 매뉴얼", "교육 일지", ("위험성 평가",),
         "정기교육 예시에는 위험성 평가에 관한 사항이 포함된다."),
    Case("H16", "축산악취 ICT 장비는 설치 위치에 따라 무엇이 달라지나요?",
         "ICT 기계·장비 설치 위치 선정 매뉴얼", "서문", ("신뢰성", "활용도"),
         "설치 위치에 따라 측정 결과의 신뢰성과 활용도가 달라진다."),
    Case("H17", "밀폐형 돈사에서 음압 환기방식의 장점은 무엇인가요?",
         "ICT 기계·장비 설치 위치 선정 매뉴얼", "밀폐형", ("설치비", "유지비", "신선한 공기"),
         "설치·유지비가 비교적 낮고 입기구 조절로 신선한 공기를 고르게 분산하기 쉽다."),
    Case("H18", "양돈분뇨 액비화 시 권장되는 공기 공급량과 기간은 얼마인가요?",
         "곽정훈", "서문", ("30ℓ/분/㎥", "15일"),
         "자료는 30ℓ/분/㎥ 이상 공기를 15일 이상 공급하는 조건을 제시한다."),
    Case("H19", "악취 개선권고의 조치기간 연장은 최대 어느 범위까지 가능한가요?",
         "악취방지법 시행령", "제6조", ("연장", "6개월"),
         "불가피한 사유가 인정되면 정해진 범위에서 조치기간을 연장할 수 있다."),
    Case("H20", "익산시 악취대책민관협의회는 어떤 일을 심의하나요?",
         "익산시 악취방지 및 저감 조례", "제9조", ("추진계획", "저감대책"),
         "악취방지 추진계획과 악취발생 실태조사·저감대책 등을 심의한다."),
]


def run(index, k: int = 3) -> dict:
    details, hits = [], 0
    for case in CASES:
        response = index.search(case.question, k=k)
        matched = None
        for result in response["results"]:
            haystack = f"{result['unit']} {result['snippet']}"
            if (case.expected_doc in result["doc"]
                    and case.expected_unit in result["unit"]
                    and any(term in haystack for term in case.evidence_any)):
                matched = result
                break
        hit = matched is not None
        hits += int(hit)
        details.append({**asdict(case), "hit": hit,
                        "matched_rank": matched["rank"] if matched else None,
                        "top1": response["results"][0] if response["results"] else None})
    result = {"backend": index.backend, "hits": hits, "n": len(CASES),
              "hit_rate": round(hits / len(CASES), 4), "details": details}
    print(f"  [홀드아웃/{index.backend}] top-{k} 적중 {hits}/{len(CASES)} = {hits/len(CASES):.0%}")
    return result
