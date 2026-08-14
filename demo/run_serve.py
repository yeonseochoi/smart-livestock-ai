"""v6 서빙 + 추천 통합 실행 (검증용)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import section, DEMO_FARM
from console import use_utf8_stdout
from ops import daily_scoring
from scoring import recommend


def main():
    use_utf8_stdout()
    section("서빙 — 예보 → v6 모델 → risk_calendar_v6")
    print(daily_scoring.run_v6())

    section("추천 — 법령(미구현) → ML → 플룸 그룹선택 → 조합 → 등급")
    for wt, sp in [("액비살포", "돼지"), ("분뇨제거", "한우"), ("청소", "육계")]:
        o = recommend.recommend_v6(wt, storage_days=12, tons=20.0,
                                      method="표면살포", tillage=False, species=sp)
        r, a = o["recommended"], o["avoid"]
        print(f"\n[{sp} 농가 · {wt}]  후보 {o['n_candidates']}시각")
        print(f"  추천 : {r['t']}  final {r['final']}  등급 {r['grade']}  {r['abc']}등급")
        print(f"         플룸 → {r['plume']}   ({r['plume_method']})")
        print(f"         그룹별 창위험도 {r['per_group']}  · 채택 {r['selected_groups']}")
        print(f"  회피 : {a['t']}  final {a['final']}  등급 {a['grade']}  {a['abc']}등급")
        print(f"         플룸 → {a['plume']}")
        print(f"  A/B/C 분포 {o['abc_counts']} · 플룸이 그룹을 좁힌 시각 "
              f"{o['plume_selected_hours']}/{o['n_candidates']}")
        print(f"  조합규칙 일치율 {o['combine_agreement']}")
        for t in o["reduction_tips"]:
            print("  조언 :", t)


if __name__ == "__main__":
    main()
