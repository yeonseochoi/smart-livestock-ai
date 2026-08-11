"""S7 — 주민 알림 초안 에이전트.

1. legacy find_receptors 로 반경 2km 주거 확보 → receptor 테이블 (가상 연락처)
2. 확정 시각 예보로 legacy dispersion 전수 계산 → factor>0 수용점 = 알림 대상
3. 지도 표시용 부채꼴(plume_half_angle) 값 산출
4. 문구 생성 → notification_log 기록 (승인 전, 실발송 없음)
"""
from __future__ import annotations

from datetime import datetime

from config import DEMO_FARM, PROV
from ops import db

# legacy import (수정 금지)
from diffusion import dispersion, wind_to_direction
from plume import plume_half_angle
from residence import find_receptors
from mock_residence import mock_buildings
from geo import latlon_to_grid
import mock_forecast


def run(work_type: str, when: datetime) -> dict:
    recs, meta = find_receptors(
        DEMO_FARM["lat"], DEMO_FARM["lon"],
        buildings=mock_buildings(DEMO_FARM["lat"], DEMO_FARM["lon"]))
    PROV.log("D7 주거건물(알림)", "legacy/mock_residence.py", real=False,
             note=f"VWorld 폴백, 수용점 {len(recs)}동")

    con = db.connect()
    con.execute("DELETE FROM receptor WHERE farm_id=?", (DEMO_FARM["farm_id"],))
    for i, r in enumerate(recs):
        con.execute("INSERT INTO receptor VALUES(?,?,?,?,?,?,?,?)",
                    (DEMO_FARM["farm_id"], r.lat, r.lon, r.dist_m, r.bearing,
                     r.purpose, f"010-0000-{i:04d}", 1))
    con.commit()

    nx, ny = latlon_to_grid(DEMO_FARM["lat"], DEMO_FARM["lon"])
    fc = mock_forecast.fetch_with_fallback(nx, ny)
    key = when.strftime("%Y%m%d %H00")
    if key not in fc:
        key = sorted(fc)[0]
        when = datetime.strptime(key, "%Y%m%d %H%M")
    v = fc[key]

    res = dispersion(v["VEC"], float(v["WSD"]), v["SKY"], when,
                     DEMO_FARM["lat"], DEMO_FARM["lon"], recs)

    # 알림 대상 = "플룸 factor>0 OR 풍하 부채꼴(유효 반각) 내" 합집합.
    # 플룸 단독 의존을 풀기 위한 보수적 확대 — 플룸 미검증(S8-4) 상태의 안전측 선택.
    from geo import angle_diff
    from plume import dispersion_factor, initial_sigmas, pasquill_class
    stability, _ = pasquill_class(float(v["WSD"]), v["SKY"], when,
                                  DEMO_FARM["lat"], DEMO_FARM["lon"])
    wind_to = wind_to_direction(v["VEC"])
    sy0, sz0 = initial_sigmas(0.0)
    targets = []
    n_by_factor = n_by_sector = 0
    for r in recs:
        off = angle_diff(wind_to, r.bearing)
        by_factor = dispersion_factor(r.dist_m, off, stability,
                                      float(v["WSD"]), sy0, sz0) > 0
        by_sector = off <= plume_half_angle(r.dist_m, stability)
        n_by_factor += by_factor
        n_by_sector += by_sector
        if by_factor or by_sector:
            targets.append(r)

    half = (plume_half_angle(res.worst.dist_m, res.stability)
            if res.worst else None)
    msg = (
        f"[악취 작업 사전 안내] {when:%m월 %d일 %H시}경 인근 농가에서 "
        f"{work_type} 작업이 예정되어 있습니다. 풍향({wind_to:.0f}도 방향 확산 예상)에 "
        f"따라 일시적 냄새가 있을 수 있으니 창문을 닫아 주시기 바랍니다. "
        f"작업은 저위험 시간대로 조정되었습니다. 문의: 농가 대표번호"
    )
    con.execute(
        "INSERT INTO notification_log(farm_id, work_type, sent_at, message, approved_yn) "
        "VALUES(?,?,?,?,0)",
        (DEMO_FARM["farm_id"], work_type, when.strftime("%Y-%m-%d %H:%M"), msg))
    con.commit()
    n_log = con.execute("SELECT COUNT(*) FROM notification_log").fetchone()[0]
    con.close()

    print(f"  수용점 {len(recs)}동 중 알림 대상 {len(targets)}동 "
          f"(factor>0: {n_by_factor} OR 부채꼴 내: {n_by_sector} — 합집합)")
    if half:
        print(f"  지도 표시용 부채꼴 반각 {half:.0f}도 — 참고: 플룸은 미검증 모델(S8-4)")
    print(f"  notification_log 기록 {n_log}건 (농장주 승인 전 — 실발송 없음)")
    return {"n_receptors": len(recs), "n_targets": len(targets),
            "n_by_factor": int(n_by_factor), "n_by_sector": int(n_by_sector),
            "half_angle": half, "message": msg, "wind_to": wind_to,
            "stability": res.stability}
