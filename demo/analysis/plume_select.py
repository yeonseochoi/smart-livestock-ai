"""플룸의 새 역할 — 등급을 보정하지 않고 '어느 수용점 그룹을 볼지'만 고른다.

절대규칙 1 준수: 곱하지 않는다. 부채꼴 안에 드는 그룹을 선택만 한다.
PLUME_GRADE_BUMP 는 영구 OFF (config.py 참조).

────────────────────────────────────────────────────────────────────
거리에 따라 판정 도구를 바꾼다  (문서 13-14 M-1)

  ≤ 3km   플룸 유효 반각을 그대로 쓴다.
          익산 농가는 vworld 필지 좌표[A]로 오차가 수십 m 인데
          부채꼴 반폭은 226~1,242m 라 좌표 오차보다 훨씬 크다.
          도달 시간도 5~17분이라 정상상태 가정이 유지된다.

  > 3km    플룸을 쓰지 않고 고정 섹터(±30도)로 대체한다.
          이유 두 가지.
            ① 15km 에서 F등급(야간 역전) 부채꼴 반폭은 814m 인데
               김제 좌표는 리(里) 중앙값[B]이라 오차가 1~2km 다.
               좌표 오차가 부채꼴보다 1.2~2.5배 크면 적중이 나와도 우연이다.
            ② 도달에 50분~1.5시간이 걸려 정상상태 전제가 깨진다
               (2시간 창 풍향 일관성 중앙값 0.67, 52%가 0.7 미만).
          이건 물리 모델이라 주장하지 않고 '노출 지표'라고 부른다.
────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

from datetime import datetime

from config import GROUP_CENTER, GROUPS

# legacy import (수정 금지, import 만)
from geo import bearing, angle_diff
from plume import pasquill_class, plume_half_angle

PLUME_MAX_KM = 3.0        # 이 거리까지만 플룸으로 판정
SECTOR_DEG = 30.0         # 그 밖은 고정 섹터 반각 (spatial_features 과 동일 값)
_R = 6371.0


def _dist_km(la1, lo1, la2, lo2) -> float:
    import math
    p = math.pi / 180.0
    return _R * math.acos(max(-1.0, min(1.0,
        math.sin(la1 * p) * math.sin(la2 * p) +
        math.cos(la1 * p) * math.cos(la2 * p) * math.cos((lo2 - lo1) * p))))


def downwind_groups(farm_lat: float, farm_lon: float,
                    wd_deg: float, ws_ms: float, sky, when: datetime,
                    centers: dict | None = None) -> tuple[list, list]:
    """농장 풍하측에 드는 수용점 그룹 목록과 판정 근거를 돌려준다.

    wd_deg 는 기상 풍향(바람이 '불어오는' 방향). 냄새가 가는 방향은 +180도.
    반환: (그룹명 리스트, 그룹별 판정내역 리스트)
    """
    centers = centers or GROUP_CENTER
    to_deg = (float(wd_deg) + 180.0) % 360.0      # 불어오는 방향 → 가는 방향
    stability, elev = pasquill_class(ws_ms, sky, when, farm_lat, farm_lon)

    hit, detail = [], []
    for g in GROUPS:
        la, lo = centers[g]
        dist_km = _dist_km(farm_lat, farm_lon, la, lo)
        brg = bearing(farm_lat, farm_lon, la, lo)
        off = angle_diff(to_deg, brg)

        if dist_km <= PLUME_MAX_KM:
            half = plume_half_angle(dist_km * 1000.0, stability)
            method = f"플룸 반각({stability}등급)"
        else:
            half = SECTOR_DEG
            method = "섹터 지표(±30도)"      # 플룸 판정 불가 구간

        inside = off <= half
        if inside:
            hit.append(g)
        detail.append({
            "group": g, "dist_km": round(dist_km, 2),
            "bearing": round(brg, 1), "angle_off": round(off, 1),
            "half_angle": round(half, 1), "method": method,
            "downwind": bool(inside),
        })
    return hit, detail


def describe(detail: list, stability: str | None = None) -> str:
    """사람이 읽을 한 줄 설명."""
    on = [d["group"] for d in detail if d["downwind"]]
    if not on:
        return "풍하측에 수용점 그룹 없음 — 보수적으로 전 그룹 최댓값 사용"
    parts = [f"{d['group']}(거리 {d['dist_km']}km · 이탈각 {d['angle_off']}도 "
             f"≤ {d['half_angle']}도, {d['method']})"
             for d in detail if d["downwind"]]
    return "풍하측: " + " / ".join(parts)


def stability_at(farm_lat, farm_lon, ws_ms, sky, when) -> str:
    return pasquill_class(ws_ms, sky, when, farm_lat, farm_lon)[0]
