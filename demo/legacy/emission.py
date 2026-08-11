"""액비 살포 시 NH3 배출량을 계산하는 순수 함수."""

from constants import (
    F_VOLAT_TN,
    LIQUID_TN_DEFAULT,
    MW_RATIO,
    REDUCE_METHOD,
    REDUCE_TILLAGE,
)


def season_of(month: int) -> str:
    """월을 배출량 계산에 사용하는 계절명으로 변환한다."""
    if month in (3, 4, 5):
        return "봄"
    if month in (6, 7, 8):
        return "여름"
    if month in (9, 10, 11):
        return "가을"
    return "겨울"


def emission_kg(
    tons: float,
    method: str = "표면살포",
    tn_kg_per_ton: float | None = None,
    tillage: bool = False,
    month: int = 9,
) -> float:
    """액비 살포 NH3 배출량(kg NH3)을 계산한다.

    계산식은 다음과 같다.

    ``E = M × C_TN × 0.078 × (1-R_method) × (1-R_till) × 1.214``
    """
    if method not in REDUCE_METHOD:
        raise ValueError(f"알 수 없는 살포방식: {method}")

    c_tn = tn_kg_per_ton if tn_kg_per_ton is not None else LIQUID_TN_DEFAULT

    n_volat = tons * c_tn * F_VOLAT_TN
    n_volat *= 1 - REDUCE_METHOD[method]
    if tillage:
        n_volat *= 1 - REDUCE_TILLAGE[season_of(month)]

    return n_volat * MW_RATIO


def advice_lines(tons: float, method: str, tillage: bool, month: int) -> list[str]:
    """아직 적용하지 않은 저감 방법에 대한 조언을 생성한다."""
    base = emission_kg(tons, method, tillage=tillage, month=month)
    out: list[str] = []

    if method != "주입식":
        alt = emission_kg(tons, "주입식", tillage=tillage, month=month)
        out.append(
            f"주입식으로 바꾸면 배출량이 {(1 - alt / base) * 100:.0f}% 감소합니다 "
            f"({base:.2f} -> {alt:.2f} kg)"
        )
    if not tillage:
        alt = emission_kg(tons, method, tillage=True, month=month)
        out.append(
            f"살포 후 즉시 경운하면 {(1 - alt / base) * 100:.0f}% 감소합니다 "
            f"({base:.2f} -> {alt:.2f} kg), 국내 포장시험 실측"
        )
    return out
