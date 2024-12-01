"""Fixed-income pricing: bond price, duration, and yield to maturity."""

from __future__ import annotations

__all__ = [
    "bond_price",
    "bond_duration",
    "bond_yield_to_maturity",
]


def bond_price(
    face_value: float,
    coupon_rate: float,
    market_rate: float,
    maturity: int,
    payments_per_year: int = 1,
) -> float:
    """Calculate the price of a coupon bond.

    Args:
        face_value: Par value of the bond.
        coupon_rate: Annual coupon rate.
        market_rate: Annual market discount rate.
        maturity: Years to maturity.
        payments_per_year: Coupon frequency (1=annual, 2=semi-annual).

    Returns:
        Present value of the bond.
    """
    coupon = face_value * coupon_rate / payments_per_year
    periods = maturity * payments_per_year
    rate = market_rate / payments_per_year

    # Closed-form annuity + principal PV
    if rate == 0:
        return coupon * periods + face_value
    pv_coupons = coupon * (1 - (1 + rate) ** (-periods)) / rate
    pv_face = face_value / (1 + rate) ** periods
    return pv_coupons + pv_face


def bond_duration(
    face_value: float,
    coupon_rate: float,
    market_rate: float,
    maturity: int,
    payments_per_year: int = 1,
) -> float:
    """Compute Macaulay duration for a coupon bond.

    Args:
        face_value: Par value.
        coupon_rate: Annual coupon rate.
        market_rate: Annual market discount rate.
        maturity: Years to maturity.
        payments_per_year: Coupon frequency.

    Returns:
        Macaulay duration in years.
    """
    coupon = face_value * coupon_rate / payments_per_year
    periods = maturity * payments_per_year
    rate = market_rate / payments_per_year

    total_price = bond_price(face_value, coupon_rate, market_rate, maturity, payments_per_year)
    weighted_sum = sum(
        (t / payments_per_year) * coupon / (1 + rate) ** t
        for t in range(1, periods + 1)
    )
    weighted_sum += maturity * face_value / (1 + rate) ** periods
    return weighted_sum / total_price


def bond_yield_to_maturity(
    face_value: float,
    coupon_rate: float,
    current_price: float,
    maturity: int,
    payments_per_year: int = 1,
    guess: float = 0.05,
    tol: float = 1e-7,
) -> float:
    """Solve for yield to maturity via bisection.

    Args:
        face_value: Par value.
        coupon_rate: Annual coupon rate.
        current_price: Observed market price.
        maturity: Years to maturity.
        payments_per_year: Coupon frequency.
        guess: Initial guess (unused; kept for API compatibility).
        tol: Convergence tolerance.

    Returns:
        Yield to maturity as a decimal.
    """
    low, high = 0.0, 1.0
    for _ in range(200):
        mid = (low + high) / 2
        diff = bond_price(face_value, coupon_rate, mid, maturity, payments_per_year) - current_price
        if abs(diff) < tol:
            return mid
        if diff > 0:
            low = mid
        else:
            high = mid
    return (low + high) / 2
