"""Black-Scholes option pricing and Greeks."""

from __future__ import annotations

import math

from scipy.stats import norm

from ..core.types import GreeksResult

__all__ = [
    "black_scholes_call",
    "black_scholes_put",
    "black_scholes_greeks",
]


def black_scholes_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Price a European call option using Black-Scholes.

    Args:
        S: Spot price.
        K: Strike price.
        T: Time to expiry in years.
        r: Risk-free rate (annualised).
        sigma: Volatility (annualised).

    Returns:
        Call option price.
    """
    if T <= 0:
        return max(S - K, 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)


def black_scholes_put(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Price a European put option using Black-Scholes.

    Args:
        S: Spot price.
        K: Strike price.
        T: Time to expiry in years.
        r: Risk-free rate (annualised).
        sigma: Volatility (annualised).

    Returns:
        Put option price.
    """
    if T <= 0:
        return max(K - S, 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def black_scholes_greeks(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
) -> GreeksResult:
    """Compute Black-Scholes Greeks for a European option.

    Args:
        S: Spot price.
        K: Strike price.
        T: Time to expiry in years.
        r: Risk-free rate.
        sigma: Volatility.
        option_type: ``"call"`` or ``"put"``.

    Returns:
        A ``GreeksResult`` dataclass.
    """
    if T <= 0:
        return GreeksResult(delta=0.0, gamma=0.0, vega=0.0, theta=0.0, rho=0.0)

    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    pdf_d1 = (1.0 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * d1**2)

    gamma = pdf_d1 / (S * sigma * math.sqrt(T))
    vega = S * math.sqrt(T) * pdf_d1

    if option_type.lower() == "call":
        delta = norm.cdf(d1)
        theta = (
            -(S * pdf_d1 * sigma) / (2 * math.sqrt(T))
            - r * K * math.exp(-r * T) * norm.cdf(d2)
        )
        rho = K * T * math.exp(-r * T) * norm.cdf(d2)
    else:
        delta = norm.cdf(d1) - 1
        theta = (
            -(S * pdf_d1 * sigma) / (2 * math.sqrt(T))
            + r * K * math.exp(-r * T) * norm.cdf(-d2)
        )
        rho = -K * T * math.exp(-r * T) * norm.cdf(-d2)

    return GreeksResult(delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho)
