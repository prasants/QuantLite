"""Basic performance metrics for return series."""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "annualised_return",
    "annualised_volatility",
    "sharpe_ratio",
    "max_drawdown",
]


def annualised_return(returns: np.ndarray | pd.Series | list[float], freq: int = 252) -> float:
    """Compute annualised return from a series of periodic returns.

    Args:
        returns: Simple periodic returns.
        freq: Periods per year (252 for daily, 12 for monthly).

    Returns:
        Annualised return as a float.
    """
    arr = np.asarray(returns, dtype=float)
    if len(arr) == 0:
        return 0.0
    cumulative = (1 + arr).prod()
    return float(cumulative ** (freq / len(arr)) - 1)


def annualised_volatility(returns: np.ndarray | pd.Series | list[float], freq: int = 252) -> float:
    """Compute annualised volatility (standard deviation) of returns.

    Args:
        returns: Simple periodic returns.
        freq: Periods per year.

    Returns:
        Annualised volatility as a float.
    """
    arr = np.asarray(returns, dtype=float)
    if len(arr) == 0:
        return 0.0
    return float(np.std(arr, ddof=1) * np.sqrt(freq))


def sharpe_ratio(
    returns: np.ndarray | pd.Series | list[float],
    risk_free_rate: float = 0.0,
    freq: int = 252,
) -> float:
    """Compute annualised Sharpe ratio.

    Args:
        returns: Simple periodic returns.
        risk_free_rate: Annualised risk-free rate.
        freq: Periods per year.

    Returns:
        Sharpe ratio, or ``nan`` if volatility is zero.
    """
    ar = annualised_return(returns, freq=freq)
    av = annualised_volatility(returns, freq=freq)
    if av == 0:
        return float("nan")
    return (ar - risk_free_rate) / av


def max_drawdown(returns: np.ndarray | pd.Series | list[float]) -> float:
    """Compute maximum drawdown from a return series.

    Args:
        returns: Simple periodic returns.

    Returns:
        Maximum drawdown as a negative float (e.g. -0.25 for a 25% drawdown).
    """
    arr = np.asarray(returns, dtype=float)
    if len(arr) == 0:
        return 0.0
    cum = np.cumprod(1 + arr)
    roll_max = np.maximum.accumulate(cum)
    drawdown = (cum - roll_max) / roll_max
    return float(drawdown.min())
