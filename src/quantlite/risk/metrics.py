"""Risk metrics: VaR, CVaR, Sortino, Calmar, and more.

All functions accept NumPy arrays or pandas Series of simple returns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from ..core.types import DrawdownInfo, ReturnMoments

__all__ = [
    "value_at_risk",
    "cvar",
    "sortino_ratio",
    "calmar_ratio",
    "max_drawdown_duration",
    "return_moments",
    "omega_ratio",
    "tail_ratio",
]


def _to_array(returns: np.ndarray | pd.Series | list[float]) -> np.ndarray:
    """Coerce returns to a clean float64 array, dropping NaNs."""
    arr = np.asarray(returns, dtype=float)
    return arr[~np.isnan(arr)]


def value_at_risk(
    returns: np.ndarray | pd.Series | list[float],
    alpha: float = 0.05,
    method: str = "historical",
) -> float:
    """Compute Value at Risk at a given confidence level.

    Args:
        returns: Simple periodic returns.
        alpha: Significance level (0.05 = 95% VaR).
        method: One of ``"historical"``, ``"parametric"``, or ``"cornish-fisher"``.

    Returns:
        VaR as a negative float (loss).

    Raises:
        ValueError: On unknown method or insufficient data.
    """
    arr = _to_array(returns)
    if len(arr) < 2:
        raise ValueError("Need at least 2 observations for VaR")

    if method == "historical":
        return float(np.percentile(arr, alpha * 100))

    mu, sigma = float(np.mean(arr)), float(np.std(arr, ddof=1))
    z = stats.norm.ppf(alpha)

    if method == "parametric":
        return float(mu + sigma * z)

    if method == "cornish-fisher":
        s = float(stats.skew(arr))
        k = float(stats.kurtosis(arr))  # excess kurtosis
        z_cf = (
            z
            + (z**2 - 1) * s / 6
            + (z**3 - 3 * z) * k / 24
            - (2 * z**3 - 5 * z) * s**2 / 36
        )
        return float(mu + sigma * z_cf)

    raise ValueError(f"Unknown method: {method}. Use 'historical', 'parametric', or 'cornish-fisher'.")


def cvar(
    returns: np.ndarray | pd.Series | list[float],
    alpha: float = 0.05,
) -> float:
    """Compute Conditional VaR (Expected Shortfall).

    The average loss in the worst ``alpha`` fraction of outcomes.

    Args:
        returns: Simple periodic returns.
        alpha: Significance level.

    Returns:
        CVaR as a negative float.
    """
    arr = _to_array(returns)
    if len(arr) < 2:
        raise ValueError("Need at least 2 observations for CVaR")
    var = value_at_risk(arr, alpha=alpha, method="historical")
    tail = arr[arr <= var]
    if len(tail) == 0:
        return float(var)
    return float(np.mean(tail))


def sortino_ratio(
    returns: np.ndarray | pd.Series | list[float],
    risk_free_rate: float = 0.0,
    freq: int = 252,
) -> float:
    """Compute the annualised Sortino ratio (downside deviation only).

    Args:
        returns: Simple periodic returns.
        risk_free_rate: Annualised risk-free rate.
        freq: Periods per year.

    Returns:
        Sortino ratio, or ``nan`` if downside deviation is zero.
    """
    arr = _to_array(returns)
    if len(arr) == 0:
        return float("nan")

    excess = arr - risk_free_rate / freq
    downside = excess[excess < 0]
    if len(downside) == 0:
        return float("inf")

    downside_dev = float(np.sqrt(np.mean(downside**2)) * np.sqrt(freq))
    ann_return = float((1 + np.mean(arr)) ** freq - 1)
    return (ann_return - risk_free_rate) / downside_dev


def calmar_ratio(
    returns: np.ndarray | pd.Series | list[float],
    freq: int = 252,
) -> float:
    """Compute the Calmar ratio (annualised return / max drawdown).

    Args:
        returns: Simple periodic returns.
        freq: Periods per year.

    Returns:
        Calmar ratio. Returns ``inf`` if no drawdown occurred.
    """
    arr = _to_array(returns)
    if len(arr) == 0:
        return float("nan")

    ann_return = float((1 + np.mean(arr)) ** freq - 1)
    dd_info = max_drawdown_duration(arr)
    if dd_info.max_drawdown == 0:
        return float("inf")
    return ann_return / abs(dd_info.max_drawdown)


def max_drawdown_duration(
    returns: np.ndarray | pd.Series | list[float],
) -> DrawdownInfo:
    """Compute maximum drawdown and its duration in periods.

    Args:
        returns: Simple periodic returns.

    Returns:
        A ``DrawdownInfo`` dataclass with max drawdown, duration,
        start index, and end index.
    """
    arr = _to_array(returns)
    if len(arr) == 0:
        return DrawdownInfo(max_drawdown=0.0, duration=0, start_idx=0, end_idx=0)

    cum = np.cumprod(1 + arr)
    roll_max = np.maximum.accumulate(cum)
    drawdowns = (cum - roll_max) / roll_max

    trough_idx = int(np.argmin(drawdowns))
    max_dd = float(drawdowns[trough_idx])

    # Find the peak before the trough
    peak_idx = int(np.argmax(cum[: trough_idx + 1])) if trough_idx > 0 else 0
    duration = trough_idx - peak_idx

    return DrawdownInfo(
        max_drawdown=max_dd,
        duration=duration,
        start_idx=peak_idx,
        end_idx=trough_idx,
    )


def return_moments(
    returns: np.ndarray | pd.Series | list[float],
) -> ReturnMoments:
    """Compute descriptive statistics of a return series.

    Args:
        returns: Simple periodic returns.

    Returns:
        A ``ReturnMoments`` dataclass (mean, volatility, skewness, kurtosis).
    """
    arr = _to_array(returns)
    if len(arr) < 4:
        raise ValueError("Need at least 4 observations for moment estimation")

    return ReturnMoments(
        mean=float(np.mean(arr)),
        volatility=float(np.std(arr, ddof=1)),
        skewness=float(stats.skew(arr)),
        kurtosis=float(stats.kurtosis(arr)),  # excess kurtosis
    )


def omega_ratio(
    returns: np.ndarray | pd.Series | list[float],
    threshold: float = 0.0,
) -> float:
    """Compute the Omega ratio: probability-weighted gains over losses.

    Args:
        returns: Simple periodic returns.
        threshold: Minimum acceptable return.

    Returns:
        Omega ratio. Returns ``inf`` if no losses below threshold.
    """
    arr = _to_array(returns)
    if len(arr) == 0:
        return float("nan")

    gains = np.sum(np.maximum(arr - threshold, 0.0))
    losses = np.sum(np.maximum(threshold - arr, 0.0))

    if losses == 0:
        return float("inf")
    return float(gains / losses)


def tail_ratio(
    returns: np.ndarray | pd.Series | list[float],
    alpha: float = 0.05,
) -> float:
    """Compute the tail ratio: right tail magnitude over left tail magnitude.

    A ratio > 1 indicates fatter right tails (more upside extremes).

    Args:
        returns: Simple periodic returns.
        alpha: Percentile for tail measurement.

    Returns:
        Tail ratio as a positive float.
    """
    arr = _to_array(returns)
    if len(arr) < 2:
        return float("nan")

    right_tail = float(np.percentile(arr, 100 - alpha * 100))
    left_tail = float(np.percentile(arr, alpha * 100))

    if left_tail == 0:
        return float("inf")
    return float(abs(right_tail / left_tail))
