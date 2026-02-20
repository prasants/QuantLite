"""Dynamic Kelly criterion with drawdown control.

Implements configurable fractional Kelly sizing with a maximum-drawdown
circuit breaker that automatically reduces position size when the
portfolio drawdown exceeds a threshold. Uses ``scipy.optimize.minimize_scalar``
for Kelly fraction estimation (not grid search).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

__all__ = [
    "KellyResult",
    "optimal_kelly_fraction",
    "fractional_kelly",
    "rolling_kelly",
    "kelly_with_drawdown_control",
]


@dataclass(frozen=True)
class KellyResult:
    """Result container for dynamic Kelly sizing.

    Attributes:
        fraction: Optimal Kelly fraction.
        equity_curve: Simulated equity curve using this fraction.
        max_drawdown: Maximum drawdown experienced.
        final_wealth: Terminal wealth (starting from 1.0).
        cagr: Compound annual growth rate.
        method: Description of the Kelly variant used.
    """

    fraction: float
    equity_curve: np.ndarray
    max_drawdown: float
    final_wealth: float
    cagr: float
    method: str


def _growth_rate(fraction: float, returns: np.ndarray) -> float:
    """Compute the negative geometric growth rate for a given Kelly fraction.

    Maximising the geometric growth rate is equivalent to minimising
    this function. Uses ``log(1 + f * r)`` for each return.

    Args:
        fraction: Kelly fraction (position size).
        returns: Array of simple returns.

    Returns:
        Negative mean log growth (for minimisation).
    """
    # Clip to avoid log(0) or log(negative)
    growth = np.log(np.maximum(1.0 + fraction * returns, 1e-15))
    return -float(np.mean(growth))


def optimal_kelly_fraction(
    returns: np.ndarray | pd.Series | list[float],
    max_fraction: float = 5.0,
) -> float:
    """Compute the optimal Kelly fraction via scipy.optimize.minimize_scalar.

    Maximises the expected geometric growth rate. This is the
    continuous-time Kelly criterion generalised to arbitrary return
    distributions.

    Args:
        returns: Array of simple returns.
        max_fraction: Upper bound on the Kelly fraction (prevents
            absurd leverage in low-variance regimes).

    Returns:
        Optimal Kelly fraction.
    """
    arr = np.asarray(returns, dtype=float)
    arr = arr[~np.isnan(arr)]

    if len(arr) < 2:
        return 0.0

    result = minimize_scalar(
        _growth_rate,
        args=(arr,),
        bounds=(0.0, max_fraction),
        method="bounded",
    )

    return float(result.x)


def fractional_kelly(
    returns: np.ndarray | pd.Series | list[float],
    fraction_of_kelly: float = 0.5,
    max_fraction: float = 5.0,
) -> KellyResult:
    """Compute a fractional Kelly allocation and simulate the equity curve.

    Common choices: 1.0 (full Kelly), 0.5 (half Kelly), 0.25 (quarter Kelly).
    Half Kelly empirically delivers ~75% of the growth with significantly
    lower variance and drawdowns.

    Args:
        returns: Array of simple returns.
        fraction_of_kelly: Fraction of the full Kelly to use (0 to 1).
        max_fraction: Upper bound on the raw Kelly fraction.

    Returns:
        ``KellyResult`` with the equity curve and statistics.
    """
    arr = np.asarray(returns, dtype=float)
    arr = arr[~np.isnan(arr)]

    full_kelly = optimal_kelly_fraction(arr, max_fraction=max_fraction)
    f = full_kelly * fraction_of_kelly

    # Simulate equity curve
    equity = np.cumprod(1.0 + f * arr)
    equity = np.insert(equity, 0, 1.0)

    # Compute drawdown
    running_max = np.maximum.accumulate(equity)
    drawdowns = (equity - running_max) / running_max
    max_dd = float(np.min(drawdowns))

    final = float(equity[-1])
    n_periods = len(arr)
    cagr = float(final ** (252.0 / max(n_periods, 1)) - 1.0) if final > 0 else -1.0

    label = "full_kelly" if fraction_of_kelly >= 1.0 else f"{fraction_of_kelly:.0%}_kelly"

    return KellyResult(
        fraction=f,
        equity_curve=equity,
        max_drawdown=max_dd,
        final_wealth=final,
        cagr=cagr,
        method=label,
    )


def rolling_kelly(
    returns: np.ndarray | pd.Series | list[float],
    window: int = 126,
    fraction_of_kelly: float = 0.5,
    max_fraction: float = 5.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute a rolling Kelly fraction over a sliding window.

    Re-estimates the optimal Kelly fraction at each step using the
    most recent ``window`` observations.

    Args:
        returns: Array of simple returns.
        window: Lookback window in periods.
        fraction_of_kelly: Fraction of full Kelly to use.
        max_fraction: Upper bound on the raw Kelly fraction.

    Returns:
        Tuple of (kelly_fractions array, equity_curve array).
        Both have the same length as returns. Kelly fractions for
        the first ``window`` periods use the initial estimate.
    """
    arr = np.asarray(returns, dtype=float)
    arr = arr[~np.isnan(arr)]
    n = len(arr)

    kelly_fracs = np.zeros(n)
    equity = np.ones(n + 1)

    # Initial estimate from first window
    init_window = arr[:min(window, n)]
    init_kelly = optimal_kelly_fraction(init_window, max_fraction=max_fraction)

    for i in range(n):
        if i >= window:
            lookback = arr[i - window:i]
            raw_kelly = optimal_kelly_fraction(lookback, max_fraction=max_fraction)
        else:
            raw_kelly = init_kelly

        f = raw_kelly * fraction_of_kelly
        kelly_fracs[i] = f
        equity[i + 1] = equity[i] * (1.0 + f * arr[i])

    return kelly_fracs, equity[1:]


def kelly_with_drawdown_control(
    returns: np.ndarray | pd.Series | list[float],
    fraction_of_kelly: float = 0.5,
    max_drawdown_threshold: float = -0.15,
    drawdown_reduction: float = 0.5,
    window: int = 126,
    max_fraction: float = 5.0,
) -> KellyResult:
    """Kelly sizing with a maximum drawdown circuit breaker.

    When the portfolio drawdown exceeds ``max_drawdown_threshold``,
    the position size is reduced by ``drawdown_reduction`` (e.g. halved).
    This prevents the geometric drag of large drawdowns from
    destroying long-run compounding.

    Args:
        returns: Array of simple returns.
        fraction_of_kelly: Base fraction of full Kelly to use.
        max_drawdown_threshold: Drawdown level that triggers the
            circuit breaker (negative number, e.g. -0.15 for 15%).
        drawdown_reduction: Multiplicative reduction when breaker
            triggers (e.g. 0.5 = halve position size).
        window: Rolling window for Kelly re-estimation.
        max_fraction: Upper bound on the raw Kelly fraction.

    Returns:
        ``KellyResult`` with the equity curve reflecting drawdown control.
    """
    arr = np.asarray(returns, dtype=float)
    arr = arr[~np.isnan(arr)]
    n = len(arr)

    equity = np.ones(n + 1)
    kelly_fracs = np.zeros(n)

    init_window = arr[:min(window, n)]
    init_kelly = optimal_kelly_fraction(init_window, max_fraction=max_fraction)

    for i in range(n):
        # Re-estimate Kelly
        if i >= window:
            lookback = arr[i - window:i]
            raw_kelly = optimal_kelly_fraction(lookback, max_fraction=max_fraction)
        else:
            raw_kelly = init_kelly

        f = raw_kelly * fraction_of_kelly

        # Check drawdown circuit breaker
        running_max = np.max(equity[:i + 1])
        current_dd = (equity[i] - running_max) / running_max if running_max > 0 else 0.0

        if current_dd < max_drawdown_threshold:
            f *= drawdown_reduction

        kelly_fracs[i] = f
        equity[i + 1] = equity[i] * (1.0 + f * arr[i])

    equity = equity[1:]

    # Stats
    running_max = np.maximum.accumulate(np.insert(equity, 0, 1.0))
    drawdowns = (np.insert(equity, 0, 1.0) - running_max) / running_max
    max_dd = float(np.min(drawdowns))
    final = float(equity[-1]) if len(equity) > 0 else 1.0
    cagr = float(final ** (252.0 / max(n, 1)) - 1.0) if final > 0 else -1.0

    return KellyResult(
        fraction=float(np.mean(kelly_fracs)),
        equity_curve=np.insert(equity, 0, 1.0),
        max_drawdown=max_dd,
        final_wealth=final,
        cagr=cagr,
        method="kelly_drawdown_control",
    )
