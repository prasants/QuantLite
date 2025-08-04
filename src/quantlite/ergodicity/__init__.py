"""Ergodicity economics: time-average vs ensemble-average growth rates.

Most of finance assumes ergodicity (time average = ensemble average).
Real markets are non-ergodic: what happens to the average person is
not what happens to any particular person over time.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

__all__ = [
    "time_average",
    "ensemble_average",
    "ergodicity_gap",
    "kelly_fraction",
    "leverage_effect",
    "geometric_mean_dominance",
]


def _to_array(returns: ArrayLike) -> np.ndarray:
    """Convert input to a flat numpy array of floats."""
    arr = np.asarray(returns, dtype=float).ravel()
    if arr.size == 0:
        raise ValueError("returns must not be empty")
    return arr


def time_average(returns: ArrayLike) -> float:
    """Compute the time-average (geometric mean) growth rate.

    This is the growth rate experienced by a single individual
    over many periods: the only rate that matters for wealth dynamics.

    Parameters
    ----------
    returns : array-like
        Simple period returns (e.g. 0.05 for 5%).

    Returns
    -------
    float
        Geometric mean growth rate per period.
    """
    r = _to_array(returns)
    log_growth = np.mean(np.log1p(r))
    return float(np.expm1(log_growth))


def ensemble_average(returns: ArrayLike) -> float:
    """Compute the ensemble-average (arithmetic mean) growth rate.

    This is what textbooks report and what misleads: the average
    across many parallel realisations at a single point in time.

    Parameters
    ----------
    returns : array-like
        Simple period returns.

    Returns
    -------
    float
        Arithmetic mean return per period.
    """
    r = _to_array(returns)
    return float(np.mean(r))


def ergodicity_gap(returns: ArrayLike) -> float:
    """Compute the gap between ensemble and time averages.

    A large positive gap means the strategy looks good on paper
    (ensemble average) but is dangerous for any individual over time.

    Parameters
    ----------
    returns : array-like
        Simple period returns.

    Returns
    -------
    float
        ensemble_average - time_average. Positive means non-ergodic danger.
    """
    return ensemble_average(returns) - time_average(returns)


def kelly_fraction(returns: ArrayLike, risk_free: float = 0.0) -> float:
    """Compute the optimal Kelly fraction for geometric growth.

    The Kelly criterion maximises the expected logarithmic growth rate.
    For a simple binary-style approximation from empirical returns,
    we optimise f to maximise E[log(1 + f * (r - risk_free))].

    Uses a numerical grid search over [0, 2] for robustness.

    Parameters
    ----------
    returns : array-like
        Simple period returns.
    risk_free : float
        Risk-free rate per period (default 0).

    Returns
    -------
    float
        Optimal fraction of capital to deploy. Can be < 0 (short)
        or > 1 (leveraged).
    """
    r = _to_array(returns)
    excess = r - risk_free

    # Grid search over fractions from -0.5 to 3.0
    fractions = np.linspace(-0.5, 3.0, 3500)
    best_f = 0.0
    best_g = -np.inf

    for f in fractions:
        portfolio = 1.0 + f * excess
        if np.any(portfolio <= 0):
            continue
        g = np.mean(np.log(portfolio))
        if g > best_g:
            best_g = g
            best_f = f

    return float(round(best_f, 4))


def leverage_effect(
    returns: ArrayLike,
    leverages: list[float] | None = None,
) -> dict[float, float]:
    """Show how leverage affects time-average growth rate.

    Demonstrates why 2x or 3x leverage can destroy wealth even when
    the underlying has positive expected returns: volatility drag
    compounds geometrically.

    Parameters
    ----------
    returns : array-like
        Simple period returns of the underlying.
    leverages : list of float, optional
        Leverage multiples to evaluate (default [1, 2, 3, 5]).

    Returns
    -------
    dict
        Mapping of leverage multiple to time-average growth rate.
    """
    if leverages is None:
        leverages = [1.0, 2.0, 3.0, 5.0]

    r = _to_array(returns)
    result = {}
    for lev in leverages:
        leveraged = lev * r
        # If any period wipes out capital (return <= -100%), growth is -100%
        if np.any(leveraged <= -1.0):
            result[lev] = -1.0
        else:
            log_growth = np.mean(np.log1p(leveraged))
            result[lev] = float(np.expm1(log_growth))
    return result


def geometric_mean_dominance(
    returns_a: ArrayLike,
    returns_b: ArrayLike,
) -> dict[str, object]:
    """Test whether strategy A dominates strategy B in geometric mean.

    Parameters
    ----------
    returns_a : array-like
        Simple period returns for strategy A.
    returns_b : array-like
        Simple period returns for strategy B.

    Returns
    -------
    dict
        Keys: 'g_mean_a', 'g_mean_b', 'dominant' ('A', 'B', or 'neither'),
        'margin' (absolute difference).
    """
    g_a = time_average(returns_a)
    g_b = time_average(returns_b)
    margin = abs(g_a - g_b)

    if g_a > g_b:
        dominant = "A"
    elif g_b > g_a:
        dominant = "B"
    else:
        dominant = "neither"

    return {
        "g_mean_a": g_a,
        "g_mean_b": g_b,
        "dominant": dominant,
        "margin": margin,
    }
