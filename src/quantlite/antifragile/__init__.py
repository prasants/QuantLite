"""Antifragility framework: measuring what gains from disorder.

Fragile things break under stress. Robust things resist it.
Antifragile things get stronger. This module quantifies where
your portfolio sits on that spectrum.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

__all__ = [
    "antifragility_score",
    "convexity_score",
    "fourth_quadrant",
    "barbell_allocation",
    "lindy_estimate",
    "skin_in_game_score",
]


def _to_array(returns: ArrayLike) -> np.ndarray:
    """Convert input to a flat numpy array of floats."""
    arr = np.asarray(returns, dtype=float).ravel()
    if arr.size == 0:
        raise ValueError("returns must not be empty")
    return arr


def antifragility_score(returns: ArrayLike) -> float:
    """Measure antifragility via payoff convexity.

    Compares the average gain from positive shocks (above median)
    to the average loss from negative shocks (below median).
    Antifragile entities gain more from upside than they lose
    from downside of equal magnitude.

    Parameters
    ----------
    returns : array-like
        Simple period returns.

    Returns
    -------
    float
        Score > 0 indicates antifragility, < 0 indicates fragility,
        near 0 indicates robustness.
    """
    r = _to_array(returns)
    median = np.median(r)
    above = r[r > median]
    below = r[r < median]

    if above.size == 0 or below.size == 0:
        return 0.0

    avg_gain = np.mean(above - median)
    avg_loss = np.mean(median - below)

    if avg_loss == 0:
        return float(avg_gain)

    # Ratio of convex gain to concave loss, centred at 0
    return float((avg_gain / avg_loss) - 1.0)


def convexity_score(returns: ArrayLike, shocks: ArrayLike) -> float:
    """Measure payoff curvature across the distribution.

    Fits a second-order polynomial to (shock, return) pairs.
    Positive curvature = convex payoff = antifragile.

    Parameters
    ----------
    returns : array-like
        Observed returns (dependent variable).
    shocks : array-like
        Shock magnitudes (independent variable).

    Returns
    -------
    float
        Second-order coefficient of the fitted polynomial.
        Positive = convex, negative = concave.
    """
    r = _to_array(returns)
    s = _to_array(shocks)
    if r.size != s.size:
        raise ValueError("returns and shocks must have the same length")
    if r.size < 3:
        raise ValueError("Need at least 3 observations for curvature estimation")

    coeffs = np.polyfit(s, r, 2)
    return float(coeffs[0])


def fourth_quadrant(returns: ArrayLike) -> dict[str, object]:
    """Detect if returns fall in Taleb's Fourth Quadrant.

    The Fourth Quadrant is where fat tails meet complex payoffs,
    making statistical models dangerous and unreliable.

    Uses excess kurtosis as a fat-tail indicator and payoff
    nonlinearity as a complexity indicator.

    Parameters
    ----------
    returns : array-like
        Simple period returns.

    Returns
    -------
    dict
        Keys: 'kurtosis' (excess), 'fat_tailed' (bool),
        'payoff_nonlinearity', 'fourth_quadrant' (bool),
        'warning' (str).
    """
    r = _to_array(returns)

    # Excess kurtosis (normal = 0)
    n = r.size
    mean = np.mean(r)
    std = np.std(r, ddof=1) if n > 1 else 1.0
    kurt = 0.0 if std == 0 else float(np.mean(((r - mean) / std) ** 4) - 3.0)

    fat_tailed = kurt > 1.0

    # Payoff nonlinearity: ratio of tail impact to body impact
    sorted_r = np.sort(r)
    tail_size = max(1, n // 10)
    left_tail = sorted_r[:tail_size]
    right_tail = sorted_r[-tail_size:]
    body = sorted_r[tail_size:-tail_size] if n > 2 * tail_size else sorted_r

    tail_impact = np.mean(np.abs(right_tail)) + np.mean(np.abs(left_tail))
    body_impact = np.mean(np.abs(body)) if body.size > 0 else 1.0
    nonlinearity = float(tail_impact / body_impact) if body_impact > 0 else 0.0

    in_fourth = fat_tailed and nonlinearity > 2.0

    if in_fourth:
        warning = (
            "Fourth Quadrant detected: fat tails with complex payoffs. "
            "Standard models are unreliable here; use extreme caution."
        )
    else:
        warning = "Not in the Fourth Quadrant, but remain vigilant."

    return {
        "kurtosis": kurt,
        "fat_tailed": fat_tailed,
        "payoff_nonlinearity": nonlinearity,
        "fourth_quadrant": in_fourth,
        "warning": warning,
    }


def barbell_allocation(
    conservative_returns: ArrayLike,
    aggressive_returns: ArrayLike,
    conservative_pct: float = 0.9,
) -> dict[str, float]:
    """Compute barbell allocation metrics.

    The barbell strategy allocates most capital to hyperconservative
    assets and a small fraction to hyperaggressive ones, with nothing
    in the mediocre middle.

    Parameters
    ----------
    conservative_returns : array-like
        Returns from the conservative leg.
    aggressive_returns : array-like
        Returns from the aggressive leg.
    conservative_pct : float
        Fraction allocated to conservative leg (default 0.9).

    Returns
    -------
    dict
        Keys: 'conservative_pct', 'aggressive_pct',
        'blended_arithmetic', 'blended_geometric',
        'max_loss' (worst-case single-period loss),
        'upside_capture' (mean of top 10% of blended returns).
    """
    c = _to_array(conservative_returns)
    a = _to_array(aggressive_returns)

    min_len = min(c.size, a.size)
    c = c[:min_len]
    a = a[:min_len]

    agg_pct = 1.0 - conservative_pct
    blended = conservative_pct * c + agg_pct * a

    arith = float(np.mean(blended))
    geo = float(np.expm1(np.mean(np.log1p(blended))))

    # Worst-case loss
    max_loss = float(np.min(blended))

    # Upside capture: mean of top 10%
    top_n = max(1, min_len // 10)
    sorted_b = np.sort(blended)
    upside = float(np.mean(sorted_b[-top_n:]))

    return {
        "conservative_pct": conservative_pct,
        "aggressive_pct": agg_pct,
        "blended_arithmetic": arith,
        "blended_geometric": geo,
        "max_loss": max_loss,
        "upside_capture": upside,
    }


def lindy_estimate(age: float, confidence: float = 0.95) -> dict[str, float]:
    """Estimate remaining life expectancy using the Lindy effect.

    For non-perishable entities (ideas, technologies, institutions),
    expected remaining lifespan is proportional to current age.

    Uses a Pareto-like survival model where the expected remaining
    life equals the current age (for the median estimate).

    Parameters
    ----------
    age : float
        Current age of the entity (in any consistent unit).
    confidence : float
        Confidence level for the survival bound (default 0.95).

    Returns
    -------
    dict
        Keys: 'age', 'expected_remaining' (median estimate),
        'lower_bound' (at given confidence), 'total_expected'.
    """
    if age <= 0:
        raise ValueError("age must be positive")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")

    # Under Lindy (Pareto with alpha=1), expected remaining life = age
    expected_remaining = age

    # Lower bound: at confidence level, survival beyond this point
    # P(survive t more) = age / (age + t), so t = age * (1/p - 1)
    lower_bound = age * (1.0 / (1.0 - confidence) - 1.0) * (1.0 - confidence)
    # Simplifies to: age * confidence / (1 - confidence) * (1 - confidence) = age * confidence
    # Actually: P(T > age + t | T > age) = age / (age + t) for Pareto
    # Set this = 1 - confidence: age/(age+t) = 1 - confidence
    # t = age * confidence / (1 - confidence)
    lower_bound = age * (1.0 - confidence) / confidence
    # That's the point we're confident we'll reach (small value)
    # More useful: expected remaining at median
    # P(T > age + t | T > age) = 0.5 => t = age (median remaining = age)

    return {
        "age": age,
        "expected_remaining": expected_remaining,
        "lower_bound": float(lower_bound),
        "total_expected": age + expected_remaining,
    }


def skin_in_game_score(
    manager_returns: ArrayLike,
    fund_returns: ArrayLike,
) -> dict[str, float]:
    """Measure principal-agent alignment via payoff asymmetry.

    Compares the manager's exposure to downside vs upside relative
    to the fund. A good score means the manager shares the pain.

    Parameters
    ----------
    manager_returns : array-like
        Returns experienced by the manager (compensation-adjusted).
    fund_returns : array-like
        Returns experienced by the fund investors.

    Returns
    -------
    dict
        Keys: 'alignment' (correlation), 'downside_sharing' (ratio of
        manager downside to fund downside), 'upside_asymmetry',
        'score' (composite: 1.0 = perfect alignment).
    """
    m = _to_array(manager_returns)
    f = _to_array(fund_returns)
    min_len = min(m.size, f.size)
    m = m[:min_len]
    f = f[:min_len]

    # Correlation as alignment measure
    alignment = 0.0 if np.std(m) == 0 or np.std(f) == 0 else float(np.corrcoef(m, f)[0, 1])

    # Downside sharing: when fund loses, does the manager also lose?
    fund_down = f < 0
    if np.sum(fund_down) > 0:
        manager_down_exposure = np.mean(m[fund_down])
        fund_down_mean = np.mean(f[fund_down])
        downside_sharing = float(manager_down_exposure / fund_down_mean) if fund_down_mean != 0 else 0.0
    else:
        downside_sharing = 1.0

    # Upside asymmetry: does the manager capture disproportionate upside?
    fund_up = f > 0
    if np.sum(fund_up) > 0:
        manager_up_mean = np.mean(m[fund_up])
        fund_up_mean = np.mean(f[fund_up])
        upside_asymmetry = float(manager_up_mean / fund_up_mean) if fund_up_mean != 0 else 0.0
    else:
        upside_asymmetry = 1.0

    # Composite score: high alignment + high downside sharing + low upside asymmetry = good
    # Normalise to [0, 1] approximately
    score = (alignment * 0.4) + (min(downside_sharing, 1.0) * 0.4) + (max(0, 1.0 - abs(upside_asymmetry - 1.0)) * 0.2)

    return {
        "alignment": alignment,
        "downside_sharing": downside_sharing,
        "upside_asymmetry": upside_asymmetry,
        "score": float(np.clip(score, 0, 1)),
    }
