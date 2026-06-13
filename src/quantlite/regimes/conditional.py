"""Regime-conditional risk metrics.

Computes risk metrics, correlations, and VaR separately for each
market regime, enabling regime-aware risk management.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..risk.metrics import cvar, sortino_ratio, value_at_risk

__all__ = [
    "conditional_metrics",
    "regime_transition_risk",
    "conditional_correlation",
    "regime_aware_var",
]


def conditional_metrics(
    returns: np.ndarray | pd.Series,
    regimes: np.ndarray,
) -> dict[int, dict[str, float]]:
    """Compute risk metrics separately for each regime.

    For each unique regime label, calculates VaR, CVaR, mean,
    volatility, Sortino ratio, and observation count.

    Args:
        returns: Simple periodic returns.
        regimes: Array of integer regime labels (same length as returns).

    Returns:
        Dict mapping regime label to a dict of metric name to value.
    """
    arr = np.asarray(returns, dtype=float)
    reg = np.asarray(regimes, dtype=int)

    result: dict[int, dict[str, float]] = {}

    for label in sorted(np.unique(reg)):
        mask = reg == label
        r = arr[mask]
        n = len(r)

        metrics: dict[str, float] = {
            "n_observations": float(n),
            "mean": float(np.mean(r)),
            "volatility": float(np.std(r, ddof=1)) if n > 1 else 0.0,
        }

        if n >= 2:
            metrics["var_95"] = value_at_risk(r, alpha=0.05)
            metrics["cvar_95"] = cvar(r, alpha=0.05)
        if n >= 10:
            metrics["sortino"] = sortino_ratio(r)

        result[int(label)] = metrics

    return result


def regime_transition_risk(
    model: Any,
) -> dict[str, Any]:
    """Compute regime transition risk measures.

    Calculates the probability of transitioning from the calmest
    regime (highest mean) to the most volatile regime (lowest mean),
    and the expected duration in each regime.

    Args:
        model: A ``RegimeModel`` from ``quantlite.regimes.hmm``.

    Returns:
        Dict with ``"calm_to_crisis_prob"``, ``"expected_durations"``,
        and ``"crisis_regime"`` keys.
    """
    trans = model.transition_matrix
    n = model.n_regimes

    # Regime 0 is crisis (lowest mean, sorted by fit_regime_model)
    crisis_regime = 0
    calm_regime = n - 1

    # Expected duration = 1 / (1 - p_ii)
    durations = {}
    for i in range(n):
        p_stay = trans[i, i]
        durations[i] = 1.0 / (1.0 - p_stay) if p_stay < 1.0 else float("inf")

    return {
        "calm_to_crisis_prob": float(trans[calm_regime, crisis_regime]),
        "expected_durations": durations,
        "crisis_regime": crisis_regime,
        "calm_regime": calm_regime,
    }


def conditional_correlation(
    returns_df: pd.DataFrame,
    regimes: np.ndarray,
) -> dict[int, pd.DataFrame]:
    """Compute the correlation matrix separately for each regime.

    Args:
        returns_df: DataFrame of asset returns.
        regimes: Array of regime labels.

    Returns:
        Dict mapping regime label to correlation DataFrame.
    """
    reg = np.asarray(regimes, dtype=int)
    result: dict[int, pd.DataFrame] = {}

    for label in sorted(np.unique(reg)):
        mask = reg == label
        sub = returns_df.iloc[mask]
        if len(sub) >= 3:
            result[int(label)] = sub.corr()

    return result


def regime_aware_var(
    returns: np.ndarray | pd.Series,
    regimes: np.ndarray,
    alpha: float = 0.05,
    current_probs: np.ndarray | None = None,
) -> float:
    """Compute VaR weighted by current regime probabilities.

    If ``current_probs`` is not provided, uses the stationary
    distribution (equal weight to each regime's VaR proportional
    to its frequency in the sample).

    Args:
        returns: Simple periodic returns.
        regimes: Array of regime labels.
        alpha: Significance level.
        current_probs: Optional probability vector over regimes.

    Returns:
        Regime-weighted VaR as a float.
    """
    arr = np.asarray(returns, dtype=float)
    reg = np.asarray(regimes, dtype=int)
    labels = sorted(np.unique(reg))

    if current_probs is None:
        # Use empirical frequencies
        current_probs = np.array([
            np.sum(reg == lbl) / len(reg) for lbl in labels
        ])

    weighted_var = 0.0
    for i, label in enumerate(labels):
        mask = reg == label
        r = arr[mask]
        if len(r) >= 2:
            var_i = value_at_risk(r, alpha=alpha)
        else:
            var_i = float(np.min(r)) if len(r) > 0 else 0.0
        weighted_var += current_probs[i] * var_i

    return float(weighted_var)
