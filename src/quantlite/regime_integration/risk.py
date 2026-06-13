"""Regime-conditional risk analytics.

Computes VaR, CVaR, and summary statistics broken down by market regime.
Also estimates transition risk (probability of moving to worse regimes).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

__all__ = [
    "regime_conditional_var",
    "regime_conditional_cvar",
    "regime_risk_summary",
    "regime_transition_risk",
]


def _to_array(returns: np.ndarray | pd.Series | list[float]) -> np.ndarray:
    """Coerce returns to a clean float64 array, dropping NaNs."""
    arr = np.asarray(returns, dtype=float)
    return arr[~np.isnan(arr)]


def _to_regime_array(
    regimes: np.ndarray | pd.Series | list[Any],
) -> np.ndarray:
    """Coerce regimes to a numpy array."""
    return np.asarray(regimes)


def regime_conditional_var(
    returns: np.ndarray | pd.Series | list[float],
    regimes: np.ndarray | pd.Series | list[Any],
    alpha: float = 0.05,
) -> dict[str, float]:
    """Compute Value at Risk separately for each regime.

    For each unique regime label, the historical VaR is computed at the
    given confidence level using only observations belonging to that regime.

    Args:
        returns: Simple periodic returns.
        regimes: Regime labels, same length as *returns*.
        alpha: Significance level (0.05 = 95% VaR).

    Returns:
        Dictionary mapping regime labels (as strings) to VaR values
        (negative floats representing losses).

    Raises:
        ValueError: If *returns* and *regimes* have different lengths.
    """
    arr = _to_array(returns)
    reg = _to_regime_array(regimes)
    if len(arr) != len(reg):
        raise ValueError(
            f"returns ({len(arr)}) and regimes ({len(reg)}) must have the same length"
        )

    result: dict[str, float] = {}
    for label in np.unique(reg):
        mask = reg == label
        regime_returns = arr[mask]
        if len(regime_returns) < 2:
            result[str(label)] = float("nan")
        else:
            result[str(label)] = float(np.percentile(regime_returns, alpha * 100))
    return result


def regime_conditional_cvar(
    returns: np.ndarray | pd.Series | list[float],
    regimes: np.ndarray | pd.Series | list[Any],
    alpha: float = 0.05,
) -> dict[str, float]:
    """Compute Conditional Value at Risk (Expected Shortfall) per regime.

    For each regime, CVaR is the mean of returns below the VaR threshold.

    Args:
        returns: Simple periodic returns.
        regimes: Regime labels, same length as *returns*.
        alpha: Significance level (0.05 = 95% CVaR).

    Returns:
        Dictionary mapping regime labels (as strings) to CVaR values.

    Raises:
        ValueError: If *returns* and *regimes* have different lengths.
    """
    arr = _to_array(returns)
    reg = _to_regime_array(regimes)
    if len(arr) != len(reg):
        raise ValueError(
            f"returns ({len(arr)}) and regimes ({len(reg)}) must have the same length"
        )

    result: dict[str, float] = {}
    for label in np.unique(reg):
        mask = reg == label
        regime_returns = arr[mask]
        if len(regime_returns) < 2:
            result[str(label)] = float("nan")
        else:
            var_threshold = float(np.percentile(regime_returns, alpha * 100))
            tail = regime_returns[regime_returns <= var_threshold]
            if len(tail) == 0:
                result[str(label)] = var_threshold
            else:
                result[str(label)] = float(np.mean(tail))
    return result


def regime_risk_summary(
    returns: np.ndarray | pd.Series | list[float],
    regimes: np.ndarray | pd.Series | list[Any],
    alpha: float = 0.05,
) -> dict[str, dict[str, float]]:
    """One-call risk summary per regime, plus overall.

    Computes VaR, CVaR, annualised volatility, skewness, and kurtosis
    for each regime and for the full sample.

    Args:
        returns: Simple periodic returns.
        regimes: Regime labels, same length as *returns*.
        alpha: Significance level for VaR and CVaR.

    Returns:
        Dictionary of dictionaries. Outer keys are regime labels plus
        ``"overall"``. Inner keys: ``"var"``, ``"cvar"``, ``"volatility"``,
        ``"skewness"``, ``"kurtosis"``, ``"count"``.
    """
    arr = _to_array(returns)
    reg = _to_regime_array(regimes)
    if len(arr) != len(reg):
        raise ValueError(
            f"returns ({len(arr)}) and regimes ({len(reg)}) must have the same length"
        )

    def _compute_stats(r: np.ndarray) -> dict[str, float]:
        if len(r) < 2:
            return {
                "var": float("nan"),
                "cvar": float("nan"),
                "volatility": float("nan"),
                "skewness": float("nan"),
                "kurtosis": float("nan"),
                "count": float(len(r)),
            }
        var_val = float(np.percentile(r, alpha * 100))
        tail = r[r <= var_val]
        cvar_val = float(np.mean(tail)) if len(tail) > 0 else var_val
        return {
            "var": var_val,
            "cvar": cvar_val,
            "volatility": float(np.std(r, ddof=1) * np.sqrt(252)),
            "skewness": float(stats.skew(r)),
            "kurtosis": float(stats.kurtosis(r)),
            "count": float(len(r)),
        }

    result: dict[str, dict[str, float]] = {}
    for label in np.unique(reg):
        mask = reg == label
        result[str(label)] = _compute_stats(arr[mask])
    result["overall"] = _compute_stats(arr)
    return result


def regime_transition_risk(
    transition_matrix: np.ndarray,
    current_regime: int,
    worse_regimes: list[int] | None = None,
) -> dict[str, float]:
    """Probability of transitioning to a worse regime within 1, 5, 21 steps.

    By default, regimes with lower indices are considered "worse"
    (matching the convention where regime 0 = crisis, sorted by mean
    return). Override with *worse_regimes*.

    Args:
        transition_matrix: Row-stochastic transition matrix of shape
            ``(n_regimes, n_regimes)``.
        current_regime: Index of the current regime.
        worse_regimes: Indices of regimes considered "worse". Defaults to
            all regimes with index less than *current_regime*.

    Returns:
        Dictionary with keys ``"1_step"``, ``"5_step"``, ``"21_step"``
        mapping to cumulative probabilities of entering a worse regime.
    """
    tm = np.asarray(transition_matrix, dtype=float)

    if worse_regimes is None:
        worse_regimes = [i for i in range(current_regime)]

    if not worse_regimes:
        return {"1_step": 0.0, "5_step": 0.0, "21_step": 0.0}

    result: dict[str, float] = {}
    for label, steps in [("1_step", 1), ("5_step", 5), ("21_step", 21)]:
        # Probability of being in any worse regime after k steps
        mat_k = np.linalg.matrix_power(tm, steps)
        prob = sum(float(mat_k[current_regime, w]) for w in worse_regimes)
        result[label] = min(prob, 1.0)
    return result
