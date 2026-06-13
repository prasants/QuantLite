"""Extreme Value Theory: GPD, GEV, Hill estimator, and tail analysis.

Provides tools for modelling the tails of return distributions where
Gaussian assumptions fail. Uses scipy.stats for maximum likelihood
estimation of the Generalised Pareto and Generalised Extreme Value
distributions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from ..core.types import GEVFit, GPDFit, HillEstimate, TailRiskSummary
from .metrics import cvar, value_at_risk

__all__ = [
    "fit_gpd",
    "fit_gev",
    "hill_estimator",
    "peaks_over_threshold",
    "return_level",
    "tail_risk_summary",
]


def _to_array(returns: np.ndarray | pd.Series | list[float]) -> np.ndarray:
    """Coerce to float64 array, dropping NaNs."""
    arr = np.asarray(returns, dtype=float)
    return arr[~np.isnan(arr)]


def fit_gpd(
    returns: np.ndarray | pd.Series | list[float],
    threshold: float | None = None,
) -> GPDFit:
    """Fit a Generalised Pareto Distribution to tail exceedances.

    Uses MLE via ``scipy.stats.genpareto``. By default the threshold
    is set at the 95th percentile of losses (negated returns).

    Args:
        returns: Simple periodic returns.
        threshold: Loss threshold. If ``None``, uses the 95th
            percentile of ``-returns``.

    Returns:
        A ``GPDFit`` dataclass with shape, scale, threshold, and counts.

    Raises:
        ValueError: If fewer than 10 exceedances are found.
    """
    arr = _to_array(returns)
    losses = -arr  # work with losses (positive values)

    if threshold is None:
        threshold = float(np.percentile(losses, 95))

    exceedances = losses[losses > threshold] - threshold
    if len(exceedances) < 10:
        raise ValueError(
            f"Only {len(exceedances)} exceedances found; need at least 10. "
            "Consider lowering the threshold."
        )

    shape, _loc, scale = stats.genpareto.fit(exceedances, floc=0)

    return GPDFit(
        shape=float(shape),
        scale=float(scale),
        threshold=float(threshold),
        n_exceedances=len(exceedances),
        n_total=len(arr),
    )


def fit_gev(
    block_maxima: np.ndarray | pd.Series | list[float],
) -> GEVFit:
    """Fit a Generalised Extreme Value distribution to block maxima.

    Typically used with annual or monthly maxima of losses.

    Args:
        block_maxima: Series of block maximum values.

    Returns:
        A ``GEVFit`` dataclass.

    Raises:
        ValueError: If fewer than 5 block maxima are provided.
    """
    arr = _to_array(block_maxima)
    if len(arr) < 5:
        raise ValueError("Need at least 5 block maxima for GEV fitting")

    shape, loc, scale = stats.genextreme.fit(arr)

    return GEVFit(
        shape=float(shape),
        loc=float(loc),
        scale=float(scale),
    )


def hill_estimator(
    returns: np.ndarray | pd.Series | list[float],
    k: int | None = None,
) -> HillEstimate:
    """Estimate the tail index using the Hill estimator.

    The Hill estimator applies to heavy-tailed distributions and
    estimates the power-law exponent alpha from the k largest
    observations.

    Args:
        returns: Simple periodic returns.
        k: Number of upper order statistics to use. If ``None``,
            defaults to ``int(sqrt(n))``.

    Returns:
        A ``HillEstimate`` dataclass with tail index and k.

    Raises:
        ValueError: If ``k`` is too small or data is insufficient.
    """
    arr = _to_array(returns)
    losses = -arr
    losses_sorted = np.sort(losses)[::-1]  # descending

    if k is None:
        k = max(int(np.sqrt(len(losses_sorted))), 2)
    if k < 2:
        raise ValueError("k must be >= 2")
    if k >= len(losses_sorted):
        raise ValueError(f"k={k} exceeds the number of observations ({len(losses_sorted)})")

    # Hill estimator: 1/alpha = (1/k) * sum(log(X_i / X_{k+1})) for i=1..k
    top_k = losses_sorted[:k]
    x_kplus1 = losses_sorted[k]
    if x_kplus1 <= 0:
        raise ValueError("The (k+1)-th order statistic is non-positive; Hill estimator undefined.")

    log_ratios = np.log(top_k / x_kplus1)
    gamma = float(np.mean(log_ratios))  # 1/alpha

    if gamma <= 0:
        raise ValueError("Hill estimator yielded non-positive gamma; data may not be heavy-tailed.")

    alpha = 1.0 / gamma
    return HillEstimate(tail_index=float(alpha), k=k)


def peaks_over_threshold(
    returns: np.ndarray | pd.Series | list[float],
    threshold: float | None = None,
) -> tuple[np.ndarray, GPDFit]:
    """Apply the Peaks Over Threshold method.

    Extracts exceedances above a threshold and fits a GPD.

    Args:
        returns: Simple periodic returns.
        threshold: Loss threshold (applied to ``-returns``). If ``None``,
            uses the 90th percentile of losses.

    Returns:
        Tuple of (exceedance values, GPDFit).
    """
    arr = _to_array(returns)
    losses = -arr

    if threshold is None:
        threshold = float(np.percentile(losses, 90))

    exceedances = losses[losses > threshold]
    gpd_fit = fit_gpd(returns, threshold=threshold)

    return exceedances, gpd_fit


def return_level(
    gpd_fit: GPDFit,
    return_period: float,
    n_obs: int | None = None,
) -> float:
    """Estimate the return level for a given return period.

    The return level is the loss magnitude expected to be exceeded
    once every ``return_period`` observations. For example, a
    return period of 250 with daily data gives a "1-in-1-year" loss.

    Args:
        gpd_fit: A fitted GPD from ``fit_gpd``.
        return_period: Number of periods (e.g. 25000 for "1-in-100-year"
            with daily data).
        n_obs: Total number of observations. If ``None``, uses
            ``gpd_fit.n_total``.

    Returns:
        Estimated loss at the given return level.
    """
    if n_obs is None:
        n_obs = gpd_fit.n_total

    zeta = gpd_fit.n_exceedances / n_obs  # exceedance probability
    xi = gpd_fit.shape
    sigma = gpd_fit.scale
    u = gpd_fit.threshold

    m = return_period
    if abs(xi) < 1e-10:
        # Exponential tail (xi ~ 0)
        level = u + sigma * np.log(m * zeta)
    else:
        level = u + (sigma / xi) * ((m * zeta) ** xi - 1)

    return float(level)


def tail_risk_summary(
    returns: np.ndarray | pd.Series | list[float],
) -> TailRiskSummary:
    """Produce a comprehensive tail risk analysis.

    Combines GPD fitting, Hill estimation, VaR, CVaR, and return
    level estimation into a single summary.

    Args:
        returns: Simple periodic returns (at least 100 observations
            recommended).

    Returns:
        A ``TailRiskSummary`` dataclass.
    """
    arr = _to_array(returns)

    gpd = fit_gpd(arr)
    hill = hill_estimator(arr)

    var_95 = value_at_risk(arr, alpha=0.05, method="historical")
    var_99 = value_at_risk(arr, alpha=0.01, method="historical")
    cvar_95 = cvar(arr, alpha=0.05)
    cvar_99 = cvar(arr, alpha=0.01)

    # 1-in-100-period return level
    rl_100 = return_level(gpd, return_period=100)

    excess_kurt = float(stats.kurtosis(arr))

    return TailRiskSummary(
        gpd_fit=gpd,
        hill_estimate=hill,
        var_95=var_95,
        var_99=var_99,
        cvar_95=cvar_95,
        cvar_99=cvar_99,
        return_level_100=rl_100,
        excess_kurtosis=excess_kurt,
    )
