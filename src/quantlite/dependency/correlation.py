"""Dynamic and stress correlation analysis.

Provides rolling, exponentially-weighted, stress-conditional, and
rank-based correlation measures for robustly capturing time-varying
dependence in financial returns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

__all__ = [
    "rolling_correlation",
    "exponential_weighted_correlation",
    "stress_correlation",
    "correlation_breakdown_test",
    "rank_correlation",
]


def rolling_correlation(
    x: pd.Series | np.ndarray,
    y: pd.Series | np.ndarray,
    window: int = 60,
) -> pd.Series:
    """Compute rolling Pearson correlation between two return series.

    Args:
        x: First return series.
        y: Second return series.
        window: Rolling window size in periods.

    Returns:
        A pandas Series of rolling correlations (NaN for initial periods).
    """
    sx = pd.Series(np.asarray(x, dtype=float))
    sy = pd.Series(np.asarray(y, dtype=float))
    return sx.rolling(window).corr(sy)


def exponential_weighted_correlation(
    x: pd.Series | np.ndarray,
    y: pd.Series | np.ndarray,
    halflife: int = 30,
) -> pd.Series:
    """Compute exponentially-weighted moving average correlation.

    Recent observations receive exponentially more weight, making
    this estimator more responsive to regime changes than simple
    rolling correlation.

    Args:
        x: First return series.
        y: Second return series.
        halflife: Decay halflife in periods.

    Returns:
        A pandas Series of EWMA correlations.
    """
    sx = pd.Series(np.asarray(x, dtype=float))
    sy = pd.Series(np.asarray(y, dtype=float))

    ewm_cov = sx.ewm(halflife=halflife).cov(sy)
    ewm_var_x = sx.ewm(halflife=halflife).var()
    ewm_var_y = sy.ewm(halflife=halflife).var()

    denom = np.sqrt(ewm_var_x * ewm_var_y)
    denom = denom.replace(0, np.nan)
    return ewm_cov / denom


def stress_correlation(
    returns_df: pd.DataFrame,
    threshold_percentile: float = 10.0,
) -> pd.DataFrame:
    """Compute correlation matrix conditional on stress periods.

    A period is classified as "stress" if any asset's return falls
    below its threshold percentile. This captures the well-documented
    phenomenon of correlation increasing during drawdowns.

    Args:
        returns_df: DataFrame with assets as columns.
        threshold_percentile: Percentile below which an observation
            is considered stressed (applied per asset).

    Returns:
        Correlation matrix of returns during stress periods.

    Raises:
        ValueError: If fewer than 10 stress observations are found.
    """
    thresholds = returns_df.quantile(threshold_percentile / 100)
    stress_mask = (returns_df <= thresholds).any(axis=1)
    stress_returns = returns_df.loc[stress_mask]

    if len(stress_returns) < 10:
        raise ValueError(
            f"Only {len(stress_returns)} stress observations found; "
            "consider raising the threshold percentile."
        )

    return stress_returns.corr()


def correlation_breakdown_test(
    returns_df: pd.DataFrame,
    threshold_percentile: float = 25.0,
) -> dict[str, float]:
    """Test whether correlation increases significantly during drawdowns.

    Compares the average pairwise correlation in calm periods against
    stress periods using a Fisher z-transformation test.

    Args:
        returns_df: DataFrame with assets as columns.
        threshold_percentile: Percentile threshold separating calm
            from stress periods (applied to equal-weighted portfolio).

    Returns:
        Dict with ``"test_statistic"``, ``"p_value"``, ``"calm_corr"``,
        and ``"stress_corr"`` keys.
    """
    portfolio = returns_df.mean(axis=1)
    threshold = portfolio.quantile(threshold_percentile / 100)

    stress_mask = portfolio <= threshold
    calm_mask = ~stress_mask

    stress_returns = returns_df.loc[stress_mask]
    calm_returns = returns_df.loc[calm_mask]

    n_stress = len(stress_returns)
    n_calm = len(calm_returns)

    if n_stress < 10 or n_calm < 10:
        raise ValueError("Insufficient observations in stress or calm regime")

    # Average pairwise correlation (upper triangle)
    stress_corr = stress_returns.corr()
    calm_corr = calm_returns.corr()

    mask_upper = np.triu(np.ones(stress_corr.shape, dtype=bool), k=1)
    avg_stress = float(stress_corr.values[mask_upper].mean())
    avg_calm = float(calm_corr.values[mask_upper].mean())

    # Fisher z-test for difference in correlations
    z_stress = np.arctanh(np.clip(avg_stress, -0.999, 0.999))
    z_calm = np.arctanh(np.clip(avg_calm, -0.999, 0.999))
    se = np.sqrt(1 / (n_stress - 3) + 1 / (n_calm - 3))
    test_stat = (z_stress - z_calm) / se
    p_value = 2 * (1 - stats.norm.cdf(abs(test_stat)))

    return {
        "test_statistic": float(test_stat),
        "p_value": float(p_value),
        "calm_corr": avg_calm,
        "stress_corr": avg_stress,
    }


def rank_correlation(
    x: pd.Series | np.ndarray,
    y: pd.Series | np.ndarray,
    method: str = "spearman",
) -> tuple[float, float]:
    """Compute rank-based correlation (more robust than Pearson for fat tails).

    Args:
        x: First series.
        y: Second series.
        method: ``"spearman"`` or ``"kendall"``.

    Returns:
        Tuple of (correlation, p-value).

    Raises:
        ValueError: On unknown method.
    """
    ax = np.asarray(x, dtype=float)
    ay = np.asarray(y, dtype=float)

    if method == "spearman":
        corr, p = stats.spearmanr(ax, ay)
    elif method == "kendall":
        corr, p = stats.kendalltau(ax, ay)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'spearman' or 'kendall'.")

    return float(corr), float(p)
