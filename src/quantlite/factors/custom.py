"""Custom factor tools: factor wrapping, significance testing, portfolios, and decay.

Provides utilities for constructing custom factors, testing their statistical
significance, analysing multicollinearity, building factor-sorted portfolios,
and measuring how quickly a factor's predictive power decays with lag.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sp_stats


class CustomFactor:
    """Wrap any time series as a named factor.

    Parameters
    ----------
    name : str
        Human-readable factor name.
    series : array-like or pd.Series
        Factor return or value series.

    Attributes
    ----------
    name : str
        Factor name.
    values : np.ndarray
        Factor values as a numpy array.
    """

    def __init__(self, name, series):
        self.name = name
        if isinstance(series, pd.Series):
            self.values = series.values.astype(float)
        else:
            self.values = np.asarray(series, dtype=float)

    def __len__(self):
        return len(self.values)

    def __repr__(self):
        return f"CustomFactor(name='{self.name}', n={len(self.values)})"

    def mean(self):
        """Mean of the factor series."""
        return float(np.mean(self.values))

    def std(self):
        """Standard deviation of the factor series."""
        return float(np.std(self.values, ddof=1))

    def correlation(self, other):
        """Correlation with another CustomFactor or array.

        Parameters
        ----------
        other : CustomFactor or array-like
            The other series to correlate with.

        Returns
        -------
        float
            Pearson correlation coefficient.
        """
        if isinstance(other, CustomFactor):
            other_vals = other.values
        else:
            other_vals = np.asarray(other, dtype=float)
        return float(np.corrcoef(self.values, other_vals)[0, 1])


def test_factor_significance(returns, factor, control_factors=None):
    """Test a factor's marginal contribution via t-test and F-test.

    Runs a regression of returns on the factor (plus any control factors)
    and tests whether the factor adds explanatory power beyond the controls.

    Parameters
    ----------
    returns : array-like
        Asset return series.
    factor : CustomFactor or array-like
        The factor to test.
    control_factors : list of CustomFactor or array-like, optional
        Control factors to include in the regression.

    Returns
    -------
    dict
        Keys: t_stat, t_pvalue, f_stat, f_pvalue, beta, r_squared_full,
        r_squared_restricted, marginal_r_squared.
    """
    y = np.asarray(returns, dtype=float)
    f_vals = factor.values if isinstance(factor, CustomFactor) else np.asarray(factor, dtype=float)

    n = len(y)

    # Build control matrix
    controls = []
    if control_factors is not None:
        for cf in control_factors:
            if isinstance(cf, CustomFactor):
                controls.append(cf.values)
            else:
                controls.append(np.asarray(cf, dtype=float))

    # Restricted model (controls only)
    X_r = np.column_stack([np.ones(n)] + controls) if controls else np.ones((n, 1))

    beta_r = np.linalg.lstsq(X_r, y, rcond=None)[0]
    resid_r = y - X_r @ beta_r
    ss_res_r = float(np.sum(resid_r ** 2))

    # Full model (controls + test factor)
    X_f = np.column_stack([X_r, f_vals])
    beta_f = np.linalg.lstsq(X_f, y, rcond=None)[0]
    resid_f = y - X_f @ beta_f
    ss_res_f = float(np.sum(resid_f ** 2))

    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2_full = 1.0 - ss_res_f / ss_tot if ss_tot > 0 else 0.0
    r2_restricted = 1.0 - ss_res_r / ss_tot if ss_tot > 0 else 0.0
    marginal_r2 = r2_full - r2_restricted

    # F-test for marginal contribution (1 restriction)
    k_f = X_f.shape[1]
    dof_f = max(n - k_f, 1)
    f_stat = ((ss_res_r - ss_res_f) / 1) / (ss_res_f / dof_f) if ss_res_f > 0 else 0.0
    f_pvalue = 1.0 - sp_stats.f.cdf(abs(f_stat), 1, dof_f)

    # t-stat for the test factor coefficient
    beta_test = float(beta_f[-1])
    mse = ss_res_f / dof_f
    try:
        cov = mse * np.linalg.inv(X_f.T @ X_f)
    except np.linalg.LinAlgError:
        cov = mse * np.linalg.pinv(X_f.T @ X_f)
    se = np.sqrt(max(cov[-1, -1], 0.0))
    t_stat = beta_test / se if se > 0 else 0.0
    t_pvalue = 2.0 * (1.0 - sp_stats.t.cdf(abs(t_stat), df=dof_f))

    return {
        "t_stat": t_stat,
        "t_pvalue": t_pvalue,
        "f_stat": f_stat,
        "f_pvalue": f_pvalue,
        "beta": beta_test,
        "r_squared_full": r2_full,
        "r_squared_restricted": r2_restricted,
        "marginal_r_squared": marginal_r2,
    }


def factor_correlation_matrix(factors):
    """Compute correlation matrix between factors.

    Useful for detecting multicollinearity in factor models.

    Parameters
    ----------
    factors : list of CustomFactor or list of array-like
        Factor series to correlate.

    Returns
    -------
    dict
        Keys: matrix (2D np.ndarray), names (list of str),
        max_offdiag (float), pairs (list of tuples with high correlations).
    """
    arrays = []
    names = []
    for f in factors:
        if isinstance(f, CustomFactor):
            arrays.append(f.values)
            names.append(f.name)
        else:
            arrays.append(np.asarray(f, dtype=float))
            names.append(f"factor_{len(names)}")

    data = np.column_stack(arrays)
    corr = np.corrcoef(data, rowvar=False)

    # Find high off-diagonal correlations
    n = len(names)
    pairs = []
    max_offdiag = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            c = abs(corr[i, j])
            if c > max_offdiag:
                max_offdiag = c
            if c > 0.5:
                pairs.append((names[i], names[j], float(corr[i, j])))

    return {
        "matrix": corr,
        "names": names,
        "max_offdiag": float(max_offdiag),
        "pairs": sorted(pairs, key=lambda x: abs(x[2]), reverse=True),
    }


def factor_portfolio(returns_df, factor_values, n_quantiles=5):
    """Build long-short portfolios from factor-sorted quantiles.

    Sorts assets into quantiles based on factor values and computes
    the mean return for each quantile, plus the long-short spread.

    Parameters
    ----------
    returns_df : pd.DataFrame
        DataFrame of asset returns (columns are assets, rows are periods).
    factor_values : array-like or pd.Series
        Factor values for each asset (same length as number of columns).
    n_quantiles : int
        Number of quantiles (default 5 for quintiles).

    Returns
    -------
    dict
        Keys: quantile_returns (dict mapping quantile number to mean return),
        spread (top minus bottom quantile return), n_quantiles,
        monotonic (bool, whether returns increase monotonically).
    """
    if isinstance(returns_df, pd.DataFrame):
        mean_returns = returns_df.mean().values
    else:
        mean_returns = np.mean(np.asarray(returns_df, dtype=float), axis=0)

    factor_vals = np.asarray(factor_values, dtype=float)
    n_assets = len(factor_vals)

    # Sort assets by factor value
    sorted_idx = np.argsort(factor_vals)

    # Split into quantiles
    quantile_returns = {}
    chunk_size = n_assets / n_quantiles

    for q in range(n_quantiles):
        start = int(round(q * chunk_size))
        end = int(round((q + 1) * chunk_size))
        if start == end:
            end = min(start + 1, n_assets)
        indices = sorted_idx[start:end]
        quantile_returns[q + 1] = float(np.mean(mean_returns[indices]))

    # Spread: top quantile minus bottom quantile
    spread = quantile_returns[n_quantiles] - quantile_returns[1]

    # Test monotonicity
    qr_vals = [quantile_returns[q + 1] for q in range(n_quantiles)]
    diffs = [qr_vals[i + 1] - qr_vals[i] for i in range(len(qr_vals) - 1)]
    monotonic = all(d >= 0 for d in diffs) or all(d <= 0 for d in diffs)

    return {
        "quantile_returns": quantile_returns,
        "spread": spread,
        "n_quantiles": n_quantiles,
        "monotonic": monotonic,
    }


def factor_decay(returns, factor, max_lag=20):
    """Measure how a factor's predictive power decays with lag.

    Computes the correlation between factor values and forward returns
    at increasing lags, and estimates the half-life of predictive power.

    Parameters
    ----------
    returns : array-like
        Asset return series.
    factor : CustomFactor or array-like
        Factor series (same length as returns).
    max_lag : int
        Maximum number of lags to test (default 20).

    Returns
    -------
    dict
        Keys: decay_curve (list of (lag, correlation) tuples),
        half_life (float or None), r_squared_curve (list of
        (lag, r_squared) tuples).
    """
    y = np.asarray(returns, dtype=float)
    f = factor.values if isinstance(factor, CustomFactor) else np.asarray(factor, dtype=float)

    n = len(y)
    decay_curve = []
    r_squared_curve = []

    for lag in range(1, max_lag + 1):
        if lag >= n:
            break
        # Factor at time t predicts returns at time t+lag
        f_slice = f[:n - lag]
        y_slice = y[lag:]

        if len(f_slice) < 3:
            break

        corr = float(np.corrcoef(f_slice, y_slice)[0, 1])
        if np.isnan(corr):
            corr = 0.0
        decay_curve.append((lag, corr))
        r_squared_curve.append((lag, corr ** 2))

    # Estimate half-life from exponential decay fit
    half_life = None
    if len(decay_curve) >= 3:
        initial_corr = abs(decay_curve[0][1])
        if initial_corr > 0:
            half_target = initial_corr / 2.0
            for lag, corr in decay_curve[1:]:
                if abs(corr) <= half_target:
                    # Linear interpolation
                    prev_lag, prev_corr = decay_curve[lag - 2] if lag > 1 else (0, initial_corr)
                    # Find the previous entry
                    idx = lag - 1  # 0-indexed in decay_curve
                    if idx > 0 and idx < len(decay_curve):
                        prev_lag_val, prev_corr_val = decay_curve[idx - 1]
                        curr_corr_val = abs(corr)
                        prev_corr_abs = abs(prev_corr_val)
                        if prev_corr_abs != curr_corr_val:
                            frac = (prev_corr_abs - half_target) / (prev_corr_abs - curr_corr_val)
                            half_life = prev_lag_val + frac * (lag - prev_lag_val)
                        else:
                            half_life = float(lag)
                    else:
                        half_life = float(lag)
                    break

    return {
        "decay_curve": decay_curve,
        "half_life": half_life,
        "r_squared_curve": r_squared_curve,
    }
