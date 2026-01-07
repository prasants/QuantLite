"""Classical factor models: Fama-French, Carhart, and generic multi-factor attribution.

Implements the standard academic factor models used in performance attribution
and risk analysis, plus a flexible generic factor regression framework.
"""

from __future__ import annotations

import numpy as np
from scipy import stats as sp_stats


def _ols_regression(y, X):
    """Run OLS regression with intercept already included in X.

    Returns dict with coefficients, t-stats, p-values, R-squared,
    adjusted R-squared, and residuals.
    """
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    n, k = X.shape

    # OLS: beta = (X'X)^-1 X'y
    XtX = X.T @ X
    Xty = X.T @ y
    try:
        beta = np.linalg.solve(XtX, Xty)
    except np.linalg.LinAlgError:
        beta = np.linalg.lstsq(X, y, rcond=None)[0]

    residuals = y - X @ beta
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))

    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    adj_r_squared = 1.0 - (1.0 - r_squared) * (n - 1) / (n - k) if n > k else 0.0

    # Standard errors
    dof = max(n - k, 1)
    mse = ss_res / dof
    try:
        cov_matrix = mse * np.linalg.inv(XtX)
    except np.linalg.LinAlgError:
        cov_matrix = mse * np.linalg.pinv(XtX)
    se = np.sqrt(np.maximum(np.diag(cov_matrix), 0.0))

    t_stats = np.where(se > 0, beta / se, 0.0)
    p_values = np.array([
        2.0 * (1.0 - sp_stats.t.cdf(abs(t), df=dof)) for t in t_stats
    ])

    return {
        "coefficients": beta,
        "t_stats": t_stats,
        "p_values": p_values,
        "r_squared": r_squared,
        "adj_r_squared": adj_r_squared,
        "residuals": residuals,
        "se": se,
    }


def _run_factor_model(returns, factor_arrays, factor_names):
    """Common logic for running a factor regression.

    Parameters
    ----------
    returns : array-like
        Asset return series.
    factor_arrays : list of array-like
        Factor return series (each same length as returns).
    factor_names : list of str
        Names for each factor.

    Returns
    -------
    dict with alpha, betas, r_squared, t_stats, p_values, residuals,
    adj_r_squared, and named beta entries.
    """
    y = np.asarray(returns, dtype=float)
    factors = [np.asarray(f, dtype=float) for f in factor_arrays]

    n = len(y)

    # Build design matrix with intercept
    X = np.column_stack([np.ones(n)] + factors)

    reg = _ols_regression(y, X)

    alpha = float(reg["coefficients"][0])
    betas = {name: float(reg["coefficients"][i + 1]) for i, name in enumerate(factor_names)}

    result = {
        "alpha": alpha,
        "betas": betas,
        "r_squared": reg["r_squared"],
        "adj_r_squared": reg["adj_r_squared"],
        "t_stats": {
            "alpha": float(reg["t_stats"][0]),
        },
        "p_values": {
            "alpha": float(reg["p_values"][0]),
        },
        "residuals": reg["residuals"],
    }

    for i, name in enumerate(factor_names):
        result["t_stats"][name] = float(reg["t_stats"][i + 1])
        result["p_values"][name] = float(reg["p_values"][i + 1])

    return result


def fama_french_three(returns, market_returns, smb, hml):
    """Fama-French three-factor model regression.

    Regresses asset returns against market excess returns, SMB (Small Minus
    Big), and HML (High Minus Low) factors.

    Parameters
    ----------
    returns : array-like
        Asset excess return series.
    market_returns : array-like
        Market excess return series.
    smb : array-like
        Size factor (Small Minus Big).
    hml : array-like
        Value factor (High Minus Low).

    Returns
    -------
    dict
        Keys: alpha, betas (dict with 'market', 'smb', 'hml'),
        r_squared, adj_r_squared, t_stats, p_values, residuals.
    """
    return _run_factor_model(
        returns,
        [market_returns, smb, hml],
        ["market", "smb", "hml"],
    )


def fama_french_five(returns, market_returns, smb, hml, rmw, cma):
    """Fama-French five-factor model regression.

    Extends the three-factor model with profitability (RMW) and
    investment (CMA) factors.

    Parameters
    ----------
    returns : array-like
        Asset excess return series.
    market_returns : array-like
        Market excess return series.
    smb : array-like
        Size factor (Small Minus Big).
    hml : array-like
        Value factor (High Minus Low).
    rmw : array-like
        Profitability factor (Robust Minus Weak).
    cma : array-like
        Investment factor (Conservative Minus Aggressive).

    Returns
    -------
    dict
        Keys: alpha, betas (dict with 'market', 'smb', 'hml', 'rmw', 'cma'),
        r_squared, adj_r_squared, t_stats, p_values, residuals.
    """
    return _run_factor_model(
        returns,
        [market_returns, smb, hml, rmw, cma],
        ["market", "smb", "hml", "rmw", "cma"],
    )


def carhart_four(returns, market_returns, smb, hml, mom):
    """Carhart four-factor model regression.

    Extends the Fama-French three-factor model with a momentum factor.

    Parameters
    ----------
    returns : array-like
        Asset excess return series.
    market_returns : array-like
        Market excess return series.
    smb : array-like
        Size factor (Small Minus Big).
    hml : array-like
        Value factor (High Minus Low).
    mom : array-like
        Momentum factor (Winners Minus Losers).

    Returns
    -------
    dict
        Keys: alpha, betas (dict with 'market', 'smb', 'hml', 'mom'),
        r_squared, adj_r_squared, t_stats, p_values, residuals.
    """
    return _run_factor_model(
        returns,
        [market_returns, smb, hml, mom],
        ["market", "smb", "hml", "mom"],
    )


def factor_attribution(returns, factor_returns, factor_names):
    """Generic multi-factor attribution.

    Decomposes total returns into factor contributions and unexplained
    (alpha) component.

    Parameters
    ----------
    returns : array-like
        Asset return series.
    factor_returns : list of array-like
        Factor return series.
    factor_names : list of str
        Names for each factor.

    Returns
    -------
    dict
        Keys: alpha, factor_contributions (dict mapping factor name to
        mean contribution), unexplained (mean residual return),
        r_squared, total_return (annualised mean).
    """
    result = _run_factor_model(returns, factor_returns, factor_names)

    y = np.asarray(returns, dtype=float)
    factors = [np.asarray(f, dtype=float) for f in factor_returns]

    contributions = {}
    for i, name in enumerate(factor_names):
        beta = result["betas"][name]
        mean_factor = float(np.mean(factors[i]))
        contributions[name] = beta * mean_factor

    total_mean = float(np.mean(y))
    explained = sum(contributions.values())
    unexplained = total_mean - explained

    return {
        "alpha": result["alpha"],
        "factor_contributions": contributions,
        "unexplained": unexplained,
        "r_squared": result["r_squared"],
        "total_return": total_mean,
    }


def factor_summary(returns, factor_returns, factor_names):
    """One-call summary table for multi-factor regression.

    Returns a comprehensive summary including alpha, each factor beta,
    t-statistics, p-values, R-squared, and adjusted R-squared.

    Parameters
    ----------
    returns : array-like
        Asset return series.
    factor_returns : list of array-like
        Factor return series.
    factor_names : list of str
        Names for each factor.

    Returns
    -------
    dict
        Keys: alpha, alpha_t, alpha_p, betas (dict), t_stats (dict),
        p_values (dict), r_squared, adj_r_squared, n_obs.
    """
    result = _run_factor_model(returns, factor_returns, factor_names)

    return {
        "alpha": result["alpha"],
        "alpha_t": result["t_stats"]["alpha"],
        "alpha_p": result["p_values"]["alpha"],
        "betas": result["betas"],
        "t_stats": {name: result["t_stats"][name] for name in factor_names},
        "p_values": {name: result["p_values"][name] for name in factor_names},
        "r_squared": result["r_squared"],
        "adj_r_squared": result["adj_r_squared"],
        "n_obs": len(np.asarray(returns)),
    }
