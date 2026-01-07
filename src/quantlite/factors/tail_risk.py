"""Tail-risk-aware factor analysis: CVaR decomposition, regime exposures, crowding, and tail betas.

Goes beyond variance-based factor models to analyse how factor exposures
behave in the tails of the return distribution and across market regimes.
"""

from __future__ import annotations

import numpy as np
from scipy import stats as sp_stats


def factor_cvar_decomposition(returns, factor_returns, factor_names, alpha=0.05):
    """Decompose CVaR into factor contributions.

    Uses component CVaR methodology: estimates each factor's contribution
    to portfolio CVaR based on factor betas and conditional tail behaviour.

    Parameters
    ----------
    returns : array-like
        Asset return series.
    factor_returns : list of array-like
        Factor return series.
    factor_names : list of str
        Names for each factor.
    alpha : float
        Tail probability (default 0.05, i.e. worst 5%).

    Returns
    -------
    dict
        Keys: total_cvar, factor_contributions (dict mapping factor name
        to CVaR contribution), residual_contribution, pct_contributions
        (dict mapping factor name to percentage of total CVaR).
    """
    y = np.asarray(returns, dtype=float)
    factors = [np.asarray(f, dtype=float) for f in factor_returns]
    n = len(y)

    # Identify tail observations
    var_threshold = np.percentile(y, alpha * 100)
    tail_mask = y <= var_threshold
    n_tail = int(np.sum(tail_mask))

    if n_tail < 2:
        return {
            "total_cvar": float(var_threshold),
            "factor_contributions": {name: 0.0 for name in factor_names},
            "residual_contribution": float(var_threshold),
            "pct_contributions": {name: 0.0 for name in factor_names},
        }

    # Total CVaR
    total_cvar = float(np.mean(y[tail_mask]))

    # OLS regression for betas
    X = np.column_stack([np.ones(n)] + factors)
    beta = np.linalg.lstsq(X, y, rcond=None)[0]

    # Component CVaR: beta_i * E[factor_i | portfolio in tail]
    contributions = {}
    for i, name in enumerate(factor_names):
        factor_tail_mean = float(np.mean(factors[i][tail_mask]))
        contributions[name] = float(beta[i + 1]) * factor_tail_mean

    # Alpha (intercept) contribution
    alpha_contrib = float(beta[0])

    # Residual contribution
    residuals = y - X @ beta
    residual_contrib = float(np.mean(residuals[tail_mask]))

    explained = sum(contributions.values()) + alpha_contrib
    residual_contribution = total_cvar - explained + residual_contrib

    # Percentage contributions
    pct_contributions = {}
    if abs(total_cvar) > 1e-12:
        for name in factor_names:
            pct_contributions[name] = contributions[name] / abs(total_cvar)
    else:
        pct_contributions = {name: 0.0 for name in factor_names}

    return {
        "total_cvar": total_cvar,
        "factor_contributions": contributions,
        "residual_contribution": residual_contribution,
        "pct_contributions": pct_contributions,
    }


def regime_factor_exposure(returns, factor_returns, factor_names, regimes):
    """Compute factor betas separately for each regime.

    Estimates regime-conditional factor loadings, revealing how factor
    exposures shift across bull, bear, and crisis periods.

    Parameters
    ----------
    returns : array-like
        Asset return series.
    factor_returns : list of array-like
        Factor return series.
    factor_names : list of str
        Names for each factor.
    regimes : array-like
        Regime labels for each observation (e.g. 'bull', 'bear', 'crisis').

    Returns
    -------
    dict
        Keys are regime labels, values are dicts with alpha, betas,
        r_squared, and n_obs for that regime.
    """
    y = np.asarray(returns, dtype=float)
    factors = [np.asarray(f, dtype=float) for f in factor_returns]
    regime_arr = np.asarray(regimes)

    unique_regimes = sorted(set(regime_arr.tolist()))
    results = {}

    for regime in unique_regimes:
        mask = regime_arr == regime
        n_regime = int(np.sum(mask))

        if n_regime < len(factor_names) + 2:
            # Not enough observations for regression
            results[regime] = {
                "alpha": None,
                "betas": {name: None for name in factor_names},
                "r_squared": None,
                "n_obs": n_regime,
            }
            continue

        y_r = y[mask]
        X_r = np.column_stack([np.ones(n_regime)] + [f[mask] for f in factors])

        beta = np.linalg.lstsq(X_r, y_r, rcond=None)[0]
        resid = y_r - X_r @ beta
        ss_res = float(np.sum(resid ** 2))
        ss_tot = float(np.sum((y_r - np.mean(y_r)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        betas = {}
        for i, name in enumerate(factor_names):
            betas[name] = float(beta[i + 1])

        results[regime] = {
            "alpha": float(beta[0]),
            "betas": betas,
            "r_squared": r2,
            "n_obs": n_regime,
        }

    return results


def factor_crowding_score(factor_returns, rolling_window=60):
    """Detect factor crowding via rolling pairwise correlations.

    Rising correlations among factor returns suggest that many investors
    are piling into the same trades, creating crowding risk.

    Parameters
    ----------
    factor_returns : list of array-like
        Factor return series (at least 2 factors).
    rolling_window : int
        Window size for rolling correlation (default 60).

    Returns
    -------
    dict
        Keys: crowding_scores (np.ndarray of rolling mean pairwise
        correlation), current_score (float, latest value),
        trend (float, slope of crowding score over time),
        is_crowded (bool, True if current score > 0.7).
    """
    factors = [np.asarray(f, dtype=float) for f in factor_returns]
    n_factors = len(factors)
    n = len(factors[0])

    if n_factors < 2:
        return {
            "crowding_scores": np.zeros(max(n - rolling_window + 1, 0)),
            "current_score": 0.0,
            "trend": 0.0,
            "is_crowded": False,
        }

    # Rolling mean pairwise correlation
    n_windows = max(n - rolling_window + 1, 0)
    scores = np.zeros(n_windows)

    for w in range(n_windows):
        start = w
        end = w + rolling_window
        corrs = []
        for i in range(n_factors):
            for j in range(i + 1, n_factors):
                c = np.corrcoef(factors[i][start:end], factors[j][start:end])[0, 1]
                if not np.isnan(c):
                    corrs.append(abs(c))
        scores[w] = float(np.mean(corrs)) if corrs else 0.0

    current_score = float(scores[-1]) if len(scores) > 0 else 0.0

    # Trend: simple linear slope
    trend = 0.0
    if len(scores) > 1:
        x = np.arange(len(scores), dtype=float)
        slope, _, _, _, _ = sp_stats.linregress(x, scores)
        trend = float(slope)

    return {
        "crowding_scores": scores,
        "current_score": current_score,
        "trend": trend,
        "is_crowded": current_score > 0.7,
    }


def tail_factor_beta(returns, factor_returns, factor_names, alpha=0.05):
    """Estimate factor betas using only tail observations.

    Restricts the regression to the worst alpha percentile of returns,
    revealing how factor exposures behave during extreme drawdowns.

    Parameters
    ----------
    returns : array-like
        Asset return series.
    factor_returns : list of array-like
        Factor return series.
    factor_names : list of str
        Names for each factor.
    alpha : float
        Tail probability (default 0.05, i.e. worst 5%).

    Returns
    -------
    dict
        Keys: tail_betas (dict mapping factor name to tail beta),
        full_betas (dict mapping factor name to full-sample beta),
        beta_ratio (dict, tail/full ratio showing amplification),
        n_tail_obs, var_threshold.
    """
    y = np.asarray(returns, dtype=float)
    factors = [np.asarray(f, dtype=float) for f in factor_returns]
    n = len(y)

    # Full-sample betas
    X_full = np.column_stack([np.ones(n)] + factors)
    beta_full = np.linalg.lstsq(X_full, y, rcond=None)[0]
    full_betas = {}
    for i, name in enumerate(factor_names):
        full_betas[name] = float(beta_full[i + 1])

    # Tail observations
    var_threshold = np.percentile(y, alpha * 100)
    tail_mask = y <= var_threshold
    n_tail = int(np.sum(tail_mask))

    if n_tail < len(factor_names) + 2:
        return {
            "tail_betas": {name: None for name in factor_names},
            "full_betas": full_betas,
            "beta_ratio": {name: None for name in factor_names},
            "n_tail_obs": n_tail,
            "var_threshold": float(var_threshold),
        }

    y_tail = y[tail_mask]
    X_tail = np.column_stack([np.ones(n_tail)] + [f[tail_mask] for f in factors])
    beta_tail = np.linalg.lstsq(X_tail, y_tail, rcond=None)[0]

    tail_betas = {}
    beta_ratio = {}
    for i, name in enumerate(factor_names):
        tb = float(beta_tail[i + 1])
        tail_betas[name] = tb
        fb = full_betas[name]
        if abs(fb) > 1e-12:
            beta_ratio[name] = tb / fb
        else:
            beta_ratio[name] = None

    return {
        "tail_betas": tail_betas,
        "full_betas": full_betas,
        "beta_ratio": beta_ratio,
        "n_tail_obs": n_tail,
        "var_threshold": float(var_threshold),
    }
