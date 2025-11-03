"""Concentration and diversification analysis.

Provides measures of portfolio diversification including effective
number of bets, entropy-based diversification, tail diversification,
marginal tail risk contributions, diversification ratio, and the
Herfindahl index.
"""

import numpy as np


def effective_number_of_bets(weights, covariance_matrix):
    """Compute eigenvalue-based Effective Number of Bets (ENB).

    ENB = exp(entropy of PCA-explained variance). A portfolio of 10
    assets might have only 3 effective bets if they are correlated.

    Parameters
    ----------
    weights : array-like
        Portfolio weights.
    covariance_matrix : array-like
        Covariance matrix of asset returns.

    Returns
    -------
    float
        Effective number of bets.
    """
    cov = np.asarray(covariance_matrix, dtype=float)

    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = eigenvalues[eigenvalues > 1e-15]

    if len(eigenvalues) == 0:
        return 1.0

    # Proportions of explained variance
    total = eigenvalues.sum()
    props = eigenvalues / total

    # Shannon entropy
    entropy = -np.sum(props * np.log(props))

    return float(np.exp(entropy))


def entropy_diversification(weights):
    """Compute information-theoretic diversification measure.

    Shannon entropy of the weight distribution, normalised to [0, 1].
    Uniform weights yield maximum diversification (1.0); a single
    concentrated position yields minimum (0.0).

    Parameters
    ----------
    weights : array-like
        Portfolio weights (must be non-negative).

    Returns
    -------
    float
        Normalised entropy in [0, 1].
    """
    w = np.asarray(weights, dtype=float)
    w = w[w > 1e-15]
    n = len(w)

    if n <= 1:
        return 0.0

    # Normalise
    w = w / w.sum()

    entropy = -np.sum(w * np.log(w))
    max_entropy = np.log(n)

    if max_entropy < 1e-15:
        return 0.0

    return float(entropy / max_entropy)


def tail_diversification(returns_df, weights, alpha=0.05):
    """Compare diversification in normal times versus the tail.

    Parameters
    ----------
    returns_df : pandas.DataFrame
        DataFrame of asset returns, one column per asset.
    weights : array-like
        Portfolio weights.
    alpha : float, optional
        Tail percentile (default 0.05).

    Returns
    -------
    dict
        Dictionary with 'normal_diversification',
        'tail_diversification', and 'tail_concentration_ratio'.
    """
    w = np.asarray(weights, dtype=float)
    returns = returns_df.values
    port_returns = returns @ w

    # Normal times: full sample correlation-based diversification
    individual_vol = np.sqrt(np.sum(w ** 2 * np.var(returns, axis=0)))
    portfolio_vol = np.std(port_returns)
    normal_div = float(individual_vol / portfolio_vol) if portfolio_vol > 1e-15 else 1.0

    # Tail: worst alpha percentile
    threshold = np.percentile(port_returns, alpha * 100)
    tail_mask = port_returns <= threshold
    if tail_mask.sum() < 2:
        tail_div = normal_div
    else:
        tail_returns = returns[tail_mask]
        tail_individual_vol = np.sqrt(np.sum(w ** 2 * np.var(tail_returns, axis=0)))
        tail_portfolio_vol = np.std(port_returns[tail_mask])
        tail_div = float(tail_individual_vol / tail_portfolio_vol) if tail_portfolio_vol > 1e-15 else 1.0

    ratio = float(tail_div / normal_div) if normal_div > 1e-15 else 1.0

    return {
        "normal_diversification": normal_div,
        "tail_diversification": tail_div,
        "tail_concentration_ratio": ratio,
    }


def marginal_tail_risk_contribution(returns_df, weights, alpha=0.05):
    """Decompose portfolio CVaR into per-asset contributions.

    Parameters
    ----------
    returns_df : pandas.DataFrame
        DataFrame of asset returns, one column per asset.
    weights : array-like
        Portfolio weights.
    alpha : float, optional
        Tail percentile (default 0.05).

    Returns
    -------
    dict
        Mapping of asset names to their marginal CVaR contribution.
    """
    w = np.asarray(weights, dtype=float)
    returns = returns_df.values
    port_returns = returns @ w

    threshold = np.percentile(port_returns, alpha * 100)
    tail_mask = port_returns <= threshold

    if tail_mask.sum() == 0:
        return {col: 0.0 for col in returns_df.columns}

    tail_returns = returns[tail_mask]
    # Mean return of each asset in the tail
    mean_tail = tail_returns.mean(axis=0)

    # Marginal contribution = weight * mean tail return of that asset
    contributions = w * mean_tail
    result = {}
    for i, col in enumerate(returns_df.columns):
        result[col] = float(contributions[i])

    return result


def diversification_ratio(weights, volatilities, covariance_matrix):
    """Compute the classic diversification ratio.

    DR = weighted average volatility / portfolio volatility. A ratio
    greater than 1 means diversification is reducing risk.

    Parameters
    ----------
    weights : array-like
        Portfolio weights.
    volatilities : array-like
        Individual asset volatilities (standard deviations).
    covariance_matrix : array-like
        Covariance matrix of asset returns.

    Returns
    -------
    float
        Diversification ratio.
    """
    w = np.asarray(weights, dtype=float)
    vol = np.asarray(volatilities, dtype=float)
    cov = np.asarray(covariance_matrix, dtype=float)

    weighted_avg_vol = float(np.dot(w, vol))
    port_var = float(w @ cov @ w)
    port_vol = np.sqrt(port_var)

    if port_vol < 1e-15:
        return 1.0

    return float(weighted_avg_vol / port_vol)


def herfindahl_index(weights):
    """Compute the Herfindahl-Hirschman Index (concentration measure).

    Sum of squared weights. Equals 1/n for equal-weighted portfolios
    and 1.0 for a single-asset portfolio.

    Parameters
    ----------
    weights : array-like
        Portfolio weights.

    Returns
    -------
    float
        Herfindahl index.
    """
    w = np.asarray(weights, dtype=float)
    return float(np.sum(w ** 2))
