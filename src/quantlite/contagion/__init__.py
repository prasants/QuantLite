"""Contagion metrics for systemic risk analysis.

Provides tools for measuring how financial distress propagates between
assets and across systems: CoVaR, delta CoVaR, marginal expected
shortfall, Granger causality testing, and causal network construction.
"""

import numpy as np
from scipy import stats


def covar(returns_a, returns_b, alpha=0.05, method="quantile"):
    """Compute CoVaR: VaR of asset B conditional on asset A being at its VaR.

    Parameters
    ----------
    returns_a : array-like
        Return series for the conditioning asset.
    returns_b : array-like
        Return series for the target asset.
    alpha : float, optional
        Significance level for VaR (default 0.05).
    method : str, optional
        Estimation method: "quantile" for conditional quantile,
        "regression" for OLS-based approximation (default "quantile").

    Returns
    -------
    dict
        Dictionary with keys: covar, var_a, var_b, delta_covar.
    """
    a = np.asarray(returns_a, dtype=float)
    b = np.asarray(returns_b, dtype=float)

    var_a = float(np.percentile(a, alpha * 100))
    var_b = float(np.percentile(b, alpha * 100))

    if method == "quantile":
        # Conditional: B returns when A <= VaR(A)
        mask = a <= var_a
        if mask.sum() < 2:
            # Fallback: use closest points
            idx = np.argsort(a)[:max(2, int(len(a) * alpha * 2))]
            conditional_b = b[idx]
        else:
            conditional_b = b[mask]
        covar_value = float(np.percentile(conditional_b, alpha * 100))

        # CoVaR at median for delta
        mask_median = np.abs(a - np.median(a)) <= np.std(a) * 0.5
        if mask_median.sum() < 2:
            covar_median = var_b
        else:
            covar_median = float(np.percentile(b[mask_median], alpha * 100))

    elif method == "regression":
        # OLS approximation of quantile regression
        X = np.column_stack([np.ones(len(a)), a])
        beta = np.linalg.lstsq(X, b, rcond=None)[0]
        covar_value = float(beta[0] + beta[1] * var_a)

        # At median
        covar_median = float(beta[0] + beta[1] * np.median(a))
    else:
        raise ValueError(f"Unknown method: {method}. Use 'quantile' or 'regression'.")

    delta = covar_value - covar_median

    return {
        "covar": covar_value,
        "var_a": var_a,
        "var_b": var_b,
        "delta_covar": delta,
    }


def delta_covar(returns_a, returns_b, alpha=0.05):
    """Compute the marginal contribution to systemic risk.

    Difference between CoVaR when asset A is in distress and CoVaR when
    asset A is at its median. A positive value indicates that A contributes
    to B's tail risk.

    Parameters
    ----------
    returns_a : array-like
        Return series for the conditioning asset.
    returns_b : array-like
        Return series for the target asset.
    alpha : float, optional
        Significance level (default 0.05).

    Returns
    -------
    float
        Delta CoVaR value.
    """
    result = covar(returns_a, returns_b, alpha=alpha, method="quantile")
    return result["delta_covar"]


def marginal_expected_shortfall(returns_system, returns_asset, alpha=0.05):
    """Compute Marginal Expected Shortfall (MES).

    Average return of the asset on days when the system is in its worst
    alpha-percentile. Shows each asset's contribution to system-wide
    tail risk.

    Parameters
    ----------
    returns_system : array-like
        System-level return series (e.g., equal-weighted portfolio).
    returns_asset : array-like
        Individual asset return series.
    alpha : float, optional
        Tail percentile (default 0.05).

    Returns
    -------
    float
        MES value (typically negative; more negative means higher
        contribution to systemic risk).
    """
    sys_r = np.asarray(returns_system, dtype=float)
    asset_r = np.asarray(returns_asset, dtype=float)

    threshold = np.percentile(sys_r, alpha * 100)
    mask = sys_r <= threshold
    if mask.sum() == 0:
        return 0.0
    return float(np.mean(asset_r[mask]))


def systemic_risk_contributions(returns_df, alpha=0.05):
    """Compute MES for every asset relative to an equal-weighted system portfolio.

    Parameters
    ----------
    returns_df : pandas.DataFrame
        DataFrame of asset returns, one column per asset.
    alpha : float, optional
        Tail percentile (default 0.05).

    Returns
    -------
    dict
        Mapping of asset names to their MES values, sorted by
        contribution (most negative first).
    """
    system_returns = returns_df.mean(axis=1).values
    contributions = {}
    for col in returns_df.columns:
        mes = marginal_expected_shortfall(system_returns, returns_df[col].values, alpha)
        contributions[col] = mes

    # Sort by MES (most negative = highest contribution)
    sorted_items = sorted(contributions.items(), key=lambda x: x[1])
    return dict(sorted_items)


def granger_causality(returns_a, returns_b, max_lag=5):
    """Test Granger causality in both directions between two return series.

    Uses OLS regression to test whether lagged values of A predict B
    beyond B's own lags, and vice versa.

    Parameters
    ----------
    returns_a : array-like
        First return series.
    returns_b : array-like
        Second return series.
    max_lag : int, optional
        Maximum number of lags to test (default 5).

    Returns
    -------
    dict
        Dictionary with keys 'a_to_b' and 'b_to_a', each containing
        f_statistic, p_value, direction, and optimal_lag.
    """
    a = np.asarray(returns_a, dtype=float)
    b = np.asarray(returns_b, dtype=float)

    def _test_direction(x, y, max_lag):
        """Test if x Granger-causes y."""
        best_result = None
        best_aic = np.inf

        for lag in range(1, max_lag + 1):
            n = len(y) - lag
            if n < lag * 2 + 2:
                continue

            # Restricted model: y ~ lagged y only
            Y = y[lag:]
            X_r = np.column_stack([y[lag - i - 1:n + lag - i - 1] for i in range(lag)])
            X_r = np.column_stack([np.ones(n), X_r])

            # Unrestricted model: y ~ lagged y + lagged x
            X_u = np.column_stack([
                X_r,
                *[x[lag - i - 1:n + lag - i - 1].reshape(-1, 1) for i in range(lag)]
            ])

            # OLS for restricted
            beta_r = np.linalg.lstsq(X_r, Y, rcond=None)[0]
            resid_r = Y - X_r @ beta_r
            ssr_r = float(np.sum(resid_r ** 2))

            # OLS for unrestricted
            beta_u = np.linalg.lstsq(X_u, Y, rcond=None)[0]
            resid_u = Y - X_u @ beta_u
            ssr_u = float(np.sum(resid_u ** 2))

            # F-test
            df1 = lag
            df2 = n - X_u.shape[1]
            if df2 <= 0 or ssr_u <= 0:
                continue

            f_stat = ((ssr_r - ssr_u) / df1) / (ssr_u / df2)
            p_val = 1.0 - stats.f.cdf(f_stat, df1, df2)

            # AIC for model selection
            aic = n * np.log(ssr_u / n) + 2 * X_u.shape[1]

            if aic < best_aic:
                best_aic = aic
                best_result = {
                    "f_statistic": float(f_stat),
                    "p_value": float(p_val),
                    "optimal_lag": lag,
                }

        if best_result is None:
            best_result = {
                "f_statistic": 0.0,
                "p_value": 1.0,
                "optimal_lag": 1,
            }
        return best_result

    a_to_b = _test_direction(a, b, max_lag)
    a_to_b["direction"] = "a_to_b"

    b_to_a = _test_direction(b, a, max_lag)
    b_to_a["direction"] = "b_to_a"

    return {"a_to_b": a_to_b, "b_to_a": b_to_a}


def causal_network(returns_df, max_lag=5, significance=0.05):
    """Build a directed causal graph from pairwise Granger causality tests.

    Parameters
    ----------
    returns_df : pandas.DataFrame
        DataFrame of asset returns, one column per asset.
    max_lag : int, optional
        Maximum number of lags to test (default 5).
    significance : float, optional
        P-value threshold for including an edge (default 0.05).

    Returns
    -------
    dict
        Dictionary with 'edges' (list of (source, target, p_value, lag)
        tuples), 'adjacency_matrix' (numpy array), and 'nodes' (list
        of column names).
    """
    cols = list(returns_df.columns)
    n = len(cols)
    adj = np.zeros((n, n))
    edges = []

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            result = granger_causality(
                returns_df.iloc[:, i].values,
                returns_df.iloc[:, j].values,
                max_lag=max_lag,
            )
            r = result["a_to_b"]
            if r["p_value"] < significance:
                adj[i, j] = 1.0
                edges.append((cols[i], cols[j], r["p_value"], r["optimal_lag"]))

    return {
        "edges": edges,
        "adjacency_matrix": adj,
        "nodes": cols,
    }
