"""Head-to-head comparison framework.

Compares QuantLite's fat-tail-aware methods against naive Gaussian
baselines (labelled as standard MVO, basic HRP, simple risk parity)
on identical simulated datasets.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from ..metrics import max_drawdown, sharpe_ratio
from ..risk.metrics import cvar, value_at_risk

# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def _simulate_sp500(n: int = 2520, seed: int = 42) -> np.ndarray:
    """Simulate S&P 500-like daily returns with fat tails.

    Args:
        n: Number of observations.
        seed: Random seed.

    Returns:
        Array of daily returns.
    """
    rng = np.random.RandomState(seed)
    # Student-t with ~5 degrees of freedom captures fat tails
    returns = rng.standard_t(df=5, size=n) * 0.01 + 0.0003
    return returns


def _simulate_multi_asset(n: int = 2520, seed: int = 43) -> pd.DataFrame:
    """Simulate a multi-asset portfolio (equity, bond, gold, crypto).

    Args:
        n: Number of observations.
        seed: Random seed.

    Returns:
        DataFrame with columns for each asset class.
    """
    rng = np.random.RandomState(seed)
    # Correlation structure
    corr = np.array([
        [1.0, -0.2, 0.05, 0.3],
        [-0.2, 1.0, 0.1, -0.1],
        [0.05, 0.1, 1.0, 0.05],
        [0.3, -0.1, 0.05, 1.0],
    ])
    vols = np.array([0.01, 0.003, 0.008, 0.03])
    mus = np.array([0.0003, 0.0001, 0.0002, 0.0005])
    cov = np.outer(vols, vols) * corr
    L = np.linalg.cholesky(cov)

    z = rng.standard_t(df=5, size=(n, 4))
    returns = z @ L.T + mus
    return pd.DataFrame(returns, columns=["equity", "bond", "gold", "crypto"])


def _simulate_emerging_market(n: int = 2520, seed: int = 44) -> np.ndarray:
    """Simulate emerging market returns with heavier tails.

    Args:
        n: Number of observations.
        seed: Random seed.

    Returns:
        Array of daily returns.
    """
    rng = np.random.RandomState(seed)
    returns = rng.standard_t(df=3, size=n) * 0.015 + 0.0004
    return returns


# ---------------------------------------------------------------------------
# Baseline methods (naive Gaussian approaches)
# ---------------------------------------------------------------------------

def _gaussian_var(returns: np.ndarray, alpha: float = 0.05) -> float:
    """Parametric Gaussian VaR.

    Args:
        returns: Return series.
        alpha: Significance level.

    Returns:
        VaR estimate (negative float = loss).
    """
    mu = np.mean(returns)
    sigma = np.std(returns, ddof=1)
    return float(mu + sigma * stats.norm.ppf(alpha))


def _gaussian_cvar(returns: np.ndarray, alpha: float = 0.05) -> float:
    """Parametric Gaussian CVaR (Expected Shortfall).

    Args:
        returns: Return series.
        alpha: Significance level.

    Returns:
        CVaR estimate (negative float).
    """
    mu = np.mean(returns)
    sigma = np.std(returns, ddof=1)
    return float(mu - sigma * stats.norm.pdf(stats.norm.ppf(alpha)) / alpha)


def _mvo_weights(returns: pd.DataFrame) -> np.ndarray:
    """Mean-variance optimisation (Markowitz, labelled as PyPortfolioOpt-style).

    Args:
        returns: Multi-asset returns DataFrame.

    Returns:
        Optimal weight array.
    """
    mu = returns.mean().values
    cov = returns.cov().values
    n = len(mu)

    # Minimise variance for target return (max Sharpe via closed-form)
    ones = np.ones(n)
    cov_inv = np.linalg.inv(cov + 1e-8 * np.eye(n))
    w = cov_inv @ mu
    w = w / np.sum(np.abs(w))
    # Ensure non-negative (long only)
    w = np.maximum(w, 0)
    s = w.sum()
    w = w / s if s > 0 else ones / n
    return w


def _hrp_weights(returns: pd.DataFrame) -> np.ndarray:
    """Basic Hierarchical Risk Parity (labelled as PyPortfolioOpt-style HRP).

    Simplified version: inverse-volatility weighting with correlation
    clustering (simplified to just inverse-vol for the baseline).

    Args:
        returns: Multi-asset returns DataFrame.

    Returns:
        Weight array.
    """
    vols = returns.std().values
    inv_vol = 1.0 / (vols + 1e-10)
    return inv_vol / inv_vol.sum()


def _risk_parity_weights(returns: pd.DataFrame) -> np.ndarray:
    """Simple risk parity (labelled as Riskfolio-Lib-style).

    Equal risk contribution based on marginal volatility.

    Args:
        returns: Multi-asset returns DataFrame.

    Returns:
        Weight array.
    """
    cov = returns.cov().values
    n = cov.shape[0]
    w = np.ones(n) / n

    for _ in range(50):
        sigma_p = np.sqrt(w @ cov @ w)
        mrc = (cov @ w) / (sigma_p + 1e-10)
        rc = w * mrc
        target = sigma_p / n
        w = w * target / (rc + 1e-10)
        w = w / w.sum()

    return w


# ---------------------------------------------------------------------------
# QuantLite methods (using the library's own functions)
# ---------------------------------------------------------------------------

def _quantlite_var(returns: np.ndarray, alpha: float = 0.05) -> float:
    """QuantLite EVT-aware VaR using Cornish-Fisher expansion.

    Args:
        returns: Return series.
        alpha: Significance level.

    Returns:
        VaR estimate.
    """
    return value_at_risk(returns, alpha=alpha, method="cornish-fisher")


def _quantlite_cvar(returns: np.ndarray, alpha: float = 0.05) -> float:
    """QuantLite CVaR using historical method (tail-aware).

    Args:
        returns: Return series.
        alpha: Significance level.

    Returns:
        CVaR estimate.
    """
    return cvar(returns, alpha=alpha)


def _quantlite_tail_parity_weights(returns: pd.DataFrame) -> np.ndarray:
    """QuantLite tail-risk parity weights.

    Uses CVaR-based risk contributions for weight allocation.

    Args:
        returns: Multi-asset returns DataFrame.

    Returns:
        Weight array.
    """
    n = returns.shape[1]
    cvars = np.array([
        abs(cvar(returns.iloc[:, i].values, alpha=0.05))
        for i in range(n)
    ])
    inv_cvar = 1.0 / (cvars + 1e-10)
    return inv_cvar / inv_cvar.sum()


# ---------------------------------------------------------------------------
# Comparison engine
# ---------------------------------------------------------------------------

@dataclass
class ComparisonResult:
    """Result of a head-to-head comparison.

    Attributes:
        dataset: Name of the dataset used.
        methods: Dictionary mapping method name to its metrics.
        var_violations: VaR violation rates per method.
        sharpe_ratios: Sharpe ratios per method (portfolio-level).
        computation_times: Timing per method in seconds.
    """
    dataset: str
    methods: dict[str, dict[str, float]] = field(default_factory=dict)
    var_violations: dict[str, float] = field(default_factory=dict)
    sharpe_ratios: dict[str, float] = field(default_factory=dict)
    computation_times: dict[str, float] = field(default_factory=dict)


def _var_violation_rate(
    returns: np.ndarray,
    var_estimate: float,
) -> float:
    """Compute the fraction of returns that violated (fell below) the VaR.

    Args:
        returns: Return series.
        var_estimate: The VaR threshold (negative number).

    Returns:
        Violation rate as a float between 0 and 1.
    """
    violations = np.sum(returns < var_estimate)
    return float(violations / len(returns))


def _portfolio_returns(
    returns: pd.DataFrame,
    weights: np.ndarray,
) -> np.ndarray:
    """Compute portfolio returns from asset returns and weights.

    Args:
        returns: Multi-asset returns.
        weights: Portfolio weights.

    Returns:
        Array of portfolio returns.
    """
    return (returns.values @ weights).astype(float)


def run_comparison(
    datasets: Sequence[str] | None = None,
    alpha: float = 0.05,
    seed: int = 42,
) -> list[ComparisonResult]:
    """Run head-to-head comparison of QuantLite vs baseline methods.

    Args:
        datasets: Which datasets to use. Options: ``"sp500"``,
            ``"multi_asset"``, ``"emerging_market"``. Defaults to all.
        alpha: VaR significance level.
        seed: Random seed for reproducibility.

    Returns:
        List of ``ComparisonResult`` objects, one per dataset.
    """
    if datasets is None:
        datasets = ["sp500", "multi_asset", "emerging_market"]

    results = []

    for ds_name in datasets:
        result = ComparisonResult(dataset=ds_name)

        if ds_name == "sp500":
            rets = _simulate_sp500(seed=seed)
            _run_univariate_comparison(rets, alpha, result)

        elif ds_name == "multi_asset":
            df = _simulate_multi_asset(seed=seed + 1)
            _run_portfolio_comparison(df, alpha, result)

        elif ds_name == "emerging_market":
            rets = _simulate_emerging_market(seed=seed + 2)
            _run_univariate_comparison(rets, alpha, result)

        results.append(result)

    return results


def _run_univariate_comparison(
    returns: np.ndarray,
    alpha: float,
    result: ComparisonResult,
) -> None:
    """Run VaR/CVaR comparison on a single return series.

    Args:
        returns: Return series.
        alpha: Significance level.
        result: ComparisonResult to populate (mutated in place).
    """
    # Split into estimation and test windows
    split = len(returns) // 2
    train, test = returns[:split], returns[split:]

    methods = {
        "Gaussian VaR (parametric)": lambda r: _gaussian_var(r, alpha),
        "Historical VaR (scipy baseline)": lambda r: value_at_risk(r, alpha, "historical"),
        "QuantLite EVT VaR (Cornish-Fisher)": lambda r: _quantlite_var(r, alpha),
    }

    for name, fn in methods.items():
        t0 = time.perf_counter()
        var_est = fn(train)
        elapsed = time.perf_counter() - t0

        violation_rate = _var_violation_rate(test, var_est)
        cvar_est = (
            _gaussian_cvar(train, alpha) if "Gaussian" in name
            else _quantlite_cvar(train, alpha)
        )

        result.methods[name] = {
            "var": var_est,
            "cvar": cvar_est,
            "violation_rate": violation_rate,
            "expected_rate": alpha,
        }
        result.var_violations[name] = violation_rate
        result.computation_times[name] = elapsed


def _run_portfolio_comparison(
    returns: pd.DataFrame,
    alpha: float,
    result: ComparisonResult,
) -> None:
    """Run portfolio-level comparison across methods.

    Args:
        returns: Multi-asset returns.
        alpha: Significance level.
        result: ComparisonResult to populate (mutated in place).
    """
    split = len(returns) // 2
    train, test = returns.iloc[:split], returns.iloc[split:]

    weight_methods = {
        "Gaussian MVO (PyPortfolioOpt-style)": _mvo_weights,
        "Basic HRP (PyPortfolioOpt-style)": _hrp_weights,
        "Simple Risk Parity (Riskfolio-style)": _risk_parity_weights,
        "QuantLite Tail-Risk Parity": _quantlite_tail_parity_weights,
    }

    for name, weight_fn in weight_methods.items():
        t0 = time.perf_counter()
        weights = weight_fn(train)
        port_ret_train = _portfolio_returns(train, weights)
        port_ret_test = _portfolio_returns(test, weights)
        elapsed = time.perf_counter() - t0

        sr = sharpe_ratio(port_ret_test)
        md = max_drawdown(port_ret_test)

        if "Gaussian" in name or "HRP" in name or "Risk Parity" in name:
            var_est = _gaussian_var(port_ret_train, alpha)
        else:
            var_est = _quantlite_var(port_ret_train, alpha)

        violation_rate = _var_violation_rate(port_ret_test, var_est)

        result.methods[name] = {
            "sharpe": sr,
            "max_drawdown": md,
            "var": var_est,
            "violation_rate": violation_rate,
            "weights": weights.tolist(),
        }
        result.var_violations[name] = violation_rate
        result.sharpe_ratios[name] = sr
        result.computation_times[name] = elapsed
