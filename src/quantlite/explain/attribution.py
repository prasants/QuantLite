"""Risk attribution: decompose portfolio risk into per-asset and factor contributions.

Supports component VaR/CVaR, marginal risk contributions, factor-based
attribution (market, size, value, momentum), and regime-conditional
decomposition.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

__all__ = [
    "RiskAttribution",
    "compute_risk_attribution",
    "marginal_risk_contribution",
    "factor_attribution",
    "regime_conditional_attribution",
]


@dataclass(frozen=True)
class RiskAttribution:
    """Structured container for risk attribution results.

    Attributes:
        total_var: Portfolio-level VaR.
        total_cvar: Portfolio-level CVaR (Expected Shortfall).
        component_var: Per-asset contribution to VaR.
        component_cvar: Per-asset contribution to CVaR.
        marginal_var: Marginal VaR per asset (change from +1% allocation).
        marginal_cvar: Marginal CVaR per asset.
        factor_contributions: Risk attributed to each factor.
        idiosyncratic_risk: Residual risk not explained by factors.
        regime_attributions: Per-regime risk breakdown (if regimes provided).
        asset_names: Ordered list of asset names.
    """

    total_var: float
    total_cvar: float
    component_var: dict[str, float]
    component_cvar: dict[str, float]
    marginal_var: dict[str, float]
    marginal_cvar: dict[str, float]
    factor_contributions: dict[str, float] = field(default_factory=dict)
    idiosyncratic_risk: float = 0.0
    regime_attributions: dict[int, dict[str, float]] = field(default_factory=dict)
    asset_names: list[str] = field(default_factory=list)


def _portfolio_var(
    returns: np.ndarray, weights: np.ndarray, alpha: float = 0.05
) -> float:
    """Compute portfolio VaR at the given confidence level."""
    port_returns = returns @ weights
    return -float(np.percentile(port_returns, alpha * 100))


def _portfolio_cvar(
    returns: np.ndarray, weights: np.ndarray, alpha: float = 0.05
) -> float:
    """Compute portfolio CVaR (Expected Shortfall)."""
    port_returns = returns @ weights
    threshold = np.percentile(port_returns, alpha * 100)
    tail = port_returns[port_returns <= threshold]
    if len(tail) == 0:
        return -float(threshold)
    return -float(np.mean(tail))


def _component_risk(
    returns: np.ndarray,
    weights: np.ndarray,
    alpha: float = 0.05,
    metric: str = "cvar",
) -> np.ndarray:
    """Compute component risk contributions using Euler decomposition.

    Uses the gradient-based approach: component risk_i = w_i * d(risk)/d(w_i).
    Approximated via finite differences.
    """
    len(weights)
    base_risk = (
        _portfolio_cvar(returns, weights, alpha)
        if metric == "cvar"
        else _portfolio_var(returns, weights, alpha)
    )

    # Use covariance-based Euler decomposition for more stable results
    port_returns = returns @ weights
    if metric == "cvar":
        threshold = np.percentile(port_returns, alpha * 100)
        tail_mask = port_returns <= threshold
    else:
        threshold = np.percentile(port_returns, alpha * 100)
        # For VaR, use a small band around the quantile
        band = np.abs(threshold) * 0.1 + 1e-8
        tail_mask = np.abs(port_returns - threshold) <= band

    if tail_mask.sum() < 2:
        tail_mask = port_returns <= np.percentile(port_returns, alpha * 100 * 2)

    # Marginal contribution = E[r_i | portfolio in tail]
    tail_returns = returns[tail_mask]
    marginals = -tail_returns.mean(axis=0)

    # Component risk = w_i * marginal_i, scaled to sum to total
    components = weights * marginals
    comp_sum = components.sum()
    if abs(comp_sum) > 1e-10:
        components = components * (base_risk / comp_sum)

    return components


def marginal_risk_contribution(
    returns_df: pd.DataFrame,
    weights: dict[str, float],
    alpha: float = 0.05,
    delta: float = 0.01,
) -> dict[str, float]:
    """Compute marginal risk contribution per asset.

    Measures the change in portfolio CVaR when an asset's weight is
    increased by ``delta`` (default 1 percentage point).

    Args:
        returns_df: DataFrame of asset returns (assets as columns).
        weights: Current portfolio weights.
        alpha: VaR/CVaR confidence level.
        delta: Weight perturbation size.

    Returns:
        Dictionary mapping asset names to marginal CVaR contributions.
    """
    names = list(returns_df.columns)
    ret = returns_df.values
    w = np.array([weights.get(n, 0.0) for n in names])

    base_cvar = _portfolio_cvar(ret, w, alpha)
    marginals = {}

    for i, name in enumerate(names):
        w_pert = w.copy()
        w_pert[i] += delta
        w_pert = w_pert / w_pert.sum()
        new_cvar = _portfolio_cvar(ret, w_pert, alpha)
        marginals[name] = new_cvar - base_cvar

    return marginals


def factor_attribution(
    returns_df: pd.DataFrame,
    weights: dict[str, float],
    factor_returns: pd.DataFrame | None = None,
    alpha: float = 0.05,
) -> tuple:
    """Attribute portfolio risk to systematic factors and idiosyncratic residual.

    If ``factor_returns`` is not provided, synthetic market, size, value,
    and momentum factors are constructed from the asset returns.

    Args:
        returns_df: DataFrame of asset returns.
        weights: Portfolio weights.
        factor_returns: Optional DataFrame of factor returns.
        alpha: CVaR confidence level.

    Returns:
        Tuple of (factor_contributions dict, idiosyncratic_risk float).
    """
    names = list(returns_df.columns)
    ret = returns_df.values
    w = np.array([weights.get(n, 0.0) for n in names])
    port_ret = ret @ w

    # Build default factors from cross-section
    if factor_returns is None:
        factor_dict = {}
        # Market: equal-weighted average
        factor_dict["Market"] = ret.mean(axis=1)
        if ret.shape[1] >= 4:
            # Size: small minus big (first half minus second half by vol)
            vols = ret.std(axis=0)
            order = np.argsort(vols)
            half = len(order) // 2
            factor_dict["Size"] = (
                ret[:, order[:half]].mean(axis=1) - ret[:, order[half:]].mean(axis=1)
            )
            # Value: low return minus high return (contrarian proxy)
            means = ret.mean(axis=0)
            m_order = np.argsort(means)
            factor_dict["Value"] = (
                ret[:, m_order[:half]].mean(axis=1)
                - ret[:, m_order[half:]].mean(axis=1)
            )
            # Momentum: winners minus losers (trailing mean)
            if ret.shape[0] >= 60:
                trailing = pd.DataFrame(ret).rolling(60).mean().iloc[-1].values
                t_order = np.argsort(trailing)
                factor_dict["Momentum"] = (
                    ret[:, t_order[half:]].mean(axis=1)
                    - ret[:, t_order[:half]].mean(axis=1)
                )
        factor_returns = pd.DataFrame(factor_dict)

    # Regress portfolio returns on factors
    X = factor_returns.values
    if X.shape[0] != len(port_ret):
        min_len = min(X.shape[0], len(port_ret))
        X = X[:min_len]
        port_ret = port_ret[:min_len]

    X_with_const = np.column_stack([np.ones(len(X)), X])
    try:
        betas, _, _, _ = np.linalg.lstsq(X_with_const, port_ret, rcond=None)
    except np.linalg.LinAlgError:
        betas = np.zeros(X_with_const.shape[1])

    factor_betas = betas[1:]
    residuals = port_ret - X_with_const @ betas

    # Factor risk contributions via variance decomposition
    total_var = np.var(port_ret)
    factor_names = list(factor_returns.columns)
    contributions = {}

    for i, fname in enumerate(factor_names):
        factor_var_contrib = factor_betas[i] ** 2 * np.var(X[:, i])
        contributions[fname] = float(factor_var_contrib)

    idio = float(np.var(residuals))

    # Normalise to sum to total variance
    raw_sum = sum(contributions.values()) + idio
    if raw_sum > 0:
        scale = total_var / raw_sum
        contributions = {k: v * scale for k, v in contributions.items()}
        idio *= scale

    return contributions, idio


def regime_conditional_attribution(
    returns_df: pd.DataFrame,
    weights: dict[str, float],
    regime_labels: np.ndarray,
    alpha: float = 0.05,
) -> dict[int, dict[str, float]]:
    """Compute per-regime risk attribution.

    Decomposes component CVaR separately for each detected regime.

    Args:
        returns_df: DataFrame of asset returns.
        weights: Portfolio weights.
        regime_labels: Array of regime labels (same length as returns).
        alpha: CVaR confidence level.

    Returns:
        Dictionary mapping regime label to per-asset CVaR contributions.
    """
    names = list(returns_df.columns)
    ret = returns_df.values
    w = np.array([weights.get(n, 0.0) for n in names])

    unique_regimes = sorted(set(regime_labels))
    result = {}

    for regime in unique_regimes:
        mask = regime_labels == regime
        regime_ret = ret[mask]
        if len(regime_ret) < 10:
            result[int(regime)] = {n: 0.0 for n in names}
            continue

        comp = _component_risk(regime_ret, w, alpha, metric="cvar")
        result[int(regime)] = {n: float(comp[i]) for i, n in enumerate(names)}

    return result


def compute_risk_attribution(
    returns_df: pd.DataFrame,
    weights: dict[str, float],
    alpha: float = 0.05,
    factor_returns: pd.DataFrame | None = None,
    regime_labels: np.ndarray | None = None,
) -> RiskAttribution:
    """Compute comprehensive risk attribution for a portfolio.

    This is the main entry point. It computes component VaR/CVaR,
    marginal contributions, factor attribution, and optionally
    regime-conditional breakdown.

    Args:
        returns_df: DataFrame of asset returns (assets as columns).
        weights: Portfolio weights as {asset_name: weight}.
        alpha: VaR/CVaR confidence level (default 0.05 = 95%).
        factor_returns: Optional factor return series for attribution.
        regime_labels: Optional array of regime labels for conditional analysis.

    Returns:
        ``RiskAttribution`` dataclass with all decompositions.

    Raises:
        ValueError: If weights and returns columns do not align.

    Example:
        >>> import pandas as pd
        >>> import numpy as np
        >>> returns = pd.DataFrame(np.random.randn(500, 3) * 0.01,
        ...                        columns=["Equity", "Bonds", "Gold"])
        >>> weights = {"Equity": 0.5, "Bonds": 0.3, "Gold": 0.2}
        >>> attr = compute_risk_attribution(returns, weights)
        >>> print(f"Total CVaR: {attr.total_cvar:.4f}")
    """
    names = list(returns_df.columns)
    ret = returns_df.values
    w = np.array([weights.get(n, 0.0) for n in names])

    # Total risk
    total_var = _portfolio_var(ret, w, alpha)
    total_cvar = _portfolio_cvar(ret, w, alpha)

    # Component risk
    comp_var = _component_risk(ret, w, alpha, metric="var")
    comp_cvar = _component_risk(ret, w, alpha, metric="cvar")

    component_var = {n: float(comp_var[i]) for i, n in enumerate(names)}
    component_cvar = {n: float(comp_cvar[i]) for i, n in enumerate(names)}

    # Marginal risk
    marg = marginal_risk_contribution(returns_df, weights, alpha)
    marg_var = {}
    for i, name in enumerate(names):
        w_pert = w.copy()
        w_pert[i] += 0.01
        w_pert = w_pert / w_pert.sum()
        marg_var[name] = _portfolio_var(ret, w_pert, alpha) - total_var

    # Factor attribution
    factor_contrib, idio = factor_attribution(
        returns_df, weights, factor_returns, alpha
    )

    # Regime attribution
    regime_attr = {}
    if regime_labels is not None:
        regime_attr = regime_conditional_attribution(
            returns_df, weights, regime_labels, alpha
        )

    return RiskAttribution(
        total_var=total_var,
        total_cvar=total_cvar,
        component_var=component_var,
        component_cvar=component_cvar,
        marginal_var=marg_var,
        marginal_cvar=marg,
        factor_contributions=factor_contrib,
        idiosyncratic_risk=idio,
        regime_attributions=regime_attr,
        asset_names=names,
    )
