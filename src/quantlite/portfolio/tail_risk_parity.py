"""Tail-risk parity allocation: CVaR parity, Expected Shortfall parity.

Goes beyond traditional risk parity (which equalises volatility
contributions) by equalising tail-risk contributions. This matters
because assets with similar volatility can have very different tail
behaviour, and a portfolio that looks balanced by volatility may be
dangerously concentrated in tail risk.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

__all__ = [
    "TailRiskParityResult",
    "cvar_parity_weights",
    "es_parity_weights",
    "vol_parity_weights",
    "compare_parity_methods",
    "regime_conditional_tail_parity",
]


@dataclass(frozen=True)
class TailRiskParityResult:
    """Result container for tail-risk parity allocation.

    Attributes:
        weights: Mapping of asset name to weight.
        method: Parity method used.
        risk_contributions: Per-asset risk contribution.
        total_risk: Total portfolio risk measure.
        risk_measure: Name of the risk measure (volatility, CVaR, ES).
    """

    weights: dict[str, float]
    method: str
    risk_contributions: dict[str, float]
    total_risk: float
    risk_measure: str


def _compute_cvar_contributions(
    returns_arr: np.ndarray,
    weights: np.ndarray,
    alpha: float = 0.05,
) -> tuple[np.ndarray, float]:
    """Compute per-asset CVaR contributions via component decomposition.

    Uses the Euler decomposition: the CVaR contribution of asset i
    equals w_i * E[r_i | r_p <= VaR_p].

    Args:
        returns_arr: Array of shape (T, n) with asset returns.
        weights: Weight vector of length n.
        alpha: Significance level for CVaR.

    Returns:
        Tuple of (contributions array, total portfolio CVaR).
    """
    port_returns = returns_arr @ weights
    var_threshold = np.percentile(port_returns, alpha * 100)
    tail_mask = port_returns <= var_threshold

    if tail_mask.sum() == 0:
        n = returns_arr.shape[1]
        return np.zeros(n), 0.0

    tail_returns = returns_arr[tail_mask]
    # Component CVaR: w_i * mean(r_i | tail event)
    mean_tail = tail_returns.mean(axis=0)
    contributions = weights * mean_tail
    total_cvar = float(port_returns[tail_mask].mean())

    return contributions, total_cvar


def _compute_vol_contributions(
    cov: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Compute per-asset volatility contributions.

    Args:
        cov: Covariance matrix.
        weights: Weight vector.

    Returns:
        Tuple of (contributions array, total portfolio volatility).
    """
    port_vol = float(np.sqrt(weights @ cov @ weights))
    if port_vol < 1e-12:
        return np.zeros(len(weights)), 0.0
    mrc = (cov @ weights) / port_vol
    contributions = weights * mrc
    return contributions, port_vol


def cvar_parity_weights(
    returns_df: pd.DataFrame,
    alpha: float = 0.05,
    risk_budget: dict[str, float] | None = None,
    max_iter: int = 1000,
) -> TailRiskParityResult:
    """Compute CVaR parity weights: equalise CVaR contributions.

    Each asset contributes equally to the portfolio's Conditional
    Value at Risk, unless a custom risk budget is specified.

    Args:
        returns_df: DataFrame of asset returns (assets as columns).
        alpha: Significance level for CVaR (e.g. 0.05 for 95%).
        risk_budget: Optional per-asset risk budget (must sum to 1).
            If None, equal risk budget is used.
        max_iter: Maximum optimisation iterations.

    Returns:
        ``TailRiskParityResult`` with CVaR parity allocations.
    """
    n = returns_df.shape[1]
    names = list(returns_df.columns)
    returns_arr = returns_df.values

    if risk_budget is not None:
        target_fracs = np.array([risk_budget[name] for name in names])
    else:
        target_fracs = np.ones(n) / n

    x0 = np.ones(n) / n

    def objective(w: np.ndarray) -> float:
        contributions, total = _compute_cvar_contributions(returns_arr, w, alpha)
        if abs(total) < 1e-12:
            return 0.0
        # Normalise contributions to fractions
        contrib_fracs = contributions / total
        return float(np.sum((contrib_fracs - target_fracs) ** 2))

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(1e-6, 1.0)] * n

    result = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-14, "maxiter": max_iter},
    )

    w = result.x
    w = w / w.sum()  # ensure exact sum to 1
    contributions, total = _compute_cvar_contributions(returns_arr, w, alpha)

    return TailRiskParityResult(
        weights=dict(zip(names, w.tolist())),
        method="cvar_parity",
        risk_contributions=dict(zip(names, contributions.tolist())),
        total_risk=total,
        risk_measure="CVaR",
    )


def es_parity_weights(
    returns_df: pd.DataFrame,
    alpha: float = 0.025,
    risk_budget: dict[str, float] | None = None,
    max_iter: int = 1000,
) -> TailRiskParityResult:
    """Compute Expected Shortfall parity weights.

    Functionally similar to CVaR parity but uses a deeper tail
    (default alpha=0.025 vs 0.05) for more extreme tail equalisation.

    Args:
        returns_df: DataFrame of asset returns (assets as columns).
        alpha: Significance level for ES (default 2.5%).
        risk_budget: Optional per-asset risk budget.
        max_iter: Maximum optimisation iterations.

    Returns:
        ``TailRiskParityResult`` with ES parity allocations.
    """
    result = cvar_parity_weights(
        returns_df, alpha=alpha, risk_budget=risk_budget, max_iter=max_iter
    )
    # Re-wrap with correct method name
    return TailRiskParityResult(
        weights=result.weights,
        method="es_parity",
        risk_contributions=result.risk_contributions,
        total_risk=result.total_risk,
        risk_measure="Expected Shortfall",
    )


def vol_parity_weights(
    returns_df: pd.DataFrame,
    risk_budget: dict[str, float] | None = None,
    max_iter: int = 1000,
) -> TailRiskParityResult:
    """Compute volatility parity weights (for comparison).

    Standard risk parity: equalise volatility contributions.

    Args:
        returns_df: DataFrame of asset returns (assets as columns).
        risk_budget: Optional per-asset risk budget.
        max_iter: Maximum optimisation iterations.

    Returns:
        ``TailRiskParityResult`` with volatility parity allocations.
    """
    n = returns_df.shape[1]
    names = list(returns_df.columns)
    cov = returns_df.cov().values

    if risk_budget is not None:
        target_fracs = np.array([risk_budget[name] for name in names])
    else:
        target_fracs = np.ones(n) / n

    x0 = np.ones(n) / n

    def objective(w: np.ndarray) -> float:
        contributions, total = _compute_vol_contributions(cov, w)
        if total < 1e-12:
            return 0.0
        contrib_fracs = contributions / total
        return float(np.sum((contrib_fracs - target_fracs) ** 2))

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(1e-6, 1.0)] * n

    result = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-15, "maxiter": max_iter},
    )

    w = result.x
    w = w / w.sum()
    contributions, total = _compute_vol_contributions(cov, w)

    return TailRiskParityResult(
        weights=dict(zip(names, w.tolist())),
        method="vol_parity",
        risk_contributions=dict(zip(names, contributions.tolist())),
        total_risk=total,
        risk_measure="Volatility",
    )


def compare_parity_methods(
    returns_df: pd.DataFrame,
    alpha: float = 0.05,
) -> dict[str, TailRiskParityResult]:
    """Compare vol parity, CVaR parity, and ES parity side by side.

    Args:
        returns_df: DataFrame of asset returns (assets as columns).
        alpha: Significance level for tail risk measures.

    Returns:
        Dictionary mapping method name to ``TailRiskParityResult``.
    """
    return {
        "vol_parity": vol_parity_weights(returns_df),
        "cvar_parity": cvar_parity_weights(returns_df, alpha=alpha),
        "es_parity": es_parity_weights(returns_df, alpha=alpha / 2),
    }


def regime_conditional_tail_parity(
    returns_df: pd.DataFrame,
    regime_labels: np.ndarray,
    alpha: float = 0.05,
    risk_budgets: dict[int, dict[str, float]] | None = None,
) -> dict[int, TailRiskParityResult]:
    """Compute CVaR parity weights conditional on detected regimes.

    Fits a separate CVaR parity allocation for each regime, allowing
    the portfolio to adapt its tail-risk budget to market conditions.

    Args:
        returns_df: DataFrame of asset returns (assets as columns).
        regime_labels: Array of integer regime labels (same length as
            returns_df).
        alpha: Significance level for CVaR.
        risk_budgets: Optional per-regime risk budgets. Keys are regime
            integers, values are per-asset budget dicts.

    Returns:
        Dictionary mapping regime label to ``TailRiskParityResult``.
    """
    labels = np.asarray(regime_labels)
    unique_regimes = sorted(set(labels.tolist()))
    results: dict[int, TailRiskParityResult] = {}

    for regime in unique_regimes:
        mask = labels == regime
        if mask.sum() < 10:
            # Not enough data for this regime; use equal weights
            names = list(returns_df.columns)
            n = len(names)
            results[regime] = TailRiskParityResult(
                weights=dict(zip(names, [1.0 / n] * n)),
                method=f"cvar_parity_regime_{regime}",
                risk_contributions=dict(zip(names, [0.0] * n)),
                total_risk=0.0,
                risk_measure="CVaR",
            )
            continue

        regime_returns = returns_df.iloc[mask]
        budget = None
        if risk_budgets is not None and regime in risk_budgets:
            budget = risk_budgets[regime]

        result = cvar_parity_weights(regime_returns, alpha=alpha, risk_budget=budget)
        results[regime] = TailRiskParityResult(
            weights=result.weights,
            method=f"cvar_parity_regime_{regime}",
            risk_contributions=result.risk_contributions,
            total_risk=result.total_risk,
            risk_measure="CVaR",
        )

    return results
