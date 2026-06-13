"""What-if analysis: explore hypothetical portfolio scenarios.

Supports asset removal, correlation stress testing, weight constraints,
and side-by-side scenario comparison.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = [
    "WhatIfResult",
    "ScenarioMetrics",
    "remove_asset",
    "stress_correlations",
    "cap_weights",
    "compare_scenarios",
]


@dataclass(frozen=True)
class ScenarioMetrics:
    """Portfolio metrics for a single scenario.

    Attributes:
        name: Scenario name.
        weights: Portfolio weights.
        annual_return: Annualised return.
        annual_vol: Annualised volatility.
        sharpe: Sharpe ratio.
        var_95: Value at Risk (95%).
        cvar_95: Conditional VaR (95%).
        max_drawdown: Maximum drawdown.
    """

    name: str
    weights: dict[str, float]
    annual_return: float
    annual_vol: float
    sharpe: float
    var_95: float
    cvar_95: float
    max_drawdown: float


@dataclass(frozen=True)
class WhatIfResult:
    """Container for what-if analysis results.

    Attributes:
        base: Metrics for the base case.
        scenarios: List of scenario metrics.
        comparison_table: DataFrame comparing all scenarios side by side.
    """

    base: ScenarioMetrics
    scenarios: list[ScenarioMetrics]
    comparison_table: pd.DataFrame


def _compute_metrics(
    returns_df: pd.DataFrame,
    weights: dict[str, float],
    name: str,
    freq: int = 252,
) -> ScenarioMetrics:
    """Compute portfolio metrics for a given weight set."""
    names = list(returns_df.columns)
    w = np.array([weights.get(n, 0.0) for n in names])
    port_ret = returns_df.values @ w

    ann_ret = float(np.mean(port_ret)) * freq
    ann_vol = float(np.std(port_ret)) * np.sqrt(freq)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0

    var_95 = -float(np.percentile(port_ret, 5))
    tail = port_ret[port_ret <= np.percentile(port_ret, 5)]
    cvar_95 = -float(np.mean(tail)) if len(tail) > 0 else var_95

    # Max drawdown
    cumulative = np.cumprod(1 + port_ret)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = (cumulative - running_max) / running_max
    max_dd = float(-np.min(drawdowns))

    return ScenarioMetrics(
        name=name,
        weights=weights,
        annual_return=ann_ret,
        annual_vol=ann_vol,
        sharpe=sharpe,
        var_95=var_95,
        cvar_95=cvar_95,
        max_drawdown=max_dd,
    )


def remove_asset(
    returns_df: pd.DataFrame,
    weights: dict[str, float],
    asset: str,
    freq: int = 252,
) -> WhatIfResult:
    """What-if: remove an asset and redistribute weights proportionally.

    Args:
        returns_df: DataFrame of asset returns.
        weights: Base portfolio weights.
        asset: Name of the asset to remove.
        freq: Annualisation frequency.

    Returns:
        ``WhatIfResult`` comparing base case to the asset-removal scenario.

    Raises:
        ValueError: If the asset is not in the portfolio.

    Example:
        >>> result = remove_asset(returns_df, weights, "Crypto")
        >>> print(result.comparison_table)
    """
    if asset not in weights:
        raise ValueError(f"Asset '{asset}' not found in weights.")

    base = _compute_metrics(returns_df, weights, "Base Case", freq)

    # Remove and renormalise
    new_weights = {k: v for k, v in weights.items() if k != asset}
    total = sum(new_weights.values())
    if total > 0:
        new_weights = {k: v / total for k, v in new_weights.items()}

    # Use only remaining columns
    remaining_cols = [c for c in returns_df.columns if c != asset]
    scenario_df = returns_df[remaining_cols]
    scenario = _compute_metrics(scenario_df, new_weights, f"Remove {asset}", freq)

    return _build_result(base, [scenario])


def stress_correlations(
    returns_df: pd.DataFrame,
    weights: dict[str, float],
    stress_factor: float = 2.0,
    freq: int = 252,
) -> WhatIfResult:
    """What-if: stress the correlation matrix toward crisis levels.

    Increases off-diagonal correlations by the stress factor (capped at
    the valid correlation range). Generates synthetic stressed returns
    using the stressed covariance matrix.

    Args:
        returns_df: DataFrame of asset returns.
        weights: Portfolio weights.
        stress_factor: Multiplier for off-diagonal correlations.
        freq: Annualisation frequency.

    Returns:
        ``WhatIfResult`` comparing normal vs stressed correlations.
    """
    base = _compute_metrics(returns_df, weights, "Base Case", freq)

    # Compute stressed correlation matrix
    corr = returns_df.corr().values
    n = corr.shape[0]
    stressed_corr = corr.copy()
    for i in range(n):
        for j in range(n):
            if i != j:
                stressed_corr[i, j] = np.clip(corr[i, j] * stress_factor, -0.99, 0.99)

    # Ensure positive semi-definite
    eigvals, eigvecs = np.linalg.eigh(stressed_corr)
    eigvals = np.maximum(eigvals, 1e-8)
    stressed_corr = eigvecs @ np.diag(eigvals) @ eigvecs.T
    # Renormalise to correlation matrix
    d = np.sqrt(np.diag(stressed_corr))
    stressed_corr = stressed_corr / np.outer(d, d)
    np.fill_diagonal(stressed_corr, 1.0)

    # Generate stressed returns using Cholesky
    stds = returns_df.std().values
    stressed_cov = np.outer(stds, stds) * stressed_corr
    try:
        L = np.linalg.cholesky(stressed_cov)
    except np.linalg.LinAlgError:
        eigvals2, eigvecs2 = np.linalg.eigh(stressed_cov)
        eigvals2 = np.maximum(eigvals2, 1e-10)
        stressed_cov = eigvecs2 @ np.diag(eigvals2) @ eigvecs2.T
        L = np.linalg.cholesky(stressed_cov)

    rng = np.random.RandomState(42)
    z = rng.randn(len(returns_df), n)
    means = returns_df.mean().values
    stressed_returns = z @ L.T + means

    stressed_df = pd.DataFrame(
        stressed_returns, columns=returns_df.columns, index=returns_df.index
    )
    scenario = _compute_metrics(stressed_df, weights, f"Stressed (x{stress_factor})", freq)

    return _build_result(base, [scenario])


def cap_weights(
    returns_df: pd.DataFrame,
    weights: dict[str, float],
    max_weight: float = 0.20,
    freq: int = 252,
) -> WhatIfResult:
    """What-if: cap any single position at a maximum weight.

    Excess weight is redistributed proportionally to uncapped assets.

    Args:
        returns_df: DataFrame of asset returns.
        weights: Base (unconstrained) portfolio weights.
        max_weight: Maximum allowed weight per asset.
        freq: Annualisation frequency.

    Returns:
        ``WhatIfResult`` comparing unconstrained vs capped portfolios.
    """
    base = _compute_metrics(returns_df, weights, "Unconstrained", freq)

    # Cap and redistribute
    capped = dict(weights)
    for _ in range(20):  # Iterate to convergence
        excess = 0.0
        uncapped_total = 0.0
        for k, v in capped.items():
            if v > max_weight:
                excess += v - max_weight
                capped[k] = max_weight
            else:
                uncapped_total += v

        if excess < 1e-8:
            break

        # Redistribute excess
        if uncapped_total > 0:
            for k in capped:
                if capped[k] < max_weight:
                    capped[k] += excess * (capped[k] / uncapped_total)

    scenario = _compute_metrics(returns_df, capped, f"Capped at {max_weight:.0%}", freq)
    return _build_result(base, [scenario])


def compare_scenarios(
    returns_df: pd.DataFrame,
    base_weights: dict[str, float],
    scenarios: dict[str, dict[str, float]],
    freq: int = 252,
) -> WhatIfResult:
    """Compare multiple what-if scenarios against a base case.

    Args:
        returns_df: DataFrame of asset returns.
        base_weights: Base portfolio weights.
        scenarios: Dictionary mapping scenario names to weight dicts.
        freq: Annualisation frequency.

    Returns:
        ``WhatIfResult`` with all scenarios compared.
    """
    base = _compute_metrics(returns_df, base_weights, "Base Case", freq)
    scenario_list = []
    for name, w in scenarios.items():
        # Filter to available columns
        available = [c for c in returns_df.columns if c in w]
        if available:
            sub_df = returns_df[available]
            sub_w = {k: v for k, v in w.items() if k in available}
            total = sum(sub_w.values())
            if total > 0:
                sub_w = {k: v / total for k, v in sub_w.items()}
            scenario_list.append(_compute_metrics(sub_df, sub_w, name, freq))
        else:
            scenario_list.append(_compute_metrics(returns_df, w, name, freq))

    return _build_result(base, scenario_list)


def _build_result(
    base: ScenarioMetrics, scenarios: list[ScenarioMetrics]
) -> WhatIfResult:
    """Build a WhatIfResult with comparison table."""
    all_scenarios = [base] + scenarios
    rows = []
    for s in all_scenarios:
        rows.append({
            "Scenario": s.name,
            "Annual Return": s.annual_return,
            "Annual Vol": s.annual_vol,
            "Sharpe": s.sharpe,
            "VaR (95%)": s.var_95,
            "CVaR (95%)": s.cvar_95,
            "Max Drawdown": s.max_drawdown,
        })

    table = pd.DataFrame(rows).set_index("Scenario")
    return WhatIfResult(base=base, scenarios=scenarios, comparison_table=table)
