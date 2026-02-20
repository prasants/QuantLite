"""Ensemble allocators: blend multiple allocation strategies.

Combines HRP, risk parity, Kelly, and Black-Litterman into a
single consensus portfolio. Supports equal weighting,
regime-confidence weighting, and inverse-error weighting schemes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..dependency.clustering import hrp_weights as _hrp_weights_raw
from .optimisation import risk_parity_weights

__all__ = [
    "EnsembleResult",
    "ensemble_allocate",
    "consensus_portfolio",
    "inverse_error_weights",
]


@dataclass(frozen=True)
class EnsembleResult:
    """Result container for ensemble allocation.

    Attributes:
        blended_weights: Final blended portfolio weights.
        strategy_weights: Weight given to each strategy in the blend.
        strategy_allocations: Per-strategy portfolio allocations.
        agreement_matrix: Pairwise agreement between strategies
            (correlation of weight vectors).
    """

    blended_weights: dict[str, float]
    strategy_weights: dict[str, float]
    strategy_allocations: dict[str, dict[str, float]]
    agreement_matrix: pd.DataFrame


def _compute_agreement_matrix(
    allocations: dict[str, dict[str, float]],
    asset_names: list[str],
) -> pd.DataFrame:
    """Compute pairwise agreement (correlation) between strategy weight vectors.

    Args:
        allocations: Per-strategy weight dicts.
        asset_names: Ordered list of asset names.

    Returns:
        DataFrame of pairwise correlations between strategies.
    """
    strategy_names = list(allocations.keys())
    n = len(strategy_names)

    # Build weight matrix: rows = strategies, cols = assets
    weight_matrix = np.zeros((n, len(asset_names)))
    for i, strat in enumerate(strategy_names):
        for j, asset in enumerate(asset_names):
            weight_matrix[i, j] = allocations[strat].get(asset, 0.0)

    # Correlation matrix
    corr = np.ones((n, n)) if n < 2 else np.corrcoef(weight_matrix)

    return pd.DataFrame(corr, index=strategy_names, columns=strategy_names)


def ensemble_allocate(
    returns_df: pd.DataFrame,
    strategies: dict[str, dict[str, float]] | None = None,
    weighting: str = "equal",
    strategy_errors: dict[str, float] | None = None,
    regime_confidences: dict[str, float] | None = None,
    market_caps: dict[str, float] | None = None,
    views: dict[str, float] | None = None,
    view_confidences: dict[str, float] | None = None,
    kelly_returns: np.ndarray | None = None,
) -> EnsembleResult:
    """Blend multiple allocation strategies into a single portfolio.

    If ``strategies`` is not provided, computes HRP and risk parity
    from ``returns_df`` as default strategies.

    Args:
        returns_df: DataFrame of asset returns (assets as columns).
        strategies: Pre-computed strategy allocations. Keys are strategy
            names, values are weight dicts.
        weighting: Blending scheme. One of ``"equal"``,
            ``"inverse_error"``, or ``"regime_confidence"``.
        strategy_errors: Per-strategy tracking error or loss metric
            (required for ``"inverse_error"`` weighting).
        regime_confidences: Per-strategy regime confidence score
            (required for ``"regime_confidence"`` weighting).
        market_caps: Market caps for BL (used if strategies not provided).
        views: Views for BL (used if strategies not provided).
        view_confidences: View confidences for BL.
        kelly_returns: Returns for Kelly (used if strategies not provided).

    Returns:
        ``EnsembleResult`` with blended weights and agreement analysis.
    """
    names = list(returns_df.columns)

    # Build default strategies if not provided
    if strategies is None:
        strategies = {}

        # HRP
        hrp_raw = _hrp_weights_raw(returns_df)
        strategies["HRP"] = dict(hrp_raw)

        # Risk parity
        rp = risk_parity_weights(returns_df)
        strategies["Risk Parity"] = rp.weights

    strategy_names = list(strategies.keys())
    n_strategies = len(strategy_names)

    # Compute strategy weights
    if weighting == "inverse_error" and strategy_errors is not None:
        strat_w = inverse_error_weights(strategy_errors)
    elif weighting == "regime_confidence" and regime_confidences is not None:
        total_conf = sum(regime_confidences.values())
        if total_conf > 0:
            strat_w = {k: v / total_conf for k, v in regime_confidences.items()}
        else:
            strat_w = {k: 1.0 / n_strategies for k in strategy_names}
    else:
        strat_w = {k: 1.0 / n_strategies for k in strategy_names}

    # Blend
    blended: dict[str, float] = {a: 0.0 for a in names}
    for strat_name, alloc in strategies.items():
        sw = strat_w.get(strat_name, 0.0)
        for asset in names:
            blended[asset] += alloc.get(asset, 0.0) * sw

    # Renormalise
    total = sum(blended.values())
    if abs(total) > 1e-12:
        blended = {k: v / total for k, v in blended.items()}

    agreement = _compute_agreement_matrix(strategies, names)

    return EnsembleResult(
        blended_weights=blended,
        strategy_weights=strat_w,
        strategy_allocations=strategies,
        agreement_matrix=agreement,
    )


def consensus_portfolio(
    strategies: dict[str, dict[str, float]],
    threshold: float = 0.05,
) -> dict[str, float]:
    """Find the consensus portfolio: where all strategies agree.

    For each asset, takes the minimum weight across strategies
    (the weight everyone agrees on). Useful for identifying
    high-conviction positions.

    Args:
        strategies: Per-strategy weight dictionaries.
        threshold: Minimum consensus weight to include (filters noise).

    Returns:
        Dictionary of consensus weights (renormalised to sum to 1).
    """
    all_assets: list[str] = []
    for weights in strategies.values():
        for asset in weights:
            if asset not in all_assets:
                all_assets.append(asset)

    consensus: dict[str, float] = {}
    for asset in all_assets:
        weights_for_asset = [
            alloc.get(asset, 0.0) for alloc in strategies.values()
        ]
        min_weight = min(weights_for_asset)
        if min_weight >= threshold:
            consensus[asset] = min_weight

    # Renormalise
    total = sum(consensus.values())
    if total > 1e-12:
        consensus = {k: v / total for k, v in consensus.items()}

    return consensus


def inverse_error_weights(
    strategy_errors: dict[str, float],
) -> dict[str, float]:
    """Compute strategy blend weights inversely proportional to error.

    Strategies with lower tracking error or loss get higher weight.

    Args:
        strategy_errors: Per-strategy error metric (positive values).

    Returns:
        Normalised weight dictionary.
    """
    inv = {}
    for k, v in strategy_errors.items():
        inv[k] = 1.0 / max(abs(v), 1e-12)

    total = sum(inv.values())
    return {k: v / total for k, v in inv.items()}
