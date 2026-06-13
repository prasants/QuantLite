"""Regime-aware portfolio construction and backtesting.

Provides weight computation that tilts defensively during crisis regimes,
regime transition detection for rebalancing signals, and a filtered
backtester that applies different weight sets per regime.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

__all__ = [
    "regime_aware_weights",
    "regime_rebalance_signals",
    "regime_filtered_backtest",
]


def regime_aware_weights(
    returns_df: pd.DataFrame,
    regimes: np.ndarray | pd.Series | list[Any],
    method: str = "hrp",
    defensive_tilt: float = 0.3,
    defensive_assets: list[str] | None = None,
) -> dict[str, float]:
    """Compute portfolio weights with automatic defensive tilting in crisis.

    During the most recent regime, if it is identified as crisis (the
    regime with the lowest mean return), weights are tilted towards
    defensive assets.

    Args:
        returns_df: DataFrame of asset returns (assets as columns).
        regimes: Array of regime labels, same length as *returns_df*.
        method: Weight method: ``"hrp"``, ``"min_variance"``, or
            ``"equal_weight"``.
        defensive_tilt: Fraction by which to increase defensive asset
            weights during crisis (0.0 to 1.0).
        defensive_assets: Column names to treat as defensive. If ``None``,
            heuristically selects assets with names containing "GLD",
            "TLT", "bond", or "gold" (case-insensitive).

    Returns:
        Dictionary mapping asset names to weights (summing to 1.0).

    Raises:
        ValueError: On unknown method or empty DataFrame.
    """
    if returns_df.empty:
        raise ValueError("returns_df must be non-empty")

    reg = np.asarray(regimes)
    assets = list(returns_df.columns)
    n = len(assets)

    # Compute base weights
    if method == "equal_weight":
        base_weights = {a: 1.0 / n for a in assets}
    elif method == "min_variance":
        cov = returns_df.cov().values
        try:
            inv_cov = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            inv_cov = np.linalg.pinv(cov)
        ones = np.ones(n)
        w = inv_cov @ ones / (ones @ inv_cov @ ones)
        w = np.maximum(w, 0)
        w = w / w.sum()
        base_weights = dict(zip(assets, w.tolist()))
    elif method == "hrp":
        # Simple quasi-HRP: inverse volatility weighting with correlation adjustment
        vols = returns_df.std().values
        vols = np.where(vols < 1e-10, 1e-10, vols)
        inv_vol = 1.0 / vols
        w = inv_vol / inv_vol.sum()
        base_weights = dict(zip(assets, w.tolist()))
    else:
        raise ValueError(f"Unknown method: {method}. Use 'hrp', 'min_variance', or 'equal_weight'.")

    # Determine current regime
    current_regime = reg[-1] if len(reg) > 0 else None

    # Identify crisis regime (lowest mean return)
    unique_regimes = np.unique(reg)
    regime_means = {}
    for label in unique_regimes:
        mask = reg == label
        if mask.sum() > 0:
            regime_means[label] = float(returns_df.iloc[mask.nonzero()[0]].mean().mean())
    crisis_regime = min(regime_means, key=regime_means.get) if regime_means else None

    # Apply defensive tilt if in crisis
    if current_regime is not None and current_regime == crisis_regime and defensive_tilt > 0:
        if defensive_assets is None:
            defensive_keywords = ["gld", "tlt", "bond", "gold", "treasury", "ief", "shy"]
            defensive_assets = [
                a for a in assets
                if any(kw in a.lower() for kw in defensive_keywords)
            ]

        if defensive_assets:
            offensive_assets = [a for a in assets if a not in defensive_assets]
            for a in defensive_assets:
                if a in base_weights:
                    base_weights[a] *= (1.0 + defensive_tilt)
            for a in offensive_assets:
                if a in base_weights:
                    ratio = len(defensive_assets) / max(len(offensive_assets), 1)
                    base_weights[a] *= (1.0 - defensive_tilt * ratio)

            # Renormalise
            total = sum(base_weights.values())
            if total > 0:
                base_weights = {k: v / total for k, v in base_weights.items()}

    return base_weights


def regime_rebalance_signals(
    regimes: np.ndarray | pd.Series | list[Any],
    lookback: int = 5,
) -> list[dict[str, Any]]:
    """Detect regime transitions and emit rebalance signals.

    A signal is emitted whenever the regime changes, confirmed by
    *lookback* consecutive observations in the new regime to avoid
    whipsaw.

    Args:
        regimes: Array of regime labels.
        lookback: Number of consecutive observations in the new regime
            required before emitting a signal.

    Returns:
        List of signal dictionaries with keys ``"index"``,
        ``"from_regime"``, ``"to_regime"``.
    """
    reg = np.asarray(regimes)
    if len(reg) < 2:
        return []

    signals: list[dict[str, Any]] = []
    prev_regime = reg[0]
    consecutive = 0
    pending_regime = None

    for i in range(1, len(reg)):
        if reg[i] != prev_regime:
            if pending_regime == reg[i]:
                consecutive += 1
            else:
                pending_regime = reg[i]
                consecutive = 1

            if consecutive >= lookback:
                signals.append({
                    "index": i,
                    "from_regime": prev_regime,
                    "to_regime": reg[i],
                })
                prev_regime = reg[i]
                pending_regime = None
                consecutive = 0
        else:
            pending_regime = None
            consecutive = 0

    return signals


def regime_filtered_backtest(
    returns_df: pd.DataFrame,
    weights_by_regime: dict[Any, dict[str, float]],
    regimes: np.ndarray | pd.Series | list[Any],
    rebalance: str = "monthly",
    initial_capital: float = 10000.0,
) -> dict[str, Any]:
    """Backtest with different weight sets per regime.

    Applies the weight dictionary corresponding to the current regime
    at each rebalance point.

    Args:
        returns_df: DataFrame of asset returns (assets as columns,
            DatetimeIndex).
        weights_by_regime: Mapping of regime label to weight dictionary.
        regimes: Array of regime labels, same length as *returns_df*.
        rebalance: Rebalance frequency: ``"daily"``, ``"weekly"``, or
            ``"monthly"``.
        initial_capital: Starting portfolio value.

    Returns:
        Dictionary with keys:
        - ``"equity_curve"``: pandas Series of portfolio values.
        - ``"regime_attribution"``: dict mapping regimes to cumulative
          return contributed.
        - ``"total_return"``: total percentage return.
        - ``"weights_history"``: list of (index, regime, weights) tuples.
    """
    reg = np.asarray(regimes)
    if len(reg) != len(returns_df):
        raise ValueError("regimes must have same length as returns_df")

    assets = list(returns_df.columns)
    n = len(returns_df)

    portfolio_values = np.zeros(n)
    pv = initial_capital
    current_weights = None
    last_rebalance_month = None
    last_rebalance_week = None
    weights_history: list[Any] = []
    regime_returns: dict[str, float] = {str(r): 0.0 for r in np.unique(reg)}

    for i in range(n):
        current_regime = reg[i]
        should_rebalance = False

        if i == 0 or rebalance == "daily":
            should_rebalance = True
        elif rebalance == "weekly":
            if hasattr(returns_df.index, 'isocalendar'):
                week = returns_df.index[i].isocalendar()[1]
                if week != last_rebalance_week:
                    should_rebalance = True
                    last_rebalance_week = week
            else:
                should_rebalance = (i % 5 == 0)
        elif rebalance == "monthly":
            if hasattr(returns_df.index, 'month'):
                month = returns_df.index[i].month
                if month != last_rebalance_month:
                    should_rebalance = True
                    last_rebalance_month = month
            else:
                should_rebalance = (i % 21 == 0)

        if should_rebalance:
            w = weights_by_regime.get(current_regime)
            if w is None:
                # Fall back to equal weight
                w = {a: 1.0 / len(assets) for a in assets}
            current_weights = w
            weights_history.append((i, current_regime, dict(w)))

        if current_weights is not None and i > 0:
            period_return = sum(
                current_weights.get(a, 0.0) * returns_df.iloc[i][a]
                for a in assets
            )
            pv *= (1.0 + period_return)
            regime_returns[str(current_regime)] += period_return

        portfolio_values[i] = pv

    equity_curve = pd.Series(portfolio_values, index=returns_df.index)
    total_return = (pv / initial_capital - 1.0) * 100

    return {
        "equity_curve": equity_curve,
        "regime_attribution": regime_returns,
        "total_return": total_return,
        "weights_history": weights_history,
    }
