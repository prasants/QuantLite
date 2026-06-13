"""Portfolio rebalancing strategies: calendar, threshold, and tactical.

All rebalancing functions accept a weights function and a returns DataFrame,
returning a ``RebalanceResult`` with portfolio returns, weight history,
rebalance dates, and turnover statistics.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = [
    "RebalanceResult",
    "rebalance_calendar",
    "rebalance_threshold",
    "rebalance_tactical",
]


@dataclass(frozen=True)
class RebalanceResult:
    """Result container for a rebalancing strategy.

    Attributes:
        portfolio_returns: Daily portfolio return series.
        weights_over_time: DataFrame of weights at each date.
        rebalance_dates: List of dates when rebalancing occurred.
        turnover: Total turnover (sum of absolute weight changes).
        n_rebalances: Number of rebalance events.
    """

    portfolio_returns: pd.Series
    weights_over_time: pd.DataFrame
    rebalance_dates: list
    turnover: float
    n_rebalances: int

    def __repr__(self) -> str:
        return (
            f"RebalanceResult(rebalances={self.n_rebalances}, "
            f"turnover={self.turnover:.4f}, periods={len(self.portfolio_returns)})"
        )


def _drift_weights(
    current_weights: np.ndarray,
    returns_row: np.ndarray,
) -> np.ndarray:
    """Drift weights forward by one period's returns."""
    new_values = current_weights * (1 + returns_row)
    total = new_values.sum()
    if total < 1e-12:
        return current_weights
    return new_values / total


def _run_rebalance_loop(
    returns_df: pd.DataFrame,
    weights_func: Callable[[pd.DataFrame], dict[str, float]],
    should_rebalance: Callable[[int, pd.DatetimeIndex, np.ndarray, np.ndarray | None], bool],
    regime_labels: np.ndarray | None = None,
) -> RebalanceResult:
    """Core rebalancing loop shared by all strategies.

    Args:
        returns_df: Asset returns DataFrame.
        weights_func: Callable that returns target weights dict.
        should_rebalance: Callable(idx, dates, current_weights, regime_labels) -> bool.
        regime_labels: Optional regime label array.

    Returns:
        ``RebalanceResult`` with full history.
    """
    names = list(returns_df.columns)
    n_assets = len(names)
    dates = returns_df.index
    n_periods = len(dates)

    weights_history = np.zeros((n_periods, n_assets))
    portfolio_rets = np.zeros(n_periods)
    rebalance_dates: list = []
    total_turnover = 0.0

    current_weights = np.zeros(n_assets)

    for i in range(n_periods):
        returns_row = returns_df.iloc[i].values

        if i == 0 or should_rebalance(i, dates, current_weights, regime_labels):
            target = weights_func(returns_df.iloc[:i + 1] if i > 0 else returns_df.iloc[:1])
            target_w = np.array([target.get(name, 0.0) for name in names])
            # Normalise
            s = target_w.sum()
            if s > 1e-12:
                target_w = target_w / s

            total_turnover += float(np.sum(np.abs(target_w - current_weights)))
            current_weights = target_w
            rebalance_dates.append(dates[i])

        # Portfolio return for this period
        port_ret = float(current_weights @ returns_row)
        portfolio_rets[i] = port_ret
        weights_history[i] = current_weights

        # Drift weights
        current_weights = _drift_weights(current_weights, returns_row)

    return RebalanceResult(
        portfolio_returns=pd.Series(portfolio_rets, index=dates),
        weights_over_time=pd.DataFrame(weights_history, index=dates, columns=names),
        rebalance_dates=rebalance_dates,
        turnover=total_turnover,
        n_rebalances=len(rebalance_dates),
    )


def rebalance_calendar(
    returns_df: pd.DataFrame,
    weights_func: Callable[[pd.DataFrame], dict[str, float]],
    freq: str = "monthly",
) -> RebalanceResult:
    """Rebalance at fixed calendar intervals.

    Args:
        returns_df: Asset returns DataFrame with DatetimeIndex.
        weights_func: Callable returning target weights dict.
        freq: One of ``"daily"``, ``"weekly"``, ``"monthly"``, ``"quarterly"``.

    Returns:
        ``RebalanceResult`` with periodic rebalancing.
    """

    def _period_key(dt: object, f: str) -> object:
        if f == "daily":
            return dt
        if f == "weekly":
            return (getattr(dt, "year", 0), getattr(dt, "isocalendar", lambda: (0, 0, 0))()[1])
        if f == "monthly":
            return (getattr(dt, "year", 0), getattr(dt, "month", 0))
        if f == "quarterly":
            return (getattr(dt, "year", 0), (getattr(dt, "month", 0) - 1) // 3)
        return dt

    prev_key: object = None

    def should_rebalance(
        idx: int,
        dates_arr: pd.DatetimeIndex,
        _w: np.ndarray,
        _r: np.ndarray | None,
    ) -> bool:
        nonlocal prev_key
        key = _period_key(dates_arr[idx], freq)
        if key != prev_key:
            prev_key = key
            return True
        return False

    return _run_rebalance_loop(returns_df, weights_func, should_rebalance)


def rebalance_threshold(
    returns_df: pd.DataFrame,
    weights_func: Callable[[pd.DataFrame], dict[str, float]],
    threshold: float = 0.05,
) -> RebalanceResult:
    """Rebalance when any weight drifts beyond the threshold.

    Args:
        returns_df: Asset returns DataFrame.
        weights_func: Callable returning target weights dict.
        threshold: Maximum allowed drift from target for any asset.

    Returns:
        ``RebalanceResult`` with threshold-triggered rebalancing.
    """
    names = list(returns_df.columns)
    last_target: np.ndarray | None = None

    def should_rebalance(
        idx: int,
        _dates: pd.DatetimeIndex,
        current_weights: np.ndarray,
        _r: np.ndarray | None,
    ) -> bool:
        nonlocal last_target
        if last_target is None:
            return True
        drift = np.max(np.abs(current_weights - last_target))
        return drift > threshold

    original_func = weights_func

    def tracking_func(df: pd.DataFrame) -> dict[str, float]:
        nonlocal last_target
        result = original_func(df)
        last_target = np.array([result.get(name, 0.0) for name in names])
        s = last_target.sum()
        if s > 1e-12:
            last_target = last_target / s
        return result

    return _run_rebalance_loop(returns_df, tracking_func, should_rebalance)


def rebalance_tactical(
    returns_df: pd.DataFrame,
    weights_func: Callable[[pd.DataFrame], dict[str, float]],
    regime_labels: np.ndarray | pd.Series,
) -> RebalanceResult:
    """Rebalance when the market regime changes.

    Triggers a rebalance whenever the regime label differs from the
    previous period, enabling tactical allocation shifts.

    Args:
        returns_df: Asset returns DataFrame.
        weights_func: Callable returning target weights dict.
        regime_labels: Array of regime labels (one per period in returns_df).

    Returns:
        ``RebalanceResult`` with regime-triggered rebalancing.
    """
    labels = np.asarray(regime_labels)
    if len(labels) != len(returns_df):
        raise ValueError(
            f"regime_labels length ({len(labels)}) must match "
            f"returns_df length ({len(returns_df)})"
        )

    def should_rebalance(
        idx: int,
        _dates: pd.DatetimeIndex,
        _w: np.ndarray,
        _r: np.ndarray | None,
    ) -> bool:
        if idx == 0:
            return True
        return bool(labels[idx] != labels[idx - 1])

    return _run_rebalance_loop(returns_df, weights_func, should_rebalance, labels)
