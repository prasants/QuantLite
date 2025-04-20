"""Multi-asset production backtesting engine.

Supports fractional shares, configurable slippage models, risk limits
with circuit breakers, and regime-aware allocation functions.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ..risk.metrics import (
    calmar_ratio,
    cvar,
    max_drawdown_duration,
    omega_ratio,
    return_moments,
    sortino_ratio,
    tail_ratio,
    value_at_risk,
)

__all__ = [
    "BacktestConfig",
    "BacktestContext",
    "BacktestResult",
    "RiskLimits",
    "SlippageModel",
    "run_backtest",
]


@dataclass
class SlippageModel:
    """Transaction cost and slippage model.

    Attributes:
        kind: Slippage type: ``"fixed"``, ``"volume_impact"``, or ``"orderbook"``.
        spread_bps: Fixed spread in basis points (used for ``"fixed"``).
        impact_coeff: Market impact coefficient (used for ``"volume_impact"``).
    """

    kind: str = "fixed"
    spread_bps: float = 10.0
    impact_coeff: float = 0.1


@dataclass
class RiskLimits:
    """Portfolio-level risk constraints.

    Attributes:
        max_drawdown: Circuit breaker threshold (e.g. -0.20 for 20% drawdown).
        max_position_pct: Maximum weight for any single position.
        max_leverage: Maximum gross exposure (1.0 = no leverage).
        daily_loss_limit: Maximum single-day loss before halting.
    """

    max_drawdown: float = -0.20
    max_position_pct: float = 0.25
    max_leverage: float = 1.0
    daily_loss_limit: float = -0.05


@dataclass
class BacktestConfig:
    """Configuration for a backtest run.

    Attributes:
        initial_capital: Starting portfolio value.
        fractional_shares: Allow fractional share positions (critical for crypto).
        slippage_model: Slippage/transaction cost model.
        risk_limits: Portfolio risk constraints.
        rebalance_freq: Rebalance frequency: ``"daily"``, ``"weekly"``, ``"monthly"``.
        fee_per_trade_pct: Transaction fee as a fraction (0.001 = 10bps).
    """

    initial_capital: float = 100_000.0
    fractional_shares: bool = True
    slippage_model: SlippageModel | None = None
    risk_limits: RiskLimits | None = None
    rebalance_freq: str = "daily"
    fee_per_trade_pct: float = 0.001


@dataclass(frozen=True)
class BacktestContext:
    """State passed to the allocation function at each rebalance.

    Attributes:
        current_prices: Most recent price per asset.
        historical_returns: Returns up to the current date.
        current_weights: Current portfolio weight per asset.
        current_drawdown: Current drawdown from peak.
        current_regime: Current regime label, if a regime model is active.
        portfolio_value: Current total portfolio value.
        date: Current date.
    """

    current_prices: pd.Series
    historical_returns: pd.DataFrame
    current_weights: dict[str, float]
    current_drawdown: float
    current_regime: int | None
    portfolio_value: float
    date: object


@dataclass(frozen=True)
class BacktestResult:
    """Complete backtest output.

    Attributes:
        portfolio_value: Time series of portfolio value.
        weights_over_time: DataFrame of weights at each date.
        trades: List of trade records (date, asset, delta_weight, cost).
        metrics: Dict of risk/return metrics computed via risk.metrics.
        drawdown_series: Drawdown time series.
        regime_labels: Regime labels per period, if provided.
    """

    portfolio_value: pd.Series
    weights_over_time: pd.DataFrame
    trades: list[dict[str, Any]]
    metrics: dict[str, float]
    drawdown_series: pd.Series
    regime_labels: np.ndarray | None = None

    def __repr__(self) -> str:
        final = self.portfolio_value.iloc[-1] if len(self.portfolio_value) > 0 else 0
        n_trades = len(self.trades)
        return (
            f"BacktestResult(final_value={final:,.2f}, trades={n_trades}, "
            f"periods={len(self.portfolio_value)})"
        )


def _compute_slippage(
    trade_value: float,
    model: SlippageModel | None,
) -> float:
    """Compute slippage cost for a trade.

    Args:
        trade_value: Absolute value of the trade.
        model: Slippage model configuration.

    Returns:
        Slippage cost (always positive).
    """
    if model is None:
        return 0.0
    if model.kind == "fixed":
        return trade_value * model.spread_bps / 10_000
    if model.kind == "volume_impact":
        return trade_value * model.impact_coeff * np.sqrt(trade_value)
    return trade_value * model.spread_bps / 10_000


def _should_rebalance(idx: int, dates: pd.DatetimeIndex, freq: str) -> bool:
    """Determine whether to rebalance at this index."""
    if idx == 0:
        return True
    if freq == "daily":
        return True
    if freq == "weekly":
        return getattr(dates[idx], "weekday", lambda: 0)() < getattr(dates[idx - 1], "weekday", lambda: 6)()
    if freq == "monthly":
        return getattr(dates[idx], "month", 0) != getattr(dates[idx - 1], "month", 0)
    return True


def _clip_weights(
    weights: dict[str, float],
    risk_limits: RiskLimits | None,
) -> dict[str, float]:
    """Enforce position size and leverage limits."""
    if risk_limits is None:
        return weights

    clipped = {}
    for asset, w in weights.items():
        clipped[asset] = max(-risk_limits.max_position_pct, min(w, risk_limits.max_position_pct))

    gross = sum(abs(v) for v in clipped.values())
    if gross > risk_limits.max_leverage and gross > 0:
        scale = risk_limits.max_leverage / gross
        clipped = {k: v * scale for k, v in clipped.items()}

    return clipped


def _compute_metrics(returns_arr: np.ndarray) -> dict[str, float]:
    """Compute comprehensive metrics from portfolio returns."""
    metrics: dict[str, float] = {}

    if len(returns_arr) < 4:
        return metrics

    try:
        moments = return_moments(returns_arr)
        metrics["mean_return"] = moments.mean
        metrics["volatility"] = moments.volatility
        metrics["skewness"] = moments.skewness
        metrics["kurtosis"] = moments.kurtosis
    except (ValueError, Exception):
        pass

    ann_ret = float((1 + np.mean(returns_arr)) ** 252 - 1)
    ann_vol = float(np.std(returns_arr, ddof=1) * np.sqrt(252))
    metrics["annualised_return"] = ann_ret
    metrics["annualised_volatility"] = ann_vol
    metrics["sharpe_ratio"] = ann_ret / ann_vol if ann_vol > 0 else float("nan")

    try:
        metrics["var_95"] = value_at_risk(returns_arr, alpha=0.05)
        metrics["cvar_95"] = cvar(returns_arr, alpha=0.05)
    except (ValueError, Exception):
        pass

    try:
        dd = max_drawdown_duration(returns_arr)
        metrics["max_drawdown"] = dd.max_drawdown
        metrics["max_drawdown_duration"] = float(dd.duration)
    except (ValueError, Exception):
        pass

    with contextlib.suppress(ValueError, Exception):
        metrics["sortino_ratio"] = sortino_ratio(returns_arr)

    with contextlib.suppress(ValueError, Exception):
        metrics["calmar_ratio"] = calmar_ratio(returns_arr)

    with contextlib.suppress(ValueError, Exception):
        metrics["omega_ratio"] = omega_ratio(returns_arr)

    with contextlib.suppress(ValueError, Exception):
        metrics["tail_ratio"] = tail_ratio(returns_arr)

    return metrics


def run_backtest(
    price_data: pd.DataFrame,
    allocation_func: Callable[[BacktestContext], dict[str, float]],
    config: BacktestConfig | None = None,
    regime_labels: np.ndarray | None = None,
) -> BacktestResult:
    """Run a multi-asset backtest.

    Args:
        price_data: DataFrame of asset prices (assets as columns, DatetimeIndex).
        allocation_func: Callable that receives a ``BacktestContext`` and returns
            target weights as ``dict[str, float]``.
        config: Backtest configuration. Uses defaults if None.
        regime_labels: Optional array of regime labels (one per period).

    Returns:
        ``BacktestResult`` with full portfolio history and metrics.

    Raises:
        ValueError: If price_data is empty or not a DataFrame.
    """
    if not isinstance(price_data, pd.DataFrame) or price_data.empty:
        raise ValueError("price_data must be a non-empty DataFrame")

    if config is None:
        config = BacktestConfig()

    price_data = price_data.sort_index()
    returns_df = price_data.pct_change().fillna(0.0)
    dates = price_data.index
    names = list(price_data.columns)
    n_assets = len(names)
    n_periods = len(dates)

    # State
    portfolio_value = config.initial_capital
    peak_value = portfolio_value
    current_weights: dict[str, float] = {name: 0.0 for name in names}

    # Output arrays
    pv_history = np.zeros(n_periods)
    dd_history = np.zeros(n_periods)
    weights_history = np.zeros((n_periods, n_assets))
    trades: list[dict[str, Any]] = []
    circuit_breaker_active = False
    daily_start_value = portfolio_value

    for i in range(n_periods):
        date = dates[i]
        prices = price_data.iloc[i]
        returns_row = returns_df.iloc[i].values

        # Apply returns to portfolio
        if i > 0:
            port_return = sum(
                current_weights.get(name, 0.0) * returns_row[j]
                for j, name in enumerate(names)
            )
            portfolio_value *= (1 + port_return)

        # Track drawdown
        peak_value = max(peak_value, portfolio_value)
        current_dd = (portfolio_value - peak_value) / peak_value if peak_value > 0 else 0.0
        dd_history[i] = current_dd

        # Daily loss check
        if i > 0 and dates[i] != dates[i - 1]:
            daily_start_value = pv_history[i - 1]
        daily_pnl = (portfolio_value - daily_start_value) / daily_start_value if daily_start_value > 0 else 0.0

        # Risk limit enforcement
        if config.risk_limits is not None:
            if current_dd <= config.risk_limits.max_drawdown:
                circuit_breaker_active = True
            if daily_pnl <= config.risk_limits.daily_loss_limit:
                circuit_breaker_active = True

        # Rebalance decision
        if circuit_breaker_active:
            # Liquidate everything
            if any(abs(v) > 1e-10 for v in current_weights.values()):
                for name in names:
                    if abs(current_weights[name]) > 1e-10:
                        trade_val = abs(current_weights[name]) * portfolio_value
                        slip = _compute_slippage(trade_val, config.slippage_model)
                        fee = trade_val * config.fee_per_trade_pct
                        portfolio_value -= (slip + fee)
                        trades.append({
                            "date": date,
                            "asset": name,
                            "old_weight": current_weights[name],
                            "new_weight": 0.0,
                            "cost": slip + fee,
                        })
                current_weights = {name: 0.0 for name in names}
        elif _should_rebalance(i, dates, config.rebalance_freq):
            hist_returns = returns_df.iloc[:i + 1] if i > 0 else returns_df.iloc[:1]
            regime = int(regime_labels[i]) if regime_labels is not None else None

            ctx = BacktestContext(
                current_prices=prices,
                historical_returns=hist_returns,
                current_weights=dict(current_weights),
                current_drawdown=current_dd,
                current_regime=regime,
                portfolio_value=portfolio_value,
                date=date,
            )

            target_weights = allocation_func(ctx)
            target_weights = _clip_weights(target_weights, config.risk_limits)

            # Execute trades
            for _j, name in enumerate(names):
                old_w = current_weights.get(name, 0.0)
                new_w = target_weights.get(name, 0.0)
                delta = abs(new_w - old_w)

                if delta > 1e-8:
                    trade_val = delta * portfolio_value
                    slip = _compute_slippage(trade_val, config.slippage_model)
                    fee = trade_val * config.fee_per_trade_pct
                    portfolio_value -= (slip + fee)
                    trades.append({
                        "date": date,
                        "asset": name,
                        "old_weight": old_w,
                        "new_weight": new_w,
                        "cost": slip + fee,
                    })

            current_weights = {name: target_weights.get(name, 0.0) for name in names}

        pv_history[i] = portfolio_value
        weights_history[i] = [current_weights.get(name, 0.0) for name in names]

    # Compute portfolio returns from value series
    pv_series = pd.Series(pv_history, index=dates)
    port_returns = pv_series.pct_change().fillna(0.0).values

    metrics = _compute_metrics(port_returns[1:] if len(port_returns) > 1 else port_returns)
    metrics["initial_capital"] = config.initial_capital
    metrics["final_value"] = float(pv_history[-1]) if n_periods > 0 else config.initial_capital
    metrics["total_return"] = (metrics["final_value"] / config.initial_capital) - 1
    metrics["n_trades"] = float(len(trades))

    return BacktestResult(
        portfolio_value=pv_series,
        weights_over_time=pd.DataFrame(weights_history, index=dates, columns=names),
        trades=trades,
        metrics=metrics,
        drawdown_series=pd.Series(dd_history, index=dates),
        regime_labels=regime_labels,
    )
