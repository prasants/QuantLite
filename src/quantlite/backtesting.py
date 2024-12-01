"""Single-asset signal-based backtesting engine."""

from __future__ import annotations

from typing import Any, Callable

import pandas as pd

from .core.types import BacktestResult

__all__ = ["run_backtest"]


def run_backtest(
    price_data: pd.Series,
    signal_function: Callable[[int, pd.Series], int],
    initial_capital: float = 10_000.0,
    fee: float = 0.0,
    partial_capital: bool = False,
    capital_fraction: float = 1.0,
    allow_short: bool = True,
    per_share_cost: float = 0.0,
) -> dict[str, Any]:
    """Run a single-asset backtest driven by a signal function.

    Args:
        price_data: Price series of the asset.
        signal_function: Callable ``(idx, price_data) -> {-1, 0, 1}``.
        initial_capital: Starting capital.
        fee: Flat transaction fee per trade.
        partial_capital: If ``True``, allocate only ``capital_fraction`` of capital.
        capital_fraction: Fraction of capital to deploy (0 to 1).
        allow_short: If ``False``, negative signals are clamped to zero.
        per_share_cost: Commission per share traded.

    Returns:
        Dict with keys ``portfolio_value``, ``positions``, ``trades``,
        and ``final_value``.

    Raises:
        ValueError: If ``price_data`` is not a ``pd.Series``.
    """
    if not isinstance(price_data, pd.Series):
        raise ValueError("price_data must be a pd.Series")

    price_data = price_data.sort_index()
    dates = price_data.index
    capital = initial_capital
    current_shares = 0
    portfolio_values: list[float] = []
    positions: list[int] = []
    trades: list[tuple[object, str, int, float, float]] = []

    for i, date in enumerate(dates):
        price = price_data.iloc[i]
        raw_signal = signal_function(i, price_data)

        if not allow_short and raw_signal < 0:
            raw_signal = 0

        target_position = raw_signal
        desired_capital_for_position = (
            capital * capital_fraction if partial_capital else capital
        )
        desired_capital_for_position = max(desired_capital_for_position, 0.0)

        if target_position == 0:
            desired_shares = 0
        else:
            desired_shares = int(desired_capital_for_position // price) * target_position

        if desired_shares != current_shares:
            delta_shares = desired_shares - current_shares
            trade_cost = fee + abs(delta_shares) * per_share_cost
            capital += delta_shares * price * -1 - trade_cost
            trades.append((
                date,
                "buy" if delta_shares > 0 else "sell",
                delta_shares,
                price,
                trade_cost,
            ))
            current_shares = desired_shares

        port_val = capital + current_shares * price
        portfolio_values.append(port_val)
        positions.append(current_shares)

    portfolio_series = pd.Series(portfolio_values, index=dates)
    positions_series = pd.Series(positions, index=dates)

    return {
        "portfolio_value": portfolio_series,
        "positions": positions_series,
        "trades": trades,
        "final_value": portfolio_series.iloc[-1],
    }
