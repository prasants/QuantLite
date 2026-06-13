"""Tests for quantlite.backtesting.legacy (single-asset backtester)."""

import pandas as pd

from quantlite.backtesting.legacy import legacy_run_backtest


def test_backtest_simple():
    prices = pd.Series([100, 102, 105], index=[1, 2, 3])
    result = legacy_run_backtest(prices, lambda idx, s: 1 if idx == 0 else 0)
    assert "final_value" in result
    assert result["final_value"] != 0
    assert len(result["portfolio_value"]) == 3


def test_backtest_partial_capital():
    prices = pd.Series([100, 105, 110], index=[0, 1, 2])
    result = legacy_run_backtest(
        prices, lambda idx, s: 1, initial_capital=10_000.0,
        partial_capital=True, capital_fraction=0.5,
    )
    expected = 10365.0
    assert abs(result["final_value"] - expected) < 1e-9


def test_backtest_per_share_cost():
    prices = pd.Series([100, 101], index=[0, 1])
    result = legacy_run_backtest(
        prices, lambda idx, s: 1 if idx == 0 else 0,
        initial_capital=10000, per_share_cost=1.0,
    )
    assert abs(result["final_value"] - 9900.0) < 1e-9
