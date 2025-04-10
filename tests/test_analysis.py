"""Tests for post-backtest analysis functions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantlite.backtesting.engine import (
    BacktestConfig,
    BacktestContext,
    run_backtest,
)
from quantlite.backtesting.analysis import (
    monthly_returns_table,
    performance_summary,
    regime_attribution,
    rolling_metrics,
    trade_analysis,
)


@pytest.fixture()
def backtest_result():
    rng = np.random.default_rng(42)
    n = 504  # ~2 years
    dates = pd.bdate_range("2020-01-01", periods=n)
    prices = pd.DataFrame({
        "A": 100 * np.cumprod(1 + rng.normal(0.0003, 0.01, n)),
        "B": 50 * np.cumprod(1 + rng.normal(0.0002, 0.015, n)),
    }, index=dates)

    def alloc(ctx: BacktestContext) -> dict[str, float]:
        return {"A": 0.6, "B": 0.4}

    labels = np.zeros(n, dtype=int)
    labels[n // 2:] = 1

    return run_backtest(prices, alloc, regime_labels=labels)


class TestPerformanceSummary:

    def test_returns_dataframe(self, backtest_result) -> None:
        summary = performance_summary(backtest_result)
        assert isinstance(summary, pd.DataFrame)
        assert "Value" in summary.columns

    def test_expected_rows(self, backtest_result) -> None:
        summary = performance_summary(backtest_result)
        assert "Sharpe Ratio" in summary.index
        assert "Max Drawdown" in summary.index
        assert "Total Trades" in summary.index


class TestMonthlyReturns:

    def test_has_12_columns(self, backtest_result) -> None:
        table = monthly_returns_table(backtest_result)
        assert table.shape[1] == 12

    def test_column_names(self, backtest_result) -> None:
        table = monthly_returns_table(backtest_result)
        assert table.columns[0] == "Jan"
        assert table.columns[11] == "Dec"


class TestRollingMetrics:

    def test_returns_dataframe(self, backtest_result) -> None:
        rm = rolling_metrics(backtest_result, window=63)
        assert isinstance(rm, pd.DataFrame)
        assert "rolling_sharpe" in rm.columns
        assert "rolling_vol" in rm.columns


class TestTradeAnalysis:

    def test_returns_dict(self, backtest_result) -> None:
        ta = trade_analysis(backtest_result)
        assert isinstance(ta, dict)
        assert "win_rate" in ta
        assert "profit_factor" in ta

    def test_known_win_rate(self) -> None:
        """Verify win rate on a trivially constructed result."""
        from quantlite.backtesting.engine import BacktestResult

        result = BacktestResult(
            portfolio_value=pd.Series([100, 101]),
            weights_over_time=pd.DataFrame({"A": [1.0, 1.0]}),
            trades=[
                {"date": "2020-01-01", "asset": "A", "old_weight": 0.0, "new_weight": 1.0, "cost": 0.0},
                {"date": "2020-01-02", "asset": "A", "old_weight": 1.0, "new_weight": 0.5, "cost": 0.0},
            ],
            metrics={},
            drawdown_series=pd.Series([0, 0]),
        )
        ta = trade_analysis(result)
        # First trade: delta +1.0 (win), second: delta -0.5 (loss)
        assert abs(ta["win_rate"] - 0.5) < 1e-6


class TestRegimeAttribution:

    def test_returns_dataframe(self, backtest_result) -> None:
        ra = regime_attribution(backtest_result)
        assert isinstance(ra, pd.DataFrame)
        assert len(ra) == 2  # two regimes

    def test_pct_time_sums_to_one(self, backtest_result) -> None:
        ra = regime_attribution(backtest_result)
        assert abs(ra["pct_time"].sum() - 1.0) < 0.01
