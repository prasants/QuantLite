"""Tests for the multi-asset backtesting engine."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantlite.backtesting.engine import (
    BacktestConfig,
    BacktestContext,
    BacktestResult,
    RiskLimits,
    SlippageModel,
    run_backtest,
)


@pytest.fixture()
def price_data() -> pd.DataFrame:
    """Synthetic 3-asset price data."""
    rng = np.random.default_rng(42)
    n = 252
    dates = pd.bdate_range("2020-01-01", periods=n)
    prices = {
        "A": 100 * np.cumprod(1 + rng.normal(0.0003, 0.01, n)),
        "B": 50 * np.cumprod(1 + rng.normal(0.0002, 0.015, n)),
        "C": 200 * np.cumprod(1 + rng.normal(0.0004, 0.008, n)),
    }
    return pd.DataFrame(prices, index=dates)


def equal_alloc(ctx: BacktestContext) -> dict[str, float]:
    return {"A": 1 / 3, "B": 1 / 3, "C": 1 / 3}


class TestRunBacktest:

    def test_portfolio_value_positive(self, price_data: pd.DataFrame) -> None:
        result = run_backtest(price_data, equal_alloc)
        assert isinstance(result, BacktestResult)
        assert (result.portfolio_value > 0).all()

    def test_metrics_populated(self, price_data: pd.DataFrame) -> None:
        result = run_backtest(price_data, equal_alloc)
        assert "total_return" in result.metrics
        assert "sharpe_ratio" in result.metrics

    def test_trades_recorded(self, price_data: pd.DataFrame) -> None:
        result = run_backtest(price_data, equal_alloc)
        assert len(result.trades) > 0

    def test_fractional_shares(self, price_data: pd.DataFrame) -> None:
        config = BacktestConfig(fractional_shares=True, initial_capital=1000.0)
        result = run_backtest(price_data, equal_alloc, config=config)
        assert result.portfolio_value.iloc[-1] > 0

    def test_empty_df_raises(self) -> None:
        with pytest.raises(ValueError):
            run_backtest(pd.DataFrame(), equal_alloc)


class TestSlippage:

    def test_slippage_reduces_returns(self, price_data: pd.DataFrame) -> None:
        config_no_slip = BacktestConfig(
            slippage_model=None, fee_per_trade_pct=0.0
        )
        config_slip = BacktestConfig(
            slippage_model=SlippageModel(kind="fixed", spread_bps=50.0),
            fee_per_trade_pct=0.005,
        )
        result_no = run_backtest(price_data, equal_alloc, config=config_no_slip)
        result_yes = run_backtest(price_data, equal_alloc, config=config_slip)
        assert result_no.portfolio_value.iloc[-1] >= result_yes.portfolio_value.iloc[-1]


class TestRiskLimits:

    def test_circuit_breaker_fires(self) -> None:
        """Inject a crash and verify the circuit breaker activates."""
        np.random.default_rng(42)
        n = 100
        dates = pd.bdate_range("2020-01-01", periods=n)
        # Normal prices then crash
        prices_a = np.ones(n) * 100.0
        prices_a[50:] = 100 * np.cumprod(1 + np.full(50, -0.03))  # daily -3%
        prices = pd.DataFrame({
            "X": prices_a,
            "Y": np.ones(n) * 50.0,
        }, index=dates)

        def all_in_x(ctx: BacktestContext) -> dict[str, float]:
            return {"X": 1.0, "Y": 0.0}

        config = BacktestConfig(
            risk_limits=RiskLimits(max_drawdown=-0.15),
            fee_per_trade_pct=0.0,
        )
        result = run_backtest(prices, all_in_x, config=config)
        # After circuit breaker, weights should be zero
        final_weights = result.weights_over_time.iloc[-1]
        assert abs(final_weights.sum()) < 1e-6


class TestRegimeLabels:

    def test_regime_labels_passed_through(self, price_data: pd.DataFrame) -> None:
        labels = np.zeros(len(price_data), dtype=int)
        labels[len(labels) // 2:] = 1
        result = run_backtest(price_data, equal_alloc, regime_labels=labels)
        assert result.regime_labels is not None
        assert len(result.regime_labels) == len(price_data)
