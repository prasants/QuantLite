"""Tests for portfolio rebalancing strategies."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantlite.portfolio.rebalancing import (
    RebalanceResult,
    rebalance_calendar,
    rebalance_threshold,
    rebalance_tactical,
)


@pytest.fixture()
def returns_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 252
    data = {
        "A": rng.normal(0.0005, 0.01, n),
        "B": rng.normal(0.0003, 0.015, n),
    }
    dates = pd.bdate_range("2020-01-01", periods=n)
    return pd.DataFrame(data, index=dates)


def equal_weights(df: pd.DataFrame) -> dict[str, float]:
    n = df.shape[1]
    return {col: 1.0 / n for col in df.columns}


class TestCalendarRebalance:

    def test_monthly_fires_correct_intervals(self, returns_df: pd.DataFrame) -> None:
        result = rebalance_calendar(returns_df, equal_weights, freq="monthly")
        assert isinstance(result, RebalanceResult)
        assert result.n_rebalances >= 10  # ~12 months
        assert result.n_rebalances <= 15

    def test_daily_rebalance(self, returns_df: pd.DataFrame) -> None:
        result = rebalance_calendar(returns_df, equal_weights, freq="daily")
        assert result.n_rebalances == len(returns_df)

    def test_result_shape(self, returns_df: pd.DataFrame) -> None:
        result = rebalance_calendar(returns_df, equal_weights, freq="monthly")
        assert len(result.portfolio_returns) == len(returns_df)
        assert result.weights_over_time.shape == (len(returns_df), 2)


class TestThresholdRebalance:

    def test_triggers_on_drift(self, returns_df: pd.DataFrame) -> None:
        result = rebalance_threshold(returns_df, equal_weights, threshold=0.05)
        assert isinstance(result, RebalanceResult)
        assert result.n_rebalances >= 1

    def test_tight_threshold_more_rebalances(self, returns_df: pd.DataFrame) -> None:
        loose = rebalance_threshold(returns_df, equal_weights, threshold=0.20)
        tight = rebalance_threshold(returns_df, equal_weights, threshold=0.02)
        assert tight.n_rebalances >= loose.n_rebalances


class TestTacticalRebalance:

    def test_regime_change_triggers(self, returns_df: pd.DataFrame) -> None:
        # Create alternating regimes
        labels = np.zeros(len(returns_df), dtype=int)
        labels[len(labels) // 2:] = 1
        result = rebalance_tactical(returns_df, equal_weights, labels)
        assert isinstance(result, RebalanceResult)
        assert result.n_rebalances == 2  # initial + one regime change

    def test_mismatched_length_raises(self, returns_df: pd.DataFrame) -> None:
        labels = np.zeros(10)
        with pytest.raises(ValueError, match="length"):
            rebalance_tactical(returns_df, equal_weights, labels)
