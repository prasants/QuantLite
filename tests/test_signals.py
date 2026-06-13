"""Tests for backtesting signal generators."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantlite.backtesting.signals import (
    mean_reversion_signal,
    momentum_signal,
    regime_filter,
    trend_following,
    volatility_targeting,
)


@pytest.fixture()
def prices() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 200
    dates = pd.bdate_range("2020-01-01", periods=n)
    return pd.DataFrame({
        "A": 100 * np.cumprod(1 + rng.normal(0.001, 0.02, n)),
        "B": 50 * np.cumprod(1 + rng.normal(0.0005, 0.015, n)),
    }, index=dates)


@pytest.fixture()
def returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().dropna()


class TestMomentum:

    def test_correct_shape(self, prices: pd.DataFrame) -> None:
        sig = momentum_signal(prices, lookback=20)
        assert sig.shape == prices.shape

    def test_nan_for_short_history(self, prices: pd.DataFrame) -> None:
        sig = momentum_signal(prices, lookback=20)
        assert sig.iloc[:20].isna().all().all()


class TestMeanReversion:

    def test_correct_shape(self, prices: pd.DataFrame) -> None:
        sig = mean_reversion_signal(prices, lookback=20, z_threshold=1.5)
        assert sig.shape == prices.shape

    def test_values_in_range(self, prices: pd.DataFrame) -> None:
        sig = mean_reversion_signal(prices, lookback=20, z_threshold=1.5)
        valid = sig.dropna()
        assert set(valid.values.flatten()).issubset({-1.0, 0.0, 1.0})


class TestVolTargeting:

    def test_scales_exposure(self, returns: pd.DataFrame) -> None:
        scalar = volatility_targeting(returns, target_vol=0.10, lookback=60)
        assert scalar.shape == returns.shape
        # Should have some values > 0
        valid = scalar.dropna()
        assert (valid > 0).any().any()

    def test_capped_at_3x(self, returns: pd.DataFrame) -> None:
        scalar = volatility_targeting(returns, target_vol=0.50, lookback=60)
        valid = scalar.dropna()
        assert (valid <= 3.0).all().all()


class TestTrendFollowing:

    def test_correct_shape(self, prices: pd.DataFrame) -> None:
        sig = trend_following(prices, fast_window=10, slow_window=30)
        assert sig.shape == prices.shape

    def test_values_in_range(self, prices: pd.DataFrame) -> None:
        sig = trend_following(prices, fast_window=10, slow_window=30)
        valid = sig.dropna()
        assert set(valid.values.flatten()).issubset({-1.0, 1.0})


class TestRegimeFilter:

    def test_zeros_out_disallowed(self, prices: pd.DataFrame) -> None:
        sig = momentum_signal(prices, lookback=20)
        regimes = np.zeros(len(prices), dtype=int)
        regimes[100:] = 1

        filtered = regime_filter(sig, regimes, allowed_regimes=[0])
        # Everything in regime 1 (index 100+) should be zero
        assert (filtered.iloc[100:] == 0).all().all()
