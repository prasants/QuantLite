"""Tests for portfolio optimisation methods."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantlite.portfolio.optimisation import (
    PortfolioWeights,
    black_litterman,
    half_kelly,
    kelly_criterion,
    max_sharpe_weights,
    mean_cvar_weights,
    mean_variance_weights,
    min_variance_weights,
    risk_parity_weights,
    hrp_weights,
)


@pytest.fixture()
def returns_df() -> pd.DataFrame:
    """Synthetic 3-asset returns DataFrame."""
    rng = np.random.default_rng(42)
    n = 500
    data = {
        "A": rng.normal(0.0005, 0.01, n),
        "B": rng.normal(0.0003, 0.015, n),
        "C": rng.normal(0.0004, 0.008, n),
    }
    dates = pd.bdate_range("2020-01-01", periods=n)
    return pd.DataFrame(data, index=dates)


class TestMeanVariance:
    """Mean-variance optimisation tests."""

    def test_weights_sum_to_one(self, returns_df: pd.DataFrame) -> None:
        result = mean_variance_weights(returns_df)
        total = sum(result.weights.values())
        assert abs(total - 1.0) < 1e-6

    def test_long_only_constraint(self, returns_df: pd.DataFrame) -> None:
        result = mean_variance_weights(returns_df, long_only=True)
        for w in result.weights.values():
            assert w >= -1e-8

    def test_returns_portfolio_weights(self, returns_df: pd.DataFrame) -> None:
        result = mean_variance_weights(returns_df)
        assert isinstance(result, PortfolioWeights)
        assert result.method == "mean_variance"

    def test_with_target_return(self, returns_df: pd.DataFrame) -> None:
        result = mean_variance_weights(returns_df, target_return=0.05)
        assert isinstance(result, PortfolioWeights)
        assert abs(sum(result.weights.values()) - 1.0) < 1e-6


class TestMinVariance:
    """Minimum variance portfolio tests."""

    def test_weights_sum_to_one(self, returns_df: pd.DataFrame) -> None:
        result = min_variance_weights(returns_df)
        assert abs(sum(result.weights.values()) - 1.0) < 1e-6


class TestMeanCVaR:
    """Mean-CVaR optimisation tests."""

    def test_returns_valid_weights(self, returns_df: pd.DataFrame) -> None:
        result = mean_cvar_weights(returns_df, alpha=0.05)
        assert isinstance(result, PortfolioWeights)
        assert result.method == "mean_cvar"
        total = sum(result.weights.values())
        assert abs(total - 1.0) < 1e-6

    def test_long_only(self, returns_df: pd.DataFrame) -> None:
        result = mean_cvar_weights(returns_df, long_only=True)
        for w in result.weights.values():
            assert w >= -1e-8


class TestRiskParity:
    """Risk parity tests."""

    def test_weights_sum_to_one(self, returns_df: pd.DataFrame) -> None:
        result = risk_parity_weights(returns_df)
        assert abs(sum(result.weights.values()) - 1.0) < 1e-6


class TestHRP:
    """HRP convenience wrapper tests."""

    def test_weights_sum_to_one(self, returns_df: pd.DataFrame) -> None:
        result = hrp_weights(returns_df)
        assert abs(sum(result.weights.values()) - 1.0) < 1e-6
        assert result.method == "hrp"


class TestMaxSharpe:
    """Maximum Sharpe portfolio tests."""

    def test_weights_sum_to_one(self, returns_df: pd.DataFrame) -> None:
        result = max_sharpe_weights(returns_df)
        assert abs(sum(result.weights.values()) - 1.0) < 1e-6
        assert result.sharpe is not None


class TestBlackLitterman:
    """Black-Litterman model tests."""

    def test_returns_posterior(self, returns_df: pd.DataFrame) -> None:
        caps = {"A": 1e9, "B": 5e8, "C": 2e8}
        views = {"A": 0.10}
        confs = {"A": 0.8}
        post_mu, post_cov = black_litterman(returns_df, caps, views, confs)
        assert isinstance(post_mu, pd.Series)
        assert isinstance(post_cov, pd.DataFrame)
        assert len(post_mu) == 3
        assert post_cov.shape == (3, 3)


class TestKelly:
    """Kelly criterion tests."""

    def test_with_probabilities(self) -> None:
        k = kelly_criterion([], win_prob=0.6, win_loss_ratio=2.0)
        # f* = 0.6 - 0.4/2.0 = 0.4
        assert abs(k - 0.4) < 1e-10

    def test_returns_float(self) -> None:
        returns = np.random.default_rng(42).normal(0.001, 0.01, 200)
        k = kelly_criterion(returns)
        assert isinstance(k, float)

    def test_half_kelly(self) -> None:
        k_full = kelly_criterion([], win_prob=0.6, win_loss_ratio=2.0)
        k_half = half_kelly([], win_prob=0.6, win_loss_ratio=2.0)
        assert abs(k_half - k_full / 2) < 1e-10
