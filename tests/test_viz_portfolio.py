"""Smoke tests for portfolio visualisation functions."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from quantlite.backtesting.engine import BacktestContext, run_backtest
from quantlite.viz.portfolio import (
    plot_backtest_summary,
    plot_correlation_network,
    plot_efficient_frontier,
    plot_monthly_returns,
    plot_regime_performance,
    plot_risk_contribution,
    plot_weights_over_time,
)


@pytest.fixture()
def returns_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 252
    dates = pd.bdate_range("2020-01-01", periods=n)
    return pd.DataFrame({
        "A": rng.normal(0.0005, 0.01, n),
        "B": rng.normal(0.0003, 0.015, n),
        "C": rng.normal(0.0004, 0.008, n),
    }, index=dates)


@pytest.fixture()
def backtest_result(returns_df: pd.DataFrame):
    prices = (1 + returns_df).cumprod() * 100

    def alloc(ctx: BacktestContext) -> dict[str, float]:
        return {"A": 0.4, "B": 0.3, "C": 0.3}

    labels = np.zeros(len(prices), dtype=int)
    labels[len(labels) // 2:] = 1
    return run_backtest(prices, alloc, regime_labels=labels)


def test_plot_efficient_frontier(returns_df: pd.DataFrame) -> None:
    fig, ax = plot_efficient_frontier(returns_df, n_portfolios=200)
    assert fig is not None


def test_plot_weights_over_time(backtest_result) -> None:
    fig, ax = plot_weights_over_time(backtest_result)
    assert fig is not None


def test_plot_monthly_returns(backtest_result) -> None:
    fig, ax = plot_monthly_returns(backtest_result)
    assert fig is not None


def test_plot_backtest_summary(backtest_result) -> None:
    fig, axes = plot_backtest_summary(backtest_result)
    assert fig is not None
    assert len(axes) == 3


def test_plot_regime_performance(backtest_result) -> None:
    fig, axes = plot_regime_performance(backtest_result)
    assert fig is not None


def test_plot_risk_contribution(returns_df: pd.DataFrame) -> None:
    weights = {"A": 0.4, "B": 0.3, "C": 0.3}
    fig, ax = plot_risk_contribution(weights, returns_df)
    assert fig is not None


def test_plot_correlation_network(returns_df: pd.DataFrame) -> None:
    corr = returns_df.corr()
    fig, ax = plot_correlation_network(corr, threshold=0.1)
    assert fig is not None
