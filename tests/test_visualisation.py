"""Tests for quantlite.visualisation (legacy charts)."""

import matplotlib
matplotlib.use("Agg")

import pandas as pd
import pytest

from quantlite.visualisation import (
    plot_equity_curve,
    plot_multiple_equity_curves,
    plot_ohlc,
    plot_return_distribution,
    plot_time_series,
)


def test_plot_time_series():
    data = pd.Series([100, 101, 102], index=[0, 1, 2])
    plot_time_series(data, title="Test Time Series")


def test_plot_ohlc():
    df = pd.DataFrame(
        {"Open": [100, 102, 101], "High": [103, 104, 103],
         "Low": [99, 100, 99], "Close": [102, 101, 102]},
        index=pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03"]),
    )
    plot_ohlc(df, title="Test OHLC", type="candle", volume=False)


def test_plot_return_distribution():
    plot_return_distribution([0.01, -0.02, 0.03, 0.0, 0.01])


def test_plot_equity_curve():
    eq = pd.Series([10000, 10200, 10150], index=[0, 1, 2])
    plot_equity_curve(eq, drawdowns=True)


def test_plot_multiple_equity_curves():
    eq1 = pd.Series([10000, 10100, 10300], index=[0, 1, 2])
    eq2 = pd.Series([10000, 9900, 9950], index=[0, 1, 2])
    plot_multiple_equity_curves({"A": eq1, "B": eq2}, rolling_sharpe=False)
