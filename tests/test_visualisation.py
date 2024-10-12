import pytest
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
from quantlite.visualisation import (
    plot_time_series,
    plot_ohlc,
    plot_return_distribution,
    plot_equity_curve,
    plot_multiple_equity_curves
)

def test_plot_time_series():
    data = pd.Series([100, 101, 102], index=[0,1,2])
    plot_time_series(data, title="Test Time Series")
    assert True

def test_plot_ohlc():
    # MUST have DatetimeIndex for mplfinance
    df = pd.DataFrame({
        "Open": [100, 102, 101],
        "High": [103, 104, 103],
        "Low":  [99, 100, 99],
        "Close":[102, 101, 102]
    }, index=pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03"]))

    plot_ohlc(df, title="Test OHLC", type="candle", volume=False)
    assert True

def test_plot_return_distribution():
    returns = [0.01, -0.02, 0.03, 0.0, 0.01]
    plot_return_distribution(returns, title="Test Return Dist")
    assert True

def test_plot_equity_curve():
    eq = pd.Series([10000, 10200, 10150], index=[0,1,2])
    plot_equity_curve(eq, drawdowns=True)
    assert True

def test_plot_multiple_equity_curves():
    eq1 = pd.Series([10000, 10100, 10300], index=[0,1,2])
    eq2 = pd.Series([10000, 9900, 9950], index=[0,1,2])
    curves = {"Strategy A": eq1, "Strategy B": eq2}
    plot_multiple_equity_curves(curves_dict=curves, rolling_sharpe=False)
    assert True
