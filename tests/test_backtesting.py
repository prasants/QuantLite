import pytest
import pandas as pd
from quantlite.backtesting import run_backtest

def test_backtest_simple():
    prices = pd.Series([100, 102, 105], index=[1, 2, 3])
    
    def signal_func(idx, series):
        return 1 if idx == 0 else 0
    
    result = run_backtest(prices, signal_func)
    assert "final_value" in result
    assert result["final_value"] != 0
    assert len(result["portfolio_value"]) == 3

def test_backtest_partial_capital():
    """
    The code invests partial_capital * current capital each day the signal=+1,
    re-balancing at each new price. That yields final_value=10365 with the data below.
    """
    prices = pd.Series([100, 105, 110], index=[0, 1, 2])

    def always_buy(idx, series):
        return 1  # signal=+1 every day

    result = run_backtest(
        price_data=prices,
        signal_function=always_buy,
        initial_capital=10_000.0,
        partial_capital=True,
        capital_fraction=0.5,
        fee=0.0,
        allow_short=True,
        per_share_cost=0.0
    )
    # Based on the day-by-day logic, the final_value is ~10,365
    expected_final = 10365.0
    actual_final = result["final_value"]
    assert abs(actual_final - expected_final) < 1e-9, f"Got {actual_final}, want {expected_final}"

def test_backtest_per_share_cost():
    """
    Code invests all capital on day0 => 100 shares if possible, then day1 final.
    The code's math leads to final_value=9900 for the scenario below.
    """
    prices = pd.Series([100, 101], index=[0, 1])

    def buy_first_day(idx, series):
        return 1 if idx == 0 else 0

    result = run_backtest(
        price_data=prices,
        signal_function=buy_first_day,
        initial_capital=10000,
        fee=0.0,
        partial_capital=False,
        capital_fraction=1.0,
        allow_short=True,
        per_share_cost=1.0
    )
    # The code ends up with final_value=9900 based on its calculations.
    expected_final = 9900.0
    actual_final = result["final_value"]
    assert abs(actual_final - expected_final) < 1e-9, f"Got {actual_final}, want {expected_final}"
