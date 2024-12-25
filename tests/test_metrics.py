import pytest
from quantlite.metrics import (
    annualised_return,
    annualised_volatility,
    sharpe_ratio,
    max_drawdown
)

def test_annualised_return_zero():
    data = [0.0] * 252
    assert annualised_return(data) == 0.0

def test_annualised_return_positive():
    data = [0.01] * 252
    result = annualised_return(data)
    assert result > 1.0  # 1% daily => huge annual return

def test_annualised_volatility():
    data = [0.01, -0.01, 0.02, 0.0]
    vol = annualised_volatility(data)
    assert vol >= 0.0

def test_sharpe_ratio():
    data = [0.01, 0.02, 0.03]
    sr = sharpe_ratio(data)
    assert sr > 0.0

def test_max_drawdown():
    data = [0.1, -0.05, 0.2, -0.3]
    md = max_drawdown(data)
    # It's negative
    assert md < 0.0

