"""Tests for quantlite.metrics."""

from quantlite.metrics import (
    annualised_return,
    annualised_volatility,
    max_drawdown,
    sharpe_ratio,
)


def test_annualised_return_zero():
    assert annualised_return([0.0] * 252) == 0.0


def test_annualised_return_positive():
    result = annualised_return([0.01] * 252)
    assert result > 1.0


def test_annualised_volatility():
    vol = annualised_volatility([0.01, -0.01, 0.02, 0.0])
    assert vol >= 0.0


def test_sharpe_ratio():
    sr = sharpe_ratio([0.01, 0.02, 0.03])
    assert sr > 0.0


def test_max_drawdown():
    md = max_drawdown([0.1, -0.05, 0.2, -0.3])
    assert md < 0.0
