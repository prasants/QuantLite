"""Tests for quantlite.risk.metrics."""

import math

import numpy as np
import pytest

from quantlite.risk.metrics import (
    calmar_ratio,
    cvar,
    max_drawdown_duration,
    omega_ratio,
    return_moments,
    sortino_ratio,
    tail_ratio,
    value_at_risk,
)


@pytest.fixture()
def sample_returns():
    """Deterministic returns for reproducible tests."""
    rng = np.random.default_rng(42)
    return rng.normal(0.0005, 0.02, size=500)


class TestValueAtRisk:
    def test_historical(self, sample_returns):
        var = value_at_risk(sample_returns, alpha=0.05, method="historical")
        assert var < 0, "VaR should be negative (a loss)"
        assert var > -0.2, "VaR should be reasonable"

    def test_parametric(self, sample_returns):
        var = value_at_risk(sample_returns, alpha=0.05, method="parametric")
        assert var < 0

    def test_cornish_fisher(self, sample_returns):
        var = value_at_risk(sample_returns, alpha=0.05, method="cornish-fisher")
        assert var < 0

    def test_unknown_method_raises(self, sample_returns):
        with pytest.raises(ValueError, match="Unknown method"):
            value_at_risk(sample_returns, method="magic")

    def test_insufficient_data(self):
        with pytest.raises(ValueError, match="at least 2"):
            value_at_risk([0.01])

    def test_handles_nan(self):
        data = [0.01, -0.02, float("nan"), 0.03, -0.01, 0.005, -0.015, 0.02, -0.03, 0.01]
        var = value_at_risk(data, alpha=0.05)
        assert math.isfinite(var)


class TestCVaR:
    def test_cvar_worse_than_var(self, sample_returns):
        var = value_at_risk(sample_returns, alpha=0.05)
        es = cvar(sample_returns, alpha=0.05)
        assert es <= var, "CVaR must be at least as bad as VaR"

    def test_insufficient_data(self):
        with pytest.raises(ValueError):
            cvar([0.01])


class TestSortino:
    def test_positive_returns(self):
        rets = [0.01] * 252
        s = sortino_ratio(rets, freq=252)
        assert s == float("inf"), "No downside returns means infinite Sortino"

    def test_mixed_returns(self, sample_returns):
        s = sortino_ratio(sample_returns)
        assert math.isfinite(s)


class TestCalmar:
    def test_basic(self, sample_returns):
        c = calmar_ratio(sample_returns)
        assert math.isfinite(c)

    def test_no_drawdown(self):
        rets = [0.01] * 10
        c = calmar_ratio(rets)
        assert c == float("inf")


class TestMaxDrawdownDuration:
    def test_basic(self, sample_returns):
        dd = max_drawdown_duration(sample_returns)
        assert dd.max_drawdown < 0
        assert dd.duration >= 0
        assert dd.start_idx <= dd.end_idx

    def test_empty(self):
        dd = max_drawdown_duration([])
        assert dd.max_drawdown == 0.0
        assert dd.duration == 0


class TestReturnMoments:
    def test_normal_returns(self, sample_returns):
        m = return_moments(sample_returns)
        assert abs(m.skewness) < 1, "Near-normal returns should have low skew"
        assert abs(m.kurtosis) < 2, "Near-normal returns should have low excess kurtosis"

    def test_insufficient_data(self):
        with pytest.raises(ValueError, match="at least 4"):
            return_moments([0.01, 0.02, 0.03])


class TestOmegaRatio:
    def test_all_positive(self):
        o = omega_ratio([0.01, 0.02, 0.03])
        assert o == float("inf")

    def test_mixed(self, sample_returns):
        o = omega_ratio(sample_returns)
        assert o > 0


class TestTailRatio:
    def test_symmetric_returns(self):
        rng = np.random.default_rng(99)
        rets = rng.normal(0, 0.01, 1000)
        tr = tail_ratio(rets, alpha=0.05)
        assert 0.5 < tr < 2.0, "Symmetric returns should have tail ratio near 1"
