"""Tests for quantlite.regime_integration module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantlite.regime_integration.portfolio import (
    regime_aware_weights,
    regime_filtered_backtest,
    regime_rebalance_signals,
)
from quantlite.regime_integration.reporting import (
    regime_comparison_table,
    regime_performance_attribution,
    regime_tearsheet,
)
from quantlite.regime_integration.risk import (
    regime_conditional_cvar,
    regime_conditional_var,
    regime_risk_summary,
    regime_transition_risk,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_returns():
    """Generate sample returns with known regime structure."""
    rng = np.random.RandomState(42)
    # Bull regime: positive drift, low vol
    bull = rng.normal(0.001, 0.01, 200)
    # Bear regime: negative drift, high vol
    bear = rng.normal(-0.002, 0.025, 100)
    # Crisis regime: large negative drift, very high vol
    crisis = rng.normal(-0.005, 0.04, 50)
    returns = np.concatenate([bull, bear, crisis])
    regimes = np.array([2] * 200 + [1] * 100 + [0] * 50)
    return returns, regimes


@pytest.fixture
def sample_returns_df():
    """Generate a multi-asset returns DataFrame with regimes."""
    rng = np.random.RandomState(42)
    n = 350
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    df = pd.DataFrame({
        "AAPL": rng.normal(0.001, 0.02, n),
        "GLD": rng.normal(0.0005, 0.008, n),
        "TLT": rng.normal(0.0003, 0.006, n),
        "BTC": rng.normal(0.002, 0.04, n),
    }, index=dates)
    regimes = np.array([2] * 200 + [1] * 100 + [0] * 50)
    return df, regimes


# ---------------------------------------------------------------------------
# Risk tests
# ---------------------------------------------------------------------------

class TestRegimeConditionalVar:
    def test_basic(self, sample_returns):
        returns, regimes = sample_returns
        result = regime_conditional_var(returns, regimes)
        assert isinstance(result, dict)
        assert len(result) == 3
        for key in ["0", "1", "2"]:
            assert key in result
            assert isinstance(result[key], float)
        # Crisis VaR should be more negative than bull VaR
        assert result["0"] < result["2"]

    def test_length_mismatch(self):
        with pytest.raises(ValueError, match="same length"):
            regime_conditional_var(np.zeros(10), np.zeros(5))

    def test_custom_alpha(self, sample_returns):
        returns, regimes = sample_returns
        var_1 = regime_conditional_var(returns, regimes, alpha=0.01)
        var_5 = regime_conditional_var(returns, regimes, alpha=0.05)
        # 1% VaR should be more extreme than 5%
        for key in var_1:
            assert var_1[key] <= var_5[key]


class TestRegimeConditionalCvar:
    def test_basic(self, sample_returns):
        returns, regimes = sample_returns
        result = regime_conditional_cvar(returns, regimes)
        assert isinstance(result, dict)
        # CVaR should be more extreme than VaR
        var_result = regime_conditional_var(returns, regimes)
        for key in result:
            assert result[key] <= var_result[key]


class TestRegimeRiskSummary:
    def test_basic(self, sample_returns):
        returns, regimes = sample_returns
        result = regime_risk_summary(returns, regimes)
        assert "overall" in result
        assert "0" in result
        for key in ["var", "cvar", "volatility", "skewness", "kurtosis", "count"]:
            assert key in result["overall"]
        # Crisis should have higher volatility
        assert result["0"]["volatility"] > result["2"]["volatility"]

    def test_counts(self, sample_returns):
        returns, regimes = sample_returns
        result = regime_risk_summary(returns, regimes)
        assert result["0"]["count"] == 50
        assert result["1"]["count"] == 100
        assert result["2"]["count"] == 200


class TestRegimeTransitionRisk:
    def test_basic(self):
        # Simple 3-state transition matrix
        tm = np.array([
            [0.8, 0.15, 0.05],
            [0.1, 0.7, 0.2],
            [0.05, 0.1, 0.85],
        ])
        result = regime_transition_risk(tm, current_regime=2)
        assert "1_step" in result
        assert "5_step" in result
        assert "21_step" in result
        assert 0 <= result["1_step"] <= 1
        assert 0 <= result["5_step"] <= 1
        assert 0 <= result["21_step"] <= 1

    def test_no_worse_regimes(self):
        tm = np.array([[0.9, 0.1], [0.2, 0.8]])
        result = regime_transition_risk(tm, current_regime=0)
        assert result["1_step"] == 0.0

    def test_custom_worse_regimes(self):
        tm = np.array([
            [0.8, 0.15, 0.05],
            [0.1, 0.7, 0.2],
            [0.05, 0.1, 0.85],
        ])
        result = regime_transition_risk(tm, current_regime=2, worse_regimes=[0])
        assert result["1_step"] == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# Portfolio tests
# ---------------------------------------------------------------------------

class TestRegimeAwareWeights:
    def test_equal_weight(self, sample_returns_df):
        df, regimes = sample_returns_df
        w = regime_aware_weights(df, regimes, method="equal_weight", defensive_tilt=0.0)
        assert len(w) == 4
        assert pytest.approx(sum(w.values()), abs=1e-10) == 1.0

    def test_defensive_tilt(self, sample_returns_df):
        df, regimes = sample_returns_df
        # Current regime is 0 (crisis), so defensive tilt should apply
        w_tilted = regime_aware_weights(df, regimes, method="equal_weight", defensive_tilt=0.3)
        w_no_tilt = regime_aware_weights(df, regimes, method="equal_weight", defensive_tilt=0.0)
        # GLD and TLT should get higher weight with tilt
        assert w_tilted["GLD"] >= w_no_tilt["GLD"]
        assert w_tilted["TLT"] >= w_no_tilt["TLT"]

    def test_min_variance(self, sample_returns_df):
        df, regimes = sample_returns_df
        w = regime_aware_weights(df, regimes, method="min_variance", defensive_tilt=0.0)
        assert pytest.approx(sum(w.values()), abs=1e-6) == 1.0

    def test_hrp(self, sample_returns_df):
        df, regimes = sample_returns_df
        w = regime_aware_weights(df, regimes, method="hrp")
        assert pytest.approx(sum(w.values()), abs=1e-10) == 1.0

    def test_unknown_method(self, sample_returns_df):
        df, regimes = sample_returns_df
        with pytest.raises(ValueError, match="Unknown method"):
            regime_aware_weights(df, regimes, method="magic")

    def test_empty_df(self, sample_returns_df):
        _, regimes = sample_returns_df
        with pytest.raises(ValueError):
            regime_aware_weights(pd.DataFrame(), regimes)


class TestRegimeRebalanceSignals:
    def test_basic(self):
        # Clear transition: 10 of regime 0, then 10 of regime 1
        regimes = np.array([0] * 10 + [1] * 10)
        signals = regime_rebalance_signals(regimes, lookback=5)
        assert len(signals) >= 1
        assert signals[0]["from_regime"] == 0
        assert signals[0]["to_regime"] == 1

    def test_no_transition(self):
        regimes = np.array([0] * 20)
        signals = regime_rebalance_signals(regimes, lookback=5)
        assert len(signals) == 0

    def test_noisy_transition(self):
        # Transition that doesn't persist long enough
        regimes = np.array([0] * 10 + [1, 0, 1, 0] + [0] * 6)
        signals = regime_rebalance_signals(regimes, lookback=5)
        assert len(signals) == 0


class TestRegimeFilteredBacktest:
    def test_basic(self, sample_returns_df):
        df, regimes = sample_returns_df
        weights_by_regime = {
            0: {"AAPL": 0.1, "GLD": 0.3, "TLT": 0.5, "BTC": 0.1},
            1: {"AAPL": 0.25, "GLD": 0.25, "TLT": 0.25, "BTC": 0.25},
            2: {"AAPL": 0.4, "GLD": 0.1, "TLT": 0.1, "BTC": 0.4},
        }
        result = regime_filtered_backtest(df, weights_by_regime, regimes)
        assert "equity_curve" in result
        assert "regime_attribution" in result
        assert "total_return" in result
        assert len(result["equity_curve"]) == len(df)

    def test_length_mismatch(self, sample_returns_df):
        df, _ = sample_returns_df
        with pytest.raises(ValueError):
            regime_filtered_backtest(df, {}, np.zeros(5))


# ---------------------------------------------------------------------------
# Reporting tests
# ---------------------------------------------------------------------------

class TestRegimeTearsheet:
    def test_basic(self, sample_returns):
        returns, regimes = sample_returns
        ts = regime_tearsheet(returns, regimes)
        assert "equity_curve" in ts
        assert "drawdowns" in ts
        assert "regime_metrics" in ts
        assert "time_in_regime" in ts
        assert "overall_metrics" in ts
        assert "overall" in ts["regime_metrics"]

    def test_with_benchmark(self, sample_returns):
        returns, regimes = sample_returns
        benchmark = np.random.RandomState(99).normal(0.0005, 0.012, len(returns))
        ts = regime_tearsheet(returns, regimes, benchmark=benchmark)
        assert "benchmark_metrics" in ts

    def test_time_in_regime(self, sample_returns):
        returns, regimes = sample_returns
        ts = regime_tearsheet(returns, regimes)
        total_time = sum(ts["time_in_regime"].values())
        assert pytest.approx(total_time) == 1.0


class TestRegimePerformanceAttribution:
    def test_basic(self, sample_returns):
        returns, regimes = sample_returns
        attr = regime_performance_attribution(returns, regimes)
        assert len(attr) == 3
        for key in attr:
            assert "cumulative_return" in attr[key]
            assert "contribution_pct" in attr[key]
            assert "mean_return" in attr[key]
            assert "count" in attr[key]

    def test_contributions_sum_to_100(self, sample_returns):
        returns, regimes = sample_returns
        attr = regime_performance_attribution(returns, regimes)
        total_pct = sum(v["contribution_pct"] for v in attr.values())
        assert pytest.approx(total_pct, abs=0.1) == 100.0


class TestRegimeComparisonTable:
    def test_basic(self, sample_returns):
        returns, regimes = sample_returns
        table = regime_comparison_table(returns, regimes)
        assert isinstance(table, str)
        assert "| Regime" in table
        assert "overall" in table
        lines = table.strip().split("\n")
        assert len(lines) >= 5  # header + separator + 3 regimes + overall
