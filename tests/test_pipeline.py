"""Tests for quantlite.pipeline (Dream API)."""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from quantlite.pipeline import (
    backtest,
    construct_portfolio,
    detect_regimes,
    tearsheet,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def returns_df():
    """Multi-asset returns DataFrame with enough data for HMM."""
    rng = np.random.RandomState(42)
    n = 500
    dates = pd.date_range("2019-01-01", periods=n, freq="B")
    return pd.DataFrame({
        "AAPL": rng.normal(0.001, 0.02, n),
        "GLD": rng.normal(0.0003, 0.008, n),
        "TLT": rng.normal(0.0002, 0.006, n),
    }, index=dates)


# ---------------------------------------------------------------------------
# detect_regimes
# ---------------------------------------------------------------------------

class TestDetectRegimes:
    def test_basic(self, returns_df):
        regimes = detect_regimes(returns_df, n_regimes=2, rng_seed=42)
        assert len(regimes) == len(returns_df)
        assert set(regimes).issubset({0, 1})

    def test_three_regimes(self, returns_df):
        regimes = detect_regimes(returns_df, n_regimes=3, rng_seed=42)
        assert len(regimes) == len(returns_df)
        assert len(set(regimes)) <= 3

    def test_invalid_method(self, returns_df):
        with pytest.raises(ValueError, match="Unsupported method"):
            detect_regimes(returns_df, method="magic")

    def test_single_column(self, returns_df):
        single = returns_df[["AAPL"]]
        regimes = detect_regimes(single, n_regimes=2, rng_seed=42)
        assert len(regimes) == len(single)


# ---------------------------------------------------------------------------
# construct_portfolio
# ---------------------------------------------------------------------------

class TestConstructPortfolio:
    def test_equal_weight(self, returns_df):
        w = construct_portfolio(returns_df, method="equal_weight", regime_aware=False)
        assert pytest.approx(sum(w.values())) == 1.0
        assert len(w) == 3

    def test_min_variance(self, returns_df):
        w = construct_portfolio(returns_df, method="min_variance", regime_aware=False)
        assert pytest.approx(sum(w.values()), abs=1e-6) == 1.0

    def test_hrp(self, returns_df):
        w = construct_portfolio(returns_df, method="hrp", regime_aware=False)
        assert pytest.approx(sum(w.values()), abs=1e-10) == 1.0

    def test_regime_aware(self, returns_df):
        regimes = detect_regimes(returns_df, n_regimes=2, rng_seed=42)
        w = construct_portfolio(returns_df, regime_aware=True, regimes=regimes)
        assert pytest.approx(sum(w.values()), abs=1e-6) == 1.0

    def test_regime_aware_no_regimes(self, returns_df):
        with pytest.raises(ValueError, match="regimes must be provided"):
            construct_portfolio(returns_df, regime_aware=True)

    def test_unknown_method(self, returns_df):
        with pytest.raises(ValueError):
            construct_portfolio(returns_df, method="magic", regime_aware=False)


# ---------------------------------------------------------------------------
# backtest
# ---------------------------------------------------------------------------

class TestBacktest:
    def test_basic(self, returns_df):
        w = {"AAPL": 0.5, "GLD": 0.3, "TLT": 0.2}
        result = backtest(returns_df, w)
        assert "equity_curve" in result
        assert "returns" in result
        assert "total_return" in result
        assert "sharpe_ratio" in result
        assert "max_drawdown" in result
        assert len(result["equity_curve"]) == len(returns_df)
        assert result["max_drawdown"] <= 0

    def test_equal_weight(self, returns_df):
        w = {"AAPL": 1.0 / 3, "GLD": 1.0 / 3, "TLT": 1.0 / 3}
        result = backtest(returns_df, w)
        assert result["equity_curve"].iloc[-1] > 0


# ---------------------------------------------------------------------------
# tearsheet
# ---------------------------------------------------------------------------

class TestTearsheet:
    def test_basic(self, returns_df):
        w = {"AAPL": 0.5, "GLD": 0.3, "TLT": 0.2}
        result = backtest(returns_df, w)
        ts = tearsheet(result)
        assert "overall_metrics" in ts

    def test_with_regimes(self, returns_df):
        regimes = detect_regimes(returns_df, n_regimes=2, rng_seed=42)
        w = {"AAPL": 0.5, "GLD": 0.3, "TLT": 0.2}
        result = backtest(returns_df, w)
        ts = tearsheet(result, regimes=regimes)
        assert "regime_metrics" in ts
        assert "attribution" in ts
        assert "comparison_table" in ts

    def test_save(self, returns_df):
        w = {"AAPL": 0.5, "GLD": 0.3, "TLT": 0.2}
        result = backtest(returns_df, w)
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            path = f.name
        try:
            tearsheet(result, save=path)
            assert os.path.exists(path)
            with open(path) as f:
                content = f.read()
            assert "QuantLite" in content
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# Integration: full pipeline
# ---------------------------------------------------------------------------

class TestFullPipeline:
    def test_end_to_end(self, returns_df):
        """Test the full dream API pipeline with synthetic data."""
        regimes = detect_regimes(returns_df, n_regimes=2, rng_seed=42)
        weights = construct_portfolio(
            returns_df, method="hrp", regime_aware=True, regimes=regimes,
        )
        result = backtest(returns_df, weights)
        ts = tearsheet(result, regimes=regimes)
        assert "equity_curve" in ts
        assert "regime_metrics" in ts
        assert result["total_return"] != 0.0
