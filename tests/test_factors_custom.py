"""Tests for quantlite.factors.custom module."""

import numpy as np
import pandas as pd
import pytest

from quantlite.factors.custom import (
    CustomFactor,
    factor_correlation_matrix,
    factor_decay,
    factor_portfolio,
    test_factor_significance as check_factor_significance,
)


@pytest.fixture
def rng():
    return np.random.RandomState(42)


class TestCustomFactor:
    def test_from_array(self):
        vals = [1.0, 2.0, 3.0]
        f = CustomFactor("test", vals)
        assert f.name == "test"
        assert len(f) == 3
        np.testing.assert_array_equal(f.values, [1.0, 2.0, 3.0])

    def test_from_series(self):
        s = pd.Series([1.0, 2.0, 3.0], name="x")
        f = CustomFactor("test", s)
        assert len(f) == 3

    def test_mean_std(self, rng):
        vals = rng.normal(0, 1, 100)
        f = CustomFactor("test", vals)
        assert abs(f.mean() - np.mean(vals)) < 1e-10
        assert abs(f.std() - np.std(vals, ddof=1)) < 1e-10

    def test_correlation(self, rng):
        a = rng.normal(0, 1, 100)
        b = a + rng.normal(0, 0.1, 100)
        fa = CustomFactor("a", a)
        fb = CustomFactor("b", b)
        assert fa.correlation(fb) > 0.9

    def test_repr(self):
        f = CustomFactor("momentum", [1.0, 2.0])
        assert "momentum" in repr(f)
        assert "n=2" in repr(f)


class TestFactorSignificance:
    def test_significant_factor(self, rng):
        n = 500
        factor = rng.normal(0, 1, n)
        returns = 0.5 * factor + rng.normal(0, 0.5, n)

        result = check_factor_significance(returns, factor)
        assert result["t_pvalue"] < 0.05
        assert result["f_pvalue"] < 0.05
        assert abs(result["beta"] - 0.5) < 0.15

    def test_insignificant_factor(self, rng):
        n = 500
        factor = rng.normal(0, 1, n)
        returns = rng.normal(0, 1, n)

        result = check_factor_significance(returns, factor)
        assert result["r_squared_full"] < 0.05

    def test_with_controls(self, rng):
        n = 500
        control = rng.normal(0, 1, n)
        test_factor = rng.normal(0, 1, n)
        returns = 0.8 * control + 0.3 * test_factor + rng.normal(0, 0.5, n)

        result = check_factor_significance(
            returns, test_factor, control_factors=[control]
        )
        assert result["marginal_r_squared"] > 0
        assert result["t_pvalue"] < 0.05

    def test_with_custom_factor(self, rng):
        n = 200
        f = CustomFactor("test", rng.normal(0, 1, n))
        returns = 0.5 * f.values + rng.normal(0, 0.5, n)

        result = check_factor_significance(returns, f)
        assert result["t_pvalue"] < 0.05

    def test_output_keys(self, rng):
        n = 100
        result = check_factor_significance(
            rng.normal(0, 1, n), rng.normal(0, 1, n)
        )
        for key in [
            "t_stat", "t_pvalue", "f_stat", "f_pvalue", "beta",
            "r_squared_full", "r_squared_restricted", "marginal_r_squared",
        ]:
            assert key in result


class TestFactorCorrelationMatrix:
    def test_basic(self, rng):
        factors = [
            CustomFactor("a", rng.normal(0, 1, 100)),
            CustomFactor("b", rng.normal(0, 1, 100)),
            CustomFactor("c", rng.normal(0, 1, 100)),
        ]
        result = factor_correlation_matrix(factors)
        assert result["matrix"].shape == (3, 3)
        assert result["names"] == ["a", "b", "c"]
        # Diagonal should be 1
        np.testing.assert_allclose(np.diag(result["matrix"]), 1.0, atol=1e-10)

    def test_high_correlation_detected(self, rng):
        a = rng.normal(0, 1, 200)
        b = a + rng.normal(0, 0.1, 200)
        factors = [CustomFactor("a", a), CustomFactor("b", b)]
        result = factor_correlation_matrix(factors)
        assert result["max_offdiag"] > 0.9
        assert len(result["pairs"]) > 0

    def test_raw_arrays(self, rng):
        factors = [rng.normal(0, 1, 50), rng.normal(0, 1, 50)]
        result = factor_correlation_matrix(factors)
        assert result["matrix"].shape == (2, 2)


class TestFactorPortfolio:
    def test_basic(self, rng):
        n_assets = 50
        n_periods = 100
        returns_df = pd.DataFrame(
            rng.normal(0.001, 0.02, (n_periods, n_assets)),
            columns=[f"asset_{i}" for i in range(n_assets)],
        )
        factor_values = rng.normal(0, 1, n_assets)

        result = factor_portfolio(returns_df, factor_values, n_quantiles=5)
        assert result["n_quantiles"] == 5
        assert len(result["quantile_returns"]) == 5
        assert "spread" in result
        assert "monotonic" in result

    def test_quantile_count(self, rng):
        n_assets = 30
        returns_df = pd.DataFrame(rng.normal(0, 0.01, (50, n_assets)))
        factor_values = rng.normal(0, 1, n_assets)

        for nq in [3, 5, 10]:
            result = factor_portfolio(returns_df, factor_values, n_quantiles=nq)
            assert result["n_quantiles"] == nq

    def test_spread_direction(self, rng):
        # Factor that predicts returns should show positive spread
        n_assets = 100
        factor_values = np.linspace(-1, 1, n_assets)
        # Returns positively related to factor
        mean_rets = 0.01 * factor_values + rng.normal(0, 0.001, n_assets)
        returns_df = pd.DataFrame(
            np.tile(mean_rets, (50, 1)),
            columns=[f"a{i}" for i in range(n_assets)],
        )
        result = factor_portfolio(returns_df, factor_values)
        assert result["spread"] > 0


class TestFactorDecay:
    def test_basic(self, rng):
        n = 300
        factor = rng.normal(0, 1, n)
        # Returns partially predicted by lagged factor
        returns = np.zeros(n)
        returns[1:] = 0.3 * factor[:-1] + rng.normal(0, 0.5, n - 1)

        result = factor_decay(returns, factor, max_lag=10)
        assert len(result["decay_curve"]) == 10
        assert len(result["r_squared_curve"]) == 10

    def test_first_lag_strongest(self, rng):
        n = 500
        factor = rng.normal(0, 1, n)
        returns = np.zeros(n)
        returns[1:] = 0.5 * factor[:-1] + rng.normal(0, 0.5, n - 1)

        result = factor_decay(returns, factor)
        # First lag should have highest absolute correlation
        first_corr = abs(result["decay_curve"][0][1])
        for lag, corr in result["decay_curve"][1:]:
            assert abs(corr) <= first_corr + 0.05  # small tolerance

    def test_with_custom_factor(self, rng):
        n = 200
        f = CustomFactor("test", rng.normal(0, 1, n))
        returns = rng.normal(0, 1, n)
        result = factor_decay(returns, f, max_lag=5)
        assert len(result["decay_curve"]) == 5

    def test_short_series(self):
        result = factor_decay([1.0, 2.0, 3.0], [0.5, 1.0, 1.5], max_lag=5)
        assert len(result["decay_curve"]) <= 2
