"""Tests for quantlite.contagion module."""

import numpy as np
import pandas as pd
import pytest

from quantlite.contagion import (
    causal_network,
    covar,
    delta_covar,
    granger_causality,
    marginal_expected_shortfall,
    systemic_risk_contributions,
)


@pytest.fixture()
def correlated_returns():
    """Generate correlated return series."""
    rng = np.random.RandomState(42)
    n = 2000
    common = rng.normal(0, 0.01, n)
    a = common + rng.normal(0, 0.005, n)
    b = common + rng.normal(0, 0.005, n)
    return a, b


@pytest.fixture()
def returns_df():
    """Generate a DataFrame of correlated returns."""
    rng = np.random.RandomState(42)
    n = 500
    common = rng.normal(0, 0.01, n)
    data = {}
    for name in ["A", "B", "C"]:
        data[name] = common + rng.normal(0, 0.005, n)
    return pd.DataFrame(data)


class TestCoVaR:
    """Tests for CoVaR computation."""

    def test_covar_returns_dict(self, correlated_returns):
        a, b = correlated_returns
        result = covar(a, b)
        assert isinstance(result, dict)
        assert "covar" in result
        assert "var_a" in result
        assert "var_b" in result
        assert "delta_covar" in result

    def test_covar_quantile_method(self, correlated_returns):
        a, b = correlated_returns
        result = covar(a, b, method="quantile")
        # CoVaR should be negative (left tail)
        assert result["covar"] < 0

    def test_covar_regression_method(self, correlated_returns):
        a, b = correlated_returns
        result = covar(a, b, method="regression")
        assert result["covar"] < 0

    def test_covar_var_a_negative(self, correlated_returns):
        a, b = correlated_returns
        result = covar(a, b)
        assert result["var_a"] < 0

    def test_covar_invalid_method(self, correlated_returns):
        a, b = correlated_returns
        with pytest.raises(ValueError, match="Unknown method"):
            covar(a, b, method="invalid")

    def test_covar_different_alpha(self, correlated_returns):
        a, b = correlated_returns
        r1 = covar(a, b, alpha=0.01)
        r5 = covar(a, b, alpha=0.05)
        # 1% VaR should be more extreme than 5%
        assert r1["var_a"] < r5["var_a"]


class TestDeltaCoVaR:
    """Tests for delta CoVaR."""

    def test_delta_covar_returns_float(self, correlated_returns):
        a, b = correlated_returns
        result = delta_covar(a, b)
        assert isinstance(result, float)

    def test_delta_covar_correlated_is_negative(self, correlated_returns):
        """For positively correlated assets, delta CoVaR should be negative."""
        a, b = correlated_returns
        result = delta_covar(a, b)
        assert result < 0


class TestMES:
    """Tests for Marginal Expected Shortfall."""

    def test_mes_returns_float(self, correlated_returns):
        a, b = correlated_returns
        result = marginal_expected_shortfall(a, b)
        assert isinstance(result, float)

    def test_mes_negative_in_tail(self, correlated_returns):
        """MES should be negative for correlated assets."""
        a, b = correlated_returns
        result = marginal_expected_shortfall(a, b)
        assert result < 0

    def test_mes_worse_than_unconditional(self, correlated_returns):
        """MES should be worse than unconditional mean for correlated assets."""
        a, b = correlated_returns
        mes = marginal_expected_shortfall(a, b)
        assert mes < np.mean(b)

    def test_mes_independent_assets(self):
        """For independent assets, MES should be close to unconditional mean."""
        rng = np.random.RandomState(123)
        a = rng.normal(0, 0.01, 10000)
        b = rng.normal(0, 0.01, 10000)
        mes = marginal_expected_shortfall(a, b)
        # Should be close to 0 (unconditional mean)
        assert abs(mes) < 0.005


class TestSystemicRiskContributions:
    """Tests for systemic risk contributions."""

    def test_returns_dict(self, returns_df):
        result = systemic_risk_contributions(returns_df)
        assert isinstance(result, dict)
        assert len(result) == 3

    def test_all_assets_present(self, returns_df):
        result = systemic_risk_contributions(returns_df)
        for col in returns_df.columns:
            assert col in result

    def test_sorted_by_contribution(self, returns_df):
        result = systemic_risk_contributions(returns_df)
        values = list(result.values())
        assert values == sorted(values)


class TestGrangerCausality:
    """Tests for Granger causality."""

    def test_returns_both_directions(self):
        rng = np.random.RandomState(42)
        a = rng.normal(0, 1, 200)
        b = rng.normal(0, 1, 200)
        result = granger_causality(a, b, max_lag=3)
        assert "a_to_b" in result
        assert "b_to_a" in result

    def test_result_keys(self):
        rng = np.random.RandomState(42)
        a = rng.normal(0, 1, 200)
        b = rng.normal(0, 1, 200)
        result = granger_causality(a, b, max_lag=3)
        for key in ["f_statistic", "p_value", "direction", "optimal_lag"]:
            assert key in result["a_to_b"]
            assert key in result["b_to_a"]

    def test_known_causal_relationship(self):
        """When A causes B with a lag, we should detect it."""
        rng = np.random.RandomState(42)
        n = 1000
        a = rng.normal(0, 1, n)
        b = np.zeros(n)
        for i in range(1, n):
            b[i] = 0.8 * a[i - 1] + rng.normal(0, 0.3)
        result = granger_causality(a, b, max_lag=3)
        # A should Granger-cause B
        assert result["a_to_b"]["p_value"] < 0.05

    def test_no_causality_independent(self):
        """Independent series should not show causality."""
        rng = np.random.RandomState(42)
        a = rng.normal(0, 1, 500)
        b = rng.normal(0, 1, 500)
        result = granger_causality(a, b, max_lag=3)
        assert result["a_to_b"]["p_value"] > 0.01


class TestCausalNetwork:
    """Tests for causal network construction."""

    def test_returns_structure(self, returns_df):
        result = causal_network(returns_df, max_lag=2, significance=0.5)
        assert "edges" in result
        assert "adjacency_matrix" in result
        assert "nodes" in result

    def test_adjacency_shape(self, returns_df):
        result = causal_network(returns_df, max_lag=2)
        n = len(returns_df.columns)
        assert result["adjacency_matrix"].shape == (n, n)

    def test_no_self_loops(self, returns_df):
        result = causal_network(returns_df, max_lag=2, significance=0.99)
        adj = result["adjacency_matrix"]
        for i in range(adj.shape[0]):
            assert adj[i, i] == 0.0
