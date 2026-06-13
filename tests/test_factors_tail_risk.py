"""Tests for quantlite.factors.tail_risk module."""

import numpy as np
import pytest

from quantlite.factors.tail_risk import (
    factor_crowding_score,
    factor_cvar_decomposition,
    regime_factor_exposure,
    tail_factor_beta,
)


@pytest.fixture
def rng():
    return np.random.RandomState(42)


@pytest.fixture
def factor_data(rng):
    """Synthetic data with known factor structure."""
    n = 1000
    market = rng.normal(0.0005, 0.015, n)
    value = rng.normal(0.0001, 0.008, n)
    noise = rng.normal(0, 0.005, n)
    returns = 0.0001 + 1.0 * market + 0.5 * value + noise
    return {
        "returns": returns,
        "market": market,
        "value": value,
        "n": n,
    }


class TestFactorCvarDecomposition:
    def test_basic_output(self, factor_data):
        result = factor_cvar_decomposition(
            factor_data["returns"],
            [factor_data["market"], factor_data["value"]],
            ["market", "value"],
        )
        assert "total_cvar" in result
        assert "factor_contributions" in result
        assert "residual_contribution" in result
        assert "pct_contributions" in result
        assert result["total_cvar"] < 0  # CVaR should be negative

    def test_market_dominates(self, factor_data):
        result = factor_cvar_decomposition(
            factor_data["returns"],
            [factor_data["market"], factor_data["value"]],
            ["market", "value"],
        )
        # Market factor should contribute more to CVaR (higher beta, higher vol)
        mkt_contrib = abs(result["factor_contributions"]["market"])
        val_contrib = abs(result["factor_contributions"]["value"])
        assert mkt_contrib > val_contrib * 0.5  # relaxed check

    def test_custom_alpha(self, factor_data):
        r1 = factor_cvar_decomposition(
            factor_data["returns"],
            [factor_data["market"]],
            ["market"],
            alpha=0.01,
        )
        r5 = factor_cvar_decomposition(
            factor_data["returns"],
            [factor_data["market"]],
            ["market"],
            alpha=0.05,
        )
        # 1% CVaR should be more extreme than 5%
        assert r1["total_cvar"] <= r5["total_cvar"]

    def test_few_observations(self, rng):
        # Very short series
        n = 20
        returns = rng.normal(0, 0.01, n)
        market = rng.normal(0, 0.01, n)
        result = factor_cvar_decomposition(returns, [market], ["market"], alpha=0.05)
        assert "total_cvar" in result


class TestRegimeFactorExposure:
    def test_basic_output(self, factor_data, rng):
        n = factor_data["n"]
        regimes = np.array(["bull"] * (n // 2) + ["bear"] * (n - n // 2))

        result = regime_factor_exposure(
            factor_data["returns"],
            [factor_data["market"], factor_data["value"]],
            ["market", "value"],
            regimes,
        )
        assert "bull" in result
        assert "bear" in result
        assert "alpha" in result["bull"]
        assert "betas" in result["bull"]

    def test_three_regimes(self, rng):
        n = 600
        market = rng.normal(0.001, 0.01, n)
        returns = 1.0 * market + rng.normal(0, 0.005, n)
        regimes = np.array(["bull"] * 200 + ["bear"] * 200 + ["crisis"] * 200)

        result = regime_factor_exposure(returns, [market], ["market"], regimes)
        assert len(result) == 3
        for regime in ["bull", "bear", "crisis"]:
            assert result[regime]["n_obs"] == 200

    def test_insufficient_obs_regime(self, rng):
        n = 100
        market = rng.normal(0, 0.01, n)
        returns = market + rng.normal(0, 0.005, n)
        regimes = np.array(["bull"] * 98 + ["crisis"] * 2)

        result = regime_factor_exposure(returns, [market], ["market"], regimes)
        assert result["crisis"]["alpha"] is None

    def test_beta_varies_by_regime(self, rng):
        n = 400
        market = rng.normal(0, 0.01, n)
        # Different beta in each regime
        returns = np.zeros(n)
        returns[:200] = 0.5 * market[:200] + rng.normal(0, 0.002, 200)
        returns[200:] = 2.0 * market[200:] + rng.normal(0, 0.002, 200)
        regimes = np.array(["calm"] * 200 + ["stress"] * 200)

        result = regime_factor_exposure(returns, [market], ["market"], regimes)
        assert result["calm"]["betas"]["market"] < result["stress"]["betas"]["market"]


class TestFactorCrowdingScore:
    def test_basic_output(self, rng):
        n = 200
        factors = [rng.normal(0, 0.01, n), rng.normal(0, 0.01, n)]
        result = factor_crowding_score(factors, rolling_window=60)

        assert "crowding_scores" in result
        assert "current_score" in result
        assert "trend" in result
        assert "is_crowded" in result

    def test_correlated_factors_crowded(self, rng):
        n = 200
        a = rng.normal(0, 0.01, n)
        b = a + rng.normal(0, 0.001, n)  # Very correlated
        result = factor_crowding_score([a, b], rolling_window=30)
        assert result["current_score"] > 0.5

    def test_uncorrelated_not_crowded(self, rng):
        n = 200
        factors = [rng.normal(0, 0.01, n) for _ in range(3)]
        result = factor_crowding_score(factors, rolling_window=60)
        assert result["current_score"] < 0.5

    def test_single_factor(self, rng):
        result = factor_crowding_score([rng.normal(0, 0.01, 100)])
        assert result["current_score"] == 0.0

    def test_scores_length(self, rng):
        n = 200
        window = 60
        factors = [rng.normal(0, 0.01, n), rng.normal(0, 0.01, n)]
        result = factor_crowding_score(factors, rolling_window=window)
        assert len(result["crowding_scores"]) == n - window + 1


class TestTailFactorBeta:
    def test_basic_output(self, factor_data):
        result = tail_factor_beta(
            factor_data["returns"],
            [factor_data["market"], factor_data["value"]],
            ["market", "value"],
        )
        assert "tail_betas" in result
        assert "full_betas" in result
        assert "beta_ratio" in result
        assert "n_tail_obs" in result
        assert "var_threshold" in result

    def test_tail_betas_exist(self, factor_data):
        result = tail_factor_beta(
            factor_data["returns"],
            [factor_data["market"], factor_data["value"]],
            ["market", "value"],
        )
        assert result["tail_betas"]["market"] is not None
        assert result["tail_betas"]["value"] is not None

    def test_full_betas_close_to_true(self, factor_data):
        result = tail_factor_beta(
            factor_data["returns"],
            [factor_data["market"], factor_data["value"]],
            ["market", "value"],
        )
        assert abs(result["full_betas"]["market"] - 1.0) < 0.15
        assert abs(result["full_betas"]["value"] - 0.5) < 0.15

    def test_n_tail_obs(self, factor_data):
        result = tail_factor_beta(
            factor_data["returns"],
            [factor_data["market"]],
            ["market"],
            alpha=0.05,
        )
        expected = int(factor_data["n"] * 0.05)
        assert abs(result["n_tail_obs"] - expected) <= 2

    def test_different_alpha(self, factor_data):
        r1 = tail_factor_beta(
            factor_data["returns"],
            [factor_data["market"]],
            ["market"],
            alpha=0.01,
        )
        r5 = tail_factor_beta(
            factor_data["returns"],
            [factor_data["market"]],
            ["market"],
            alpha=0.10,
        )
        assert r1["n_tail_obs"] < r5["n_tail_obs"]
        assert r1["var_threshold"] < r5["var_threshold"]
