"""Tests for quantlite.factors.classical module."""

import numpy as np
import pytest

from quantlite.factors.classical import (
    carhart_four,
    factor_attribution,
    factor_summary,
    fama_french_five,
    fama_french_three,
)


@pytest.fixture
def synthetic_factor_data():
    """Generate realistic synthetic factor data."""
    rng = np.random.RandomState(42)
    n = 500

    # Factors
    market = rng.normal(0.0005, 0.01, n)
    smb = rng.normal(0.0002, 0.005, n)
    hml = rng.normal(0.0001, 0.004, n)
    mom = rng.normal(0.0003, 0.006, n)
    rmw = rng.normal(0.0001, 0.003, n)
    cma = rng.normal(0.0001, 0.003, n)

    # Asset returns with known betas
    alpha = 0.0001
    noise = rng.normal(0, 0.003, n)
    returns = alpha + 1.1 * market + 0.3 * smb - 0.2 * hml + noise

    return {
        "returns": returns,
        "market": market,
        "smb": smb,
        "hml": hml,
        "mom": mom,
        "rmw": rmw,
        "cma": cma,
        "n": n,
    }


class TestFamaFrenchThree:
    def test_basic_output(self, synthetic_factor_data):
        d = synthetic_factor_data
        result = fama_french_three(d["returns"], d["market"], d["smb"], d["hml"])

        assert "alpha" in result
        assert "betas" in result
        assert "r_squared" in result
        assert "t_stats" in result
        assert "p_values" in result
        assert "residuals" in result

    def test_beta_recovery(self, synthetic_factor_data):
        d = synthetic_factor_data
        result = fama_french_three(d["returns"], d["market"], d["smb"], d["hml"])

        # Should approximately recover true betas
        assert abs(result["betas"]["market"] - 1.1) < 0.15
        assert abs(result["betas"]["smb"] - 0.3) < 0.15
        assert abs(result["betas"]["hml"] - (-0.2)) < 0.15

    def test_r_squared_reasonable(self, synthetic_factor_data):
        d = synthetic_factor_data
        result = fama_french_three(d["returns"], d["market"], d["smb"], d["hml"])

        assert 0.5 < result["r_squared"] < 1.0
        assert result["adj_r_squared"] <= result["r_squared"]

    def test_residuals_length(self, synthetic_factor_data):
        d = synthetic_factor_data
        result = fama_french_three(d["returns"], d["market"], d["smb"], d["hml"])
        assert len(result["residuals"]) == d["n"]

    def test_t_stats_structure(self, synthetic_factor_data):
        d = synthetic_factor_data
        result = fama_french_three(d["returns"], d["market"], d["smb"], d["hml"])

        assert "alpha" in result["t_stats"]
        assert "market" in result["t_stats"]
        assert "smb" in result["t_stats"]
        assert "hml" in result["t_stats"]

    def test_market_beta_significant(self, synthetic_factor_data):
        d = synthetic_factor_data
        result = fama_french_three(d["returns"], d["market"], d["smb"], d["hml"])
        assert result["p_values"]["market"] < 0.05


class TestFamaFrenchFive:
    def test_basic_output(self, synthetic_factor_data):
        d = synthetic_factor_data
        result = fama_french_five(
            d["returns"], d["market"], d["smb"], d["hml"], d["rmw"], d["cma"]
        )
        assert "betas" in result
        assert "rmw" in result["betas"]
        assert "cma" in result["betas"]
        assert len(result["betas"]) == 5

    def test_five_factor_r_squared(self, synthetic_factor_data):
        d = synthetic_factor_data
        r3 = fama_french_three(d["returns"], d["market"], d["smb"], d["hml"])
        r5 = fama_french_five(
            d["returns"], d["market"], d["smb"], d["hml"], d["rmw"], d["cma"]
        )
        # Five factor R-squared should be >= three factor
        assert r5["r_squared"] >= r3["r_squared"] - 0.001


class TestCarhartFour:
    def test_basic_output(self, synthetic_factor_data):
        d = synthetic_factor_data
        result = carhart_four(d["returns"], d["market"], d["smb"], d["hml"], d["mom"])
        assert "mom" in result["betas"]
        assert len(result["betas"]) == 4

    def test_momentum_exposure(self, synthetic_factor_data):
        d = synthetic_factor_data
        # Build returns with known momentum exposure
        rng = np.random.RandomState(99)
        n = 500
        noise = rng.normal(0, 0.002, n)
        ret_mom = 0.5 * d["mom"] + noise

        result = carhart_four(ret_mom, d["market"], d["smb"], d["hml"], d["mom"])
        assert abs(result["betas"]["mom"] - 0.5) < 0.2


class TestFactorAttribution:
    def test_basic_output(self, synthetic_factor_data):
        d = synthetic_factor_data
        result = factor_attribution(
            d["returns"],
            [d["market"], d["smb"], d["hml"]],
            ["market", "smb", "hml"],
        )
        assert "alpha" in result
        assert "factor_contributions" in result
        assert "unexplained" in result
        assert "r_squared" in result
        assert "total_return" in result

    def test_contributions_sum(self, synthetic_factor_data):
        d = synthetic_factor_data
        result = factor_attribution(
            d["returns"],
            [d["market"], d["smb"], d["hml"]],
            ["market", "smb", "hml"],
        )
        total = sum(result["factor_contributions"].values()) + result["unexplained"]
        assert abs(total - result["total_return"]) < 1e-8


class TestFactorSummary:
    def test_basic_output(self, synthetic_factor_data):
        d = synthetic_factor_data
        result = factor_summary(
            d["returns"],
            [d["market"], d["smb"], d["hml"]],
            ["market", "smb", "hml"],
        )
        assert "alpha" in result
        assert "alpha_t" in result
        assert "alpha_p" in result
        assert "betas" in result
        assert "t_stats" in result
        assert "p_values" in result
        assert "r_squared" in result
        assert "adj_r_squared" in result
        assert result["n_obs"] == d["n"]

    def test_consistency_with_three_factor(self, synthetic_factor_data):
        d = synthetic_factor_data
        r3 = fama_french_three(d["returns"], d["market"], d["smb"], d["hml"])
        summary = factor_summary(
            d["returns"],
            [d["market"], d["smb"], d["hml"]],
            ["market", "smb", "hml"],
        )
        assert abs(r3["alpha"] - summary["alpha"]) < 1e-10
        assert abs(r3["r_squared"] - summary["r_squared"]) < 1e-10


class TestEdgeCases:
    def test_short_series(self):
        rng = np.random.RandomState(10)
        n = 10
        returns = rng.normal(0, 0.01, n)
        market = rng.normal(0, 0.01, n)
        smb = rng.normal(0, 0.005, n)
        hml = rng.normal(0, 0.004, n)
        result = fama_french_three(returns, market, smb, hml)
        assert "alpha" in result

    def test_constant_returns(self):
        n = 100
        returns = np.ones(n) * 0.001
        market = np.random.RandomState(1).normal(0, 0.01, n)
        smb = np.random.RandomState(2).normal(0, 0.005, n)
        hml = np.random.RandomState(3).normal(0, 0.004, n)
        result = fama_french_three(returns, market, smb, hml)
        # R-squared can be negative for constant returns; just check it runs
        assert "r_squared" in result
