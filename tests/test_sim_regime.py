"""Tests for quantlite.simulation.regime_mc."""

from __future__ import annotations

import numpy as np
import pytest

from quantlite.simulation.regime_mc import (
    regime_switching_simulation,
    reverse_stress_test,
    simulation_summary,
    stress_test_scenario,
)


@pytest.fixture()
def regime_params():
    return [
        {"mu": 0.0003, "sigma": 0.008},   # calm
        {"mu": -0.001, "sigma": 0.025},    # volatile
    ]


@pytest.fixture()
def transition_matrix():
    return np.array([
        [0.95, 0.05],
        [0.10, 0.90],
    ])


@pytest.fixture()
def sample_returns():
    rng = np.random.default_rng(42)
    return rng.standard_t(4, 500) * 0.01


class TestRegimeSwitchingSimulation:
    def test_output_keys(self, regime_params, transition_matrix):
        result = regime_switching_simulation(
            regime_params, transition_matrix, n_steps=50, n_scenarios=100, seed=42,
        )
        assert "returns" in result
        assert "regimes" in result
        assert "cumulative_returns" in result
        assert "regime_params" in result

    def test_output_shapes(self, regime_params, transition_matrix):
        result = regime_switching_simulation(
            regime_params, transition_matrix, n_steps=50, n_scenarios=100, seed=42,
        )
        assert result["returns"].shape == (100, 50)
        assert result["regimes"].shape == (100, 50)
        assert result["cumulative_returns"].shape == (100, 50)

    def test_deterministic(self, regime_params, transition_matrix):
        r1 = regime_switching_simulation(
            regime_params, transition_matrix, n_steps=20, n_scenarios=50, seed=42,
        )
        r2 = regime_switching_simulation(
            regime_params, transition_matrix, n_steps=20, n_scenarios=50, seed=42,
        )
        np.testing.assert_array_equal(r1["returns"], r2["returns"])

    def test_regimes_valid(self, regime_params, transition_matrix):
        result = regime_switching_simulation(
            regime_params, transition_matrix, n_steps=100, n_scenarios=50, seed=42,
        )
        assert np.all(result["regimes"] >= 0)
        assert np.all(result["regimes"] < len(regime_params))

    def test_both_regimes_visited(self, regime_params, transition_matrix):
        result = regime_switching_simulation(
            regime_params, transition_matrix, n_steps=252, n_scenarios=100, seed=42,
        )
        unique_regimes = np.unique(result["regimes"])
        assert len(unique_regimes) == 2

    def test_invalid_transition_matrix(self, regime_params):
        bad_matrix = np.array([[0.5, 0.3], [0.1, 0.9]])
        with pytest.raises(ValueError, match="rows must sum to 1"):
            regime_switching_simulation(
                regime_params, bad_matrix, n_steps=10, n_scenarios=10, seed=42,
            )

    def test_three_regimes(self):
        params = [
            {"mu": 0.001, "sigma": 0.005},
            {"mu": 0.0, "sigma": 0.015},
            {"mu": -0.002, "sigma": 0.035},
        ]
        trans = np.array([
            [0.90, 0.08, 0.02],
            [0.05, 0.85, 0.10],
            [0.10, 0.20, 0.70],
        ])
        result = regime_switching_simulation(
            params, trans, n_steps=100, n_scenarios=50, seed=42,
        )
        assert result["returns"].shape == (50, 100)


class TestStressTestScenario:
    @pytest.mark.parametrize("shock_type", [
        "market_crash", "vol_spike", "correlation_breakdown", "liquidity_freeze",
    ])
    def test_all_shock_types(self, sample_returns, shock_type):
        result = stress_test_scenario(
            sample_returns, shock_type=shock_type, magnitude=0.20, horizon=21,
        )
        assert "stressed_returns" in result
        assert "cumulative_impact" in result
        assert "max_drawdown" in result
        assert len(result["stressed_returns"]) == 21

    def test_market_crash_negative(self, sample_returns):
        result = stress_test_scenario(
            sample_returns, shock_type="market_crash", magnitude=0.30, horizon=21,
        )
        assert result["cumulative_impact"] < 0

    def test_invalid_shock_type(self, sample_returns):
        with pytest.raises(ValueError, match="shock_type must be one of"):
            stress_test_scenario(
                sample_returns, shock_type="alien_invasion", magnitude=0.5,
            )

    def test_max_drawdown_negative(self, sample_returns):
        result = stress_test_scenario(
            sample_returns, shock_type="market_crash", magnitude=0.20, horizon=21,
        )
        assert result["max_drawdown"] <= 0


class TestReverseStressTest:
    def test_output_keys(self, sample_returns):
        result = reverse_stress_test(
            sample_returns, target_loss=-0.10, n_scenarios=5000, seed=42,
        )
        assert "target_loss" in result
        assert "closest_scenarios" in result
        assert "closest_cumulative" in result
        assert "mean_path" in result
        assert "worst_day_mean" in result

    def test_closest_near_target(self, sample_returns):
        result = reverse_stress_test(
            sample_returns, target_loss=-0.10, n_scenarios=10000, seed=42,
        )
        # Closest scenarios should be near -10%
        mean_cum = np.mean(result["closest_cumulative"])
        assert abs(mean_cum - (-0.10)) < 0.05

    def test_deterministic(self, sample_returns):
        r1 = reverse_stress_test(sample_returns, target_loss=-0.15, n_scenarios=1000, seed=42)
        r2 = reverse_stress_test(sample_returns, target_loss=-0.15, n_scenarios=1000, seed=42)
        np.testing.assert_array_equal(r1["mean_path"], r2["mean_path"])

    def test_mean_path_shape(self, sample_returns):
        result = reverse_stress_test(
            sample_returns, target_loss=-0.20, n_scenarios=5000, seed=42,
        )
        assert result["mean_path"].shape == (21,)


class TestSimulationSummary:
    def test_1d_input(self):
        rng = np.random.default_rng(42)
        returns = rng.standard_t(4, 10000) * 0.01
        result = simulation_summary(returns)

        assert "var" in result
        assert "cvar" in result
        assert "max_drawdown" in result
        assert "probability_of_ruin" in result
        assert "mean_return" in result
        assert "skewness" in result
        assert "kurtosis" in result

    def test_2d_input(self):
        rng = np.random.default_rng(42)
        returns = rng.normal(0.0003, 0.01, (1000, 252))
        result = simulation_summary(returns)

        assert result["var"]["95%"] > 0  # positive loss
        assert result["cvar"]["95%"] >= result["var"]["95%"]

    def test_var_ordering(self):
        rng = np.random.default_rng(42)
        returns = rng.standard_t(4, 10000) * 0.01
        result = simulation_summary(returns)

        # Higher confidence = larger VaR
        assert result["var"]["99%"] >= result["var"]["95%"]
        assert result["var"]["95%"] >= result["var"]["90%"]

    def test_cvar_exceeds_var(self):
        rng = np.random.default_rng(42)
        returns = rng.standard_t(4, 10000) * 0.01
        result = simulation_summary(returns)

        for level in ["90%", "95%", "99%"]:
            assert result["cvar"][level] >= result["var"][level] - 1e-10

    def test_ruin_probabilities(self):
        rng = np.random.default_rng(42)
        returns = rng.normal(0.001, 0.005, 10000)
        result = simulation_summary(returns)

        # With small vol, ruin probability should be very low
        assert result["probability_of_ruin"]["50%"] < 0.01
