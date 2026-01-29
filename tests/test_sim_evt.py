"""Tests for quantlite.simulation.evt_simulation."""

from __future__ import annotations

import numpy as np
import pytest

from quantlite.simulation.evt_simulation import (
    evt_tail_simulation,
    historical_bootstrap_evt,
    parametric_tail_simulation,
    scenario_fan,
)


@pytest.fixture()
def fat_tailed_returns():
    """Generate synthetic fat-tailed returns for testing."""
    rng = np.random.default_rng(123)
    # Mix of normal body with occasional large moves
    body = rng.normal(0.0005, 0.01, 900)
    tails = rng.standard_t(3, 100) * 0.03
    returns = np.concatenate([body, tails])
    rng.shuffle(returns)
    return returns


class TestEvtTailSimulation:
    def test_output_shape(self, fat_tailed_returns):
        result = evt_tail_simulation(fat_tailed_returns, n_scenarios=500, seed=42)
        assert result.shape == (500,)

    def test_deterministic(self, fat_tailed_returns):
        r1 = evt_tail_simulation(fat_tailed_returns, n_scenarios=100, seed=42)
        r2 = evt_tail_simulation(fat_tailed_returns, n_scenarios=100, seed=42)
        np.testing.assert_array_equal(r1, r2)

    def test_different_seeds(self, fat_tailed_returns):
        r1 = evt_tail_simulation(fat_tailed_returns, n_scenarios=100, seed=42)
        r2 = evt_tail_simulation(fat_tailed_returns, n_scenarios=100, seed=99)
        assert not np.array_equal(r1, r2)

    def test_no_nans(self, fat_tailed_returns):
        result = evt_tail_simulation(fat_tailed_returns, n_scenarios=1000, seed=42)
        assert not np.any(np.isnan(result))

    def test_reasonable_range(self, fat_tailed_returns):
        result = evt_tail_simulation(fat_tailed_returns, n_scenarios=5000, seed=42)
        # Most values should be within a reasonable range
        assert np.percentile(result, 1) > -0.5
        assert np.percentile(result, 99) < 0.5

    def test_tails_present(self, fat_tailed_returns):
        result = evt_tail_simulation(fat_tailed_returns, n_scenarios=10000, seed=42)
        hist_min = np.min(fat_tailed_returns)
        # EVT should produce some values beyond observed minimum
        assert np.min(result) < hist_min * 0.5 or np.min(result) < hist_min

    def test_alpha_parameter(self, fat_tailed_returns):
        r1 = evt_tail_simulation(fat_tailed_returns, n_scenarios=500, alpha=0.01, seed=42)
        r2 = evt_tail_simulation(fat_tailed_returns, n_scenarios=500, alpha=0.10, seed=42)
        # Different alpha should produce different distributions
        assert abs(np.std(r1) - np.std(r2)) > 0 or not np.array_equal(r1, r2)


class TestParametricTailSimulation:
    def test_output_shape(self):
        result = parametric_tail_simulation(
            shape=0.2, scale=0.01, threshold=0.03,
            n_body=1000, body_mean=0.0005, body_std=0.01,
            n_scenarios=500, seed=42,
        )
        assert result.shape == (500,)

    def test_deterministic(self):
        kwargs = dict(
            shape=0.2, scale=0.01, threshold=0.03,
            n_body=1000, body_mean=0.0005, body_std=0.01,
            n_scenarios=200, seed=42,
        )
        r1 = parametric_tail_simulation(**kwargs)
        r2 = parametric_tail_simulation(**kwargs)
        np.testing.assert_array_equal(r1, r2)

    def test_body_centered(self):
        result = parametric_tail_simulation(
            shape=0.1, scale=0.005, threshold=0.05,
            n_body=1000, body_mean=0.001, body_std=0.01,
            n_scenarios=5000, seed=42,
        )
        # Median should be near body mean
        assert abs(np.median(result) - 0.001) < 0.01

    def test_no_nans(self):
        result = parametric_tail_simulation(
            shape=0.3, scale=0.02, threshold=0.04,
            n_body=500, body_mean=0.0, body_std=0.015,
            n_scenarios=1000, seed=42,
        )
        assert not np.any(np.isnan(result))


class TestHistoricalBootstrapEvt:
    def test_output_shape(self, fat_tailed_returns):
        result = historical_bootstrap_evt(fat_tailed_returns, n_scenarios=500, seed=42)
        assert result.shape == (500,)

    def test_deterministic(self, fat_tailed_returns):
        r1 = historical_bootstrap_evt(fat_tailed_returns, n_scenarios=200, seed=42)
        r2 = historical_bootstrap_evt(fat_tailed_returns, n_scenarios=200, seed=42)
        np.testing.assert_array_equal(r1, r2)

    def test_no_nans(self, fat_tailed_returns):
        result = historical_bootstrap_evt(fat_tailed_returns, n_scenarios=1000, seed=42)
        assert not np.any(np.isnan(result))

    def test_tail_fraction(self, fat_tailed_returns):
        r1 = historical_bootstrap_evt(
            fat_tailed_returns, n_scenarios=500, tail_fraction=0.02, seed=42,
        )
        r2 = historical_bootstrap_evt(
            fat_tailed_returns, n_scenarios=500, tail_fraction=0.10, seed=42,
        )
        assert not np.array_equal(r1, r2)


class TestScenarioFan:
    def test_basic_output(self, fat_tailed_returns):
        horizons = [1, 5, 21]
        result = scenario_fan(fat_tailed_returns, horizons, n_scenarios=500, seed=42)

        assert result["horizons"] == horizons
        assert result["percentiles"] == [5, 25, 50, 75, 95]
        assert set(result["fans"].keys()) == {1, 5, 21}
        assert set(result["scenarios"].keys()) == {1, 5, 21}

    def test_scenario_shapes(self, fat_tailed_returns):
        result = scenario_fan(fat_tailed_returns, [1, 5], n_scenarios=300, seed=42)
        for h in [1, 5]:
            assert result["scenarios"][h].shape == (300,)

    def test_longer_horizons_wider(self, fat_tailed_returns):
        result = scenario_fan(
            fat_tailed_returns, [1, 21, 252], n_scenarios=2000, seed=42,
        )
        # Longer horizons should have wider distributions
        std_1 = np.std(result["scenarios"][1])
        std_252 = np.std(result["scenarios"][252])
        assert std_252 > std_1

    def test_fan_percentile_ordering(self, fat_tailed_returns):
        result = scenario_fan(fat_tailed_returns, [21], n_scenarios=1000, seed=42)
        fan = result["fans"][21]
        assert float(fan["5"]) <= float(fan["25"])
        assert float(fan["25"]) <= float(fan["50"])
        assert float(fan["50"]) <= float(fan["75"])
        assert float(fan["75"]) <= float(fan["95"])

    def test_deterministic(self, fat_tailed_returns):
        r1 = scenario_fan(fat_tailed_returns, [5], n_scenarios=100, seed=42)
        r2 = scenario_fan(fat_tailed_returns, [5], n_scenarios=100, seed=42)
        np.testing.assert_array_equal(
            r1["scenarios"][5], r2["scenarios"][5],
        )
