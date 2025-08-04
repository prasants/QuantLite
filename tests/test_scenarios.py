"""Tests for quantlite.scenarios module."""

from __future__ import annotations

import numpy as np
import pytest

from quantlite.scenarios import (
    SCENARIO_LIBRARY,
    Scenario,
    fragility_heatmap,
    shock_propagation,
    stress_test,
)


class TestScenario:
    """Tests for the Scenario fluent API."""

    def test_basic_construction(self):
        s = Scenario("test")
        assert s.name == "test"
        assert s.shocks == {}

    def test_fluent_api(self):
        s = Scenario("crisis") \
            .shock("BTC", -0.5) \
            .shock("ETH", -0.6) \
            .correlations(spike_to=0.9) \
            .duration(days=30)
        assert s.shocks == {"BTC": -0.5, "ETH": -0.6}
        assert s.correlation_spike == 0.9
        assert s.duration_days == 30

    def test_repr(self):
        s = Scenario("test").shock("BTC", -0.5)
        r = repr(s)
        assert "test" in r
        assert "BTC" in r

    def test_overwrite_shock(self):
        s = Scenario("x").shock("BTC", -0.3).shock("BTC", -0.5)
        assert s.shocks["BTC"] == -0.5


class TestScenarioLibrary:
    """Tests for the pre-built scenario library."""

    def test_all_scenarios_present(self):
        expected = {"2008 GFC", "2020 COVID", "2022 Luna/FTX", "USDT depeg", "rates +200bps"}
        assert set(SCENARIO_LIBRARY.keys()) == expected

    def test_scenarios_have_shocks(self):
        for name, scenario in SCENARIO_LIBRARY.items():
            assert len(scenario.shocks) > 0, f"{name} has no shocks"
            assert scenario.duration_days is not None
            assert scenario.correlation_spike is not None


class TestStressTest:
    """Tests for stress_test."""

    def test_basic_stress(self):
        weights = {"BTC": 0.5, "ETH": 0.3, "CASH": 0.2}
        scenario = Scenario("crash").shock("BTC", -0.50).shock("ETH", -0.60)
        result = stress_test(weights, scenario)
        # BTC: 0.5 * -0.5 = -0.25, ETH: 0.3 * -0.6 = -0.18, CASH: 0
        assert result["portfolio_impact"] == pytest.approx(-0.43)
        assert result["worst_asset"] == "BTC"
        assert result["survival"] is True  # 1 - 0.43 > 0

    def test_total_wipeout(self):
        weights = {"BTC": 1.0}
        scenario = Scenario("death").shock("BTC", -1.0)
        result = stress_test(weights, scenario)
        assert result["portfolio_impact"] == pytest.approx(-1.0)
        assert result["survival"] is False

    def test_no_shocks_zero_impact(self):
        weights = {"BTC": 0.5, "ETH": 0.5}
        scenario = Scenario("nothing")
        result = stress_test(weights, scenario)
        assert result["portfolio_impact"] == pytest.approx(0.0)

    def test_with_returns(self):
        rng = np.random.default_rng(42)
        weights = {"A": 0.5, "B": 0.5}
        scenario = Scenario("test").shock("A", -0.3)
        returns = {"A": rng.normal(0, 0.05, 100), "B": rng.normal(0, 0.01, 100)}
        result = stress_test(weights, scenario, returns)
        assert result["portfolio_impact"] != 0

    def test_returns_correct_keys(self):
        result = stress_test({"A": 1.0}, Scenario("x").shock("A", -0.1))
        expected = {"scenario_name", "portfolio_impact", "asset_impacts",
                    "worst_asset", "best_asset", "survival"}
        assert set(result.keys()) == expected


class TestFragilityHeatmap:
    """Tests for fragility_heatmap."""

    def test_basic_heatmap(self):
        weights = {"BTC": 0.5, "ETH": 0.5}
        scenarios = [
            Scenario("crash").shock("BTC", -0.5),
            Scenario("boom").shock("ETH", 0.3),
        ]
        heatmap = fragility_heatmap(weights, scenarios)
        assert "crash" in heatmap
        assert "boom" in heatmap
        assert "BTC" in heatmap["crash"]

    def test_empty_scenarios(self):
        heatmap = fragility_heatmap({"A": 1.0}, [])
        assert heatmap == {}

    def test_library_scenarios(self):
        weights = {"BTC": 0.4, "SPX": 0.3, "BONDS_10Y": 0.3}
        scenarios = list(SCENARIO_LIBRARY.values())
        heatmap = fragility_heatmap(weights, scenarios)
        assert len(heatmap) == 5


class TestShockPropagation:
    """Tests for shock_propagation."""

    def test_correlated_assets(self):
        rng = np.random.default_rng(42)
        base = rng.normal(0, 0.02, 500)
        returns = {
            "A": base + rng.normal(0, 0.005, 500),
            "B": base + rng.normal(0, 0.005, 500),
            "C": rng.normal(0, 0.02, 500),
        }
        result = shock_propagation(returns, {"A": -0.20})
        # B should be more impacted than C (more correlated with A)
        assert abs(result["B"]) > abs(result["C"]) * 0.5

    def test_single_asset(self):
        returns = {"A": [0.01, -0.02, 0.03]}
        result = shock_propagation(returns, {"A": -0.10})
        assert result["A"] == pytest.approx(-0.10)

    def test_custom_correlation_matrix(self):
        returns = {"A": [0.01, 0.02, 0.03], "B": [0.01, 0.02, 0.03]}
        corr = np.array([[1.0, 0.5], [0.5, 1.0]])
        result = shock_propagation(returns, {"A": -0.20}, correlation_matrix=corr)
        assert result["A"] == pytest.approx(-0.20)
        assert result["B"] == pytest.approx(-0.10)  # 0.5 * -0.20

    def test_no_shock_no_propagation(self):
        returns = {"A": [0.01, 0.02], "B": [0.01, 0.02]}
        result = shock_propagation(returns, {})
        assert result["A"] == pytest.approx(0.0)
        assert result["B"] == pytest.approx(0.0)
