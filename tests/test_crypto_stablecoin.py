"""Tests for quantlite.crypto.stablecoin module."""

import numpy as np
import pytest

from quantlite.crypto.stablecoin import (
    HISTORICAL_DEPEGS,
    depeg_probability,
    depeg_recovery_time,
    peg_deviation_tracker,
    reserve_risk_score,
)


class TestHistoricalDepegs:
    def test_known_events_present(self):
        assert "UST" in HISTORICAL_DEPEGS
        assert "USDC_2023" in HISTORICAL_DEPEGS
        assert "DAI_2020" in HISTORICAL_DEPEGS
        assert "USDT_2022" in HISTORICAL_DEPEGS

    def test_ust_never_recovered(self):
        assert HISTORICAL_DEPEGS["UST"]["recovery_days"] is None

    def test_usdc_recovered_quickly(self):
        assert HISTORICAL_DEPEGS["USDC_2023"]["recovery_days"] == 3

    def test_all_events_have_required_fields(self):
        required = {"name", "date", "trough_price", "magnitude", "recovery_days", "cause"}
        for event in HISTORICAL_DEPEGS.values():
            assert required.issubset(event.keys())


class TestDepegProbability:
    def test_stable_coin_low_risk(self):
        rng = np.random.default_rng(42)
        prices = 1.0 + rng.normal(0, 0.001, 1000)
        result = depeg_probability(prices, threshold=0.005)
        assert result["risk_rating"] == "low"
        assert result["empirical_prob"] < 0.05
        assert result["n_observations"] == 1000

    def test_volatile_coin_high_risk(self):
        rng = np.random.default_rng(42)
        prices = 1.0 + rng.normal(0, 0.05, 1000)
        result = depeg_probability(prices, threshold=0.005)
        assert result["empirical_prob"] > 0.5
        assert result["risk_rating"] in ("high", "critical")

    def test_perfect_peg(self):
        prices = np.ones(100)
        result = depeg_probability(prices, threshold=0.005)
        assert result["empirical_prob"] == 0.0
        assert result["n_breaches"] == 0
        assert result["mean_deviation"] == 0.0

    def test_custom_threshold(self):
        prices = np.array([1.0, 1.003, 0.997, 1.01, 0.99])
        result = depeg_probability(prices, threshold=0.008)
        assert result["n_breaches"] == 2  # 1.01 and 0.99

    def test_return_keys(self):
        result = depeg_probability([1.0, 1.001, 0.999])
        expected_keys = {
            "empirical_prob", "parametric_prob", "n_breaches",
            "n_observations", "mean_deviation", "max_deviation", "risk_rating",
        }
        assert expected_keys == set(result.keys())


class TestPegDeviationTracker:
    def test_perfect_peg(self):
        prices = np.ones(100)
        result = peg_deviation_tracker(prices)
        assert result["mean_deviation"] == 0.0
        assert result["max_deviation"] == 0.0
        assert len(result["excursions"]) == 0

    def test_single_excursion(self):
        prices = np.ones(100)
        prices[20:30] = 1.01  # 10-period excursion
        result = peg_deviation_tracker(prices)
        assert len(result["excursions"]) == 1
        assert result["excursions"][0]["duration"] == 10
        assert result["excursions"][0]["direction"] == "above"

    def test_below_peg_excursion(self):
        prices = np.ones(50)
        prices[10:15] = 0.98
        result = peg_deviation_tracker(prices)
        assert len(result["excursions"]) == 1
        assert result["excursions"][0]["direction"] == "below"

    def test_custom_peg(self):
        prices = np.full(50, 2.0)
        prices[5:10] = 2.05
        result = peg_deviation_tracker(prices, peg=2.0)
        assert result["max_deviation"] == pytest.approx(0.05)

    def test_excursion_at_end(self):
        prices = np.ones(50)
        prices[45:] = 1.02  # Excursion extends to end
        result = peg_deviation_tracker(prices)
        excursions = result["excursions"]
        assert len(excursions) >= 1
        assert excursions[-1]["end"] == 49

    def test_pct_time_off_peg(self):
        prices = np.ones(100)
        prices[0:50] = 1.01  # 50% off peg
        result = peg_deviation_tracker(prices)
        assert result["pct_time_off_peg"] == pytest.approx(50.0)


class TestDepegRecoveryTime:
    def test_no_depeg(self):
        prices = np.ones(100)
        result = depeg_recovery_time(prices)
        assert result["total_events"] == 0
        assert result["mean_recovery_time"] == 0.0

    def test_single_recovered_event(self):
        prices = np.ones(100)
        prices[20:30] = 0.98  # depeg for 10 periods, then recover
        result = depeg_recovery_time(prices, threshold=0.005)
        assert result["total_events"] == 1
        assert result["events"][0]["recovered"] is True
        assert result["events"][0]["duration"] == 10

    def test_unrecovered_event(self):
        prices = np.ones(100)
        prices[90:] = 0.98  # depeg at end, never recovers
        result = depeg_recovery_time(prices, threshold=0.005)
        assert result["unrecovered"] == 1
        assert result["events"][-1]["recovered"] is False

    def test_multiple_events(self):
        prices = np.ones(100)
        prices[10:15] = 0.99  # 5-period event
        prices[30:40] = 1.02  # 10-period event
        result = depeg_recovery_time(prices, threshold=0.005)
        assert result["total_events"] == 2

    def test_max_deviation_tracked(self):
        prices = np.ones(100)
        prices[20] = 0.95
        prices[21] = 0.96
        prices[22] = 0.97
        result = depeg_recovery_time(prices, threshold=0.005)
        assert result["events"][0]["max_deviation"] == pytest.approx(0.05)


class TestReserveRiskScore:
    def test_excellent_reserves(self):
        composition = {"cash": 50, "treasuries": 45, "money_market": 5}
        result = reserve_risk_score(composition)
        assert result["rating"] == "excellent"
        assert result["score"] > 0.85

    def test_poor_reserves(self):
        composition = {"crypto": 60, "other": 40}
        result = reserve_risk_score(composition)
        assert result["rating"] in ("poor", "critical")
        assert result["score"] < 0.30

    def test_warnings_generated(self):
        composition = {"crypto": 30, "other": 40, "cash": 10, "treasuries": 5}
        result = reserve_risk_score(composition)
        assert len(result["warnings"]) > 0

    def test_breakdown_present(self):
        composition = {"cash": 80, "treasuries": 20}
        result = reserve_risk_score(composition)
        assert "cash" in result["breakdown"]
        assert "treasuries" in result["breakdown"]

    def test_incomplete_allocation_warning(self):
        composition = {"cash": 40}
        result = reserve_risk_score(composition)
        warnings_text = " ".join(result["warnings"])
        assert "incomplete" in warnings_text.lower() or "sum" in warnings_text.lower()

    def test_score_bounded(self):
        composition = {"cash": 100}
        result = reserve_risk_score(composition)
        assert 0.0 <= result["score"] <= 1.0
