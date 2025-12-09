"""Tests for quantlite.crypto.exchange module."""

import pytest

from quantlite.crypto.exchange import (
    concentration_score,
    liquidity_risk,
    proof_of_reserves_check,
    slippage_estimate,
    wallet_risk_assessment,
)


class TestConcentrationScore:
    def test_single_exchange_max_concentration(self):
        result = concentration_score({"Binance": 1_000_000})
        assert result["hhi"] == pytest.approx(1.0)
        assert result["normalised_hhi"] == pytest.approx(1.0)
        assert result["risk_rating"] == "critical"

    def test_equal_split_two(self):
        result = concentration_score({"Binance": 500, "Coinbase": 500})
        assert result["hhi"] == pytest.approx(0.5)
        assert result["normalised_hhi"] == pytest.approx(0.0)
        assert result["risk_rating"] == "low"

    def test_equal_split_four(self):
        balances = {"A": 250, "B": 250, "C": 250, "D": 250}
        result = concentration_score(balances)
        assert result["hhi"] == pytest.approx(0.25)
        assert result["risk_rating"] == "low"

    def test_dominant_exchange(self):
        balances = {"Binance": 900, "Coinbase": 50, "Kraken": 50}
        result = concentration_score(balances)
        assert result["dominant_exchange"] == "Binance"
        assert result["risk_rating"] in ("high", "critical")

    def test_empty_balances(self):
        result = concentration_score({})
        assert result["risk_rating"] == "critical"

    def test_shares_sum_to_one(self):
        balances = {"A": 100, "B": 200, "C": 300}
        result = concentration_score(balances)
        assert sum(result["shares"].values()) == pytest.approx(1.0)


class TestWalletRiskAssessment:
    def test_safe_allocation(self):
        result = wallet_risk_assessment(2, 98, 1_000_000)
        assert result["risk_rating"] == "low"
        assert result["risk_score"] < 0.20

    def test_dangerous_allocation(self):
        result = wallet_risk_assessment(60, 40, 500_000_000)
        assert result["risk_rating"] == "critical"
        assert result["risk_score"] > 0.80

    def test_values_computed(self):
        result = wallet_risk_assessment(10, 90, 1_000_000)
        assert result["hot_value"] == pytest.approx(100_000)
        assert result["cold_value"] == pytest.approx(900_000)

    def test_recommendations_generated(self):
        result = wallet_risk_assessment(25, 75, 1_000_000_000)
        assert len(result["recommendations"]) > 0

    def test_low_hot_no_recommendations(self):
        result = wallet_risk_assessment(1, 99, 100_000)
        # Very safe allocation, minimal recommendations
        assert result["risk_rating"] == "low"


class TestProofOfReservesCheck:
    def test_fully_verified(self):
        claimed = {"BTC": 1000, "ETH": 5000}
        verified = {"BTC": 1000, "ETH": 5000}
        result = proof_of_reserves_check(claimed, verified)
        assert result["fully_verified"] is True
        assert result["overall_ratio"] == pytest.approx(1.0)
        assert result["risk_rating"] == "low"

    def test_shortfall(self):
        claimed = {"BTC": 1000, "ETH": 5000}
        verified = {"BTC": 800, "ETH": 4000}
        result = proof_of_reserves_check(claimed, verified)
        assert result["fully_verified"] is False
        assert result["overall_ratio"] < 1.0

    def test_nothing_verified(self):
        claimed = {"BTC": 1000}
        verified = {"BTC": 0}
        result = proof_of_reserves_check(claimed, verified)
        assert result["risk_rating"] == "critical"
        assert len(result["warnings"]) > 0

    def test_per_asset_details(self):
        claimed = {"BTC": 100, "ETH": 200}
        verified = {"BTC": 95, "ETH": 200}
        result = proof_of_reserves_check(claimed, verified)
        assert "BTC" in result["per_asset"]
        assert result["per_asset"]["BTC"]["shortfall"] == pytest.approx(5.0)
        assert result["per_asset"]["ETH"]["ratio"] == pytest.approx(1.0)

    def test_over_collateralised(self):
        claimed = {"BTC": 100}
        verified = {"BTC": 120}
        result = proof_of_reserves_check(claimed, verified)
        assert result["overall_ratio"] > 1.0
        assert result["fully_verified"] is True


class TestLiquidityRisk:
    def test_small_position(self):
        result = liquidity_risk(1_000_000, 100_000)
        assert result["risk_rating"] == "low"
        assert result["periods_to_unwind"] < 1.0

    def test_large_position(self):
        result = liquidity_risk(100_000, 5_000_000)
        assert result["risk_rating"] == "critical"
        assert result["periods_to_unwind"] == pytest.approx(50.0)

    def test_zero_depth(self):
        result = liquidity_risk(0, 100_000)
        assert result["risk_rating"] == "critical"
        assert result["periods_to_unwind"] == float("inf")

    def test_recommended_periods(self):
        result = liquidity_risk(1_000_000, 500_000)
        assert result["recommended_unwind_periods"] > result["periods_to_unwind"]


class TestSlippageEstimate:
    def test_no_slippage(self):
        book = [(100.0, 1000)]
        result = slippage_estimate(book, 100)
        assert result["slippage_pct"] == pytest.approx(0.0)
        assert result["vwap"] == pytest.approx(100.0)

    def test_multi_level_slippage(self):
        book = [(100.0, 50), (101.0, 50), (102.0, 50)]
        result = slippage_estimate(book, 100)
        assert result["vwap"] == pytest.approx(100.5)
        assert result["slippage_pct"] > 0
        assert result["levels_consumed"] == 2

    def test_partial_fill(self):
        book = [(100.0, 30)]
        result = slippage_estimate(book, 100)
        assert result["unfilled"] == pytest.approx(70.0)

    def test_empty_book(self):
        result = slippage_estimate([], 100)
        assert result["risk_rating"] == "critical"
        assert result["unfilled"] == pytest.approx(100.0)

    def test_zero_trade_size(self):
        result = slippage_estimate([(100, 50)], 0)
        assert result["risk_rating"] == "low"

    def test_worst_price(self):
        book = [(100.0, 10), (101.0, 10), (105.0, 100)]
        result = slippage_estimate(book, 25)
        assert result["worst_price"] == pytest.approx(105.0)
        assert result["best_price"] == pytest.approx(100.0)
