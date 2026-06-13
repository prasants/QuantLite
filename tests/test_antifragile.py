"""Tests for quantlite.antifragile module."""

from __future__ import annotations

import numpy as np
import pytest

from quantlite.antifragile import (
    antifragility_score,
    barbell_allocation,
    convexity_score,
    fourth_quadrant,
    lindy_estimate,
    skin_in_game_score,
)


class TestAntifragilityScore:
    """Tests for antifragility_score."""

    def test_symmetric_returns_near_zero(self):
        rng = np.random.default_rng(42)
        r = rng.normal(0.0, 0.05, 10000)
        score = antifragility_score(r)
        assert abs(score) < 0.2, "Symmetric returns should have near-zero antifragility"

    def test_convex_payoff_positive(self):
        """Returns with larger upside than downside should score positive."""
        rng = np.random.default_rng(42)
        # Skewed: small losses, occasional large gains
        r = np.concatenate([
            rng.uniform(-0.02, 0.0, 800),
            rng.uniform(0.0, 0.20, 200),
        ])
        score = antifragility_score(r)
        assert score > 0, "Convex payoff should be antifragile"

    def test_concave_payoff_negative(self):
        """Returns with larger downside than upside should score negative."""
        rng = np.random.default_rng(42)
        r = np.concatenate([
            rng.uniform(-0.20, 0.0, 200),
            rng.uniform(0.0, 0.02, 800),
        ])
        score = antifragility_score(r)
        assert score < 0, "Concave payoff should be fragile"

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            antifragility_score([])


class TestConvexityScore:
    """Tests for convexity_score."""

    def test_quadratic_payoff(self):
        shocks = np.linspace(-1, 1, 100)
        # Convex payoff: returns = shocks^2
        returns = shocks ** 2
        score = convexity_score(returns, shocks)
        assert score > 0, "Quadratic payoff should have positive convexity"

    def test_concave_payoff(self):
        shocks = np.linspace(-1, 1, 100)
        returns = -(shocks ** 2)
        score = convexity_score(returns, shocks)
        assert score < 0, "Concave payoff should have negative convexity"

    def test_linear_payoff_near_zero(self):
        shocks = np.linspace(-1, 1, 100)
        returns = 2.0 * shocks + 0.5
        score = convexity_score(returns, shocks)
        assert abs(score) < 0.01, "Linear payoff should have near-zero convexity"

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="same length"):
            convexity_score([1, 2, 3], [1, 2])

    def test_too_few_raises(self):
        with pytest.raises(ValueError, match="at least 3"):
            convexity_score([1, 2], [1, 2])


class TestFourthQuadrant:
    """Tests for fourth_quadrant detection."""

    def test_normal_returns_not_fourth(self):
        rng = np.random.default_rng(42)
        r = rng.normal(0.0, 0.02, 10000)
        result = fourth_quadrant(r)
        assert not result["fourth_quadrant"]

    def test_fat_tailed_returns(self):
        rng = np.random.default_rng(42)
        # Student-t with 2 df: very fat tails
        r = rng.standard_t(2, 10000) * 0.01
        result = fourth_quadrant(r)
        assert result["fat_tailed"], "t(2) should be fat-tailed"
        assert result["kurtosis"] > 1

    def test_returns_correct_keys(self):
        result = fourth_quadrant([0.01, -0.01, 0.02, -0.02, 0.0])
        expected_keys = {"kurtosis", "fat_tailed", "payoff_nonlinearity", "fourth_quadrant", "warning"}
        assert set(result.keys()) == expected_keys


class TestBarbellAllocation:
    """Tests for barbell_allocation."""

    def test_basic_allocation(self):
        rng = np.random.default_rng(42)
        conservative = rng.normal(0.002, 0.005, 1000)
        aggressive = rng.normal(0.01, 0.10, 1000)
        result = barbell_allocation(conservative, aggressive)
        assert result["conservative_pct"] == 0.9
        assert result["aggressive_pct"] == pytest.approx(0.1)

    def test_custom_split(self):
        c = [0.01] * 100
        a = [0.05] * 100
        result = barbell_allocation(c, a, conservative_pct=0.8)
        assert result["conservative_pct"] == 0.8
        assert result["aggressive_pct"] == pytest.approx(0.2)

    def test_blended_between_components(self):
        rng = np.random.default_rng(42)
        c = rng.normal(0.001, 0.002, 500)
        a = rng.normal(0.005, 0.05, 500)
        result = barbell_allocation(c, a)
        # Blended arithmetic should be between pure conservative and pure aggressive
        assert isinstance(result["blended_arithmetic"], float)
        assert isinstance(result["blended_geometric"], float)
        assert isinstance(result["max_loss"], float)
        assert isinstance(result["upside_capture"], float)

    def test_returns_all_keys(self):
        result = barbell_allocation([0.01, 0.02], [0.05, -0.03])
        expected = {"conservative_pct", "aggressive_pct", "blended_arithmetic",
                    "blended_geometric", "max_loss", "upside_capture"}
        assert set(result.keys()) == expected


class TestLindyEstimate:
    """Tests for lindy_estimate."""

    def test_expected_remaining_equals_age(self):
        result = lindy_estimate(100)
        assert result["expected_remaining"] == 100
        assert result["total_expected"] == 200

    def test_young_entity(self):
        result = lindy_estimate(1)
        assert result["expected_remaining"] == 1

    def test_negative_age_raises(self):
        with pytest.raises(ValueError, match="positive"):
            lindy_estimate(-5)

    def test_zero_age_raises(self):
        with pytest.raises(ValueError, match="positive"):
            lindy_estimate(0)

    def test_bad_confidence_raises(self):
        with pytest.raises(ValueError, match="between 0 and 1"):
            lindy_estimate(10, confidence=1.5)

    def test_returns_correct_keys(self):
        result = lindy_estimate(50)
        assert set(result.keys()) == {"age", "expected_remaining", "lower_bound", "total_expected"}
        assert result["age"] == 50


class TestSkinInGameScore:
    """Tests for skin_in_game_score."""

    def test_perfect_alignment(self):
        r = [0.05, -0.03, 0.02, -0.01, 0.04]
        result = skin_in_game_score(r, r)
        assert result["alignment"] == pytest.approx(1.0, abs=1e-5)

    def test_no_correlation(self):
        rng = np.random.default_rng(42)
        m = rng.normal(0.01, 0.05, 1000)
        f = rng.normal(0.01, 0.05, 1000)
        result = skin_in_game_score(m, f)
        assert abs(result["alignment"]) < 0.15

    def test_returns_all_keys(self):
        result = skin_in_game_score([0.01, -0.02, 0.03], [0.01, -0.02, 0.03])
        assert set(result.keys()) == {"alignment", "downside_sharing", "upside_asymmetry", "score"}

    def test_score_bounded(self):
        rng = np.random.default_rng(42)
        m = rng.normal(0.01, 0.05, 100)
        f = rng.normal(0.01, 0.05, 100)
        result = skin_in_game_score(m, f)
        assert 0.0 <= result["score"] <= 1.0
