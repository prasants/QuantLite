"""Tests for quantlite.ergodicity module."""

from __future__ import annotations

import numpy as np
import pytest

from quantlite.ergodicity import (
    ensemble_average,
    ergodicity_gap,
    geometric_mean_dominance,
    kelly_fraction,
    leverage_effect,
    time_average,
)


class TestTimeAverage:
    """Tests for time_average (geometric mean growth rate)."""

    def test_constant_returns(self):
        r = [0.05] * 100
        assert time_average(r) == pytest.approx(0.05, abs=1e-10)

    def test_zero_returns(self):
        r = [0.0] * 50
        assert time_average(r) == pytest.approx(0.0, abs=1e-10)

    def test_volatile_returns_lower_than_arithmetic(self):
        """Volatile returns should have geometric < arithmetic mean."""
        rng = np.random.default_rng(42)
        r = rng.normal(0.01, 0.05, 1000)
        assert time_average(r) < ensemble_average(r)

    def test_single_return(self):
        assert time_average([0.10]) == pytest.approx(0.10, abs=1e-10)

    def test_negative_returns(self):
        r = [-0.01, -0.02, -0.01]
        result = time_average(r)
        assert result < 0

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="must not be empty"):
            time_average([])

    def test_numpy_array_input(self):
        r = np.array([0.02, 0.03, 0.01])
        result = time_average(r)
        assert isinstance(result, float)


class TestEnsembleAverage:
    """Tests for ensemble_average (arithmetic mean)."""

    def test_simple(self):
        assert ensemble_average([0.1, 0.2, 0.3]) == pytest.approx(0.2, abs=1e-10)

    def test_symmetric(self):
        assert ensemble_average([-0.1, 0.1]) == pytest.approx(0.0, abs=1e-10)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            ensemble_average([])


class TestErgodicityGap:
    """Tests for ergodicity_gap."""

    def test_constant_returns_zero_gap(self):
        r = [0.05] * 100
        assert ergodicity_gap(r) == pytest.approx(0.0, abs=1e-10)

    def test_volatile_returns_positive_gap(self):
        rng = np.random.default_rng(123)
        r = rng.normal(0.02, 0.10, 5000)
        gap = ergodicity_gap(r)
        assert gap > 0, "Volatile returns should have positive ergodicity gap"

    def test_gap_increases_with_volatility(self):
        rng = np.random.default_rng(99)
        r_low_vol = rng.normal(0.01, 0.02, 5000)
        r_high_vol = rng.normal(0.01, 0.10, 5000)
        assert ergodicity_gap(r_high_vol) > ergodicity_gap(r_low_vol)


class TestKellyFraction:
    """Tests for kelly_fraction."""

    def test_positive_expected_return(self):
        rng = np.random.default_rng(42)
        r = rng.normal(0.05, 0.10, 500)
        f = kelly_fraction(r)
        assert f > 0, "Positive expected returns should give positive Kelly fraction"

    def test_with_risk_free(self):
        rng = np.random.default_rng(42)
        r = rng.normal(0.05, 0.10, 500)
        f_zero = kelly_fraction(r, risk_free=0.0)
        f_high = kelly_fraction(r, risk_free=0.04)
        assert f_high < f_zero, "Higher risk-free should reduce Kelly fraction"

    def test_bad_returns_low_kelly(self):
        rng = np.random.default_rng(42)
        r = rng.normal(-0.05, 0.10, 500)
        f = kelly_fraction(r)
        assert f <= 0.5, "Negative expected returns should have low Kelly"

    def test_returns_float(self):
        f = kelly_fraction([0.1, -0.05, 0.08, -0.02])
        assert isinstance(f, float)


class TestLeverageEffect:
    """Tests for leverage_effect."""

    def test_default_leverages(self):
        rng = np.random.default_rng(42)
        r = rng.normal(0.01, 0.05, 1000)
        result = leverage_effect(r)
        assert set(result.keys()) == {1.0, 2.0, 3.0, 5.0}

    def test_custom_leverages(self):
        r = [0.01, 0.02, -0.01, 0.03]
        result = leverage_effect(r, leverages=[1.0, 1.5])
        assert set(result.keys()) == {1.0, 1.5}

    def test_higher_leverage_lower_growth_volatile(self):
        """With sufficient volatility, higher leverage should reduce time-average growth."""
        rng = np.random.default_rng(42)
        r = rng.normal(0.001, 0.05, 10000)
        result = leverage_effect(r, leverages=[1.0, 5.0])
        assert result[5.0] < result[1.0], (
            "5x leverage should have lower time-average growth with high volatility"
        )

    def test_unleveraged_matches_time_average(self):
        r = [0.01, 0.02, -0.005, 0.015]
        result = leverage_effect(r, leverages=[1.0])
        assert result[1.0] == pytest.approx(time_average(r), abs=1e-10)


class TestGeometricMeanDominance:
    """Tests for geometric_mean_dominance."""

    def test_a_dominates(self):
        a = [0.05, 0.04, 0.06]
        b = [0.01, 0.02, 0.01]
        result = geometric_mean_dominance(a, b)
        assert result["dominant"] == "A"
        assert result["margin"] > 0

    def test_b_dominates(self):
        a = [0.01, 0.00, 0.01]
        b = [0.05, 0.04, 0.06]
        result = geometric_mean_dominance(a, b)
        assert result["dominant"] == "B"

    def test_equal(self):
        a = [0.05, 0.05]
        result = geometric_mean_dominance(a, a)
        assert result["dominant"] == "neither"
        assert result["margin"] == pytest.approx(0.0, abs=1e-10)

    def test_returns_dict_keys(self):
        result = geometric_mean_dominance([0.01], [0.02])
        assert set(result.keys()) == {"g_mean_a", "g_mean_b", "dominant", "margin"}
