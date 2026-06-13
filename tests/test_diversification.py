"""Tests for quantlite.diversification module."""

import numpy as np
import pandas as pd
import pytest

from quantlite.diversification import (
    diversification_ratio,
    effective_number_of_bets,
    entropy_diversification,
    herfindahl_index,
    marginal_tail_risk_contribution,
    tail_diversification,
)


class TestEffectiveNumberOfBets:
    """Tests for effective number of bets."""

    def test_identity_covariance_equals_n(self):
        """With identity covariance, ENB should equal n."""
        n = 5
        w = np.ones(n) / n
        cov = np.eye(n)
        enb = effective_number_of_bets(w, cov)
        assert enb == pytest.approx(n, abs=0.1)

    def test_single_asset(self):
        enb = effective_number_of_bets([1.0], [[0.04]])
        assert enb == pytest.approx(1.0)

    def test_perfectly_correlated(self):
        """Perfectly correlated assets should have ENB close to 1."""
        n = 4
        cov = np.ones((n, n)) * 0.04
        w = np.ones(n) / n
        enb = effective_number_of_bets(w, cov)
        assert enb < 1.5

    def test_returns_float(self):
        enb = effective_number_of_bets([0.5, 0.5], np.eye(2))
        assert isinstance(enb, float)


class TestEntropyDiversification:
    """Tests for entropy diversification."""

    def test_equal_weights_maximum(self):
        n = 10
        w = np.ones(n) / n
        result = entropy_diversification(w)
        assert result == pytest.approx(1.0, abs=1e-6)

    def test_single_weight_zero(self):
        result = entropy_diversification([1.0])
        assert result == 0.0

    def test_concentrated_low(self):
        w = [0.9, 0.05, 0.03, 0.02]
        result = entropy_diversification(w)
        assert result < 0.5

    def test_bounded_zero_one(self):
        w = [0.6, 0.3, 0.1]
        result = entropy_diversification(w)
        assert 0.0 <= result <= 1.0


class TestTailDiversification:
    """Tests for tail diversification."""

    def test_returns_dict(self):
        rng = np.random.RandomState(42)
        df = pd.DataFrame(rng.normal(0, 0.01, (500, 3)), columns=["A", "B", "C"])
        w = np.array([1 / 3, 1 / 3, 1 / 3])
        result = tail_diversification(df, w)
        assert "normal_diversification" in result
        assert "tail_diversification" in result
        assert "tail_concentration_ratio" in result

    def test_positive_values(self):
        rng = np.random.RandomState(42)
        df = pd.DataFrame(rng.normal(0, 0.01, (500, 3)), columns=["A", "B", "C"])
        w = np.array([1 / 3, 1 / 3, 1 / 3])
        result = tail_diversification(df, w)
        assert result["normal_diversification"] > 0
        assert result["tail_diversification"] > 0


class TestMarginalTailRiskContribution:
    """Tests for marginal tail risk contribution."""

    def test_returns_all_assets(self):
        rng = np.random.RandomState(42)
        df = pd.DataFrame(rng.normal(0, 0.01, (500, 3)), columns=["X", "Y", "Z"])
        w = np.array([1 / 3, 1 / 3, 1 / 3])
        result = marginal_tail_risk_contribution(df, w)
        assert set(result.keys()) == {"X", "Y", "Z"}

    def test_contributions_are_float(self):
        rng = np.random.RandomState(42)
        df = pd.DataFrame(rng.normal(0, 0.01, (500, 2)), columns=["A", "B"])
        w = np.array([0.5, 0.5])
        result = marginal_tail_risk_contribution(df, w)
        for v in result.values():
            assert isinstance(v, float)


class TestDiversificationRatio:
    """Tests for diversification ratio."""

    def test_uncorrelated_greater_than_one(self):
        w = np.array([0.5, 0.5])
        vol = np.array([0.2, 0.2])
        cov = np.diag(vol ** 2)
        dr = diversification_ratio(w, vol, cov)
        assert dr > 1.0

    def test_perfectly_correlated_equals_one(self):
        w = np.array([0.5, 0.5])
        vol = np.array([0.2, 0.2])
        cov = np.outer(vol, vol)
        dr = diversification_ratio(w, vol, cov)
        assert dr == pytest.approx(1.0, abs=0.01)

    def test_single_asset(self):
        dr = diversification_ratio([1.0], [0.2], [[0.04]])
        assert dr == pytest.approx(1.0, abs=0.01)


class TestHerfindahlIndex:
    """Tests for Herfindahl index."""

    def test_equal_weights(self):
        n = 10
        w = np.ones(n) / n
        assert herfindahl_index(w) == pytest.approx(1.0 / n, abs=1e-10)

    def test_single_asset(self):
        assert herfindahl_index([1.0]) == pytest.approx(1.0)

    def test_two_assets_equal(self):
        assert herfindahl_index([0.5, 0.5]) == pytest.approx(0.5)

    def test_concentrated(self):
        h = herfindahl_index([0.9, 0.05, 0.05])
        assert h > 0.8
