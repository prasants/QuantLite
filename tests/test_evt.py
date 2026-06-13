"""Tests for quantlite.risk.evt (Extreme Value Theory)."""

import numpy as np
import pytest

from quantlite.risk.evt import (
    fit_gev,
    fit_gpd,
    hill_estimator,
    peaks_over_threshold,
    return_level,
    tail_risk_summary,
)


@pytest.fixture()
def heavy_tailed_returns():
    """Returns from a Student-t(4) distribution: heavy tails."""
    rng = np.random.default_rng(42)
    # Scale Student-t to realistic daily returns
    return rng.standard_t(df=4, size=2000) * 0.015


@pytest.fixture()
def block_maxima():
    """Simulated annual block maxima for GEV testing."""
    rng = np.random.default_rng(123)
    # 20 years of annual maxima from a Frechet-like distribution
    return np.abs(rng.standard_t(df=4, size=20)) * 0.05


class TestFitGPD:
    def test_basic_fit(self, heavy_tailed_returns):
        gpd = fit_gpd(heavy_tailed_returns)
        assert gpd.n_exceedances >= 10
        assert gpd.scale > 0
        assert gpd.n_total == 2000

    def test_custom_threshold(self, heavy_tailed_returns):
        losses = -heavy_tailed_returns
        threshold = float(np.percentile(losses, 90))
        gpd = fit_gpd(heavy_tailed_returns, threshold=threshold)
        assert gpd.threshold == threshold
        assert gpd.n_exceedances > gpd.n_total * 0.05

    def test_too_few_exceedances(self):
        """Very high threshold should yield too few exceedances."""
        small = np.random.default_rng(1).normal(0, 0.01, 50)
        with pytest.raises(ValueError, match="exceedances"):
            fit_gpd(small, threshold=100.0)

    def test_repr(self, heavy_tailed_returns):
        gpd = fit_gpd(heavy_tailed_returns)
        r = repr(gpd)
        assert "GPDFit" in r
        assert "shape=" in r


class TestFitGEV:
    def test_basic_fit(self, block_maxima):
        gev = fit_gev(block_maxima)
        assert gev.scale > 0

    def test_too_few_blocks(self):
        with pytest.raises(ValueError, match="at least 5"):
            fit_gev([0.01, 0.02, 0.03])


class TestHillEstimator:
    def test_heavy_tails(self, heavy_tailed_returns):
        hill = hill_estimator(heavy_tailed_returns)
        # Student-t(4) should have tail index around 4
        assert 1 < hill.tail_index < 15, f"Expected reasonable tail index, got {hill.tail_index}"
        assert hill.k >= 2

    def test_custom_k(self, heavy_tailed_returns):
        hill = hill_estimator(heavy_tailed_returns, k=50)
        assert hill.k == 50

    def test_k_too_large(self, heavy_tailed_returns):
        with pytest.raises(ValueError, match="exceeds"):
            hill_estimator(heavy_tailed_returns, k=len(heavy_tailed_returns))


class TestPeaksOverThreshold:
    def test_returns_exceedances_and_fit(self, heavy_tailed_returns):
        exceedances, gpd = peaks_over_threshold(heavy_tailed_returns)
        assert len(exceedances) > 0
        assert gpd.scale > 0


class TestReturnLevel:
    def test_increasing_with_period(self, heavy_tailed_returns):
        gpd = fit_gpd(heavy_tailed_returns)
        rl_100 = return_level(gpd, 100)
        rl_1000 = return_level(gpd, 1000)
        assert rl_1000 > rl_100, "Longer return period should give larger loss"

    def test_custom_n_obs(self, heavy_tailed_returns):
        gpd = fit_gpd(heavy_tailed_returns)
        rl = return_level(gpd, 100, n_obs=5000)
        assert rl > 0


class TestTailRiskSummary:
    def test_comprehensive(self, heavy_tailed_returns):
        summary = tail_risk_summary(heavy_tailed_returns)
        assert summary.var_99 < summary.var_95, "99% VaR should be more extreme"
        assert summary.cvar_99 <= summary.var_99
        assert summary.excess_kurtosis > 0, "Student-t should have positive excess kurtosis"
        assert summary.return_level_100 > 0

    def test_repr(self, heavy_tailed_returns):
        summary = tail_risk_summary(heavy_tailed_returns)
        r = repr(summary)
        assert "TailRiskSummary" in r
