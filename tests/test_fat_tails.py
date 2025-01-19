"""Tests for quantlite.distributions.fat_tails."""

import numpy as np
import pytest
from scipy import stats

from quantlite.distributions.fat_tails import (
    RegimeParams,
    kou_double_exponential_jump,
    levy_stable_process,
    regime_switching_gbm,
    student_t_process,
)


class TestStudentTProcess:
    def test_correct_length(self):
        rets = student_t_process(nu=4, n_steps=500, rng_seed=42)
        assert len(rets) == 500

    def test_excess_kurtosis(self):
        """Student-t(4) has theoretical excess kurtosis = 6."""
        rets = student_t_process(nu=4, n_steps=50000, rng_seed=42)
        k = float(stats.kurtosis(rets))
        assert k > 3, f"Expected excess kurtosis > 3, got {k}"

    def test_nu_too_low(self):
        with pytest.raises(ValueError, match="nu must be > 2"):
            student_t_process(nu=1.5)

    def test_reproducibility(self):
        a = student_t_process(rng_seed=99)
        b = student_t_process(rng_seed=99)
        np.testing.assert_array_equal(a, b)


class TestLevyStableProcess:
    def test_correct_length(self):
        rets = levy_stable_process(alpha=1.7, n_steps=200, rng_seed=42)
        assert len(rets) == 200

    def test_gaussian_limit(self):
        """alpha=2 should produce approximately Gaussian returns."""
        rets = levy_stable_process(alpha=2.0, sigma=0.01, n_steps=10000, rng_seed=42)
        k = float(stats.kurtosis(rets))
        assert abs(k) < 1, f"alpha=2 should be near-Gaussian, got kurtosis={k}"

    def test_invalid_alpha(self):
        with pytest.raises(ValueError, match="alpha"):
            levy_stable_process(alpha=0)

    def test_invalid_beta(self):
        with pytest.raises(ValueError, match="beta"):
            levy_stable_process(beta=2.0)


class TestRegimeSwitchingGBM:
    def test_correct_shape(self):
        params = [RegimeParams(mu=0.05, sigma=0.1), RegimeParams(mu=-0.02, sigma=0.3)]
        trans = np.array([[0.95, 0.05], [0.10, 0.90]])
        prices, regimes = regime_switching_gbm(params, trans, n_steps=100, rng_seed=42)
        assert len(prices) == 101
        assert len(regimes) == 100

    def test_regime_values(self):
        params = [RegimeParams(mu=0.05, sigma=0.1), RegimeParams(mu=-0.02, sigma=0.3)]
        trans = np.array([[0.95, 0.05], [0.10, 0.90]])
        _, regimes = regime_switching_gbm(params, trans, n_steps=500, rng_seed=42)
        assert set(regimes).issubset({0, 1})

    def test_dimension_mismatch(self):
        params = [RegimeParams(mu=0.05, sigma=0.1)]
        trans = np.array([[0.9, 0.1], [0.1, 0.9]])
        with pytest.raises(ValueError, match="regimes"):
            regime_switching_gbm(params, trans)

    def test_higher_vol_in_crisis_regime(self):
        """Crisis regime (higher sigma) should produce more volatile paths."""
        calm = RegimeParams(mu=0.05, sigma=0.05)
        crisis = RegimeParams(mu=-0.10, sigma=0.50)
        # Force all-calm
        calm_trans = np.array([[1.0, 0.0], [1.0, 0.0]])
        crisis_trans = np.array([[0.0, 1.0], [0.0, 1.0]])

        prices_calm, _ = regime_switching_gbm([calm, crisis], calm_trans, n_steps=1000, rng_seed=1)
        prices_crisis, _ = regime_switching_gbm([calm, crisis], crisis_trans, n_steps=1000, rng_seed=1)

        vol_calm = np.std(np.diff(np.log(prices_calm)))
        vol_crisis = np.std(np.diff(np.log(prices_crisis)))
        assert vol_crisis > vol_calm * 2


class TestKouDoubleExponentialJump:
    def test_correct_length(self):
        prices = kou_double_exponential_jump(n_steps=100, rng_seed=42)
        assert len(prices) == 101

    def test_positive_prices(self):
        prices = kou_double_exponential_jump(n_steps=500, rng_seed=42)
        assert np.all(prices > 0), "All prices should be positive"

    def test_eta1_too_low(self):
        with pytest.raises(ValueError, match="eta1 must be > 1"):
            kou_double_exponential_jump(eta1=0.5)

    def test_reproducibility(self):
        a = kou_double_exponential_jump(rng_seed=77)
        b = kou_double_exponential_jump(rng_seed=77)
        np.testing.assert_array_equal(a, b)

    def test_jumps_produce_fatter_tails(self):
        """Kou model with jumps should have fatter tails than pure GBM."""
        prices_kou = kou_double_exponential_jump(
            lam=5.0, n_steps=10000, rng_seed=42
        )
        rets_kou = np.diff(np.log(prices_kou))
        k_kou = float(stats.kurtosis(rets_kou))
        assert k_kou > 0.5, f"Expected excess kurtosis from jumps, got {k_kou}"
