"""Tests for HMM regime detection."""

from __future__ import annotations

import numpy as np
import pytest

try:
    import hmmlearn  # noqa: F401
    HAS_HMMLEARN = True
except ImportError:
    HAS_HMMLEARN = False

pytestmark = pytest.mark.skipif(not HAS_HMMLEARN, reason="hmmlearn not installed")


@pytest.fixture()
def regime_data() -> np.ndarray:
    """Synthetic two-regime data: concat low-vol and high-vol normals."""
    rng = np.random.default_rng(42)
    calm = rng.normal(0.001, 0.005, 200)
    crisis = rng.normal(-0.002, 0.02, 200)
    return np.concatenate([calm, crisis, calm])


class TestFitRegimeModel:
    def test_two_regime_fit(self, regime_data: np.ndarray) -> None:
        from quantlite.regimes.hmm import fit_regime_model

        model = fit_regime_model(regime_data, n_regimes=2, rng_seed=42)
        assert model.n_regimes == 2
        assert len(model.means) == 2
        assert len(model.variances) == 2
        assert model.transition_matrix.shape == (2, 2)

    def test_means_approximately_correct(self, regime_data: np.ndarray) -> None:
        from quantlite.regimes.hmm import fit_regime_model

        model = fit_regime_model(regime_data, n_regimes=2, rng_seed=42)
        # Regime 0 should have lower mean (crisis)
        assert model.means[0] < model.means[1]

    def test_regime_labels_shape(self, regime_data: np.ndarray) -> None:
        from quantlite.regimes.hmm import fit_regime_model

        model = fit_regime_model(regime_data, n_regimes=2, rng_seed=42)
        assert len(model.regime_labels) == len(regime_data)

    def test_stationary_distribution_sums_to_one(self, regime_data: np.ndarray) -> None:
        from quantlite.regimes.hmm import fit_regime_model

        model = fit_regime_model(regime_data, n_regimes=2, rng_seed=42)
        assert abs(model.stationary_distribution.sum() - 1.0) < 1e-6


class TestPredictRegimes:
    def test_output_shape(self, regime_data: np.ndarray) -> None:
        from quantlite.regimes.hmm import fit_regime_model, predict_regimes

        model = fit_regime_model(regime_data, n_regimes=2, rng_seed=42)
        labels = predict_regimes(model, regime_data)
        assert len(labels) == len(regime_data)


class TestRegimeProbabilities:
    def test_output_shape(self, regime_data: np.ndarray) -> None:
        from quantlite.regimes.hmm import fit_regime_model, regime_probabilities

        model = fit_regime_model(regime_data, n_regimes=2, rng_seed=42)
        probs = regime_probabilities(model, regime_data)
        assert probs.shape == (len(regime_data), 2)

    def test_probs_sum_to_one(self, regime_data: np.ndarray) -> None:
        from quantlite.regimes.hmm import fit_regime_model, regime_probabilities

        model = fit_regime_model(regime_data, n_regimes=2, rng_seed=42)
        probs = regime_probabilities(model, regime_data)
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-6)


class TestSelectNRegimes:
    def test_returns_best_model(self, regime_data: np.ndarray) -> None:
        from quantlite.regimes.hmm import select_n_regimes

        model = select_n_regimes(regime_data, max_regimes=3, rng_seed=42)
        assert model.n_regimes in {2, 3}
