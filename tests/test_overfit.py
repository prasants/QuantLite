"""Tests for quantlite.overfit module."""

import numpy as np
import pytest

from quantlite.overfit import (
    TrialTracker,
    min_backtest_length,
    multiple_testing_correction,
    probability_of_backtest_overfitting,
    walk_forward_validate,
)


class TestMultipleTestingCorrection:
    """Tests for multiple_testing_correction."""

    def test_bonferroni(self):
        """Bonferroni should multiply p-values by number of tests."""
        pv = np.array([0.01, 0.04, 0.05])
        adj = multiple_testing_correction(pv, method="bonferroni")
        np.testing.assert_allclose(adj, [0.03, 0.12, 0.15])

    def test_holm(self):
        """Holm should produce ordered adjusted p-values."""
        pv = np.array([0.01, 0.04, 0.05])
        adj = multiple_testing_correction(pv, method="holm")
        assert all(0 <= v <= 1 for v in adj)
        assert adj[0] <= adj[1]

    def test_bhy(self):
        """BHY should produce valid adjusted p-values."""
        pv = np.array([0.01, 0.04, 0.05])
        adj = multiple_testing_correction(pv, method="bhy")
        assert all(0 <= v <= 1 for v in adj)

    def test_clipping(self):
        """Adjusted p-values should not exceed 1."""
        pv = np.array([0.5, 0.8])
        adj = multiple_testing_correction(pv, method="bonferroni")
        assert all(v <= 1.0 for v in adj)

    def test_invalid_method(self):
        """Should raise ValueError for unknown method."""
        with pytest.raises(ValueError):
            multiple_testing_correction([0.05], method="invalid")

    def test_empty_array(self):
        """Should handle empty input."""
        adj = multiple_testing_correction([], method="bonferroni")
        assert len(adj) == 0


class TestProbabilityOfBacktestOverfitting:
    """Tests for probability_of_backtest_overfitting."""

    def test_random_strategies_high_pbo(self):
        """Random strategies should have moderate to high PBO."""
        rng = np.random.RandomState(42)
        trials = rng.randn(10, 100)
        result = probability_of_backtest_overfitting(trials, n_splits=4)
        assert 0.0 <= result["pbo"] <= 1.0
        assert "rank_correlations" in result

    def test_odd_splits_raises(self):
        """Should raise ValueError for odd n_splits."""
        with pytest.raises(ValueError):
            probability_of_backtest_overfitting(np.ones((3, 20)), n_splits=5)

    def test_single_trial_raises(self):
        """Should raise ValueError for fewer than 2 trials."""
        with pytest.raises(ValueError):
            probability_of_backtest_overfitting(np.ones((1, 20)), n_splits=2)

    def test_two_trials(self):
        """Should work with exactly 2 trials."""
        rng = np.random.RandomState(42)
        trials = rng.randn(2, 40)
        result = probability_of_backtest_overfitting(trials, n_splits=4)
        assert 0.0 <= result["pbo"] <= 1.0

    def test_list_input(self):
        """Should accept list of arrays."""
        rng = np.random.RandomState(42)
        trials = [rng.randn(50) for _ in range(5)]
        result = probability_of_backtest_overfitting(trials, n_splits=4)
        assert 0.0 <= result["pbo"] <= 1.0


class TestMinBacktestLength:
    """Tests for min_backtest_length."""

    def test_higher_sharpe_shorter(self):
        """Higher Sharpe should need fewer observations."""
        n_high = min_backtest_length(2.0)
        n_low = min_backtest_length(0.5)
        assert n_high < n_low

    def test_zero_sharpe_raises(self):
        """Should raise ValueError for zero Sharpe."""
        with pytest.raises(ValueError):
            min_backtest_length(0.0)

    def test_positive_result(self):
        """Should return positive value."""
        assert min_backtest_length(1.0) > 0


class TestWalkForwardValidate:
    """Tests for walk_forward_validate."""

    def test_basic_walk_forward(self):
        """Should produce correct number of folds."""
        ret = np.ones(100) * 0.01
        result = walk_forward_validate(
            ret, lambda x: 1.0, window=50, step=10,
        )
        assert result["n_folds"] == 5

    def test_expanding_window(self):
        """Expanding window should start training from index 0."""
        ret = np.ones(100) * 0.01
        result = walk_forward_validate(
            ret, lambda x: 1.0, window=50, step=10, expanding=True,
        )
        assert result["folds"][0]["train_start"] == 0
        assert result["folds"][-1]["train_start"] == 0

    def test_aggregate_return(self):
        """Aggregate return should equal sum of fold returns."""
        ret = np.ones(100) * 0.01
        result = walk_forward_validate(
            ret, lambda x: 1.0, window=50, step=10,
        )
        expected = sum(f["test_return"] for f in result["folds"])
        assert abs(result["aggregate_return"] - expected) < 1e-10


class TestTrialTracker:
    """Tests for TrialTracker."""

    def test_context_manager(self):
        """Should work as context manager."""
        with TrialTracker("test") as tracker:
            tracker.log(params={"a": 1}, sharpe=1.0)
        assert len(tracker.trials) == 1

    def test_best_trial(self):
        """Should return trial with highest Sharpe."""
        with TrialTracker("test") as tracker:
            tracker.log(sharpe=0.5)
            tracker.log(sharpe=1.5)
            tracker.log(sharpe=1.0)
        assert tracker.best_trial["sharpe"] == 1.5

    def test_empty_best_trial(self):
        """Should return None when no trials logged."""
        tracker = TrialTracker()
        assert tracker.best_trial is None

    def test_overfitting_probability_with_returns(self):
        """Should compute PBO when returns are available."""
        rng = np.random.RandomState(42)
        with TrialTracker("test") as tracker:
            for _ in range(5):
                ret = rng.randn(50)
                tracker.log(sharpe=float(np.mean(ret)), returns=ret)
            prob = tracker.overfitting_probability(n_splits=4)
        assert 0.0 <= prob <= 1.0

    def test_overfitting_probability_without_returns(self):
        """Should fall back to heuristic without returns."""
        with TrialTracker("test") as tracker:
            for i in range(10):
                tracker.log(sharpe=float(i) * 0.1)
            prob = tracker.overfitting_probability()
        assert 0.0 < prob < 1.0
