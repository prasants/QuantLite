"""Tests for quantlite.resample module."""


import numpy as np
import pytest

from quantlite.resample import (
    block_bootstrap,
    bootstrap_confidence_interval,
    bootstrap_drawdown_distribution,
    bootstrap_sharpe_distribution,
    stationary_bootstrap,
)


class TestBlockBootstrap:
    """Tests for block_bootstrap."""

    def test_output_shape(self):
        """Should return correct shape."""
        ret = np.random.RandomState(42).randn(100)
        samples = block_bootstrap(ret, block_size=10, n_samples=50, seed=42)
        assert samples.shape == (50, 100)

    def test_reproducibility(self):
        """Same seed should give same results."""
        ret = np.random.RandomState(42).randn(100)
        s1 = block_bootstrap(ret, 10, 20, seed=123)
        s2 = block_bootstrap(ret, 10, 20, seed=123)
        np.testing.assert_array_equal(s1, s2)

    def test_different_seeds(self):
        """Different seeds should give different results."""
        ret = np.random.RandomState(42).randn(100)
        s1 = block_bootstrap(ret, 10, 20, seed=1)
        s2 = block_bootstrap(ret, 10, 20, seed=2)
        assert not np.array_equal(s1, s2)

    def test_block_too_large_raises(self):
        """Should raise ValueError if block_size > len(returns)."""
        with pytest.raises(ValueError):
            block_bootstrap(np.ones(5), block_size=10, n_samples=1)

    def test_values_from_original(self):
        """All values should come from the original series."""
        ret = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        samples = block_bootstrap(ret, block_size=2, n_samples=10, seed=42)
        original_set = set(ret)
        for row in samples:
            for val in row:
                assert val in original_set


class TestStationaryBootstrap:
    """Tests for stationary_bootstrap."""

    def test_output_shape(self):
        """Should return correct shape."""
        ret = np.random.RandomState(42).randn(100)
        samples = stationary_bootstrap(ret, avg_block_size=10, n_samples=50, seed=42)
        assert samples.shape == (50, 100)

    def test_reproducibility(self):
        """Same seed should give same results."""
        ret = np.random.RandomState(42).randn(100)
        s1 = stationary_bootstrap(ret, 10, 20, seed=123)
        s2 = stationary_bootstrap(ret, 10, 20, seed=123)
        np.testing.assert_array_equal(s1, s2)

    def test_values_from_original(self):
        """All values should come from the original series."""
        ret = np.array([1.0, 2.0, 3.0, 4.0])
        samples = stationary_bootstrap(ret, avg_block_size=2, n_samples=10, seed=42)
        original_set = set(ret)
        for row in samples:
            for val in row:
                assert val in original_set

    def test_invalid_block_size(self):
        """Should raise for avg_block_size < 1."""
        with pytest.raises(ValueError):
            stationary_bootstrap(np.ones(10), avg_block_size=0)


class TestBootstrapConfidenceInterval:
    """Tests for bootstrap_confidence_interval."""

    def test_ci_contains_point_estimate(self):
        """CI should typically contain the point estimate."""
        rng = np.random.RandomState(42)
        ret = rng.randn(200) * 0.01 + 0.001
        result = bootstrap_confidence_interval(
            ret, np.mean, n_samples=200, seed=42,
        )
        assert result["ci_lower"] <= result["point_estimate"] <= result["ci_upper"]

    def test_distribution_length(self):
        """Distribution should have n_samples elements."""
        ret = np.random.RandomState(42).randn(50)
        result = bootstrap_confidence_interval(
            ret, np.mean, n_samples=100, seed=42,
        )
        assert len(result["distribution"]) == 100

    def test_stationary_method(self):
        """Should work with stationary bootstrap method."""
        ret = np.random.RandomState(42).randn(50)
        result = bootstrap_confidence_interval(
            ret, np.mean, n_samples=50, method="stationary", seed=42,
        )
        assert result["ci_lower"] < result["ci_upper"]

    def test_higher_confidence_wider_ci(self):
        """Higher confidence should give wider CI."""
        ret = np.random.RandomState(42).randn(100)
        ci_90 = bootstrap_confidence_interval(
            ret, np.mean, n_samples=500, confidence=0.90, seed=42,
        )
        ci_99 = bootstrap_confidence_interval(
            ret, np.mean, n_samples=500, confidence=0.99, seed=42,
        )
        width_90 = ci_90["ci_upper"] - ci_90["ci_lower"]
        width_99 = ci_99["ci_upper"] - ci_99["ci_lower"]
        assert width_99 > width_90


class TestBootstrapSharpeDistribution:
    """Tests for bootstrap_sharpe_distribution."""

    def test_basic_output(self):
        """Should return dict with expected keys."""
        ret = np.random.RandomState(42).randn(100) * 0.01
        result = bootstrap_sharpe_distribution(ret, n_samples=50, seed=42)
        assert "point_estimate" in result
        assert "ci_lower" in result
        assert "ci_upper" in result
        assert "distribution" in result

    def test_positive_returns_positive_sharpe(self):
        """Positive mean returns should yield positive point estimate."""
        ret = np.random.RandomState(42).randn(200) * 0.01 + 0.005
        result = bootstrap_sharpe_distribution(ret, n_samples=50, seed=42)
        assert result["point_estimate"] > 0


class TestBootstrapDrawdownDistribution:
    """Tests for bootstrap_drawdown_distribution."""

    def test_basic_output(self):
        """Should return dict with expected keys."""
        ret = np.random.RandomState(42).randn(100) * 0.01
        result = bootstrap_drawdown_distribution(ret, n_samples=50, seed=42)
        assert "point_estimate" in result
        assert result["point_estimate"] >= 0

    def test_drawdown_non_negative(self):
        """All drawdown values should be non-negative."""
        ret = np.random.RandomState(42).randn(100) * 0.01
        result = bootstrap_drawdown_distribution(ret, n_samples=50, seed=42)
        assert all(v >= 0 for v in result["distribution"])
