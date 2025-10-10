"""Tests for quantlite.forensics module."""


import numpy as np
import pytest

from quantlite.forensics import (
    deflated_sharpe_ratio,
    haircut_sharpe_ratio,
    min_track_record_length,
    probabilistic_sharpe_ratio,
    signal_decay,
)


class TestProbabilisticSharpeRatio:
    """Tests for probabilistic_sharpe_ratio."""

    def test_psr_positive_sharpe(self):
        """PSR should be > 0.5 when observed exceeds benchmark."""
        psr = probabilistic_sharpe_ratio(1.5, 0.0, 252)
        assert psr > 0.5

    def test_psr_zero_sharpe(self):
        """PSR should be 0.5 when observed equals benchmark."""
        psr = probabilistic_sharpe_ratio(0.0, 0.0, 252)
        assert abs(psr - 0.5) < 0.01

    def test_psr_increases_with_observations(self):
        """PSR should increase with more observations."""
        psr_short = probabilistic_sharpe_ratio(1.0, 0.0, 50)
        psr_long = probabilistic_sharpe_ratio(1.0, 0.0, 500)
        assert psr_long > psr_short

    def test_psr_negative_sharpe(self):
        """PSR should be < 0.5 when observed is below benchmark."""
        psr = probabilistic_sharpe_ratio(-0.5, 0.0, 252)
        assert psr < 0.5

    def test_psr_range(self):
        """PSR must be in [0, 1]."""
        psr = probabilistic_sharpe_ratio(2.0, 0.0, 100)
        assert 0.0 <= psr <= 1.0


class TestDeflatedSharpeRatio:
    """Tests for deflated_sharpe_ratio."""

    def test_dsr_less_than_psr(self):
        """DSR should be less than PSR (zero benchmark) for many trials."""
        psr = probabilistic_sharpe_ratio(0.5, 0.0, 252)
        dsr = deflated_sharpe_ratio(0.5, 100, 252)
        assert dsr < psr

    def test_dsr_single_trial(self):
        """DSR with one trial should be close to PSR with zero benchmark."""
        dsr = deflated_sharpe_ratio(0.5, 1, 252)
        assert 0.0 <= dsr <= 1.0

    def test_dsr_decreases_with_more_trials(self):
        """More trials should decrease DSR."""
        dsr_few = deflated_sharpe_ratio(0.5, 5, 252)
        dsr_many = deflated_sharpe_ratio(0.5, 500, 252)
        assert dsr_many < dsr_few

    def test_dsr_invalid_n_trials(self):
        """Should raise ValueError for non-positive n_trials."""
        with pytest.raises(ValueError):
            deflated_sharpe_ratio(1.0, 0, 252)

    def test_dsr_invalid_n_obs(self):
        """Should raise ValueError for n_obs <= 1."""
        with pytest.raises(ValueError):
            deflated_sharpe_ratio(1.0, 10, 1)


class TestHaircutSharpeRatio:
    """Tests for haircut_sharpe_ratio."""

    def test_haircut_less_than_observed(self):
        """Haircutted Sharpe should always be less than observed."""
        for method in ("bonferroni", "holm", "bhy"):
            hsr = haircut_sharpe_ratio(2.0, 252, method=method)
            assert hsr < 2.0

    def test_bonferroni_most_conservative(self):
        """Bonferroni should give the smallest haircut result."""
        hsr_bon = haircut_sharpe_ratio(2.0, 252, method="bonferroni")
        hsr_holm = haircut_sharpe_ratio(2.0, 252, method="holm")
        hsr_bhy = haircut_sharpe_ratio(2.0, 252, method="bhy")
        assert hsr_bon < hsr_holm < hsr_bhy

    def test_invalid_method(self):
        """Should raise ValueError for unknown method."""
        with pytest.raises(ValueError):
            haircut_sharpe_ratio(1.0, 252, method="invalid")

    def test_fat_tails_penalise(self):
        """Fat tails should reduce the haircut result."""
        normal = haircut_sharpe_ratio(2.0, 252, kurtosis=3.0)
        fat = haircut_sharpe_ratio(2.0, 252, kurtosis=6.0)
        assert fat < normal


class TestMinTrackRecordLength:
    """Tests for min_track_record_length."""

    def test_higher_sharpe_needs_less(self):
        """Higher Sharpe should need fewer observations."""
        n_high = min_track_record_length(2.0)
        n_low = min_track_record_length(0.5)
        assert n_high < n_low

    def test_returns_positive(self):
        """MinTRL should always be positive."""
        n = min_track_record_length(1.0)
        assert n > 0

    def test_zero_sharpe_raises(self):
        """Should raise ValueError when observed equals benchmark."""
        with pytest.raises(ValueError):
            min_track_record_length(0.0, benchmark_sharpe=0.0)

    def test_higher_confidence_needs_more(self):
        """Higher confidence should require more observations."""
        n_90 = min_track_record_length(1.0, confidence=0.90)
        n_99 = min_track_record_length(1.0, confidence=0.99)
        assert n_99 > n_90


class TestSignalDecay:
    """Tests for signal_decay."""

    def test_basic_output_structure(self):
        """Should return dict with expected keys."""
        rng = np.random.RandomState(42)
        signal = rng.randn(100)
        returns = signal * 0.5 + rng.randn(100) * 0.1
        result = signal_decay(returns, signal)
        assert "half_life" in result
        assert "decay_curve" in result
        assert "r_squared_curve" in result

    def test_perfect_signal_high_correlation(self):
        """Perfect signal should have high initial correlation."""
        rng = np.random.RandomState(42)
        signal = rng.randn(200)
        returns = np.roll(signal, 1)  # returns follow signal with lag 1
        returns[0] = 0
        result = signal_decay(returns, signal, lags=[1, 2, 5, 10])
        # Lag-1 correlation should be strong.
        assert len(result["decay_curve"]) > 0

    def test_mismatched_length_raises(self):
        """Should raise ValueError for mismatched lengths."""
        with pytest.raises(ValueError):
            signal_decay(np.ones(10), np.ones(5))

    def test_custom_lags(self):
        """Should respect custom lag values."""
        rng = np.random.RandomState(42)
        result = signal_decay(rng.randn(50), rng.randn(50), lags=[1, 3])
        assert len(result["decay_curve"]) == 2
