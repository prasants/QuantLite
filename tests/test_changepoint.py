"""Tests for change-point detection."""

from __future__ import annotations

import numpy as np
import pytest

from quantlite.regimes.changepoint import ChangePoint, detect_changepoints


@pytest.fixture()
def mean_shift_data() -> np.ndarray:
    """Data with a clear mean shift at index 200."""
    rng = np.random.default_rng(42)
    low = rng.normal(0.001, 0.005, 200)
    high = rng.normal(0.01, 0.005, 200)
    return np.concatenate([low, high])


class TestCUSUM:
    def test_detects_mean_shift(self, mean_shift_data: np.ndarray) -> None:
        cps = detect_changepoints(mean_shift_data, method="cusum", penalty=4.0)
        assert len(cps) > 0
        # At least one detection should be near index 200 (+/- 50)
        indices = [cp.index for cp in cps]
        assert any(150 < idx < 250 for idx in indices)

    def test_returns_changepoint_objects(self, mean_shift_data: np.ndarray) -> None:
        cps = detect_changepoints(mean_shift_data, method="cusum", penalty=4.0)
        for cp in cps:
            assert isinstance(cp, ChangePoint)
            assert 0 <= cp.confidence <= 1
            assert cp.direction in {"increase_mean", "decrease_mean"}


class TestBayesian:
    def test_detects_mean_shift(self, mean_shift_data: np.ndarray) -> None:
        cps = detect_changepoints(mean_shift_data, method="bayesian", penalty=20.0)
        assert len(cps) > 0

    def test_returns_changepoint_objects(self, mean_shift_data: np.ndarray) -> None:
        cps = detect_changepoints(mean_shift_data, method="bayesian", penalty=20.0)
        for cp in cps:
            assert isinstance(cp, ChangePoint)


class TestEdgeCases:
    def test_unknown_method(self) -> None:
        with pytest.raises(ValueError, match="Unknown method"):
            detect_changepoints(np.zeros(100), method="unknown")

    def test_constant_data(self) -> None:
        cps = detect_changepoints(np.ones(100), method="cusum", penalty=5.0)
        assert len(cps) == 0
