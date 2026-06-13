"""Tests for regime-conditional risk metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantlite.regimes.conditional import (
    conditional_correlation,
    conditional_metrics,
    regime_aware_var,
)


@pytest.fixture()
def returns_and_regimes() -> tuple[np.ndarray, np.ndarray]:
    """Synthetic returns with two regimes."""
    rng = np.random.default_rng(42)
    n = 400
    returns = rng.normal(0, 0.01, n)
    regimes = np.array([0] * 200 + [1] * 200)
    # Make regime 1 more volatile
    returns[200:] *= 3
    return returns, regimes


class TestConditionalMetrics:
    def test_returns_dict_per_regime(
        self, returns_and_regimes: tuple,
    ) -> None:
        returns, regimes = returns_and_regimes
        result = conditional_metrics(returns, regimes)
        assert 0 in result
        assert 1 in result

    def test_has_expected_keys(
        self, returns_and_regimes: tuple,
    ) -> None:
        returns, regimes = returns_and_regimes
        result = conditional_metrics(returns, regimes)
        for regime_metrics in result.values():
            assert "mean" in regime_metrics
            assert "volatility" in regime_metrics
            assert "n_observations" in regime_metrics


class TestConditionalCorrelation:
    def test_returns_corr_per_regime(self) -> None:
        rng = np.random.default_rng(42)
        df = pd.DataFrame(rng.normal(0, 0.01, (200, 3)), columns=["A", "B", "C"])
        regimes = np.array([0] * 100 + [1] * 100)
        result = conditional_correlation(df, regimes)
        assert 0 in result
        assert 1 in result
        assert result[0].shape == (3, 3)


class TestRegimeAwareVaR:
    def test_returns_float(
        self, returns_and_regimes: tuple,
    ) -> None:
        returns, regimes = returns_and_regimes
        var = regime_aware_var(returns, regimes, alpha=0.05)
        assert isinstance(var, float)
        assert np.isfinite(var)

    def test_with_custom_probs(
        self, returns_and_regimes: tuple,
    ) -> None:
        returns, regimes = returns_and_regimes
        probs = np.array([0.3, 0.7])
        var = regime_aware_var(returns, regimes, alpha=0.05, current_probs=probs)
        assert isinstance(var, float)
