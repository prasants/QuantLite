"""Tests for dynamic and stress correlation analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from quantlite.dependency.correlation import (
    correlation_breakdown_test,
    exponential_weighted_correlation,
    rank_correlation,
    rolling_correlation,
    stress_correlation,
)


@pytest.fixture()
def returns_pair() -> tuple[pd.Series, pd.Series]:
    """Generate two correlated return series."""
    rng = np.random.default_rng(42)
    n = 500
    x = rng.normal(0, 0.01, n)
    y = 0.6 * x + 0.4 * rng.normal(0, 0.01, n)
    return pd.Series(x), pd.Series(y)


@pytest.fixture()
def returns_df() -> pd.DataFrame:
    """Generate multi-asset returns DataFrame."""
    rng = np.random.default_rng(42)
    n = 500
    data = rng.normal(0, 0.01, (n, 4))
    # Add some correlation
    data[:, 1] += 0.5 * data[:, 0]
    data[:, 3] += 0.3 * data[:, 2]
    return pd.DataFrame(data, columns=["A", "B", "C", "D"])


class TestRollingCorrelation:
    def test_output_length(self, returns_pair: tuple) -> None:
        x, y = returns_pair
        result = rolling_correlation(x, y, window=30)
        assert len(result) == len(x)

    def test_matches_pandas(self, returns_pair: tuple) -> None:
        x, y = returns_pair
        result = rolling_correlation(x, y, window=30)
        expected = x.rolling(30).corr(y)
        np.testing.assert_allclose(
            result.dropna().values, expected.dropna().values, atol=1e-10
        )


class TestEWMACorrelation:
    def test_output_length(self, returns_pair: tuple) -> None:
        x, y = returns_pair
        result = exponential_weighted_correlation(x, y, halflife=20)
        assert len(result) == len(x)

    def test_range(self, returns_pair: tuple) -> None:
        x, y = returns_pair
        result = exponential_weighted_correlation(x, y, halflife=20)
        valid = result.dropna()
        assert (valid >= -1.01).all() and (valid <= 1.01).all()


class TestStressCorrelation:
    def test_returns_valid_matrix(self, returns_df: pd.DataFrame) -> None:
        result = stress_correlation(returns_df, threshold_percentile=20)
        assert result.shape == (4, 4)
        np.testing.assert_allclose(np.diag(result.values), 1.0, atol=1e-10)


class TestCorrelationBreakdownTest:
    def test_returns_dict_keys(self, returns_df: pd.DataFrame) -> None:
        result = correlation_breakdown_test(returns_df)
        assert "test_statistic" in result
        assert "p_value" in result
        assert "calm_corr" in result
        assert "stress_corr" in result
        assert 0 <= result["p_value"] <= 1


class TestRankCorrelation:
    def test_spearman_matches_scipy(self, returns_pair: tuple) -> None:
        x, y = returns_pair
        corr, p = rank_correlation(x, y, method="spearman")
        expected_corr, expected_p = stats.spearmanr(x, y)
        assert abs(corr - expected_corr) < 1e-10
        assert abs(p - expected_p) < 1e-10

    def test_kendall_matches_scipy(self, returns_pair: tuple) -> None:
        x, y = returns_pair
        corr, p = rank_correlation(x, y, method="kendall")
        expected_corr, expected_p = stats.kendalltau(x, y)
        assert abs(corr - expected_corr) < 1e-10

    def test_unknown_method(self, returns_pair: tuple) -> None:
        x, y = returns_pair
        with pytest.raises(ValueError, match="Unknown method"):
            rank_correlation(x, y, method="pearson")
