"""Tests for hierarchical clustering and HRP weights."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantlite.dependency.clustering import (
    correlation_distance,
    hierarchical_cluster,
    hrp_weights,
    quasi_diagonalise,
)


@pytest.fixture()
def returns_df() -> pd.DataFrame:
    """Generate multi-asset returns for clustering tests."""
    rng = np.random.default_rng(42)
    n = 300
    data = rng.normal(0, 0.01, (n, 5))
    data[:, 1] += 0.7 * data[:, 0]
    data[:, 3] += 0.5 * data[:, 2]
    return pd.DataFrame(data, columns=["A", "B", "C", "D", "E"])


class TestCorrelationDistance:
    def test_diagonal_zero(self) -> None:
        corr = np.array([[1.0, 0.5], [0.5, 1.0]])
        dist = correlation_distance(corr)
        np.testing.assert_allclose(np.diag(dist), 0.0, atol=1e-10)

    def test_range(self) -> None:
        corr = np.array([[1.0, -1.0], [-1.0, 1.0]])
        dist = correlation_distance(corr)
        assert dist[0, 1] == pytest.approx(1.0)


class TestHierarchicalCluster:
    def test_returns_linkage(self, returns_df: pd.DataFrame) -> None:
        corr = returns_df.corr()
        link = hierarchical_cluster(corr)
        assert link.shape[1] == 4  # scipy linkage format
        assert link.shape[0] == len(corr) - 1


class TestQuasiDiagonalise:
    def test_preserves_shape(self, returns_df: pd.DataFrame) -> None:
        corr = returns_df.corr()
        link = hierarchical_cluster(corr)
        reordered = quasi_diagonalise(link, corr)
        assert reordered.shape == corr.shape


class TestHRPWeights:
    def test_weights_sum_to_one(self, returns_df: pd.DataFrame) -> None:
        weights = hrp_weights(returns_df)
        assert abs(sum(weights.values()) - 1.0) < 1e-10

    def test_all_positive(self, returns_df: pd.DataFrame) -> None:
        weights = hrp_weights(returns_df)
        assert all(w > 0 for w in weights.values())

    def test_correct_assets(self, returns_df: pd.DataFrame) -> None:
        weights = hrp_weights(returns_df)
        assert set(weights.keys()) == set(returns_df.columns)
