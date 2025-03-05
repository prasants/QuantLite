"""Smoke tests for dependency visualisation."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from quantlite.dependency.copulas import GaussianCopula, ClaytonCopula
from quantlite.viz.dependency import (
    plot_copula_contour,
    plot_correlation_dynamics,
    plot_correlation_matrix,
    plot_stress_correlation,
)


@pytest.fixture()
def correlated_data() -> np.ndarray:
    rng = np.random.default_rng(42)
    cov = [[1.0, 0.6], [0.6, 1.0]]
    return rng.multivariate_normal([0, 0], cov, size=300)


@pytest.fixture()
def returns_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        rng.normal(0, 0.01, (200, 3)), columns=["A", "B", "C"],
    )


def test_plot_copula_contour(correlated_data: np.ndarray) -> None:
    cop = ClaytonCopula()
    cop.fit(correlated_data)
    fig, ax = plot_copula_contour(cop, correlated_data)
    assert fig is not None
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_plot_correlation_matrix(returns_df: pd.DataFrame) -> None:
    fig, ax = plot_correlation_matrix(returns_df.corr())
    assert fig is not None
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_plot_stress_correlation(returns_df: pd.DataFrame) -> None:
    calm = returns_df.corr()
    stress = returns_df.corr()  # same for smoke test
    fig, axes = plot_stress_correlation(calm, stress)
    assert fig is not None
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_plot_correlation_dynamics(returns_df: pd.DataFrame) -> None:
    fig, ax = plot_correlation_dynamics(
        returns_df["A"].values, returns_df["B"].values, window=30,
    )
    assert fig is not None
    import matplotlib.pyplot as plt
    plt.close(fig)
