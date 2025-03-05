"""Smoke tests for regime visualisation."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest

from quantlite.viz.regimes import (
    plot_regime_distributions,
    plot_regime_summary,
    plot_regime_timeline,
)


@pytest.fixture()
def returns_and_regimes() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    returns = rng.normal(0, 0.01, 400)
    regimes = np.array([0] * 200 + [1] * 200)
    return returns, regimes


def test_plot_regime_timeline(
    returns_and_regimes: tuple[np.ndarray, np.ndarray],
) -> None:
    returns, regimes = returns_and_regimes
    fig, ax = plot_regime_timeline(returns, regimes)
    assert fig is not None
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_plot_regime_distributions(
    returns_and_regimes: tuple[np.ndarray, np.ndarray],
) -> None:
    returns, regimes = returns_and_regimes
    fig, axes = plot_regime_distributions(returns, regimes)
    assert fig is not None
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_plot_regime_summary(
    returns_and_regimes: tuple[np.ndarray, np.ndarray],
) -> None:
    returns, regimes = returns_and_regimes
    fig, axes = plot_regime_summary(returns, regimes)
    assert fig is not None
    import matplotlib.pyplot as plt
    plt.close(fig)
