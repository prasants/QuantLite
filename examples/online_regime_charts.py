"""Generate online regime detection visualisations for documentation."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from quantlite.viz.online_regimes import (
    plot_detection_lag,
    plot_regime_evolution,
    plot_regime_transition_live,
)
from quantlite.viz.theme import apply_few_theme

DOCS_IMAGES = Path(__file__).resolve().parent.parent / "docs" / "images"
DOCS_IMAGES.mkdir(parents=True, exist_ok=True)

apply_few_theme()


def _simulate_regime_data(n: int = 500, seed: int = 42):
    """Generate synthetic regime-switching return data."""
    rng = np.random.default_rng(seed)
    regimes = np.zeros(n, dtype=int)
    returns = np.zeros(n)

    # Regime parameters: (mean, std)
    params = {0: (0.0005, 0.008), 1: (0.0, 0.015), 2: (-0.001, 0.03)}
    transition = {0: {0: 0.97, 1: 0.025, 2: 0.005},
                  1: {0: 0.05, 1: 0.90, 2: 0.05},
                  2: {0: 0.01, 1: 0.04, 2: 0.95}}

    current = 0
    for i in range(n):
        regimes[i] = current
        mu, sigma = params[current]
        returns[i] = rng.normal(mu, sigma)
        probs = transition[current]
        current = rng.choice([0, 1, 2], p=[probs[0], probs[1], probs[2]])

    return regimes, returns


def chart_regime_evolution() -> None:
    """Regime posterior probabilities evolving over time."""
    rng = np.random.default_rng(42)
    true_regimes, returns = _simulate_regime_data(300)
    timestamps = np.arange(len(returns))

    # Simulate posterior probabilities (smoothed one-hot with noise)
    probs = np.zeros((len(returns), 3))
    for i, r in enumerate(true_regimes):
        probs[i, r] = 0.7 + rng.uniform(0, 0.25)
        remaining = 1.0 - probs[i, r]
        others = [j for j in range(3) if j != r]
        split = rng.dirichlet([1, 1])
        for k, j in enumerate(others):
            probs[i, j] = remaining * split[k]

    # Smooth the probabilities
    from scipy.ndimage import uniform_filter1d
    for col in range(3):
        probs[:, col] = uniform_filter1d(probs[:, col], size=10)
    # Re-normalise
    probs /= probs.sum(axis=1, keepdims=True)

    fig, ax = plot_regime_evolution(timestamps, probs)
    fig.savefig(DOCS_IMAGES / "online_regime_evolution.png", bbox_inches="tight")
    plt.close(fig)
    print("✓ online_regime_evolution.png")


def chart_regime_transition_live() -> None:
    """Return series with regime background shading and change points."""
    true_regimes, returns = _simulate_regime_data(400, seed=77)
    timestamps = np.arange(len(returns))

    fig, ax = plot_regime_transition_live(timestamps, returns, true_regimes)
    fig.savefig(DOCS_IMAGES / "online_regime_transitions.png", bbox_inches="tight")
    plt.close(fig)
    print("✓ online_regime_transitions.png")


def chart_detection_lag() -> None:
    """Online vs batch regime detection comparison."""
    true_regimes, returns = _simulate_regime_data(300, seed=55)
    timestamps = np.arange(len(returns))

    # Batch detects perfectly (same as truth)
    batch = true_regimes.copy()

    # Online lags by 5-15 observations at each transition
    rng = np.random.default_rng(55)
    online = true_regimes.copy()
    transitions = np.where(np.diff(true_regimes) != 0)[0] + 1
    for t in transitions:
        lag = rng.integers(5, 16)
        end = min(t + lag, len(online))
        online[t:end] = true_regimes[t - 1]  # Keeps old regime during lag

    fig, ax = plot_detection_lag(timestamps, true_regimes, online, batch)
    fig.savefig(DOCS_IMAGES / "online_detection_lag.png", bbox_inches="tight")
    plt.close(fig)
    print("✓ online_detection_lag.png")


if __name__ == "__main__":
    chart_regime_evolution()
    chart_regime_transition_live()
    chart_detection_lag()
    print("All online regime charts generated.")
