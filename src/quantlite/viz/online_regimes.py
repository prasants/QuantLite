"""Online regime detection visualisation charts using the Stephen Few theme.

All charts follow Few's principles: high data-ink ratio, muted palette,
direct labels, horizontal gridlines only, and no chartjunk.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .theme import FEW_PALETTE, apply_few_theme, direct_label

__all__ = [
    "plot_regime_evolution",
    "plot_regime_transition_live",
    "plot_detection_lag",
]

_REGIME_COLOURS = {
    0: FEW_PALETTE["positive"],   # Calm
    1: FEW_PALETTE["secondary"],  # Transitional
    2: FEW_PALETTE["negative"],   # Crisis
}

_REGIME_LABELS = {
    0: "Calm",
    1: "Transitional",
    2: "Crisis",
}


def plot_regime_evolution(
    timestamps: np.ndarray | Any,
    regime_probs: np.ndarray | Any,
    regime_labels: dict[int, str] | None = None,
    figsize: tuple[float, float] = (10, 5),
) -> tuple[Figure, Axes]:
    """Plot regime state over time with confidence bands.

    Shows how the regime posterior probabilities evolve as new
    observations arrive, giving a sense of classification confidence.

    Args:
        timestamps: Time axis.
        regime_probs: Array of shape (n_obs, n_regimes) with posterior
            probabilities.
        regime_labels: Optional mapping from regime index to label.
        figsize: Figure dimensions.

    Returns:
        Tuple of (Figure, Axes).
    """
    apply_few_theme()
    fig, ax = plt.subplots(figsize=figsize)

    labels = regime_labels or _REGIME_LABELS
    colours = [_REGIME_COLOURS.get(i, FEW_PALETTE["neutral"]) for i in range(regime_probs.shape[1])]

    for i in range(regime_probs.shape[1]):
        label = labels.get(i, f"Regime {i}")
        ax.plot(timestamps, regime_probs[:, i], color=colours[i], linewidth=1.4)
        # Direct label at the end
        direct_label(
            ax, timestamps[-1], regime_probs[-1, i],
            f"  {label}",
            colour=colours[i],
        )

    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Time")
    ax.set_ylabel("Posterior probability")
    ax.set_title("Regime Evolution (Online Detection)")

    return fig, ax


def plot_regime_transition_live(
    timestamps: np.ndarray | Any,
    returns: np.ndarray | Any,
    regimes: np.ndarray | Any,
    regime_labels: dict[int, str] | None = None,
    figsize: tuple[float, float] = (10, 5),
) -> tuple[Figure, Axes]:
    """Plot a return series with regime change points highlighted.

    Background colour bands indicate the active regime; vertical
    lines mark transition points.

    Args:
        timestamps: Time axis.
        returns: Return series.
        regimes: Integer regime labels per observation.
        regime_labels: Optional mapping from regime index to label.
        figsize: Figure dimensions.

    Returns:
        Tuple of (Figure, Axes).
    """
    apply_few_theme()
    fig, ax = plt.subplots(figsize=figsize)

    labels = regime_labels or _REGIME_LABELS

    # Background shading for regimes
    prev_regime = regimes[0]
    start_idx = 0
    for i in range(1, len(regimes)):
        if regimes[i] != prev_regime or i == len(regimes) - 1:
            end_idx = i if regimes[i] != prev_regime else i + 1
            colour = _REGIME_COLOURS.get(int(prev_regime), FEW_PALETTE["neutral"])
            ax.axvspan(timestamps[start_idx], timestamps[min(end_idx, len(timestamps) - 1)],
                       alpha=0.15, color=colour)
            if regimes[i] != prev_regime:
                ax.axvline(timestamps[i], color=FEW_PALETTE["grey_mid"],
                           linewidth=0.8, linestyle=":")
            prev_regime = regimes[i]
            start_idx = i

    ax.plot(timestamps, returns, color=FEW_PALETTE["primary"], linewidth=1.0)

    ax.set_xlabel("Time")
    ax.set_ylabel("Return")
    ax.set_title("Returns with Online Regime Change Points")

    # Add regime labels to the legend area via direct labels
    unique_regimes = sorted(set(int(r) for r in regimes))
    for j, r in enumerate(unique_regimes):
        colour = _REGIME_COLOURS.get(r, FEW_PALETTE["neutral"])
        label = labels.get(r, f"Regime {r}")
        ax.plot([], [], color=colour, linewidth=6, alpha=0.3, label=label)
    ax.legend(loc="upper left", frameon=False)

    return fig, ax


def plot_detection_lag(
    timestamps: np.ndarray | Any,
    true_regimes: np.ndarray | Any,
    online_regimes: np.ndarray | Any,
    batch_regimes: np.ndarray | Any,
    figsize: tuple[float, float] = (10, 5),
) -> tuple[Figure, Axes]:
    """Compare online vs batch regime detection to show detection lag.

    Plots both detection series against the ground truth, making it
    easy to see where the online detector lags behind batch.

    Args:
        timestamps: Time axis.
        true_regimes: Ground truth regime labels.
        online_regimes: Online-detected regime labels.
        batch_regimes: Batch-detected regime labels.
        figsize: Figure dimensions.

    Returns:
        Tuple of (Figure, Axes).
    """
    apply_few_theme()
    fig, ax = plt.subplots(figsize=figsize)

    ax.step(timestamps, true_regimes, where="post",
            color=FEW_PALETTE["grey_mid"], linewidth=2.0, alpha=0.5)
    direct_label(ax, timestamps[-1], true_regimes[-1], "  Truth",
                 colour=FEW_PALETTE["grey_mid"])

    ax.step(timestamps, batch_regimes, where="post",
            color=FEW_PALETTE["positive"], linewidth=1.4)
    direct_label(ax, timestamps[-1], float(batch_regimes[-1]) + 0.08, "  Batch",
                 colour=FEW_PALETTE["positive"])

    ax.step(timestamps, online_regimes, where="post",
            color=FEW_PALETTE["secondary"], linewidth=1.4, linestyle="--")
    direct_label(ax, timestamps[-1], float(online_regimes[-1]) - 0.08, "  Online",
                 colour=FEW_PALETTE["secondary"])

    ax.set_xlabel("Time")
    ax.set_ylabel("Regime")
    ax.set_title("Online vs Batch Regime Detection Lag")
    ax.set_yticks(sorted(set(int(r) for r in true_regimes)))

    return fig, ax
