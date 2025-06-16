"""Regime visualisation charts using the Stephen Few theme.

All charts follow Few's principles: muted palette, small multiples,
direct labels, no chartjunk.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .theme import FEW_PALETTE, apply_few_theme

__all__ = [
    "plot_regime_timeline",
    "plot_regime_distributions",
    "plot_transition_matrix",
    "plot_regime_summary",
]

# Muted palette for regime colouring (max 5 regimes)
_REGIME_COLOURS = [
    FEW_PALETTE["negative"],   # regime 0 (crisis): muted red
    FEW_PALETTE["neutral"],    # regime 1: teal
    FEW_PALETTE["primary"],    # regime 2: blue
    FEW_PALETTE["positive"],   # regime 3: green
    FEW_PALETTE["secondary"],  # regime 4: orange
]


def plot_regime_timeline(
    returns: np.ndarray | pd.Series,
    regimes: np.ndarray,
    changepoints: list[int] | None = None,
    figsize: tuple[float, float] = (12, 4.5),
    backend: str = "matplotlib",
) -> tuple[Figure, Axes] | Any:
    """Plot cumulative returns with a regime colour band below.

    The main chart shows cumulative returns. A thin horizontal band
    below indicates the active regime using muted colours. Change
    points are shown as subtle vertical rules.

    Args:
        returns: Simple periodic returns.
        regimes: Array of integer regime labels.
        changepoints: Optional list of change-point indices.
        figsize: Figure size.

    Returns:
        Tuple of (Figure, Axes).
    """
    if backend == "plotly":
        from .plotly_backend.regimes import plot_regime_timeline as _plotly
        return _plotly(returns, regimes, changepoints=changepoints)

    apply_few_theme()
    arr = np.asarray(returns, dtype=float)
    reg = np.asarray(regimes, dtype=int)
    cum = np.cumprod(1 + arr)

    fig, (ax_main, ax_regime) = plt.subplots(
        2, 1, figsize=figsize,
        gridspec_kw={"height_ratios": [5, 1], "hspace": 0.05},
        sharex=True,
    )

    # Cumulative return line
    ax_main.plot(cum, color=FEW_PALETTE["grey_dark"], linewidth=1.2)
    ax_main.set_ylabel("Cumulative Return")
    ax_main.set_title("Returns with Regime Timeline")

    # Regime colour band
    unique_regimes = sorted(np.unique(reg))
    for i in range(len(reg)):
        colour = _REGIME_COLOURS[reg[i] % len(_REGIME_COLOURS)]
        ax_regime.axvspan(i - 0.5, i + 0.5, color=colour, alpha=0.7)

    ax_regime.set_yticks([])
    ax_regime.set_ylabel("Regime", fontsize=9)
    ax_regime.set_xlim(-0.5, len(arr) - 0.5)

    # Regime legend via direct labels
    for r in unique_regimes:
        first_idx = int(np.argmax(reg == r))
        colour = _REGIME_COLOURS[r % len(_REGIME_COLOURS)]
        ax_regime.text(
            first_idx, 0.5, f"R{r}",
            fontsize=8, color="white", fontweight="bold",
            ha="center", va="center",
        )

    # Change points as vertical rules
    if changepoints:
        for cp in changepoints:
            ax_main.axvline(cp, color=FEW_PALETTE["grey_mid"],
                            linewidth=0.7, linestyle=":", alpha=0.6)

    fig.tight_layout()
    return fig, ax_main


def plot_regime_distributions(
    returns: np.ndarray | pd.Series,
    regimes: np.ndarray,
    bins: int = 40,
    figsize: tuple[float, float] | None = None,
    backend: str = "matplotlib",
) -> tuple[Figure, Any]:
    """Plot return distributions as small multiples, one per regime.

    All histograms share the same x-axis scale for direct comparison.

    Args:
        returns: Simple periodic returns.
        regimes: Array of regime labels.
        bins: Number of histogram bins.
        figsize: Figure size.

    Returns:
        Tuple of (Figure, array of Axes).
    """
    if backend == "plotly":
        from .plotly_backend.regimes import plot_regime_distributions as _plotly
        return _plotly(returns, regimes, bins=bins)

    apply_few_theme()
    arr = np.asarray(returns, dtype=float)
    reg = np.asarray(regimes, dtype=int)
    unique = sorted(np.unique(reg))
    n_regimes = len(unique)

    if figsize is None:
        figsize = (4 * n_regimes + 1, 3.5)

    fig, axes = plt.subplots(1, n_regimes, figsize=figsize, sharey=True)
    if n_regimes == 1:
        axes = [axes]

    x_min, x_max = arr.min(), arr.max()

    for i, r in enumerate(unique):
        ax = axes[i]
        r_data = arr[reg == r]
        colour = _REGIME_COLOURS[r % len(_REGIME_COLOURS)]
        ax.hist(r_data, bins=bins, density=True, alpha=0.7,
                color=colour, edgecolor="white", linewidth=0.5)
        ax.set_xlim(x_min, x_max)
        ax.set_title(f"Regime {r} (n={len(r_data)})", fontsize=10)
        ax.set_xlabel("Return")

        # Annotate mean and vol
        mu = np.mean(r_data)
        sigma = np.std(r_data, ddof=1) if len(r_data) > 1 else 0
        ax.text(
            0.95, 0.95,
            f"mean: {mu:.4f}\nvol: {sigma:.4f}",
            transform=ax.transAxes, fontsize=8,
            va="top", ha="right", color=FEW_PALETTE["grey_dark"],
        )

    axes[0].set_ylabel("Density")
    fig.suptitle("Return Distributions by Regime", fontsize=12,
                 color=FEW_PALETTE["grey_dark"])
    fig.tight_layout()
    return fig, axes


def plot_transition_matrix(
    model: Any,
    figsize: tuple[float, float] = (5, 4),
    backend: str = "matplotlib",
) -> tuple[Figure, Axes] | Any:
    """Plot the regime transition probability matrix.

    Cell colour intensity proportional to probability. Values
    annotated in each cell.

    Args:
        model: A ``RegimeModel`` with ``transition_matrix`` and
            ``n_regimes`` attributes.
        figsize: Figure size.

    Returns:
        Tuple of (Figure, Axes).
    """
    if backend == "plotly":
        from .plotly_backend.regimes import plot_transition_matrix as _plotly
        return _plotly(model)

    apply_few_theme()
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    trans = model.transition_matrix
    n = model.n_regimes

    # Single-hue colourmap (light to dark blue)
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list(
        "few_sequential",
        ["#FFFFFF", FEW_PALETTE["primary"]],
    )

    ax.imshow(trans, cmap=cmap, vmin=0, vmax=1, aspect="auto")

    for i in range(n):
        for j in range(n):
            val = trans[i, j]
            colour = "white" if val > 0.6 else FEW_PALETTE["grey_dark"]
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=11, color=colour)

    labels = [f"R{i}" for i in range(n)]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("To Regime")
    ax.set_ylabel("From Regime")
    ax.set_title("Transition Probabilities")
    fig.tight_layout()
    return fig, ax


def plot_regime_summary(
    returns: np.ndarray | pd.Series,
    regimes: np.ndarray,
    figsize: tuple[float, float] = (12, 10),
) -> tuple[Figure, Any]:
    """Composite regime analysis: timeline, distributions, and metrics.

    Layout: timeline on top, conditional distributions in middle,
    key metrics table at bottom.

    Args:
        returns: Simple periodic returns.
        regimes: Array of regime labels.
        figsize: Figure size.

    Returns:
        Tuple of (Figure, array of Axes).
    """
    apply_few_theme()
    arr = np.asarray(returns, dtype=float)
    reg = np.asarray(regimes, dtype=int)
    unique = sorted(np.unique(reg))
    n_regimes = len(unique)

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, n_regimes, height_ratios=[3, 2, 1.5], hspace=0.35)

    # Top: timeline spanning all columns
    ax_timeline = fig.add_subplot(gs[0, :])
    cum = np.cumprod(1 + arr)
    ax_timeline.plot(cum, color=FEW_PALETTE["grey_dark"], linewidth=1.2)

    # Shade regime backgrounds
    for i in range(len(reg)):
        colour = _REGIME_COLOURS[reg[i] % len(_REGIME_COLOURS)]
        ax_timeline.axvspan(i - 0.5, i + 0.5, color=colour, alpha=0.1)

    ax_timeline.set_ylabel("Cumulative Return")
    ax_timeline.set_title("Regime Summary", fontsize=13)

    # Middle: one distribution per regime
    x_min, x_max = arr.min(), arr.max()
    for i, r in enumerate(unique):
        ax = fig.add_subplot(gs[1, i])
        r_data = arr[reg == r]
        colour = _REGIME_COLOURS[r % len(_REGIME_COLOURS)]
        ax.hist(r_data, bins=30, density=True, alpha=0.7,
                color=colour, edgecolor="white", linewidth=0.5)
        ax.set_xlim(x_min, x_max)
        ax.set_title(f"Regime {r}", fontsize=10)
        if i == 0:
            ax.set_ylabel("Density")

    # Bottom: metrics table
    ax_table = fig.add_subplot(gs[2, :])
    ax_table.axis("off")

    from ..risk.metrics import cvar, value_at_risk

    headers = ["Regime", "N", "Mean", "Vol", "VaR 95%", "CVaR 95%"]
    rows = []
    for r in unique:
        r_data = arr[reg == r]
        n = len(r_data)
        mu = np.mean(r_data)
        sigma = np.std(r_data, ddof=1) if n > 1 else 0
        var95 = value_at_risk(r_data, 0.05) if n >= 2 else float("nan")
        cvar95 = cvar(r_data, 0.05) if n >= 2 else float("nan")
        rows.append([
            f"R{r}", str(n), f"{mu:.5f}", f"{sigma:.5f}",
            f"{var95:.5f}", f"{cvar95:.5f}",
        ])

    table = ax_table.table(
        cellText=rows, colLabels=headers,
        loc="center", cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)

    # Style the table
    for key, cell in table.get_celld().items():
        cell.set_edgecolor(FEW_PALETTE["grey_light"])
        if key[0] == 0:
            cell.set_facecolor(FEW_PALETTE["grey_light"])
            cell.set_text_props(fontweight="bold")

    fig.tight_layout()
    return fig, fig.axes
