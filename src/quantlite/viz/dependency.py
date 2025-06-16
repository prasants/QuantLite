"""Dependency visualisation charts using the Stephen Few theme.

All charts follow Few's principles: high data-ink ratio, contour
lines without fill, muted palette, direct labels, and horizontal
gridlines only.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy.stats import rankdata

from .theme import FEW_PALETTE, apply_few_theme, direct_label

__all__ = [
    "plot_copula_contour",
    "plot_correlation_matrix",
    "plot_stress_correlation",
    "plot_correlation_dynamics",
]


def plot_copula_contour(
    copula: Any,
    data: np.ndarray,
    figsize: tuple[float, float] = (8, 7),
    backend: str = "matplotlib",
) -> tuple[Figure, Axes] | Any:
    """Plot bivariate copula contour with marginal distributions.

    Shows contour lines (no fill) for the fitted copula in the
    primary palette colour, with the Gaussian copula in grey for
    comparison. Marginal histograms on top and right axes. Tail
    dependence coefficients annotated directly on the plot.

    Args:
        copula: A fitted copula with ``simulate`` and ``tail_dependence``
            methods.
        data: Original data array of shape ``(n, 2)``.
        figsize: Figure size.

    Returns:
        Tuple of (Figure, central Axes).
    """
    if backend == "plotly":
        from .plotly_backend.dependency import plot_copula_contour as _plotly
        return _plotly(copula, data)

    apply_few_theme()

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(
        2, 2, width_ratios=[4, 1], height_ratios=[1, 4],
        hspace=0.05, wspace=0.05,
    )

    ax_main = fig.add_subplot(gs[1, 0])
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax_main)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)

    # Transform data to uniform margins
    n = data.shape[0]
    u = np.column_stack([
        rankdata(data[:, j], method="ordinal") / (n + 1)
        for j in range(2)
    ])

    # Scatter of uniform data
    ax_main.scatter(
        u[:, 0], u[:, 1], s=3, alpha=0.3,
        color=FEW_PALETTE["grey_mid"], zorder=1,
    )

    # Gaussian copula contour (grey reference)
    from ..dependency.copulas import GaussianCopula
    gauss = GaussianCopula()
    gauss.fit(data)
    gauss_samples = gauss.simulate(5000, rng_seed=42)
    grid_x = np.linspace(0.01, 0.99, 50)
    grid_y = np.linspace(0.01, 0.99, 50)
    X, Y = np.meshgrid(grid_x, grid_y)

    try:
        from scipy.stats import gaussian_kde
        gauss_kde = gaussian_kde(gauss_samples.T)
        Z_gauss = gauss_kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
        ax_main.contour(
            X, Y, Z_gauss, levels=5,
            colors=FEW_PALETTE["grey_mid"], linewidths=0.8,
            alpha=0.6, zorder=2,
        )
    except Exception:
        pass

    # Fitted copula contour (blue)
    cop_samples = copula.simulate(5000, rng_seed=42)
    try:
        cop_kde = gaussian_kde(cop_samples.T)
        Z_cop = cop_kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
        ax_main.contour(
            X, Y, Z_cop, levels=5,
            colors=FEW_PALETTE["primary"], linewidths=1.2,
            zorder=3,
        )
    except Exception:
        pass

    ax_main.set_xlim(0, 1)
    ax_main.set_ylim(0, 1)
    ax_main.set_xlabel("U1 (uniform margin)")
    ax_main.set_ylabel("U2 (uniform margin)")

    # Tail dependence annotation
    td = copula.tail_dependence()
    ax_main.text(
        0.05, 0.05,
        f"Lower tail dep: {td['lower']:.3f}",
        transform=ax_main.transAxes,
        fontsize=9, color=FEW_PALETTE["negative"],
    )
    ax_main.text(
        0.65, 0.95,
        f"Upper tail dep: {td['upper']:.3f}",
        transform=ax_main.transAxes,
        fontsize=9, color=FEW_PALETTE["positive"],
    )

    # Marginal histograms
    ax_top.hist(u[:, 0], bins=40, color=FEW_PALETTE["grey_light"],
                edgecolor=FEW_PALETTE["grey_mid"], density=True)
    ax_top.axis("off")

    ax_right.hist(u[:, 1], bins=40, color=FEW_PALETTE["grey_light"],
                  edgecolor=FEW_PALETTE["grey_mid"], density=True,
                  orientation="horizontal")
    ax_right.axis("off")

    copula_name = type(copula).__name__
    ax_main.set_title(f"{copula_name} vs Gaussian (grey)", fontsize=11)

    return fig, ax_main


def plot_correlation_matrix(
    corr_matrix: pd.DataFrame,
    figsize: tuple[float, float] | None = None,
    backend: str = "matplotlib",
) -> tuple[Figure, Axes] | Any:
    """Plot a correlation heatmap with Few's diverging palette.

    Uses a muted blue-white-red diverging colourmap. Values annotated
    in each cell. Optionally reordered by hierarchical clustering.

    Args:
        corr_matrix: Correlation DataFrame.
        figsize: Figure size. Defaults to a square proportional to
            the matrix size.

    Returns:
        Tuple of (Figure, Axes).
    """
    if backend == "plotly":
        from .plotly_backend.dependency import plot_correlation_matrix as _plotly
        return _plotly(corr_matrix)

    apply_few_theme()
    n = len(corr_matrix)
    if figsize is None:
        figsize = (max(6, n * 0.8), max(5, n * 0.7))

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Muted diverging colourmap
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list(
        "few_diverging",
        [FEW_PALETTE["primary"], "#FFFFFF", FEW_PALETTE["negative"]],
    )

    im = ax.imshow(corr_matrix.values, cmap=cmap, vmin=-1, vmax=1, aspect="auto")

    # Annotate cells
    for i in range(n):
        for j in range(n):
            val = corr_matrix.values[i, j]
            colour = "white" if abs(val) > 0.7 else FEW_PALETTE["grey_dark"]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=max(7, 11 - n // 3), color=colour)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(corr_matrix.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr_matrix.index)
    ax.set_title("Correlation Matrix")

    fig.colorbar(im, ax=ax, shrink=0.8, label="Correlation")
    fig.tight_layout()
    return fig, ax


def plot_stress_correlation(
    calm_corr: pd.DataFrame,
    stress_corr: pd.DataFrame,
    figsize: tuple[float, float] = (14, 5.5),
    backend: str = "matplotlib",
) -> tuple[Figure, Any]:
    """Plot calm and stress correlation matrices side by side.

    Same colour scale for both. Title annotates the average
    correlation change.

    Args:
        calm_corr: Correlation matrix during calm periods.
        stress_corr: Correlation matrix during stress periods.
        figsize: Figure size.

    Returns:
        Tuple of (Figure, array of Axes).
    """
    if backend == "plotly":
        from .plotly_backend.dependency import plot_stress_correlation as _plotly
        return _plotly(calm_corr, stress_corr)

    apply_few_theme()
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list(
        "few_diverging",
        [FEW_PALETTE["primary"], "#FFFFFF", FEW_PALETTE["negative"]],
    )

    n = len(calm_corr)

    for ax, corr, title in [(axes[0], calm_corr, "Calm"), (axes[1], stress_corr, "Stress")]:
        ax.imshow(corr.values, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
        for i in range(n):
            for j in range(n):
                val = corr.values[i, j]
                colour = "white" if abs(val) > 0.7 else FEW_PALETTE["grey_dark"]
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=max(7, 10 - n // 3), color=colour)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(corr.columns, rotation=45, ha="right")
        ax.set_yticklabels(corr.index)
        ax.set_title(f"{title} Period")

    # Average correlation change
    mask = np.triu(np.ones(n, dtype=bool), k=1)
    if n > 1:
        avg_change = float(
            stress_corr.values[np.ix_(mask.any(1), mask.any(0))].mean()
            - calm_corr.values[np.ix_(mask.any(1), mask.any(0))].mean()
        )
    else:
        avg_change = 0.0

    fig.suptitle(
        f"Correlation: Calm vs Stress (avg change: {avg_change:+.3f})",
        fontsize=13, color=FEW_PALETTE["grey_dark"],
    )
    fig.tight_layout()
    return fig, axes


def plot_correlation_dynamics(
    x: np.ndarray | pd.Series,
    y: np.ndarray | pd.Series,
    window: int = 60,
    figsize: tuple[float, float] = (10, 4),
    backend: str = "matplotlib",
) -> tuple[Figure, Axes] | Any:
    """Plot rolling correlation over time.

    Shows the rolling Pearson correlation as a line chart with
    reference lines at 0 and the overall correlation.

    Args:
        x: First return series.
        y: Second return series.
        window: Rolling window size.
        figsize: Figure size.

    Returns:
        Tuple of (Figure, Axes).
    """
    if backend == "plotly":
        from .plotly_backend.dependency import plot_correlation_dynamics as _plotly
        return _plotly(x, y, window=window)

    apply_few_theme()
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    sx = pd.Series(np.asarray(x, dtype=float))
    sy = pd.Series(np.asarray(y, dtype=float))
    rolling = sx.rolling(window).corr(sy)

    ax.plot(rolling, color=FEW_PALETTE["primary"], linewidth=1.5)
    ax.axhline(0, color=FEW_PALETTE["grey_mid"], linewidth=0.8, linestyle="--")

    overall = float(sx.corr(sy))
    ax.axhline(overall, color=FEW_PALETTE["secondary"], linewidth=1.0, linestyle=":")
    direct_label(ax, len(sx) * 0.02, overall + 0.03,
                 f"Overall: {overall:.3f}",
                 colour=FEW_PALETTE["secondary"], fontsize=9)

    ax.set_xlabel("Period")
    ax.set_ylabel("Correlation")
    ax.set_title(f"Rolling {window}-Period Correlation")
    ax.set_ylim(-1.05, 1.05)
    fig.tight_layout()
    return fig, ax
