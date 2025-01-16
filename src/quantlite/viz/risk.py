"""Risk visualisation charts using the Stephen Few theme.

All charts follow Few's principles: high data-ink ratio, muted palette,
direct labels, horizontal gridlines only, and no chartjunk.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy import stats

from ..core.types import GPDFit
from ..risk.metrics import cvar, max_drawdown_duration, return_moments, value_at_risk
from .theme import FEW_PALETTE, apply_few_theme, bullet_graph, direct_label, few_figure, sparkline

__all__ = [
    "plot_tail_distribution",
    "plot_return_levels",
    "plot_drawdown",
    "plot_risk_dashboard",
]


def plot_tail_distribution(
    returns: np.ndarray | Any,
    gpd_fit: GPDFit | None = None,
    bins: int = 80,
    figsize: tuple[float, float] = (10, 5),
) -> tuple[Figure, Axes]:
    """Plot return distribution with normal overlay and tail analysis.

    Shows a histogram of returns, a fitted normal curve (grey),
    the GPD tail fit (blue), and VaR/CVaR reference lines.

    Args:
        returns: Simple periodic returns.
        gpd_fit: Optional fitted GPD. If ``None``, only the histogram
            and normal overlay are shown.
        bins: Number of histogram bins.
        figsize: Figure size.

    Returns:
        Tuple of (Figure, Axes).
    """
    apply_few_theme()
    arr = np.asarray(returns, dtype=float)
    arr = arr[~np.isnan(arr)]

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Histogram
    ax.hist(
        arr, bins=bins, density=True, alpha=0.5,
        color=FEW_PALETTE["grey_light"], edgecolor=FEW_PALETTE["grey_mid"],
        label="_nolegend_",
    )

    # Normal overlay
    x_range = np.linspace(arr.min(), arr.max(), 300)
    mu, sigma = float(np.mean(arr)), float(np.std(arr, ddof=1))
    normal_pdf = stats.norm.pdf(x_range, mu, sigma)
    ax.plot(x_range, normal_pdf, color=FEW_PALETTE["grey_mid"], linewidth=1.5, linestyle="--")
    direct_label(ax, x_range[-1], normal_pdf[-1], "Normal", colour=FEW_PALETTE["grey_mid"])

    # GPD tail overlay
    if gpd_fit is not None:
        threshold = -gpd_fit.threshold  # convert back to return space
        tail_x = x_range[x_range < threshold]
        if len(tail_x) > 0:
            losses = -tail_x - gpd_fit.threshold
            tail_pdf = stats.genpareto.pdf(losses, gpd_fit.shape, scale=gpd_fit.scale)
            # Scale by exceedance probability
            zeta = gpd_fit.n_exceedances / gpd_fit.n_total
            ax.plot(tail_x, tail_pdf * zeta, color=FEW_PALETTE["primary"], linewidth=2)
            direct_label(ax, tail_x[0], tail_pdf[0] * zeta, "GPD tail",
                         colour=FEW_PALETTE["primary"])

    # VaR and CVaR lines
    if len(arr) >= 2:
        var_95 = value_at_risk(arr, alpha=0.05)
        cvar_95 = cvar(arr, alpha=0.05)
        ax.axvline(var_95, color=FEW_PALETTE["secondary"], linewidth=1.5, linestyle="-")
        direct_label(ax, var_95, ax.get_ylim()[1] * 0.9, f"VaR 95%: {var_95:.4f}",
                     colour=FEW_PALETTE["secondary"], ha="right")

        ax.axvline(cvar_95, color=FEW_PALETTE["negative"], linewidth=1.5, linestyle="-")
        direct_label(ax, cvar_95, ax.get_ylim()[1] * 0.8, f"CVaR 95%: {cvar_95:.4f}",
                     colour=FEW_PALETTE["negative"], ha="right")

    ax.set_xlabel("Return")
    ax.set_ylabel("Density")
    ax.set_title("Return Distribution with Tail Analysis")
    fig.tight_layout()
    return fig, ax


def plot_return_levels(
    gpd_fit: GPDFit,
    max_period: int = 10000,
    n_points: int = 50,
    figsize: tuple[float, float] = (8, 5),
) -> tuple[Figure, Axes]:
    """Plot return levels against return periods with confidence bands.

    Args:
        gpd_fit: Fitted GPD from ``fit_gpd``.
        max_period: Maximum return period to plot.
        n_points: Number of points on the curve.
        figsize: Figure size.

    Returns:
        Tuple of (Figure, Axes).
    """
    from ..risk.evt import return_level as calc_return_level

    apply_few_theme()
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    periods = np.logspace(1, np.log10(max_period), n_points)
    levels = [calc_return_level(gpd_fit, rp) for rp in periods]

    ax.plot(periods, levels, color=FEW_PALETTE["primary"], linewidth=2)

    # Approximate confidence band using delta method
    xi, sigma = gpd_fit.shape, gpd_fit.scale
    se_factor = sigma / np.sqrt(gpd_fit.n_exceedances)
    upper = [lv + 1.96 * se_factor for lv in levels]
    lower = [lv - 1.96 * se_factor for lv in levels]
    ax.fill_between(periods, lower, upper, alpha=0.15, color=FEW_PALETTE["primary"])

    ax.set_xscale("log")
    ax.set_xlabel("Return period (observations)")
    ax.set_ylabel("Estimated loss")
    ax.set_title("Return Level Plot")

    # Annotate key return levels
    for rp_label in [100, 1000, 5000]:
        if rp_label <= max_period:
            rl = calc_return_level(gpd_fit, rp_label)
            ax.plot(rp_label, rl, "o", color=FEW_PALETTE["negative"], markersize=5)
            direct_label(ax, rp_label * 1.1, rl, f"1-in-{rp_label}: {rl:.4f}",
                         colour=FEW_PALETTE["negative"], fontsize=9)

    fig.tight_layout()
    return fig, ax


def plot_drawdown(
    returns: np.ndarray | Any,
    figsize: tuple[float, float] = (10, 4),
) -> tuple[Figure, Axes]:
    """Plot an underwater (drawdown) chart with duration annotations.

    Args:
        returns: Simple periodic returns.
        figsize: Figure size.

    Returns:
        Tuple of (Figure, Axes).
    """
    apply_few_theme()
    arr = np.asarray(returns, dtype=float)
    arr = arr[~np.isnan(arr)]

    cum = np.cumprod(1 + arr)
    roll_max = np.maximum.accumulate(cum)
    drawdowns = (cum - roll_max) / roll_max

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.fill_between(range(len(drawdowns)), drawdowns, 0,
                    color=FEW_PALETTE["negative"], alpha=0.35)
    ax.plot(drawdowns, color=FEW_PALETTE["negative"], linewidth=1.0)

    # Annotate the maximum drawdown
    dd_info = max_drawdown_duration(arr)
    ax.annotate(
        f"Max DD: {dd_info.max_drawdown:.2%}\nDuration: {dd_info.duration} periods",
        xy=(dd_info.end_idx, dd_info.max_drawdown),
        xytext=(dd_info.end_idx + len(arr) * 0.05, dd_info.max_drawdown * 0.7),
        fontsize=9,
        color=FEW_PALETTE["grey_dark"],
        arrowprops={"arrowstyle": "->", "color": FEW_PALETTE["grey_mid"]},
    )

    ax.set_xlabel("Period")
    ax.set_ylabel("Drawdown")
    ax.set_title("Underwater Chart")
    ax.set_xlim(0, len(drawdowns) - 1)
    fig.tight_layout()
    return fig, ax


def plot_risk_dashboard(
    returns: np.ndarray | Any,
    figsize: tuple[float, float] = (12, 8),
) -> tuple[Figure, Any]:
    """Render a single-page risk dashboard.

    Layout (2x2):
    - Top-left: return distribution histogram
    - Top-right: drawdown chart
    - Bottom-left: bullet graphs for key metrics
    - Bottom-right: rolling volatility sparkline

    Args:
        returns: Simple periodic returns.
        figsize: Figure size.

    Returns:
        Tuple of (Figure, array of Axes).
    """
    apply_few_theme()
    arr = np.asarray(returns, dtype=float)
    arr = arr[~np.isnan(arr)]

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Top-left: distribution
    ax_dist = axes[0, 0]
    ax_dist.hist(arr, bins=50, density=True, alpha=0.5,
                 color=FEW_PALETTE["grey_light"], edgecolor=FEW_PALETTE["grey_mid"])
    x_range = np.linspace(arr.min(), arr.max(), 200)
    ax_dist.plot(x_range, stats.norm.pdf(x_range, np.mean(arr), np.std(arr, ddof=1)),
                 color=FEW_PALETTE["grey_mid"], linewidth=1.2, linestyle="--")
    ax_dist.set_title("Return Distribution")
    ax_dist.set_xlabel("Return")

    # Top-right: drawdown
    ax_dd = axes[0, 1]
    cum = np.cumprod(1 + arr)
    roll_max = np.maximum.accumulate(cum)
    dd = (cum - roll_max) / roll_max
    ax_dd.fill_between(range(len(dd)), dd, 0, color=FEW_PALETTE["negative"], alpha=0.35)
    ax_dd.set_title("Drawdown")
    ax_dd.set_xlabel("Period")

    # Bottom-left: key metrics as text (cleaner than bullet graphs for small space)
    ax_metrics = axes[1, 0]
    ax_metrics.axis("off")
    moments = return_moments(arr) if len(arr) >= 4 else None
    var95 = value_at_risk(arr, 0.05) if len(arr) >= 2 else float("nan")
    cvar95 = cvar(arr, 0.05) if len(arr) >= 2 else float("nan")
    dd_info = max_drawdown_duration(arr)

    lines = [
        f"VaR (95%):      {var95:.4f}",
        f"CVaR (95%):     {cvar95:.4f}",
        f"Max Drawdown:   {dd_info.max_drawdown:.4f}",
        f"DD Duration:    {dd_info.duration} periods",
    ]
    if moments:
        lines += [
            f"Skewness:       {moments.skewness:.4f}",
            f"Excess Kurt:    {moments.kurtosis:.4f}",
        ]
    ax_metrics.text(
        0.05, 0.95, "\n".join(lines),
        transform=ax_metrics.transAxes,
        fontsize=11, fontfamily="monospace",
        verticalalignment="top",
        color=FEW_PALETTE["grey_dark"],
    )
    ax_metrics.set_title("Key Risk Metrics")

    # Bottom-right: rolling volatility sparkline
    ax_vol = axes[1, 1]
    window = min(21, len(arr) // 3) if len(arr) > 6 else max(len(arr) // 2, 1)
    if window >= 2:
        rolling_vol = [
            np.std(arr[max(0, i - window + 1): i + 1], ddof=1) if i >= window - 1 else np.nan
            for i in range(len(arr))
        ]
        ax_vol.plot(rolling_vol, color=FEW_PALETTE["primary"], linewidth=1.2)
    ax_vol.set_title("Rolling Volatility")
    ax_vol.set_xlabel("Period")

    fig.suptitle("Risk Dashboard", fontsize=14, color=FEW_PALETTE["grey_dark"], y=1.01)
    fig.tight_layout()
    return fig, axes
