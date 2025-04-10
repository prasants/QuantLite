"""Portfolio visualisation: efficient frontiers, weight evolution, and dashboards.

All charts follow Stephen Few's principles: maximum data-ink ratio,
direct labels, muted palette, no chartjunk.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .theme import (
    FEW_PALETTE,
    apply_few_theme,
    bullet_graph,
    direct_label,
    few_figure,
)

__all__ = [
    "plot_efficient_frontier",
    "plot_weights_over_time",
    "plot_monthly_returns",
    "plot_backtest_summary",
    "plot_regime_performance",
    "plot_risk_contribution",
    "plot_correlation_network",
]

_AREA_PALETTE = [
    FEW_PALETTE["primary"],
    FEW_PALETTE["secondary"],
    FEW_PALETTE["positive"],
    FEW_PALETTE["negative"],
    FEW_PALETTE["neutral"],
    FEW_PALETTE["grey_mid"],
    "#B07AA1",
    "#FF9DA7",
    "#9C755F",
    "#BAB0AC",
]


def plot_efficient_frontier(
    returns_df: pd.DataFrame,
    n_portfolios: int = 2000,
    risk_free_rate: float = 0.0,
    freq: int = 252,
) -> tuple[Figure, Axes]:
    """Plot the efficient frontier with random portfolios.

    Scatter of random portfolios in grey, efficient frontier line in
    primary blue, minimum variance point in green, maximum Sharpe
    point in orange. Direct labels instead of a legend.

    Args:
        returns_df: Asset returns DataFrame.
        n_portfolios: Number of random portfolios to simulate.
        risk_free_rate: Annualised risk-free rate.
        freq: Periods per year.

    Returns:
        Tuple of (Figure, Axes).
    """
    from ..portfolio.optimisation import max_sharpe_weights, min_variance_weights

    apply_few_theme()
    n_assets = returns_df.shape[1]
    mu = returns_df.mean().values
    cov = returns_df.cov().values

    # Random portfolios
    rng = np.random.default_rng(42)
    rand_rets: list[float] = []
    rand_vols: list[float] = []

    for _ in range(n_portfolios):
        w = rng.random(n_assets)
        w = w / w.sum()
        ret = float((1 + w @ mu) ** freq - 1)
        vol = float(np.sqrt(w @ cov @ w) * np.sqrt(freq))
        rand_rets.append(ret)
        rand_vols.append(vol)

    fig, ax = few_figure(figsize=(8, 5))
    ax.scatter(rand_vols, rand_rets, c=FEW_PALETTE["grey_light"],
               s=8, alpha=0.6, edgecolors="none", zorder=1)

    # Min variance
    mv = min_variance_weights(returns_df, freq=freq)
    ax.scatter(mv.expected_risk, mv.expected_return,
               c=FEW_PALETTE["positive"], s=80, zorder=3, marker="o")
    direct_label(ax, mv.expected_risk + 0.005, mv.expected_return,
                 "Min Variance", colour=FEW_PALETTE["positive"])

    # Max Sharpe
    ms = max_sharpe_weights(returns_df, risk_free_rate=risk_free_rate, freq=freq)
    ax.scatter(ms.expected_risk, ms.expected_return,
               c=FEW_PALETTE["secondary"], s=80, zorder=3, marker="D")
    direct_label(ax, ms.expected_risk + 0.005, ms.expected_return,
                 f"Max Sharpe ({ms.sharpe:.2f})", colour=FEW_PALETTE["secondary"])

    ax.set_xlabel("Annualised Volatility")
    ax.set_ylabel("Annualised Return")
    ax.set_title("Efficient Frontier")
    fig.tight_layout()
    return fig, ax


def plot_weights_over_time(result: Any) -> tuple[Figure, Axes]:
    """Stacked area chart of portfolio weight evolution.

    Muted palette, no border lines on areas, direct labels on the
    largest allocations.

    Args:
        result: A ``BacktestResult`` or ``RebalanceResult`` with
            ``weights_over_time`` attribute.

    Returns:
        Tuple of (Figure, Axes).
    """
    apply_few_theme()
    weights = result.weights_over_time
    fig, ax = few_figure(figsize=(10, 4))

    colours = _AREA_PALETTE[:weights.shape[1]]
    ax.stackplot(weights.index, weights.values.T, labels=weights.columns,
                 colors=colours, linewidth=0)

    # Direct label the largest allocation at the midpoint
    mid_idx = len(weights) // 2
    cum = 0.0
    for j, col in enumerate(weights.columns):
        w = weights.iloc[mid_idx, j]
        if w > 0.1:
            y_pos = cum + w / 2
            direct_label(ax, weights.index[mid_idx], y_pos, col,
                         colour="white", fontsize=9)
        cum += w

    ax.set_xlim(weights.index[0], weights.index[-1])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Weight")
    ax.set_title("Portfolio Weights Over Time")
    fig.tight_layout()
    return fig, ax


def plot_monthly_returns(result: Any) -> tuple[Figure, Axes]:
    """Heatmap of monthly returns.

    Diverging blue-white-red colourmap. Values displayed in cells.
    Grey for missing months.

    Args:
        result: A ``BacktestResult`` from the backtesting engine.

    Returns:
        Tuple of (Figure, Axes).
    """
    from ..backtesting.analysis import monthly_returns_table

    apply_few_theme()
    table = monthly_returns_table(result)
    if table.empty:
        fig, ax = few_figure()
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center")
        return fig, ax

    fig, ax = few_figure(figsize=(10, max(2, len(table) * 0.5 + 1)))

    # Create masked array for NaN
    data = table.values
    masked = np.ma.masked_invalid(data)

    cmap = plt.cm.RdBu_r.copy()
    cmap.set_bad(color=FEW_PALETTE["grey_light"])

    vmax = max(abs(np.nanmin(data)), abs(np.nanmax(data))) if not np.all(np.isnan(data)) else 0.1
    im = ax.imshow(masked, cmap=cmap, aspect="auto", vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(12))
    ax.set_xticklabels(table.columns, fontsize=9)
    ax.set_yticks(range(len(table)))
    ax.set_yticklabels(table.index, fontsize=9)

    # Values in cells
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if not np.isnan(data[i, j]):
                ax.text(j, i, f"{data[i, j]:.1%}",
                        ha="center", va="center", fontsize=8,
                        color="white" if abs(data[i, j]) > vmax * 0.5 else FEW_PALETTE["grey_dark"])

    ax.set_title("Monthly Returns")
    fig.tight_layout()
    return fig, ax


def plot_backtest_summary(result: Any) -> tuple[Figure, list[Axes]]:
    """Composite backtest dashboard: equity curve, drawdown, weights, metrics.

    Single-page layout with four panels.

    Args:
        result: A ``BacktestResult`` from the backtesting engine.

    Returns:
        Tuple of (Figure, list of Axes).
    """
    apply_few_theme()
    fig, axes = plt.subplots(3, 1, figsize=(10, 9),
                             gridspec_kw={"height_ratios": [3, 1.5, 1.5]})
    fig.subplots_adjust(hspace=0.35)

    # 1. Equity curve
    ax = axes[0]
    pv = result.portfolio_value
    ax.plot(pv.index, pv.values, color=FEW_PALETTE["primary"], linewidth=1.5)
    ax.set_ylabel("Portfolio Value")
    ax.set_title("Backtest Summary")
    # Add key metrics as text
    m = result.metrics
    info = (
        f"Return: {m.get('total_return', 0):.1%}  "
        f"Sharpe: {m.get('sharpe_ratio', 0):.2f}  "
        f"Max DD: {m.get('max_drawdown', 0):.1%}"
    )
    ax.text(0.02, 0.95, info, transform=ax.transAxes,
            fontsize=9, va="top", color=FEW_PALETTE["grey_dark"])

    # 2. Drawdown
    ax = axes[1]
    dd = result.drawdown_series
    ax.fill_between(dd.index, dd.values, 0,
                    color=FEW_PALETTE["negative"], alpha=0.4, linewidth=0)
    ax.plot(dd.index, dd.values, color=FEW_PALETTE["negative"], linewidth=0.8)
    ax.set_ylabel("Drawdown")
    ax.set_ylim(dd.min() * 1.1 if dd.min() < 0 else -0.01, 0.01)

    # 3. Weights
    ax = axes[2]
    weights = result.weights_over_time
    colours = _AREA_PALETTE[:weights.shape[1]]
    ax.stackplot(weights.index, weights.values.T,
                 colors=colours, linewidth=0)
    ax.set_ylabel("Weights")
    ax.set_ylim(0, 1)

    fig.tight_layout()
    return fig, list(axes)


def plot_regime_performance(result: Any) -> tuple[Figure, list[Axes]]:
    """Small multiples of equity curve segments coloured by regime.

    Args:
        result: A ``BacktestResult`` with regime_labels.

    Returns:
        Tuple of (Figure, list of Axes).
    """
    apply_few_theme()

    if result.regime_labels is None:
        fig, ax = few_figure()
        ax.text(0.5, 0.5, "No regime labels available", ha="center", va="center")
        return fig, [ax]

    pv = result.portfolio_value
    labels = np.asarray(result.regime_labels)
    unique = sorted(set(labels))
    n_regimes = len(unique)

    regime_colours = [FEW_PALETTE["positive"], FEW_PALETTE["negative"],
                      FEW_PALETTE["primary"], FEW_PALETTE["secondary"],
                      FEW_PALETTE["neutral"]]

    fig, axes = plt.subplots(n_regimes, 1, figsize=(10, 2.5 * n_regimes), sharex=True)
    if n_regimes == 1:
        axes = [axes]

    for i, regime in enumerate(unique):
        ax = axes[i]
        mask = labels == regime
        ax.plot(pv.index, pv.values, color=FEW_PALETTE["grey_light"], linewidth=0.8)

        # Highlight regime segments
        regime_pv = pv.copy()
        regime_pv[~mask] = np.nan
        ax.plot(regime_pv.index, regime_pv.values,
                color=regime_colours[i % len(regime_colours)], linewidth=1.5)

        pct_time = mask.sum() / len(mask)
        ax.set_ylabel(f"Regime {regime}")
        direct_label(ax, pv.index[-1], pv.values[-1],
                     f"  {pct_time:.0%} of time",
                     colour=regime_colours[i % len(regime_colours)], fontsize=9)

    axes[0].set_title("Performance by Regime")
    fig.tight_layout()
    return fig, list(axes)


def plot_risk_contribution(
    weights: dict[str, float],
    returns_df: pd.DataFrame,
) -> tuple[Figure, Axes]:
    """Horizontal bar chart of marginal risk contributions.

    Args:
        weights: Asset weights dict.
        returns_df: Asset returns DataFrame.

    Returns:
        Tuple of (Figure, Axes).
    """
    apply_few_theme()
    names = list(weights.keys())
    w = np.array([weights[n] for n in names])
    cov = returns_df[names].cov().values

    port_vol = np.sqrt(w @ cov @ w)
    if port_vol < 1e-12:
        port_vol = 1.0
    mrc = (cov @ w) / port_vol
    rc = w * mrc

    # Sort by contribution
    order = np.argsort(rc)
    sorted_names = [names[i] for i in order]
    sorted_rc = rc[order]

    fig, ax = few_figure(figsize=(6, max(2, len(names) * 0.4 + 1)))
    colours = [FEW_PALETTE["primary"] if v >= 0 else FEW_PALETTE["negative"]
               for v in sorted_rc]
    ax.barh(range(len(sorted_names)), sorted_rc, color=colours, height=0.6)
    ax.set_yticks(range(len(sorted_names)))
    ax.set_yticklabels(sorted_names)
    ax.set_xlabel("Risk Contribution")
    ax.set_title("Marginal Risk Contributions")
    fig.tight_layout()
    return fig, ax


def plot_correlation_network(
    corr_matrix: pd.DataFrame,
    threshold: float = 0.5,
    weights: dict[str, float] | None = None,
) -> tuple[Figure, Axes]:
    """Network graph of asset correlations.

    Edges represent correlations above the threshold. Node size is
    proportional to portfolio weight (if provided).

    Args:
        corr_matrix: Correlation matrix as DataFrame.
        threshold: Minimum absolute correlation for an edge.
        weights: Optional asset weights for node sizing.

    Returns:
        Tuple of (Figure, Axes).
    """
    apply_few_theme()
    names = list(corr_matrix.columns)
    n = len(names)

    fig, ax = few_figure(figsize=(6, 6))

    # Circular layout
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x_pos = np.cos(angles)
    y_pos = np.sin(angles)

    # Edges
    for i in range(n):
        for j in range(i + 1, n):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) >= threshold:
                colour = FEW_PALETTE["primary"] if corr_val > 0 else FEW_PALETTE["negative"]
                alpha = min(abs(corr_val), 1.0)
                ax.plot([x_pos[i], x_pos[j]], [y_pos[i], y_pos[j]],
                        color=colour, alpha=alpha, linewidth=1.5)

    # Nodes
    if weights is not None:
        sizes = np.array([weights.get(name, 0.1) for name in names])
        sizes = 100 + 800 * sizes / max(sizes.max(), 1e-6)
    else:
        sizes = np.full(n, 200.0)

    ax.scatter(x_pos, y_pos, s=sizes, c=FEW_PALETTE["primary"],
               zorder=3, edgecolors="white", linewidth=1.5)

    for i, name in enumerate(names):
        offset = 0.12
        ax.text(x_pos[i] * (1 + offset), y_pos[i] * (1 + offset),
                name, ha="center", va="center", fontsize=9,
                color=FEW_PALETTE["grey_dark"])

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"Correlation Network (threshold={threshold})")
    fig.tight_layout()
    return fig, ax
