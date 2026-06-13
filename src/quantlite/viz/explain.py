"""Explainability visualisations: attribution, narratives, what-if, and audit charts.

All charts follow Stephen Few design principles: muted palette, maximum
data-ink ratio, direct labels, horizontal gridlines only, no box frames.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from .theme import FEW_PALETTE, apply_few_theme, few_figure

__all__ = [
    "plot_risk_waterfall",
    "plot_marginal_risk",
    "plot_factor_attribution",
    "plot_regime_summary",
    "plot_regime_transition_matrix",
    "plot_whatif_comparison",
    "plot_whatif_tornado",
    "plot_correlation_stress",
    "plot_weight_changes",
]


def _save_or_show(fig: Figure, save_path: str | None) -> Figure:
    """Save figure if path provided, otherwise return for display."""
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# Risk Attribution Charts
# ---------------------------------------------------------------------------


def plot_risk_waterfall(
    component_cvar: dict[str, float],
    total_cvar: float,
    title: str = "Portfolio CVaR Attribution",
    save_path: str | None = None,
) -> Figure:
    """Waterfall chart of per-asset CVaR contributions.

    Positive contributors stack upward; diversification benefit pulls
    downward to reach the total portfolio CVaR.

    Args:
        component_cvar: Per-asset CVaR contributions.
        total_cvar: Total portfolio CVaR.
        title: Chart title.
        save_path: Optional file path to save the figure.

    Returns:
        Matplotlib Figure.
    """
    fig, ax = few_figure(figsize=(12, 6))
    apply_few_theme()

    names = list(component_cvar.keys())
    values = [component_cvar[n] for n in names]

    # Add diversification benefit
    sum_components = sum(values)
    diversification = total_cvar - sum_components
    names_ext = names + ["Diversification", "Total CVaR"]
    values_ext = values + [diversification, total_cvar]

    # Compute waterfall positions
    n = len(names_ext)
    cumulative = np.zeros(n + 1)
    for i, v in enumerate(values_ext[:-1]):
        cumulative[i + 1] = cumulative[i] + v

    bottoms = np.zeros(n)
    heights = np.zeros(n)
    colours = []

    for i in range(n - 1):  # All but total
        if values_ext[i] >= 0:
            bottoms[i] = cumulative[i]
            heights[i] = values_ext[i]
            colours.append(FEW_PALETTE["negative"] if i < len(names) else FEW_PALETTE["positive"])
        else:
            bottoms[i] = cumulative[i + 1]
            heights[i] = abs(values_ext[i])
            colours.append(FEW_PALETTE["positive"])

    # Total bar starts from 0
    bottoms[-1] = 0
    heights[-1] = total_cvar
    colours.append(FEW_PALETTE["primary"])

    x = np.arange(n)
    bars = ax.bar(x, heights, bottom=bottoms, color=colours, width=0.6, edgecolor="white", linewidth=0.5)

    # Direct labels
    for i, bar in enumerate(bars):
        val = values_ext[i]
        y_pos = bar.get_y() + bar.get_height() + total_cvar * 0.02
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y_pos,
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold" if i == n - 1 else "normal",
            color=FEW_PALETTE["grey_dark"],
        )

    # Connector lines between bars
    for i in range(n - 2):
        top = cumulative[i + 1]
        ax.plot(
            [x[i] + 0.3, x[i + 1] - 0.3],
            [top, top],
            color=FEW_PALETTE["grey_mid"],
            linewidth=0.8,
            linestyle="--",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(names_ext, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("CVaR Contribution", fontsize=10)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)

    return _save_or_show(fig, save_path)


def plot_marginal_risk(
    marginal_cvar: dict[str, float],
    title: str = "Marginal Risk Contributions",
    save_path: str | None = None,
) -> Figure:
    """Horizontal bar chart of marginal CVaR contributions, sorted by magnitude.

    Args:
        marginal_cvar: Per-asset marginal CVaR.
        title: Chart title.
        save_path: Optional file path.

    Returns:
        Matplotlib Figure.
    """
    fig, ax = few_figure(figsize=(12, 6))
    apply_few_theme()

    # Sort by absolute magnitude
    sorted_items = sorted(marginal_cvar.items(), key=lambda x: abs(x[1]))
    names = [item[0] for item in sorted_items]
    values = [item[1] for item in sorted_items]

    colours = [
        FEW_PALETTE["negative"] if v > 0 else FEW_PALETTE["positive"]
        for v in values
    ]

    y = np.arange(len(names))
    bars = ax.barh(y, values, color=colours, height=0.6, edgecolor="white", linewidth=0.5)

    # Direct labels
    for bar, val in zip(bars, values):
        x_pos = bar.get_width() + 0.0001 * np.sign(val)
        ha = "left" if val >= 0 else "right"
        ax.text(
            x_pos, bar.get_y() + bar.get_height() / 2,
            f"{val:+.4f}", ha=ha, va="center", fontsize=9,
            color=FEW_PALETTE["grey_dark"],
        )

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=10)
    ax.axvline(0, color=FEW_PALETTE["grey_mid"], linewidth=0.8)
    ax.set_xlabel("Marginal CVaR (change from +1% allocation)", fontsize=10)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)

    return _save_or_show(fig, save_path)


def plot_factor_attribution(
    factor_contributions: dict[str, float],
    idiosyncratic_risk: float,
    title: str = "Factor Risk Attribution",
    save_path: str | None = None,
) -> Figure:
    """Grouped bar chart of factor vs idiosyncratic risk decomposition.

    Args:
        factor_contributions: Risk attributed to each factor.
        idiosyncratic_risk: Residual unexplained risk.
        title: Chart title.
        save_path: Optional file path.

    Returns:
        Matplotlib Figure.
    """
    fig, ax = few_figure(figsize=(12, 6))
    apply_few_theme()

    names = list(factor_contributions.keys()) + ["Idiosyncratic"]
    values = list(factor_contributions.values()) + [idiosyncratic_risk]
    total = sum(values)
    pcts = [v / total * 100 if total > 0 else 0 for v in values]

    palette = [
        FEW_PALETTE["primary"],
        FEW_PALETTE["secondary"],
        FEW_PALETTE["neutral"],
        FEW_PALETTE["positive"],
        FEW_PALETTE["negative"],
    ]
    colours = [palette[i % len(palette)] for i in range(len(names))]

    x = np.arange(len(names))
    bars = ax.bar(x, pcts, color=colours, width=0.6, edgecolor="white", linewidth=0.5)

    for bar, pct in zip(bars, pcts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{pct:.1f}%",
            ha="center", va="bottom", fontsize=10, fontweight="bold",
            color=FEW_PALETTE["grey_dark"],
        )

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylabel("Share of Total Risk (%)", fontsize=10)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)

    return _save_or_show(fig, save_path)


# ---------------------------------------------------------------------------
# Regime Narrative Charts
# ---------------------------------------------------------------------------


def plot_regime_summary(
    returns: pd.Series | np.ndarray,
    regime_labels: np.ndarray,
    regime_stats: list | None = None,
    dates: pd.DatetimeIndex | None = None,
    title: str = "Regime Summary",
    save_path: str | None = None,
) -> Figure:
    """Multi-panel regime summary: timeline on top, stats table below.

    Args:
        returns: Portfolio or asset returns.
        regime_labels: Array of regime labels.
        regime_stats: Optional list of RegimeStats dataclasses.
        dates: Optional datetime index for the x-axis.
        title: Chart title.
        save_path: Optional file path.

    Returns:
        Matplotlib Figure.
    """
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(14, 8), height_ratios=[3, 1],
        gridspec_kw={"hspace": 0.35},
    )
    apply_few_theme()

    labels = np.asarray(regime_labels)
    ret = returns.values if isinstance(returns, pd.Series) else np.asarray(returns)

    unique = sorted(set(labels))
    regime_colours = [
        FEW_PALETTE["positive"],
        FEW_PALETTE["secondary"],
        FEW_PALETTE["negative"],
        FEW_PALETTE["neutral"],
        FEW_PALETTE["primary"],
    ]

    x = dates if dates is not None else np.arange(len(ret))

    # Timeline with coloured background
    for i, regime in enumerate(unique):
        mask = labels == regime
        colour = regime_colours[i % len(regime_colours)]
        # Find contiguous blocks
        changes = np.diff(mask.astype(int))
        starts = np.where(changes == 1)[0] + 1
        ends = np.where(changes == -1)[0] + 1
        if mask[0]:
            starts = np.concatenate([[0], starts])
        if mask[-1]:
            ends = np.concatenate([ends, [len(mask)]])

        for s, e in zip(starts, ends):
            if dates is not None:
                ax_top.axvspan(dates[s], dates[min(e, len(dates) - 1)], alpha=0.15, color=colour)
            else:
                ax_top.axvspan(s, e, alpha=0.15, color=colour)

    # Cumulative returns
    cum = np.cumprod(1 + ret)
    ax_top.plot(x, cum, color=FEW_PALETTE["grey_dark"], linewidth=1.2)
    ax_top.set_ylabel("Cumulative Return", fontsize=10)
    ax_top.set_title(title, fontsize=13, fontweight="bold", pad=12)

    # Stats table
    ax_bot.axis("off")
    if regime_stats:
        col_labels = ["Regime", "Name", "Proportion", "Ann. Vol", "Avg Duration", "Sharpe"]
        table_data = []
        for s in regime_stats:
            table_data.append([
                str(s.label), s.name, f"{s.proportion:.0%}",
                f"{s.volatility:.1%}", f"{s.mean_duration:.0f}d", f"{s.sharpe:.2f}",
            ])
    else:
        col_labels = ["Regime", "Observations", "Mean Return", "Volatility"]
        table_data = []
        for regime in unique:
            mask = labels == regime
            r = ret[mask]
            table_data.append([
                str(regime), str(mask.sum()),
                f"{np.mean(r):.4%}", f"{np.std(r) * np.sqrt(252):.1%}",
            ])

    table = ax_bot.table(
        cellText=table_data, colLabels=col_labels,
        cellLoc="center", loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)

    # Style header
    for j in range(len(col_labels)):
        cell = table[0, j]
        cell.set_facecolor(FEW_PALETTE["primary"])
        cell.set_text_props(color="white", fontweight="bold")

    return _save_or_show(fig, save_path)


def plot_regime_transition_matrix(
    transition_matrix: pd.DataFrame,
    title: str = "Regime Transition Probabilities",
    save_path: str | None = None,
) -> Figure:
    """Annotated heatmap of regime transition probabilities.

    Args:
        transition_matrix: DataFrame of transition probabilities.
        title: Chart title.
        save_path: Optional file path.

    Returns:
        Matplotlib Figure.
    """
    fig, ax = few_figure(figsize=(10, 8))

    data = transition_matrix.values
    n = data.shape[0]

    # Custom colourmap from white to Few primary blue
    from matplotlib.colors import LinearSegmentedColormap
    _few_seq = LinearSegmentedColormap.from_list(
        "few_seq", ["#FFFFFF", FEW_PALETTE["primary"]], N=256,
    )
    ax.imshow(data, cmap=_few_seq, vmin=0, vmax=1, aspect="auto")

    # Annotate cells
    for i in range(n):
        for j in range(n):
            val = data[i, j]
            colour = "white" if val > 0.6 else FEW_PALETTE["grey_dark"]
            ax.text(
                j, i, f"{val:.1%}", ha="center", va="center",
                fontsize=12, fontweight="bold", color=colour,
            )

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([f"To {c}" for c in transition_matrix.columns], fontsize=10)
    ax.set_yticklabels([f"From {c}" for c in transition_matrix.index], fontsize=10)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)

    # Remove spines
    for spine in ax.spines.values():
        spine.set_visible(False)

    return _save_or_show(fig, save_path)


# ---------------------------------------------------------------------------
# What-If Analysis Charts
# ---------------------------------------------------------------------------


def plot_whatif_comparison(
    comparison_table: pd.DataFrame,
    metrics: list[str] | None = None,
    title: str = "What-If Scenario Comparison",
    save_path: str | None = None,
) -> Figure:
    """Grouped bar chart comparing scenarios across key metrics.

    Args:
        comparison_table: DataFrame with scenarios as rows, metrics as columns.
        metrics: Subset of metrics to plot. Defaults to Sharpe, CVaR, Max Drawdown.
        title: Chart title.
        save_path: Optional file path.

    Returns:
        Matplotlib Figure.
    """
    fig, ax = few_figure(figsize=(12, 6))
    apply_few_theme()

    if metrics is None:
        available = comparison_table.columns.tolist()
        metrics = [m for m in ["Sharpe", "CVaR (95%)", "Max Drawdown"] if m in available]
        if not metrics:
            metrics = available[:3]

    data = comparison_table[metrics]
    n_scenarios = len(data)
    n_metrics = len(metrics)
    x = np.arange(n_metrics)
    width = 0.8 / n_scenarios

    colours = [
        FEW_PALETTE["primary"],
        FEW_PALETTE["secondary"],
        FEW_PALETTE["negative"],
        FEW_PALETTE["positive"],
        FEW_PALETTE["neutral"],
    ]

    for i, (scenario, row) in enumerate(data.iterrows()):
        offset = (i - n_scenarios / 2 + 0.5) * width
        bars = ax.bar(
            x + offset, row.values, width * 0.9,
            color=colours[i % len(colours)], label=scenario,
            edgecolor="white", linewidth=0.5,
        )
        for bar, val in zip(bars, row.values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8,
                color=FEW_PALETTE["grey_dark"],
            )

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=10)
    ax.legend(fontsize=9, frameon=False)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)

    return _save_or_show(fig, save_path)


def plot_whatif_tornado(
    base_metrics: dict[str, float],
    scenario_metrics: dict[str, dict[str, float]],
    metric: str = "cvar_95",
    title: str = "What-If Impact Analysis",
    save_path: str | None = None,
) -> Figure:
    """Tornado chart showing which scenarios have the biggest impact.

    Args:
        base_metrics: Base case metrics dict.
        scenario_metrics: Mapping of scenario name to metrics dict.
        metric: Metric to compare (key in the metrics dicts).
        title: Chart title.
        save_path: Optional file path.

    Returns:
        Matplotlib Figure.
    """
    fig, ax = few_figure(figsize=(12, 6))
    apply_few_theme()

    base_val = base_metrics.get(metric, 0)
    deltas = {}
    for name, metrics_dict in scenario_metrics.items():
        deltas[name] = metrics_dict.get(metric, 0) - base_val

    # Sort by absolute impact
    sorted_items = sorted(deltas.items(), key=lambda x: abs(x[1]))
    names = [item[0] for item in sorted_items]
    values = [item[1] for item in sorted_items]

    colours = [
        FEW_PALETTE["negative"] if v > 0 else FEW_PALETTE["positive"]
        for v in values
    ]

    y = np.arange(len(names))
    bars = ax.barh(y, values, color=colours, height=0.6, edgecolor="white", linewidth=0.5)

    for bar, val in zip(bars, values):
        x_pos = bar.get_width() + 0.0002 * np.sign(val)
        ha = "left" if val >= 0 else "right"
        ax.text(
            x_pos, bar.get_y() + bar.get_height() / 2,
            f"{val:+.4f}", ha=ha, va="center", fontsize=9,
            color=FEW_PALETTE["grey_dark"],
        )

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=10)
    ax.axvline(0, color=FEW_PALETTE["grey_mid"], linewidth=0.8)
    ax.set_xlabel(f"Change in {metric} vs Base Case", fontsize=10)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)

    return _save_or_show(fig, save_path)


def plot_correlation_stress(
    normal_corr: pd.DataFrame,
    stressed_corr: pd.DataFrame,
    title: str = "Correlation: Normal vs Stressed",
    save_path: str | None = None,
) -> Figure:
    """Side-by-side correlation matrices: normal vs stressed.

    Args:
        normal_corr: Normal correlation matrix.
        stressed_corr: Stressed correlation matrix.
        title: Chart title.
        save_path: Optional file path.

    Returns:
        Matplotlib Figure.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    apply_few_theme()
    pass

    for ax, data, subtitle in [
        (ax1, normal_corr.values, "Normal"),
        (ax2, stressed_corr.values, "Stressed"),
    ]:
        n = data.shape[0]
        ax.imshow(data, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")

        for i in range(n):
            for j in range(n):
                colour = "white" if abs(data[i, j]) > 0.6 else FEW_PALETTE["grey_dark"]
                ax.text(
                    j, i, f"{data[i, j]:.2f}", ha="center", va="center",
                    fontsize=9, color=colour,
                )

        labels = normal_corr.columns.tolist()
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(labels, fontsize=9, rotation=45, ha="right")
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_title(subtitle, fontsize=11, fontweight="bold")

        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()

    return _save_or_show(fig, save_path)


# ---------------------------------------------------------------------------
# Audit Trail Chart
# ---------------------------------------------------------------------------


def plot_weight_changes(
    previous: dict[str, float],
    current: dict[str, float],
    title: str = "Portfolio Weight Changes",
    save_path: str | None = None,
) -> Figure:
    """Before/after weight comparison with arrows showing changes.

    Increases are coloured green, decreases red.

    Args:
        previous: Previous portfolio weights.
        current: Current portfolio weights.
        title: Chart title.
        save_path: Optional file path.

    Returns:
        Matplotlib Figure.
    """
    fig, ax = few_figure(figsize=(12, 6))
    apply_few_theme()

    all_assets = sorted(set(list(previous.keys()) + list(current.keys())))
    n = len(all_assets)
    y = np.arange(n)

    prev_vals = [previous.get(a, 0) for a in all_assets]
    curr_vals = [current.get(a, 0) for a in all_assets]

    # Plot dots for before and after
    ax.scatter(prev_vals, y, color=FEW_PALETTE["grey_mid"], s=80, zorder=3, label="Previous")
    ax.scatter(curr_vals, y, color=FEW_PALETTE["primary"], s=80, zorder=3, label="Current")

    # Arrows
    for i in range(n):
        delta = curr_vals[i] - prev_vals[i]
        if abs(delta) < 1e-6:
            continue
        colour = FEW_PALETTE["positive"] if delta > 0 else FEW_PALETTE["negative"]
        ax.annotate(
            "",
            xy=(curr_vals[i], y[i]),
            xytext=(prev_vals[i], y[i]),
            arrowprops=dict(
                arrowstyle="->",
                color=colour,
                lw=2,
                shrinkA=5,
                shrinkB=5,
            ),
        )
        # Label the delta
        mid_x = (prev_vals[i] + curr_vals[i]) / 2
        ax.text(
            mid_x, y[i] + 0.25,
            f"{delta:+.3f}",
            ha="center", va="bottom", fontsize=9,
            color=colour, fontweight="bold",
        )

    ax.set_yticks(y)
    ax.set_yticklabels(all_assets, fontsize=10)
    ax.set_xlabel("Weight", fontsize=10)
    ax.legend(fontsize=9, frameon=False, loc="lower right")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)

    return _save_or_show(fig, save_path)
