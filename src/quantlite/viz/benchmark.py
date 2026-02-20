"""Benchmark visualisations following Stephen Few principles.

Muted palette, maximum data-ink ratio, direct labels, no chartjunk.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ..benchmark.compare import ComparisonResult
from ..benchmark.speed import SpeedResult
from ..benchmark.tail_events import CrisisResult

# ---------------------------------------------------------------------------
# Palette & theme
# ---------------------------------------------------------------------------

PRIMARY = "#4E79A7"
SECONDARY = "#F28E2B"
NEGATIVE = "#E15759"
POSITIVE = "#59A14F"
NEUTRAL = "#76B7B2"
LIGHT_GREY = "#E0E0E0"
TEXT_GREY = "#4A4A4A"

_PALETTE = [PRIMARY, SECONDARY, NEGATIVE, POSITIVE, NEUTRAL, "#EDC948", "#B07AA1"]


def _apply_theme(ax: Axes) -> None:
    """Apply Stephen Few theme to an axes.

    Args:
        ax: Matplotlib axes to style.
    """
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color(LIGHT_GREY)
    ax.yaxis.grid(True, color=LIGHT_GREY, linewidth=0.5)
    ax.xaxis.grid(False)
    ax.tick_params(colors=TEXT_GREY, labelsize=9)
    ax.set_axisbelow(True)


def _direct_label(
    ax: Axes,
    x: float,
    y: float,
    text: str,
    colour: str = TEXT_GREY,
    fontsize: int = 8,
    va: str = "bottom",
) -> None:
    """Add a direct label to a chart element.

    Args:
        ax: Axes to annotate.
        x: X position.
        y: Y position.
        text: Label text.
        colour: Text colour.
        fontsize: Font size.
        va: Vertical alignment.
    """
    ax.text(x, y, text, ha="center", va=va, fontsize=fontsize,
            color=colour, fontweight="medium")


# ---------------------------------------------------------------------------
# Head-to-Head charts
# ---------------------------------------------------------------------------

def plot_var_accuracy(
    results: list[ComparisonResult],
    alpha: float = 0.05,
    figsize: tuple[float, float] = (12, 6),
) -> Figure:
    """Bar chart of VaR violation rates by method.

    Expected rate shown as horizontal reference line. Gaussian methods
    overshoot; QuantLite EVT stays closest to target.

    Args:
        results: Comparison results from ``run_comparison()``.
        alpha: Expected VaR violation rate.
        figsize: Figure size.

    Returns:
        Matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=150)
    _apply_theme(ax)

    # Collect all violation rates across datasets
    all_methods = {}  # type: Dict[str, List[float]]
    for r in results:
        for method, rate in r.var_violations.items():
            all_methods.setdefault(method, []).append(rate)

    methods = list(all_methods.keys())
    avg_rates = [np.mean(v) for v in all_methods.values()]

    colours = []
    for rate in avg_rates:
        if abs(rate - alpha) < alpha * 0.3:
            colours.append(POSITIVE)
        elif rate > alpha * 1.5:
            colours.append(NEGATIVE)
        else:
            colours.append(SECONDARY)

    x = np.arange(len(methods))
    bars = ax.bar(x, avg_rates, color=colours, width=0.6, edgecolor="none")

    # Reference line
    ax.axhline(y=alpha, color=PRIMARY, linewidth=1.5, linestyle="--", alpha=0.7)
    ax.text(len(methods) - 0.5, alpha + 0.003, f"Expected rate ({alpha:.0%})",
            ha="right", va="bottom", fontsize=8, color=PRIMARY, fontstyle="italic")

    # Direct labels
    for i, (bar, rate) in enumerate(zip(bars, avg_rates)):
        _direct_label(ax, i, rate + 0.003, f"{rate:.1%}",
                      colour=colours[i])

    ax.set_xticks(x)
    ax.set_xticklabels([m.split("(")[0].strip() for m in methods],
                       rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("VaR Violation Rate", fontsize=10, color=TEXT_GREY)
    ax.set_title("VaR Accuracy: Violation Rates by Method",
                 fontsize=13, color=TEXT_GREY, fontweight="bold", pad=15)

    fig.tight_layout()
    return fig


def plot_method_comparison(
    results: list[ComparisonResult],
    figsize: tuple[float, float] = (12, 6),
) -> Figure:
    """Grouped bar chart comparing Sharpe ratios across methods.

    Args:
        results: Comparison results (should include multi_asset dataset).
        figsize: Figure size.

    Returns:
        Matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=150)
    _apply_theme(ax)

    # Get portfolio-level results
    sharpe_data = {}  # type: Dict[str, float]
    for r in results:
        for method, sr in r.sharpe_ratios.items():
            sharpe_data[method] = sr

    if not sharpe_data:
        ax.text(0.5, 0.5, "No portfolio-level data available",
                transform=ax.transAxes, ha="center", fontsize=12, color=TEXT_GREY)
        return fig

    methods = list(sharpe_data.keys())
    values = list(sharpe_data.values())

    colours = [NEGATIVE if "Gaussian" in m or "HRP" in m or "Risk Parity" in m
               else POSITIVE for m in methods]

    x = np.arange(len(methods))
    bars = ax.bar(x, values, color=colours, width=0.6, edgecolor="none")

    for i, (bar, val) in enumerate(zip(bars, values)):
        _direct_label(ax, i, val + 0.02 if val >= 0 else val - 0.05,
                      f"{val:.2f}", colour=colours[i])

    ax.set_xticks(x)
    ax.set_xticklabels([m.split("(")[0].strip() for m in methods],
                       rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Sharpe Ratio", fontsize=10, color=TEXT_GREY)
    ax.set_title("Portfolio Sharpe Ratios: Baseline vs QuantLite Methods",
                 fontsize=13, color=TEXT_GREY, fontweight="bold", pad=15)

    fig.tight_layout()
    return fig


def plot_risk_estimate_scatter(
    results: list[ComparisonResult],
    figsize: tuple[float, float] = (12, 6),
) -> Figure:
    """Scatter of predicted vs actual losses by method.

    45-degree line = perfect prediction. Gaussian underestimates risk.

    Args:
        results: Comparison results.
        figsize: Figure size.

    Returns:
        Matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=150)
    _apply_theme(ax)

    for r in results:
        for method, metrics in r.methods.items():
            if "var" in metrics and "violation_rate" in metrics:
                predicted = abs(metrics["var"])
                # Actual = implied by violation rate
                actual = predicted * (metrics["violation_rate"] / 0.05)
                colour = NEGATIVE if "Gaussian" in method else (
                    POSITIVE if "QuantLite" in method else NEUTRAL)
                marker = "o" if "Gaussian" in method else "s"
                ax.scatter(predicted, actual, c=colour, s=80, marker=marker,
                           alpha=0.7, edgecolors="white", linewidth=0.5,
                           label=method.split("(")[0].strip())

    # 45-degree line
    lims = ax.get_xlim()
    ax.plot([0, max(lims[1], 0.1)], [0, max(lims[1], 0.1)],
            "--", color=NEUTRAL, linewidth=1, alpha=0.7)
    ax.text(max(lims[1], 0.08) * 0.7, max(lims[1], 0.08) * 0.75,
            "Perfect prediction", fontsize=8, color=NEUTRAL, fontstyle="italic",
            rotation=35)

    ax.set_xlabel("Predicted Loss (|VaR|)", fontsize=10, color=TEXT_GREY)
    ax.set_ylabel("Implied Actual Loss", fontsize=10, color=TEXT_GREY)
    ax.set_title("Risk Estimation Accuracy: Predicted vs Actual",
                 fontsize=13, color=TEXT_GREY, fontweight="bold", pad=15)

    # De-duplicate legend
    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    unique = [(h, lab) for h, lab in zip(handles, labels) if lab not in seen and not seen.add(lab)]
    if unique:
        ax.legend(*zip(*unique), frameon=False, fontsize=8, loc="upper left")

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Tail Event charts
# ---------------------------------------------------------------------------

def plot_crisis_var_comparison(
    results: list[CrisisResult],
    figsize: tuple[float, float] = (14, 6),
) -> Figure:
    """Grouped bar chart: predicted VaR vs actual worst loss per crisis.

    Args:
        results: Crisis analysis results.
        figsize: Figure size.

    Returns:
        Matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=150)
    _apply_theme(ax)

    names = [r.crisis_name for r in results]
    gaussian = [abs(r.gaussian_var) for r in results]
    evt = [abs(r.evt_var) for r in results]
    actual = [abs(r.actual_worst_loss) for r in results]

    x = np.arange(len(names))
    w = 0.25

    ax.bar(x - w, gaussian, w, color=NEGATIVE, label="Gaussian VaR", edgecolor="none")
    ax.bar(x, evt, w, color=POSITIVE, label="EVT VaR", edgecolor="none")
    ax.bar(x + w, actual, w, color=PRIMARY, label="Actual worst loss", edgecolor="none")

    for i in range(len(names)):
        _direct_label(ax, i - w, gaussian[i] + 0.002, f"{gaussian[i]:.1%}",
                      colour=NEGATIVE, fontsize=7)
        _direct_label(ax, i, evt[i] + 0.002, f"{evt[i]:.1%}",
                      colour=POSITIVE, fontsize=7)
        _direct_label(ax, i + w, actual[i] + 0.002, f"{actual[i]:.1%}",
                      colour=PRIMARY, fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{n}\n({r.year})" for n, r in zip(names, results)],
                       fontsize=8)
    ax.set_ylabel("Loss Magnitude", fontsize=10, color=TEXT_GREY)
    ax.set_title("Crisis Performance: Predicted VaR vs Actual Worst Loss",
                 fontsize=13, color=TEXT_GREY, fontweight="bold", pad=15)
    ax.legend(frameon=False, fontsize=9, loc="upper left")

    fig.tight_layout()
    return fig


def plot_var_violations(
    crisis_result: CrisisResult,
    alpha: float = 0.05,
    figsize: tuple[float, float] = (14, 6),
) -> Figure:
    """Timeline of daily returns with VaR threshold lines and violations.

    Args:
        crisis_result: A single crisis result.
        alpha: VaR level used.
        figsize: Figure size.

    Returns:
        Matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=150)
    _apply_theme(ax)

    rets = crisis_result.crisis_returns
    days = np.arange(len(rets))
    g_var = crisis_result.gaussian_var
    e_var = crisis_result.evt_var

    # Plot returns
    ax.bar(days, rets, width=0.8, color=NEUTRAL, alpha=0.6, edgecolor="none")

    # VaR lines
    ax.axhline(y=g_var, color=NEGATIVE, linewidth=1.5, linestyle="--")
    ax.axhline(y=e_var, color=POSITIVE, linewidth=1.5, linestyle="-")

    # Violations
    g_mask = rets < g_var
    e_mask = rets < e_var
    if np.any(g_mask):
        ax.scatter(days[g_mask], rets[g_mask], c=NEGATIVE, s=30, zorder=5,
                   label=f"Gaussian violations ({int(g_mask.sum())})")
    if np.any(e_mask):
        ax.scatter(days[e_mask], rets[e_mask], c=POSITIVE, s=30, zorder=5,
                   marker="D",
                   label=f"EVT violations ({int(e_mask.sum())})")

    # Labels on lines
    ax.text(len(rets) - 1, g_var - 0.003, "Gaussian VaR",
            ha="right", fontsize=8, color=NEGATIVE, fontstyle="italic")
    ax.text(len(rets) - 1, e_var + 0.003, "EVT VaR",
            ha="right", fontsize=8, color=POSITIVE, fontstyle="italic")

    ax.set_xlabel("Trading Day", fontsize=10, color=TEXT_GREY)
    ax.set_ylabel("Daily Return", fontsize=10, color=TEXT_GREY)
    ax.set_title(f"VaR Violations: {crisis_result.crisis_name} ({crisis_result.year})",
        fontsize=13, color=TEXT_GREY, fontweight="bold", pad=15)
    ax.legend(frameon=False, fontsize=9, loc="lower left")

    fig.tight_layout()
    return fig


def plot_crisis_timeline(
    results: list[CrisisResult],
    figsize: tuple[float, float] = (14, 10),
) -> Figure:
    """Multi-panel: cumulative loss paths with VaR envelopes per crisis.

    Args:
        results: Crisis analysis results.
        figsize: Figure size.

    Returns:
        Matplotlib Figure.
    """
    n_crises = len(results)
    fig, axes = plt.subplots(n_crises, 1, figsize=figsize, dpi=150)
    if n_crises == 1:
        axes = [axes]

    for ax, cr in zip(axes, results):
        _apply_theme(ax)
        cum = np.cumprod(1 + cr.crisis_returns) - 1
        days = np.arange(len(cum))

        ax.plot(days, cum, color=PRIMARY, linewidth=1.5)
        ax.fill_between(days, cum, 0, where=(cum < 0), color=PRIMARY, alpha=0.1)

        # VaR envelopes (cumulative)
        g_cum = np.arange(1, len(cum) + 1) * cr.gaussian_var
        e_cum = np.arange(1, len(cum) + 1) * cr.evt_var
        ax.plot(days, g_cum, "--", color=NEGATIVE, linewidth=1, alpha=0.7)
        ax.plot(days, e_cum, "-", color=POSITIVE, linewidth=1, alpha=0.7)

        ax.set_title(f"{cr.crisis_name} ({cr.year})",
                     fontsize=10, color=TEXT_GREY, fontweight="bold", loc="left")
        ax.set_ylabel("Cumulative Return", fontsize=8, color=TEXT_GREY)

    axes[-1].set_xlabel("Trading Day", fontsize=10, color=TEXT_GREY)
    fig.suptitle("Crisis Timelines: Cumulative Losses with VaR Envelopes",
                 fontsize=13, color=TEXT_GREY, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Speed charts
# ---------------------------------------------------------------------------

def plot_scaling(
    results: list[SpeedResult],
    figsize: tuple[float, float] = (12, 6),
) -> Figure:
    """Log-log chart of computation time vs data size.

    Args:
        results: Speed benchmark results.
        figsize: Figure size.

    Returns:
        Matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=150)
    _apply_theme(ax)

    # Group by operation
    ops = {}  # type: Dict[str, Tuple[List[int], List[float]]]
    for r in results:
        ops.setdefault(r.operation, ([], []))
        ops[r.operation][0].append(r.data_size)
        ops[r.operation][1].append(r.quantlite_time)

    for i, (op, (sizes, times)) in enumerate(ops.items()):
        colour = _PALETTE[i % len(_PALETTE)]
        ax.plot(sizes, times, "o-", color=colour, linewidth=1.5,
                markersize=5, label=op.split("(")[0].strip())

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of Observations", fontsize=10, color=TEXT_GREY)
    ax.set_ylabel("Time (seconds)", fontsize=10, color=TEXT_GREY)
    ax.set_title("Computational Scaling: Time vs Data Size",
                 fontsize=13, color=TEXT_GREY, fontweight="bold", pad=15)
    ax.legend(frameon=False, fontsize=8, loc="upper left")

    fig.tight_layout()
    return fig


def plot_speed_comparison(
    results: list[SpeedResult],
    target_size: int = 10_000,
    figsize: tuple[float, float] = (12, 6),
) -> Figure:
    """Horizontal bar chart: QuantLite vs naive speed at a given data size.

    Args:
        results: Speed benchmark results.
        target_size: Data size to show.
        figsize: Figure size.

    Returns:
        Matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=150)
    _apply_theme(ax)

    filtered = [r for r in results if r.data_size == target_size]
    if not filtered:
        # Fall back to closest available size
        available = set(r.data_size for r in results)
        if available:
            closest = min(available, key=lambda s: abs(s - target_size))
            filtered = [r for r in results if r.data_size == closest]

    ops = [r.operation.split("(")[0].strip() for r in filtered]
    ql_times = [r.quantlite_time * 1000 for r in filtered]  # ms
    bl_times = [r.baseline_time * 1000 for r in filtered]

    y = np.arange(len(ops))
    h = 0.35

    ax.barh(y - h / 2, ql_times, h, color=PRIMARY, label="QuantLite", edgecolor="none")
    ax.barh(y + h / 2, bl_times, h, color=SECONDARY, label="Baseline", edgecolor="none")

    # Speedup labels
    for i, r in enumerate(filtered):
        factor = r.speedup
        x_pos = max(ql_times[i], bl_times[i]) + 0.5
        ax.text(x_pos, i, f"{factor:.1f}x",
                va="center", fontsize=9, color=TEXT_GREY, fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels(ops, fontsize=9)
    ax.set_xlabel("Time (ms)", fontsize=10, color=TEXT_GREY)
    ax.set_title(f"Speed Comparison at {filtered[0].data_size if filtered else target_size:,} Observations",
        fontsize=13, color=TEXT_GREY, fontweight="bold", pad=15)
    ax.legend(frameon=False, fontsize=9, loc="lower right")

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Summary dashboard
# ---------------------------------------------------------------------------

def plot_benchmark_summary(
    comparison_results: list[ComparisonResult],
    crisis_results: list[CrisisResult],
    speed_results: list[SpeedResult],
    alpha: float = 0.05,
    figsize: tuple[float, float] = (16, 10),
) -> Figure:
    """2x2 dashboard: VaR accuracy, Sharpe, crisis performance, speed.

    Args:
        comparison_results: Head-to-head results.
        crisis_results: Tail event results.
        speed_results: Speed benchmark results.
        alpha: VaR significance level.
        figsize: Figure size.

    Returns:
        Matplotlib Figure.
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize, dpi=150)

    # (1) VaR accuracy — top left
    ax = axes[0, 0]
    _apply_theme(ax)
    all_methods = {}  # type: Dict[str, List[float]]
    for r in comparison_results:
        for method, rate in r.var_violations.items():
            all_methods.setdefault(method, []).append(rate)

    if all_methods:
        methods = list(all_methods.keys())
        avg_rates = [np.mean(v) for v in all_methods.values()]
        short_names = [m.split("(")[0].strip()[:20] for m in methods]
        colours = [POSITIVE if abs(r - alpha) < alpha * 0.3 else NEGATIVE
                   for r in avg_rates]
        ax.bar(range(len(methods)), avg_rates, color=colours, edgecolor="none")
        ax.axhline(y=alpha, color=PRIMARY, linewidth=1, linestyle="--", alpha=0.7)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(short_names, rotation=30, ha="right", fontsize=7)
    ax.set_title("VaR Accuracy", fontsize=10, color=TEXT_GREY, fontweight="bold")

    # (2) Sharpe comparison — top right
    ax = axes[0, 1]
    _apply_theme(ax)
    sharpe_data = {}  # type: Dict[str, float]
    for r in comparison_results:
        for m, s in r.sharpe_ratios.items():
            sharpe_data[m] = s
    if sharpe_data:
        methods = list(sharpe_data.keys())
        values = list(sharpe_data.values())
        short_names = [m.split("(")[0].strip()[:20] for m in methods]
        colours = [POSITIVE if "QuantLite" in m else SECONDARY for m in methods]
        ax.bar(range(len(methods)), values, color=colours, edgecolor="none")
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(short_names, rotation=30, ha="right", fontsize=7)
    ax.set_title("Sharpe Ratios", fontsize=10, color=TEXT_GREY, fontweight="bold")

    # (3) Crisis performance — bottom left
    ax = axes[1, 0]
    _apply_theme(ax)
    if crisis_results:
        names = [f"{r.crisis_name[:12]}\n{r.year}" for r in crisis_results]
        g_rates = [r.gaussian_violation_rate for r in crisis_results]
        e_rates = [r.evt_violation_rate for r in crisis_results]
        x = np.arange(len(names))
        w = 0.35
        ax.bar(x - w / 2, g_rates, w, color=NEGATIVE, edgecolor="none")
        ax.bar(x + w / 2, e_rates, w, color=POSITIVE, edgecolor="none")
        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=7)
    ax.set_title("Crisis VaR Violations", fontsize=10, color=TEXT_GREY, fontweight="bold")

    # (4) Speed — bottom right
    ax = axes[1, 1]
    _apply_theme(ax)
    filtered = [r for r in speed_results if r.data_size == 10_000]
    if not filtered:
        available = set(r.data_size for r in speed_results)
        if available:
            closest = min(available, key=lambda s: abs(s - 10_000))
            filtered = [r for r in speed_results if r.data_size == closest]
    if filtered:
        ops = [r.operation.split("(")[0].strip()[:18] for r in filtered]
        speedups = [r.speedup for r in filtered]
        colours = [POSITIVE if s >= 1 else NEGATIVE for s in speedups]
        y = np.arange(len(ops))
        ax.barh(y, speedups, color=colours, edgecolor="none")
        ax.set_yticks(y)
        ax.set_yticklabels(ops, fontsize=7)
        ax.axvline(x=1.0, color=NEUTRAL, linewidth=1, linestyle="--", alpha=0.5)
    ax.set_title("Speed (QuantLite vs Baseline)", fontsize=10,
                 color=TEXT_GREY, fontweight="bold")

    fig.suptitle("QuantLite Benchmark Summary",
                 fontsize=15, color=TEXT_GREY, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig
