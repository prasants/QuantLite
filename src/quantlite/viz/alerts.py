"""Alert visualisation charts using the Stephen Few theme.

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
    "plot_alert_timeline",
    "plot_threshold_monitor",
    "plot_alert_history",
]


def plot_alert_timeline(
    timestamps: np.ndarray | Any,
    prices: np.ndarray | Any,
    alert_indices: np.ndarray | Any,
    alert_types: list[str] | None = None,
    symbol: str = "BTC-USD",
    figsize: tuple[float, float] = (10, 5),
) -> tuple[Figure, Axes]:
    """Plot a price series with alert trigger points marked.

    Each alert is shown as a marker on the price line, colour-coded
    by type (regime change vs threshold breach).

    Args:
        timestamps: Time axis.
        prices: Price series.
        alert_indices: Indices into timestamps/prices where alerts fired.
        alert_types: Type label per alert (e.g. "regime", "threshold").
        symbol: Instrument label.
        figsize: Figure dimensions.

    Returns:
        Tuple of (Figure, Axes).
    """
    apply_few_theme()
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(timestamps, prices, color=FEW_PALETTE["primary"], linewidth=1.2)

    type_colours = {
        "regime": FEW_PALETTE["secondary"],
        "threshold": FEW_PALETTE["negative"],
    }

    if alert_types is None:
        alert_types = ["threshold"] * len(alert_indices)

    plotted_types: set[str] = set()
    for idx, atype in zip(alert_indices, alert_types):
        colour = type_colours.get(atype, FEW_PALETTE["negative"])
        label = atype.capitalize() if atype not in plotted_types else None
        ax.plot(timestamps[idx], prices[idx], "v", color=colour,
                markersize=8, label=label)
        plotted_types.add(atype)

    ax.set_xlabel("Time")
    ax.set_ylabel("Price ($)")
    ax.set_title(f"{symbol} Price with Alert Triggers")
    ax.legend(loc="upper left", frameon=False)

    return fig, ax


def plot_threshold_monitor(
    timestamps: np.ndarray | Any,
    values: np.ndarray | Any,
    upper_threshold: float | None = None,
    lower_threshold: float | None = None,
    metric_name: str = "Volatility",
    figsize: tuple[float, float] = (10, 5),
) -> tuple[Figure, Axes]:
    """Plot a monitored metric with threshold lines and alert zones.

    Regions beyond thresholds are shaded to indicate alert zones.

    Args:
        timestamps: Time axis.
        values: Metric values being monitored.
        upper_threshold: Upper alert threshold.
        lower_threshold: Lower alert threshold.
        metric_name: Label for the metric.
        figsize: Figure dimensions.

    Returns:
        Tuple of (Figure, Axes).
    """
    apply_few_theme()
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(timestamps, values, color=FEW_PALETTE["primary"], linewidth=1.4)

    if upper_threshold is not None:
        ax.axhline(upper_threshold, color=FEW_PALETTE["negative"],
                    linewidth=1.2, linestyle="--")
        ax.fill_between(timestamps, upper_threshold, values,
                        where=values > upper_threshold,
                        alpha=0.15, color=FEW_PALETTE["negative"],
                        interpolate=True)
        direct_label(ax, timestamps[-1], upper_threshold,
                     f"  Upper: {upper_threshold:.2f}",
                     colour=FEW_PALETTE["negative"])

    if lower_threshold is not None:
        ax.axhline(lower_threshold, color=FEW_PALETTE["secondary"],
                    linewidth=1.2, linestyle="--")
        ax.fill_between(timestamps, lower_threshold, values,
                        where=values < lower_threshold,
                        alpha=0.15, color=FEW_PALETTE["secondary"],
                        interpolate=True)
        direct_label(ax, timestamps[-1], lower_threshold,
                     f"  Lower: {lower_threshold:.2f}",
                     colour=FEW_PALETTE["secondary"])

    ax.set_xlabel("Time")
    ax.set_ylabel(metric_name)
    ax.set_title(f"{metric_name} Threshold Monitor")

    return fig, ax


def plot_alert_history(
    periods: np.ndarray | Any,
    regime_counts: np.ndarray | Any,
    threshold_counts: np.ndarray | Any,
    figsize: tuple[float, float] = (10, 5),
) -> tuple[Figure, Axes]:
    """Plot a summary of alerts fired over time.

    Stacked bars show the frequency and type breakdown of alerts
    per period (e.g. per day or per hour).

    Args:
        periods: Period labels (e.g. dates or hour indices).
        regime_counts: Number of regime-change alerts per period.
        threshold_counts: Number of threshold-breach alerts per period.
        figsize: Figure dimensions.

    Returns:
        Tuple of (Figure, Axes).
    """
    apply_few_theme()
    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(periods))
    width = 0.6

    ax.bar(x, threshold_counts, width, color=FEW_PALETTE["negative"],
           alpha=0.85, label="Threshold breach")
    ax.bar(x, regime_counts, width, bottom=threshold_counts,
           color=FEW_PALETTE["secondary"], alpha=0.85, label="Regime change")

    ax.set_xticks(x)
    ax.set_xticklabels(periods, rotation=45, ha="right")
    ax.set_xlabel("Period")
    ax.set_ylabel("Alert count")
    ax.set_title("Alert History by Type")
    ax.legend(loc="upper left", frameon=False)

    return fig, ax
