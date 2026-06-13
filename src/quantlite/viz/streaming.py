"""Streaming visualisation charts using the Stephen Few theme.

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
    "plot_live_feed",
    "plot_tick_density",
    "plot_stream_latency",
]


def plot_live_feed(
    timestamps: np.ndarray | Any,
    prices: np.ndarray | Any,
    bid: np.ndarray | Any | None = None,
    ask: np.ndarray | Any | None = None,
    symbol: str = "BTC-USD",
    figsize: tuple[float, float] = (10, 5),
) -> tuple[Figure, Axes]:
    """Plot a simulated live price feed with bid/ask spread shading.

    Args:
        timestamps: Array of timestamps (or sequential indices).
        prices: Mid/last prices.
        bid: Bid prices (optional, for spread shading).
        ask: Ask prices (optional, for spread shading).
        symbol: Instrument label.
        figsize: Figure dimensions.

    Returns:
        Tuple of (Figure, Axes).
    """
    apply_few_theme()
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(timestamps, prices, color=FEW_PALETTE["primary"], linewidth=1.4, label="Price")

    if bid is not None and ask is not None:
        ax.fill_between(
            timestamps,
            bid,
            ask,
            alpha=0.2,
            color=FEW_PALETTE["secondary"],
            label="Bid-ask spread",
        )

    ax.set_xlabel("Time")
    ax.set_ylabel("Price ($)")
    ax.set_title(f"{symbol} Live Feed")

    # Direct labels — offset vertically to avoid overlap
    price_range = prices.max() - prices.min()
    label_offset = price_range * 0.04
    direct_label(
        ax, timestamps[int(len(timestamps) * 0.75)], prices[int(len(timestamps) * 0.75)] + label_offset,
        symbol, colour=FEW_PALETTE["primary"], fontsize=10,
    )
    if bid is not None and ask is not None:
        direct_label(
            ax, timestamps[int(len(timestamps) * 0.45)], bid[int(len(timestamps) * 0.45)] - label_offset,
            "Bid–ask spread", colour=FEW_PALETTE["secondary"], fontsize=9,
        )

    return fig, ax


def plot_tick_density(
    inter_arrival_ms: np.ndarray | Any,
    bins: int = 50,
    symbol: str = "BTC-USD",
    figsize: tuple[float, float] = (10, 5),
) -> tuple[Figure, Axes]:
    """Plot a histogram of tick inter-arrival times.

    Shows market activity patterns: tight clustering indicates
    bursts of activity, long tails indicate quiet periods.

    Args:
        inter_arrival_ms: Inter-arrival times in milliseconds.
        bins: Number of histogram bins.
        symbol: Instrument label.
        figsize: Figure dimensions.

    Returns:
        Tuple of (Figure, Axes).
    """
    apply_few_theme()
    fig, ax = plt.subplots(figsize=figsize)

    ax.hist(
        inter_arrival_ms,
        bins=bins,
        color=FEW_PALETTE["primary"],
        edgecolor=FEW_PALETTE["bg"],
        linewidth=0.5,
        alpha=0.85,
    )

    median_val = float(np.median(inter_arrival_ms))
    ax.axvline(median_val, color=FEW_PALETTE["secondary"], linewidth=1.8, linestyle="--")
    direct_label(
        ax, median_val, ax.get_ylim()[1] * 0.9,
        f"  Median: {median_val:.0f} ms",
        colour=FEW_PALETTE["secondary"],
    )

    ax.set_xlabel("Inter-arrival time (ms)")
    ax.set_ylabel("Frequency")
    ax.set_title(f"{symbol} Tick Arrival Density")

    return fig, ax


def plot_stream_latency(
    latencies_ms: np.ndarray | Any,
    figsize: tuple[float, float] = (10, 5),
    percentiles: tuple[float, ...] = (50, 95, 99),
) -> tuple[Figure, Axes]:
    """Plot latency distribution for stream feed monitoring.

    Displays a histogram of latencies with key percentile markers,
    useful for assessing feed health and identifying degradation.

    Args:
        latencies_ms: Observed latencies in milliseconds.
        figsize: Figure dimensions.
        percentiles: Percentile lines to draw.

    Returns:
        Tuple of (Figure, Axes).
    """
    apply_few_theme()
    fig, ax = plt.subplots(figsize=figsize)

    ax.hist(
        latencies_ms,
        bins=60,
        color=FEW_PALETTE["primary"],
        edgecolor=FEW_PALETTE["bg"],
        linewidth=0.5,
        alpha=0.85,
    )

    colours = [FEW_PALETTE["primary"], FEW_PALETTE["secondary"], FEW_PALETTE["negative"]]
    for i, p in enumerate(percentiles):
        val = float(np.percentile(latencies_ms, p))
        c = colours[i % len(colours)]
        ax.axvline(val, color=c, linewidth=1.8, linestyle="--")
        direct_label(
            ax, val, ax.get_ylim()[1] * (0.95 - i * 0.1),
            f"  p{int(p)}: {val:.1f} ms",
            colour=c,
        )

    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("Frequency")
    ax.set_title("Stream Feed Latency Distribution")

    return fig, ax
