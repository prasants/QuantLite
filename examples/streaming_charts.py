"""Generate streaming visualisations for documentation."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from quantlite.viz.streaming import plot_live_feed, plot_stream_latency, plot_tick_density
from quantlite.viz.theme import apply_few_theme

DOCS_IMAGES = Path(__file__).resolve().parent.parent / "docs" / "images"
DOCS_IMAGES.mkdir(parents=True, exist_ok=True)

apply_few_theme()


def chart_live_feed() -> None:
    """Simulated BTC-USD live feed with bid/ask spread."""
    import pandas as pd

    rng = np.random.default_rng(42)
    n = 500
    timestamps = pd.date_range("2024-03-15 14:00", periods=n, freq="6s")
    # Random walk for price
    returns = rng.normal(0, 0.001, n)
    prices = 42_000 * np.exp(np.cumsum(returns))
    # Spread widens during volatile periods
    spread_base = prices * 0.0003
    spread_noise = np.abs(rng.normal(0, prices * 0.0001, n))
    half_spread = spread_base + spread_noise
    bid = prices - half_spread
    ask = prices + half_spread

    fig, ax = plot_live_feed(timestamps, prices, bid, ask, symbol="BTC-USD")
    import matplotlib.dates as mdates
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=10))
    fig.savefig(DOCS_IMAGES / "streaming_live_feed.png", bbox_inches="tight")
    plt.close(fig)
    print("✓ streaming_live_feed.png")


def chart_tick_density() -> None:
    """Tick inter-arrival time histogram showing bursty activity."""
    rng = np.random.default_rng(99)
    # Mix of fast bursts and quiet periods (bimodal)
    fast = rng.exponential(50, 3000)   # Burst ticks ~50ms apart
    slow = rng.exponential(500, 1000)  # Quiet ticks ~500ms apart
    inter_arrival = np.concatenate([fast, slow])
    rng.shuffle(inter_arrival)

    fig, ax = plot_tick_density(inter_arrival, bins=60, symbol="ETH-USD")
    fig.savefig(DOCS_IMAGES / "streaming_tick_density.png", bbox_inches="tight")
    plt.close(fig)
    print("✓ streaming_tick_density.png")


def chart_stream_latency() -> None:
    """Latency distribution with percentile markers."""
    rng = np.random.default_rng(7)
    # Log-normal latency: mostly fast, occasional spikes
    latencies = rng.lognormal(mean=2.5, sigma=0.8, size=5000)

    fig, ax = plot_stream_latency(latencies)
    fig.savefig(DOCS_IMAGES / "streaming_latency.png", bbox_inches="tight")
    plt.close(fig)
    print("✓ streaming_latency.png")


if __name__ == "__main__":
    chart_live_feed()
    chart_tick_density()
    chart_stream_latency()
    print("All streaming charts generated.")
