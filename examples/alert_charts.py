"""Generate alert visualisations for documentation."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from quantlite.viz.alerts import plot_alert_history, plot_alert_timeline, plot_threshold_monitor
from quantlite.viz.theme import apply_few_theme

DOCS_IMAGES = Path(__file__).resolve().parent.parent / "docs" / "images"
DOCS_IMAGES.mkdir(parents=True, exist_ok=True)

apply_few_theme()


def chart_alert_timeline() -> None:
    """BTC-USD price with alert trigger markers."""
    rng = np.random.default_rng(42)
    n = 400
    returns = rng.normal(0.0002, 0.012, n)
    # Inject a crash and recovery
    returns[150:170] = rng.normal(-0.02, 0.025, 20)
    returns[250:260] = rng.normal(0.015, 0.02, 10)
    prices = 43_000 * np.exp(np.cumsum(returns))
    timestamps = np.arange(n)

    # Alerts at large moves
    alert_indices = []
    alert_types = []
    for i in range(1, n):
        ret = abs(returns[i])
        if ret > 0.03:
            alert_indices.append(i)
            alert_types.append("threshold")
        elif i in [152, 251]:
            alert_indices.append(i)
            alert_types.append("regime")

    fig, ax = plot_alert_timeline(
        timestamps, prices,
        np.array(alert_indices), alert_types,
        symbol="BTC-USD",
    )
    fig.savefig(DOCS_IMAGES / "alert_timeline.png", bbox_inches="tight")
    plt.close(fig)
    print("✓ alert_timeline.png")


def chart_threshold_monitor() -> None:
    """Rolling volatility with upper/lower thresholds and alert zones."""
    rng = np.random.default_rng(99)
    n = 300
    returns = rng.normal(0, 0.01, n)
    # Inject volatile period
    returns[100:140] = rng.normal(0, 0.035, 40)
    # Rolling 20-day volatility
    vol = np.array([np.std(returns[max(0, i - 20):i + 1]) for i in range(n)])
    timestamps = np.arange(n)

    fig, ax = plot_threshold_monitor(
        timestamps, vol,
        upper_threshold=0.025,
        lower_threshold=0.005,
        metric_name="20-day Rolling Volatility",
    )
    fig.savefig(DOCS_IMAGES / "alert_threshold_monitor.png", bbox_inches="tight")
    plt.close(fig)
    print("✓ alert_threshold_monitor.png")


def chart_alert_history() -> None:
    """Alert frequency breakdown by type over 14 days."""
    rng = np.random.default_rng(77)
    periods = [f"Day {i + 1}" for i in range(14)]
    # More alerts during the "crash" in the middle
    base_threshold = rng.poisson(2, 14)
    base_regime = rng.poisson(0.5, 14)
    # Spike in days 7-9
    base_threshold[6:9] += rng.integers(3, 8, 3)
    base_regime[6:9] += rng.integers(1, 4, 3)

    fig, ax = plot_alert_history(
        np.array(periods),
        base_regime,
        base_threshold,
    )
    fig.savefig(DOCS_IMAGES / "alert_history.png", bbox_inches="tight")
    plt.close(fig)
    print("✓ alert_history.png")


if __name__ == "__main__":
    chart_alert_timeline()
    chart_threshold_monitor()
    chart_alert_history()
    print("All alert charts generated.")
