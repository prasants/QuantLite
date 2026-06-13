"""Generate ergodicity visualisations for documentation."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Ensure quantlite is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from quantlite.ergodicity import kelly_fraction, time_average
from quantlite.viz.theme import FEW_PALETTE, apply_few_theme, direct_label

DOCS_IMAGES = Path(__file__).resolve().parent.parent / "docs" / "images"
DOCS_IMAGES.mkdir(parents=True, exist_ok=True)

apply_few_theme()


def chart_ergodicity_gap() -> None:
    """50 wealth paths showing ensemble average soaring while median collapses."""
    rng = np.random.default_rng(42)
    n_paths, n_periods = 50, 200
    outcomes = rng.choice([0.50, -0.40], size=(n_paths, n_periods))

    # Build wealth paths
    wealth = np.ones((n_paths, n_periods + 1)) * 100.0
    for t in range(n_periods):
        wealth[:, t + 1] = wealth[:, t] * (1 + outcomes[:, t])

    ensemble_avg = wealth.mean(axis=0)
    median_path = np.median(wealth, axis=0)

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(n_periods + 1)

    # Individual paths (thin, subtle)
    for i in range(n_paths):
        ax.plot(x, wealth[i], color="#CCCCCC", alpha=0.25, linewidth=0.4)

    # Key paths
    ax.plot(x, ensemble_avg, color=FEW_PALETTE["secondary"], linewidth=2.5,
            zorder=3)
    ax.plot(x, median_path, color=FEW_PALETTE["primary"], linewidth=2.5,
            zorder=3)

    # Starting capital reference
    ax.axhline(100, color="#E0E0E0", linewidth=0.8, linestyle=":", zorder=1)

    # Direct labels
    direct_label(
        ax, n_periods + 3, ensemble_avg[-1], "Ensemble\naverage",
        colour=FEW_PALETTE["secondary"], fontsize=10,
    )
    direct_label(
        ax, n_periods + 3, median_path[-1], "Median\npath",
        colour=FEW_PALETTE["primary"], fontsize=10,
    )

    # Annotate the divergence: place in clear space
    ax.annotate(
        "Same game, same odds.\nThe average soars. You go broke.",
        xy=(100, median_path[100]),
        xytext=(30, 1e-5),
        fontsize=9, color=FEW_PALETTE["grey_dark"], fontstyle="italic",
        arrowprops={"arrowstyle": "->", "color": FEW_PALETTE["grey_mid"],
                     "lw": 0.8},
    )

    ax.set_title("The Ergodicity Illusion: Ensemble Average vs Median Path",
                 fontsize=13)
    ax.set_xlabel("Period")
    ax.set_ylabel("Wealth ($)")
    ax.set_yscale("log")
    ax.set_xlim(0, n_periods + 25)

    fig.savefig(DOCS_IMAGES / "ergodicity_gap.png", bbox_inches="tight")
    plt.close(fig)
    print("Saved ergodicity_gap.png")


def chart_leverage_growth() -> None:
    """Leverage vs time-average growth rate with Kelly peak and ruin zone."""
    rng = np.random.default_rng(42)
    returns = rng.normal(0.0005, 0.02, 10000)

    leverages = np.linspace(0.1, 5.0, 80)
    growth_rates = np.array([time_average(lev * returns) for lev in leverages])

    # Find Kelly optimal
    kelly_idx = int(np.argmax(growth_rates))
    kelly_lev = leverages[kelly_idx]
    kelly_growth = growth_rates[kelly_idx]

    # Find ruin threshold (where growth crosses zero)
    zero_crossings = np.where(np.diff(np.sign(growth_rates)))[0]
    ruin_lev = None
    if len(zero_crossings) > 0:
        idx = zero_crossings[0]
        ruin_lev = leverages[idx] + (
            leverages[idx + 1] - leverages[idx]
        ) * (-growth_rates[idx]) / (growth_rates[idx + 1] - growth_rates[idx])

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(leverages, growth_rates, color=FEW_PALETTE["primary"], linewidth=2.5)
    ax.axhline(0, color="#CCCCCC", linewidth=0.8)

    # Kelly optimal
    ax.plot(kelly_lev, kelly_growth, "o", color=FEW_PALETTE["secondary"],
            markersize=10, zorder=5)
    ax.annotate(
        f"Kelly optimal ({kelly_lev:.1f}x)",
        xy=(kelly_lev, kelly_growth),
        xytext=(kelly_lev + 0.8, kelly_growth + 0.0001),
        fontsize=9, color=FEW_PALETTE["secondary"], fontweight="bold",
        arrowprops={"arrowstyle": "->", "color": FEW_PALETTE["secondary"],
                     "lw": 1},
    )

    # Ruin threshold
    if ruin_lev is not None:
        ax.axvline(ruin_lev, color=FEW_PALETTE["negative"], linewidth=1.5,
                   linestyle="--", alpha=0.7)
        ax.fill_betweenx(
            [growth_rates.min() * 1.1, growth_rates.max() * 1.1],
            ruin_lev, leverages[-1],
            color=FEW_PALETTE["negative"], alpha=0.06,
        )
        ax.text(ruin_lev + 0.15, growth_rates.min() * 0.4,
                f"Ruin zone (>{ruin_lev:.1f}x)\nNegative geometric growth",
                fontsize=9, color=FEW_PALETTE["negative"], fontstyle="italic")

    ax.set_title("Leverage vs Time-Average Growth", fontsize=13)
    ax.set_xlabel("Leverage multiple")
    ax.set_ylabel("Time-average growth rate (per period)")

    fig.savefig(DOCS_IMAGES / "ergodicity_leverage.png", bbox_inches="tight")
    plt.close(fig)
    print("Saved ergodicity_leverage.png")


def chart_kelly_sensitivity() -> None:
    """Kelly fraction shrinks as volatility rises."""
    rng = np.random.default_rng(42)
    volatilities = [0.01, 0.02, 0.03, 0.05, 0.08]
    kelly_fracs = []

    for vol in volatilities:
        rets = rng.normal(0.001, vol, 5000)
        kf = kelly_fraction(rets)
        kelly_fracs.append(kf)

    labels = [f"{v:.0%}" for v in volatilities]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, kelly_fracs, color=FEW_PALETTE["primary"], width=0.5,
                  edgecolor="none")

    # Direct labels above each bar
    for bar, kf in zip(bars, kelly_fracs):  # noqa: B905
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.06,
            f"{kf:.1f}x", ha="center", va="bottom",
            color=FEW_PALETTE["grey_dark"], fontsize=11, fontweight="bold",
        )

    ax.set_title("Kelly Fraction Shrinks as Volatility Rises", fontsize=13)
    ax.set_xlabel("Volatility (per period)")
    ax.set_ylabel("Kelly fraction (optimal leverage)")
    ax.set_ylim(0, max(kelly_fracs) * 1.2)
    ax.yaxis.grid(True)

    fig.savefig(DOCS_IMAGES / "kelly_sensitivity.png", bbox_inches="tight")
    plt.close(fig)
    print("Saved kelly_sensitivity.png")


if __name__ == "__main__":
    chart_ergodicity_gap()
    chart_leverage_growth()
    chart_kelly_sensitivity()
