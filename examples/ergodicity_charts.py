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
    """Chart 1: Ensemble average vs median wealth paths."""
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

    # Individual paths
    for i in range(n_paths):
        ax.plot(x, wealth[i], color="#999999", alpha=0.3, linewidth=0.5)

    # Ensemble average and median
    ax.plot(x, ensemble_avg, color=FEW_PALETTE["secondary"], linewidth=2.5)
    ax.plot(x, median_path, color=FEW_PALETTE["primary"], linewidth=2.5)

    # Direct labels
    direct_label(
        ax, n_periods + 2, ensemble_avg[-1], "Ensemble\naverage",
        colour=FEW_PALETTE["secondary"], fontsize=10,
    )
    direct_label(
        ax, n_periods + 2, median_path[-1], "Median\npath",
        colour=FEW_PALETTE["primary"], fontsize=10,
    )

    ax.set_title("The Ergodicity Illusion: Ensemble Average vs Median Path")
    ax.set_xlabel("Period")
    ax.set_ylabel("Wealth ($)")
    ax.set_yscale("log")
    ax.set_xlim(0, n_periods + 20)

    fig.savefig(DOCS_IMAGES / "ergodicity_gap.png")
    plt.close(fig)
    print("Saved ergodicity_gap.png")


def chart_leverage_growth() -> None:
    """Chart 2: Leverage vs time-average growth rate."""
    rng = np.random.default_rng(42)
    returns = rng.normal(0.0005, 0.02, 10000)

    leverages = np.linspace(0.1, 5.0, 50)
    growth_rates = [time_average(lev * returns) for lev in leverages]
    growth_rates = np.array(growth_rates)

    # Find Kelly optimal
    kelly_idx = np.argmax(growth_rates)
    kelly_lev = leverages[kelly_idx]
    kelly_growth = growth_rates[kelly_idx]

    # Find ruin threshold (where growth crosses zero)
    zero_crossings = np.where(np.diff(np.sign(growth_rates)))[0]
    ruin_lev = None
    if len(zero_crossings) > 0:
        idx = zero_crossings[0]
        # Linear interpolation
        ruin_lev = leverages[idx] + (
            leverages[idx + 1] - leverages[idx]
        ) * (-growth_rates[idx]) / (growth_rates[idx + 1] - growth_rates[idx])

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(leverages, growth_rates, color=FEW_PALETTE["primary"], linewidth=2)
    ax.axhline(0, color="#999999", linewidth=0.8, linestyle="-")

    # Kelly optimal
    ax.plot(kelly_lev, kelly_growth, "o", color=FEW_PALETTE["secondary"], markersize=10, zorder=5)
    direct_label(
        ax, kelly_lev + 0.1, kelly_growth,
        f"Kelly optimal ({kelly_lev:.1f}x)",
        colour=FEW_PALETTE["secondary"],
    )

    # Ruin threshold
    if ruin_lev is not None:
        ax.axvline(ruin_lev, color=FEW_PALETTE["negative"], linewidth=1.5, linestyle="--")
        ax.fill_betweenx(
            [growth_rates.min(), growth_rates.max()],
            ruin_lev, leverages[-1],
            color=FEW_PALETTE["negative"], alpha=0.08,
        )
        direct_label(
            ax, ruin_lev + 0.1, growth_rates.min() * 0.5,
            "Ruin zone", colour=FEW_PALETTE["negative"],
        )

    ax.set_title("Leverage vs Time-Average Growth")
    ax.set_xlabel("Leverage multiple")
    ax.set_ylabel("Time-average growth rate")

    fig.savefig(DOCS_IMAGES / "ergodicity_leverage.png")
    plt.close(fig)
    print("Saved ergodicity_leverage.png")


def chart_kelly_sensitivity() -> None:
    """Chart 3: Kelly fraction sensitivity to volatility."""
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

    # Direct labels
    for bar, kf in zip(bars, kelly_fracs):  # noqa: B905
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
            f"{kf:.1f}x", ha="center", va="bottom",
            color=FEW_PALETTE["grey_dark"], fontsize=11,
        )

    ax.set_title("Kelly Fraction Shrinks as Volatility Rises")
    ax.set_xlabel("Volatility (per period)")
    ax.set_ylabel("Kelly fraction")

    fig.savefig(DOCS_IMAGES / "kelly_sensitivity.png")
    plt.close(fig)
    print("Saved kelly_sensitivity.png")


if __name__ == "__main__":
    chart_ergodicity_gap()
    chart_leverage_growth()
    chart_kelly_sensitivity()
