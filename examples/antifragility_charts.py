"""Generate antifragility visualisations for documentation."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from quantlite.viz.theme import FEW_PALETTE, apply_few_theme, direct_label

DOCS_IMAGES = Path(__file__).resolve().parent.parent / "docs" / "images"
DOCS_IMAGES.mkdir(parents=True, exist_ok=True)

apply_few_theme()


def chart_payoff_convexity() -> None:
    """Chart 4: Fragile, robust, antifragile payoff curves."""
    shocks = np.linspace(-0.3, 0.3, 200)
    fragile = shocks - 2 * shocks**2
    robust = shocks
    antifragile = shocks + 2 * shocks**2

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(shocks, fragile, color=FEW_PALETTE["negative"], linewidth=2)
    ax.plot(shocks, robust, color="#999999", linewidth=2)
    ax.plot(shocks, antifragile, color=FEW_PALETTE["positive"], linewidth=2)

    # Direct labels at right edge
    direct_label(ax, 0.31, fragile[-1], "Fragile", colour=FEW_PALETTE["negative"])
    direct_label(ax, 0.31, robust[-1], "Robust", colour="#999999")
    direct_label(ax, 0.31, antifragile[-1], "Antifragile", colour=FEW_PALETTE["positive"])

    ax.axhline(0, color="#CCCCCC", linewidth=0.5)
    ax.axvline(0, color="#CCCCCC", linewidth=0.5)
    ax.set_title("Payoff Convexity: Fragile, Robust, Antifragile")
    ax.set_xlabel("Shock magnitude")
    ax.set_ylabel("Portfolio response")

    fig.savefig(DOCS_IMAGES / "payoff_convexity.png")
    plt.close(fig)
    print("Saved payoff_convexity.png")


def chart_fourth_quadrant() -> None:
    """Chart 5: Taleb's four quadrants."""
    rng = np.random.default_rng(42)

    # Generate synthetic points in each quadrant
    points = []
    colours = []
    # Q1: thin-tailed, simple (low kurtosis, low nonlinearity)
    for _ in range(8):
        points.append((rng.uniform(0, 1), rng.uniform(0, 2)))
        colours.append(FEW_PALETTE["primary"])
    # Q2: thin-tailed, complex
    for _ in range(6):
        points.append((rng.uniform(0, 1), rng.uniform(2, 6)))
        colours.append(FEW_PALETTE["secondary"])
    # Q3: fat-tailed, simple
    for _ in range(7):
        points.append((rng.uniform(1, 15), rng.uniform(0, 2)))
        colours.append(FEW_PALETTE["neutral"])
    # Q4: fat-tailed, complex (DANGER)
    for _ in range(9):
        points.append((rng.uniform(1, 15), rng.uniform(2, 6)))
        colours.append(FEW_PALETTE["negative"])

    xs, ys = zip(*points)  # noqa: B905

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(xs, ys, c=colours, s=60, zorder=5, edgecolors="white", linewidth=0.5)

    # Quadrant lines
    ax.axvline(1, color="#999999", linestyle="--", linewidth=1)
    ax.axhline(2, color="#999999", linestyle="--", linewidth=1)

    # Labels
    ax.text(0.5, 0.5, "First:\nThin-tailed, Simple", ha="center", fontsize=9, color=FEW_PALETTE["grey_dark"])
    ax.text(0.5, 4.0, "Second:\nThin-tailed, Complex", ha="center", fontsize=9, color=FEW_PALETTE["grey_dark"])
    ax.text(8.0, 0.5, "Third:\nFat-tailed, Simple", ha="center", fontsize=9, color=FEW_PALETTE["grey_dark"])
    ax.text(8.0, 4.0, "Fourth:\nFat-tailed, Complex\n(DANGER)", ha="center", fontsize=9, color=FEW_PALETTE["negative"], fontweight="bold")

    ax.set_xlim(0, 15)
    ax.set_ylim(0, 6)
    ax.set_title("Taleb's Four Quadrants")
    ax.set_xlabel("Excess kurtosis")
    ax.set_ylabel("Payoff nonlinearity")

    fig.savefig(DOCS_IMAGES / "fourth_quadrant_map.png")
    plt.close(fig)
    print("Saved fourth_quadrant_map.png")


def chart_barbell_vs_balanced() -> None:
    """Chart 6: Barbell vs balanced cumulative wealth."""
    rng = np.random.default_rng(42)
    n = 1000
    bond_returns = rng.normal(0.0001, 0.002, n)
    crypto_returns = rng.standard_t(3, n) * 0.03

    barbell = 0.90 * bond_returns + 0.10 * crypto_returns
    balanced = 0.50 * bond_returns + 0.50 * crypto_returns

    barbell_wealth = 100 * np.cumprod(1 + barbell)
    balanced_wealth = 100 * np.cumprod(1 + balanced)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(1, n + 1)
    ax.plot(x, barbell_wealth, color=FEW_PALETTE["primary"], linewidth=2)
    ax.plot(x, balanced_wealth, color=FEW_PALETTE["secondary"], linewidth=2)

    direct_label(ax, n + 10, barbell_wealth[-1], "Barbell\n(90/10)", colour=FEW_PALETTE["primary"])
    direct_label(ax, n + 10, balanced_wealth[-1], "Balanced\n(50/50)", colour=FEW_PALETTE["secondary"])

    ax.set_title("Barbell vs Balanced: Cumulative Wealth")
    ax.set_xlabel("Period")
    ax.set_ylabel("Wealth ($)")
    ax.set_xlim(0, n + 80)

    fig.savefig(DOCS_IMAGES / "barbell_vs_balanced.png")
    plt.close(fig)
    print("Saved barbell_vs_balanced.png")


if __name__ == "__main__":
    chart_payoff_convexity()
    chart_fourth_quadrant()
    chart_barbell_vs_balanced()
