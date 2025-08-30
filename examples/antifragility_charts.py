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
    """Payoff curves: fragile (concave), robust (linear), antifragile (convex)."""
    shocks = np.linspace(-0.3, 0.3, 200)
    fragile = shocks - 2 * shocks**2
    robust = shocks
    antifragile = shocks + 2 * shocks**2

    fig, ax = plt.subplots(figsize=(9, 5.5))

    ax.plot(shocks, antifragile, color=FEW_PALETTE["positive"], linewidth=2.5,
            label="Antifragile")
    ax.plot(shocks, robust, color=FEW_PALETTE["grey_mid"], linewidth=2,
            label="Robust")
    ax.plot(shocks, fragile, color=FEW_PALETTE["negative"], linewidth=2.5,
            label="Fragile")

    # Direct labels at right edge with padding
    direct_label(ax, 0.315, antifragile[-1], "Antifragile",
                 colour=FEW_PALETTE["positive"], fontsize=10)
    direct_label(ax, 0.315, robust[-1], "Robust",
                 colour=FEW_PALETTE["grey_mid"], fontsize=10)
    direct_label(ax, 0.315, fragile[-1], "Fragile",
                 colour=FEW_PALETTE["negative"], fontsize=10)

    # Subtle zero lines
    ax.axhline(0, color="#E0E0E0", linewidth=0.8, zorder=0)
    ax.axvline(0, color="#E0E0E0", linewidth=0.8, zorder=0)

    # Annotate key insight: at shock = -0.25, show the asymmetry
    shock_pt = -0.25
    idx = np.argmin(np.abs(shocks - shock_pt))
    ax.annotate(
        f"Same shock ({shock_pt:+.0%}),\nvery different outcomes",
        xy=(shock_pt, fragile[idx]),
        xytext=(-0.12, -0.38),
        fontsize=8, color=FEW_PALETTE["grey_dark"],
        arrowprops={"arrowstyle": "->", "color": FEW_PALETTE["grey_mid"],
                     "lw": 0.8},
    )

    ax.set_title("Payoff Convexity: Fragile, Robust, Antifragile", fontsize=13)
    ax.set_xlabel("Shock magnitude")
    ax.set_ylabel("Portfolio response")
    ax.set_xlim(-0.32, 0.42)

    fig.savefig(DOCS_IMAGES / "payoff_convexity.png", bbox_inches="tight")
    plt.close(fig)
    print("Saved payoff_convexity.png")


def chart_fourth_quadrant() -> None:
    """Taleb's Four Quadrants: kurtosis vs payoff complexity."""
    rng = np.random.default_rng(42)

    # Use log scale for x-axis to give thin-tailed points proper space.
    # Kurtosis boundary at 3 (more realistic than 1).
    kurt_boundary = 3.0
    nonlin_boundary = 2.0

    # Generate synthetic points spread across quadrants
    points = []
    colours = []
    labels = []

    # Q1: Thin-tailed, simple (safe zone)
    for _ in range(8):
        points.append((rng.uniform(0.1, kurt_boundary), rng.uniform(0.2, nonlin_boundary)))
        colours.append(FEW_PALETTE["primary"])
        labels.append("Q1")

    # Q2: Thin-tailed, complex
    for _ in range(6):
        points.append((rng.uniform(0.1, kurt_boundary), rng.uniform(nonlin_boundary, 5.5)))
        colours.append(FEW_PALETTE["secondary"])
        labels.append("Q2")

    # Q3: Fat-tailed, simple
    for _ in range(6):
        points.append((rng.uniform(kurt_boundary, 30), rng.uniform(0.2, nonlin_boundary)))
        colours.append(FEW_PALETTE["neutral"])
        labels.append("Q3")

    # Q4: Fat-tailed, complex (DANGER)
    for _ in range(10):
        points.append((rng.uniform(kurt_boundary, 30), rng.uniform(nonlin_boundary, 5.5)))
        colours.append(FEW_PALETTE["negative"])
        labels.append("Q4")

    xs, ys = zip(*points)  # noqa: B905

    fig, ax = plt.subplots(figsize=(10, 6.5))

    # Shade the Fourth Quadrant
    ax.axvspan(kurt_boundary, 35, ymin=nonlin_boundary / 6.0, ymax=1.0,
               color=FEW_PALETTE["negative"], alpha=0.06, zorder=0)

    ax.scatter(xs, ys, c=colours, s=70, zorder=5, edgecolors="white", linewidth=0.8)

    # Quadrant boundary lines
    ax.axvline(kurt_boundary, color=FEW_PALETTE["grey_mid"], linestyle="--",
               linewidth=1, alpha=0.6)
    ax.axhline(nonlin_boundary, color=FEW_PALETTE["grey_mid"], linestyle="--",
               linewidth=1, alpha=0.6)

    # Quadrant labels: positioned in clear areas, not on data points
    label_style = {"fontsize": 10, "ha": "center", "va": "center"}

    ax.text(1.5, 1.0, "First Quadrant\nThin-tailed, Simple",
            color=FEW_PALETTE["primary"], alpha=0.7, **label_style)
    ax.text(1.5, 4.2, "Second Quadrant\nThin-tailed, Complex",
            color=FEW_PALETTE["secondary"], alpha=0.7, **label_style)
    ax.text(18.0, 1.0, "Third Quadrant\nFat-tailed, Simple",
            color=FEW_PALETTE["neutral"], alpha=0.8, **label_style)
    ax.text(18.0, 4.2, "FOURTH QUADRANT\nFat-tailed + Complex",
            color=FEW_PALETTE["negative"], fontsize=11, fontweight="bold",
            ha="center", va="center")
    ax.text(18.0, 3.6, "Models fail here",
            color=FEW_PALETTE["negative"], fontsize=9, fontstyle="italic",
            ha="center", va="center", alpha=0.7)

    ax.set_xlim(0, 33)
    ax.set_ylim(0, 5.8)
    ax.set_title("Taleb's Four Quadrants", fontsize=13)
    ax.set_xlabel("Excess kurtosis (fat-tailedness)")
    ax.set_ylabel("Payoff nonlinearity (complexity)")

    fig.savefig(DOCS_IMAGES / "fourth_quadrant_map.png", bbox_inches="tight")
    plt.close(fig)
    print("Saved fourth_quadrant_map.png")


def chart_barbell_vs_balanced() -> None:
    """Barbell (90/10) vs balanced (50/50) cumulative wealth."""
    rng = np.random.default_rng(42)
    n = 1000
    bond_returns = rng.normal(0.0001, 0.002, n)
    crypto_returns = rng.standard_t(3, n) * 0.03

    barbell = 0.90 * bond_returns + 0.10 * crypto_returns
    balanced = 0.50 * bond_returns + 0.50 * crypto_returns

    barbell_wealth = 100 * np.cumprod(1 + barbell)
    balanced_wealth = 100 * np.cumprod(1 + balanced)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(1, n + 1)
    ax.plot(x, barbell_wealth, color=FEW_PALETTE["primary"], linewidth=2)
    ax.plot(x, balanced_wealth, color=FEW_PALETTE["secondary"], linewidth=1.8)

    # Starting capital reference line
    ax.axhline(100, color="#E0E0E0", linewidth=0.8, linestyle=":", zorder=0)

    # Direct labels
    direct_label(ax, n + 15, barbell_wealth[-1], "Barbell (90/10)",
                 colour=FEW_PALETTE["primary"], fontsize=10)
    direct_label(ax, n + 15, balanced_wealth[-1], "Balanced (50/50)",
                 colour=FEW_PALETTE["secondary"], fontsize=10)

    # Annotate max drawdown of balanced
    balanced_peak_idx = np.argmax(balanced_wealth)
    balanced_trough_idx = balanced_peak_idx + np.argmin(
        balanced_wealth[balanced_peak_idx:]
    )
    dd_pct = (balanced_wealth[balanced_trough_idx] / balanced_wealth[balanced_peak_idx] - 1)

    ax.annotate(
        f"Balanced max DD: {dd_pct:.0%}",
        xy=(balanced_trough_idx, balanced_wealth[balanced_trough_idx]),
        xytext=(balanced_trough_idx - 200, balanced_wealth[balanced_trough_idx] + 15),
        fontsize=8, color=FEW_PALETTE["secondary"],
        arrowprops={"arrowstyle": "->", "color": FEW_PALETTE["secondary"],
                     "lw": 0.8},
    )

    ax.set_title("Barbell vs Balanced: Cumulative Wealth", fontsize=13)
    ax.set_xlabel("Period")
    ax.set_ylabel("Wealth ($)")
    ax.set_xlim(0, n + 100)

    fig.savefig(DOCS_IMAGES / "barbell_vs_balanced.png", bbox_inches="tight")
    plt.close(fig)
    print("Saved barbell_vs_balanced.png")


if __name__ == "__main__":
    chart_payoff_convexity()
    chart_fourth_quadrant()
    chart_barbell_vs_balanced()
