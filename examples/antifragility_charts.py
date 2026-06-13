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
    """Taleb's Four Quadrants: kurtosis vs payoff complexity.

    Uses sqrt-transformed x-axis so thin-tailed and fat-tailed regions
    get equal visual space. Labels are positioned with white backgrounds
    to avoid overlap with data points.
    """
    rng = np.random.default_rng(42)

    kurt_boundary = 3.0
    nonlin_boundary = 2.5

    def tx(x):
        """Sqrt transform for x-axis."""
        return np.sqrt(np.asarray(x))

    # Generate synthetic points in each quadrant
    q1 = [(rng.uniform(0.2, kurt_boundary), rng.uniform(0.3, nonlin_boundary))
           for _ in range(8)]
    q2 = [(rng.uniform(0.2, kurt_boundary), rng.uniform(nonlin_boundary, 5.5))
           for _ in range(6)]
    q3 = [(rng.uniform(kurt_boundary, 25), rng.uniform(0.3, nonlin_boundary))
           for _ in range(6)]
    q4 = [(rng.uniform(kurt_boundary, 25), rng.uniform(nonlin_boundary, 5.5))
           for _ in range(10)]

    fig, ax = plt.subplots(figsize=(11, 7))

    for pts, colour in [
        (q1, FEW_PALETTE["primary"]),
        (q2, FEW_PALETTE["secondary"]),
        (q3, FEW_PALETTE["neutral"]),
        (q4, FEW_PALETTE["negative"]),
    ]:
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.scatter(tx(xs), ys, c=colour, s=65, zorder=5,
                   edgecolors="white", linewidth=0.8)

    # Quadrant boundaries
    bx = tx(kurt_boundary)
    ax.axvline(bx, color=FEW_PALETTE["grey_mid"], linestyle="--",
               linewidth=1, alpha=0.5)
    ax.axhline(nonlin_boundary, color=FEW_PALETTE["grey_mid"], linestyle="--",
               linewidth=1, alpha=0.5)

    # Shade fourth quadrant
    x_max = tx(30)
    ax.fill_between([bx, x_max], nonlin_boundary, 5.8,
                    color=FEW_PALETTE["negative"], alpha=0.06, zorder=0)

    # Quadrant labels with white background for legibility
    bbox_style = {"facecolor": "white", "edgecolor": "none",
                  "alpha": 0.7, "pad": 3}

    ax.text(tx(1.5), 1.2, "FIRST QUADRANT\nThin-tailed + Simple\n(models work here)",
            ha="center", va="center", fontsize=9,
            color=FEW_PALETTE["primary"], fontweight="bold", bbox=bbox_style)
    ax.text(tx(1.2), 5.3, "SECOND QUADRANT\nThin-tailed + Complex",
            ha="center", va="center", fontsize=9,
            color=FEW_PALETTE["secondary"], fontweight="bold", bbox=bbox_style)
    ax.text(tx(14), 1.2, "THIRD QUADRANT\nFat-tailed + Simple",
            ha="center", va="center", fontsize=9,
            color=FEW_PALETTE["neutral"], fontweight="bold", bbox=bbox_style)
    ax.text(tx(14), 4.5, "FOURTH QUADRANT\nFat-tailed + Complex",
            ha="center", va="center", fontsize=12,
            color=FEW_PALETTE["negative"], fontweight="bold",
            bbox={"facecolor": "white", "edgecolor": FEW_PALETTE["negative"],
                  "alpha": 0.85, "pad": 5, "linewidth": 1.5})
    ax.text(tx(14), 3.7, "Models are dangerous here.\nUse extreme caution.",
            ha="center", va="center", fontsize=9,
            color=FEW_PALETTE["negative"], fontstyle="italic", alpha=0.7)

    # Custom x-ticks in original kurtosis space
    tick_vals = [0, 1, 2, 3, 5, 10, 15, 20, 25]
    ax.set_xticks(tx(tick_vals))
    ax.set_xticklabels([str(v) for v in tick_vals])

    ax.set_xlim(0, tx(30))
    ax.set_ylim(0, 5.8)
    ax.set_title("Taleb's Four Quadrants", fontsize=14, pad=12)
    ax.set_xlabel("Excess kurtosis (fat-tailedness)")
    ax.set_ylabel("Payoff nonlinearity (complexity)")
    ax.yaxis.grid(False)
    ax.xaxis.grid(False)

    fig.savefig(DOCS_IMAGES / "fourth_quadrant_map.png",
                bbox_inches="tight", dpi=150)
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
