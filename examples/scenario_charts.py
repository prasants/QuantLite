"""Generate scenario stress testing visualisations for documentation."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from quantlite.scenarios import SCENARIO_LIBRARY, stress_test
from quantlite.viz.theme import FEW_PALETTE, apply_few_theme

DOCS_IMAGES = Path(__file__).resolve().parent.parent / "docs" / "images"
DOCS_IMAGES.mkdir(parents=True, exist_ok=True)

apply_few_theme()


def chart_portfolio_impact() -> None:
    """Chart 7: Portfolio impact by crisis scenario."""
    weights = {"SPX": 0.30, "BTC": 0.15, "ETH": 0.10, "BONDS_10Y": 0.30, "GLD": 0.15}

    names = []
    impacts = []
    for name, scenario in SCENARIO_LIBRARY.items():
        result = stress_test(weights, scenario)
        names.append(name)
        impacts.append(result["portfolio_impact"])

    # Sort from worst to least bad
    order = np.argsort(impacts)
    names = [names[i] for i in order]
    impacts = [impacts[i] for i in order]

    colours = [
        FEW_PALETTE["negative"] if imp < -0.15 else FEW_PALETTE["secondary"]
        for imp in impacts
    ]

    fig, ax = plt.subplots(figsize=(10, 5))
    y_pos = range(len(names))
    bars = ax.barh(y_pos, impacts, color=colours, height=0.5, edgecolor="none")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)

    # Direct labels (always outside bar end for readability)
    for i, (_bar, imp) in enumerate(zip(bars, impacts)):  # noqa: B905
        ax.text(
            imp + 0.004, i,
            f"{imp:.1%}", ha="left", va="center",
            color=FEW_PALETTE["grey_dark"], fontsize=10, fontweight="bold",
        )

    ax.set_title("Portfolio Impact by Crisis Scenario")
    ax.set_xlabel("Portfolio impact")
    ax.axvline(0, color="#999999", linewidth=0.8)
    ax.set_xlim(min(impacts) * 1.15, 0.02)
    ax.yaxis.grid(False)

    fig.savefig(DOCS_IMAGES / "scenario_portfolio_impact.png", bbox_inches="tight")
    plt.close(fig)
    print("Saved scenario_portfolio_impact.png")


def chart_shock_propagation() -> None:
    """Chart 8: Shock propagation network diagram."""
    # Simulated propagation from BTC -50%
    assets = {
        "BTC": {"x": 0.5, "y": 0.5, "shock": -0.50, "role": "source"},
        "ETH": {"x": 0.85, "y": 0.8, "shock": -0.38, "role": "secondary"},
        "SPX": {"x": 0.15, "y": 0.8, "shock": -0.12, "role": "secondary"},
        "GLD": {"x": 0.15, "y": 0.2, "shock": +0.03, "role": "secondary"},
        "BONDS_10Y": {"x": 0.85, "y": 0.2, "shock": -0.05, "role": "secondary"},
    }

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.05, 1.05)
    ax.axis("off")
    ax.set_title("Shock Propagation: BTC -50% Crash", pad=20)

    # Draw arrows from BTC to others
    btc = assets["BTC"]
    for name, info in assets.items():
        if name == "BTC":
            continue
        shock_val = info["shock"]
        thickness = max(1, abs(shock_val) * 15)
        arrow_colour = FEW_PALETTE["negative"] if shock_val < 0 else FEW_PALETTE["secondary"]

        ax.annotate(
            "",
            xy=(info["x"], info["y"]),
            xytext=(btc["x"], btc["y"]),
            arrowprops={
                "arrowstyle": "-|>",
                "color": arrow_colour,
                "lw": thickness,
                "mutation_scale": 15,
            },
        )

        # Label on the arrow (midpoint)
        mx = (btc["x"] + info["x"]) / 2
        my = (btc["y"] + info["y"]) / 2
        ax.text(
            mx, my, f"{shock_val:+.0%}",
            fontsize=10, fontweight="bold",
            color=arrow_colour, ha="center", va="center",
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 2, "alpha": 0.8},
        )

    # Draw asset circles
    for name, info in assets.items():
        colour = FEW_PALETTE["negative"] if name == "BTC" else FEW_PALETTE["primary"]
        label = "BNDS" if name == "BONDS_10Y" else name
        size = 2000 if name == "BTC" else 1600
        ax.scatter(info["x"], info["y"], s=size, c=colour, zorder=5, edgecolors="white", linewidth=2)
        ax.text(
            info["x"], info["y"], label,
            ha="center", va="center", fontsize=9,
            fontweight="bold", color="white", zorder=6,
        )
        # Full name label below for bonds
        if name == "BONDS_10Y":
            ax.text(
                info["x"], info["y"] - 0.07, "BONDS_10Y",
                ha="center", va="top", fontsize=7,
                color=FEW_PALETTE["grey_mid"],
            )

    fig.savefig(DOCS_IMAGES / "shock_propagation_network.png")
    plt.close(fig)
    print("Saved shock_propagation_network.png")


if __name__ == "__main__":
    chart_portfolio_impact()
    chart_shock_propagation()
