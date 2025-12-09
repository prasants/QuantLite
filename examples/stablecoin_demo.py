"""Stablecoin risk analysis demo for Acme Fund.

Generates charts for docs/images/:
- stablecoin_depeg_probability.png
- stablecoin_reserve_comparison.png
- stablecoin_deviation_tracker.png
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from quantlite.crypto.stablecoin import (
    depeg_probability,
    peg_deviation_tracker,
    reserve_risk_score,
)

# Chart style constants
PRIMARY = "#4E79A7"
SECONDARY = "#F28E2B"
NEGATIVE = "#E15759"
POSITIVE = "#59A14F"
NEUTRAL = "#76B7B2"
BG = "white"

IMG_DIR = os.path.join(os.path.dirname(__file__), "..", "docs", "images")
os.makedirs(IMG_DIR, exist_ok=True)


def _style_ax(ax, title, xlabel="", ylabel=""):
    ax.set_facecolor(BG)
    ax.figure.set_facecolor(BG)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10)
    ax.yaxis.grid(True, alpha=0.3, color="#cccccc")
    ax.xaxis.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# --- Chart 1: Depeg probability across stablecoins ---
rng = np.random.default_rng(42)

# Simulate price series for different stability profiles
stable_usdc = 1.0 + rng.normal(0, 0.0008, 365)
moderate_dai = 1.0 + rng.normal(0, 0.003, 365)
volatile_ust = np.concatenate([
    1.0 + rng.normal(0, 0.002, 300),
    np.linspace(1.0, 0.15, 65),
])

coins = {
    "USDC (stable)": stable_usdc,
    "DAI (moderate)": moderate_dai,
    "UST-like (collapse)": volatile_ust,
}

fig, ax = plt.subplots(figsize=(8, 4.5))
_style_ax(ax, "Depeg Probability by Stablecoin Profile", ylabel="Empirical depeg probability")

probs = []
labels = []
colours = []
for name, series in coins.items():
    result = depeg_probability(series, threshold=0.005)
    probs.append(result["empirical_prob"])
    labels.append(name)

colours = [POSITIVE, SECONDARY, NEGATIVE]
bars = ax.barh(labels, probs, color=colours, height=0.5, edgecolor="none")

for bar, prob in zip(bars, probs):
    ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
            f"{prob:.1%}", va="center", fontsize=10, fontweight="bold")

ax.set_xlim(0, max(probs) * 1.3)
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "stablecoin_depeg_probability.png"), dpi=150, facecolor=BG)
plt.close()
print("Saved stablecoin_depeg_probability.png")


# --- Chart 2: Reserve composition comparison ---
reserves = {
    "Acme Fund\n(best practice)": {"cash": 55, "treasuries": 35, "money_market": 10},
    "Exchange B\n(moderate)": {"cash": 20, "treasuries": 30, "commercial_paper": 30, "crypto": 20},
    "Exchange C\n(risky)": {"crypto": 50, "other": 30, "secured_loans": 20},
}

fig, ax = plt.subplots(figsize=(8, 4.5))
_style_ax(ax, "Reserve Quality Score Comparison", ylabel="Reserve quality score")

names = list(reserves.keys())
scores = []
ratings = []
for name, comp in reserves.items():
    result = reserve_risk_score(comp)
    scores.append(result["score"])
    ratings.append(result["rating"])

colour_map = {"excellent": POSITIVE, "good": PRIMARY, "fair": SECONDARY, "poor": NEGATIVE, "critical": NEGATIVE}
bar_colours = [colour_map.get(r, NEUTRAL) for r in ratings]

bars = ax.bar(names, scores, color=bar_colours, width=0.5, edgecolor="none")
for bar, score, rating in zip(bars, scores, ratings):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
            f"{score:.2f}\n({rating})", ha="center", fontsize=9, fontweight="bold")

ax.set_ylim(0, 1.15)
ax.axhline(y=0.70, color=NEUTRAL, linestyle="--", alpha=0.5, linewidth=1)
ax.text(2.35, 0.71, "Good threshold", fontsize=8, color=NEUTRAL)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "stablecoin_reserve_comparison.png"), dpi=150, facecolor=BG)
plt.close()
print("Saved stablecoin_reserve_comparison.png")


# --- Chart 3: Peg deviation tracker ---
# Simulate USDC-like price with a brief depeg event
prices = np.ones(365)
prices += rng.normal(0, 0.0005, 365)
# Simulate March 2023 SVB-style depeg at day 70
prices[70:73] = [0.95, 0.88, 0.92]
prices[73:76] = [0.97, 0.995, 1.001]

result = peg_deviation_tracker(prices)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), height_ratios=[2, 1], sharex=True)

_style_ax(ax1, "Acme USDC Holdings: Peg Deviation Tracker", ylabel="Price (USD)")
ax1.plot(prices, color=PRIMARY, linewidth=1.2, label="USDC price")
ax1.axhline(y=1.0, color=NEUTRAL, linestyle="--", alpha=0.5, linewidth=1)
ax1.fill_between(range(len(prices)), 0.995, 1.005, alpha=0.1, color=POSITIVE, label="Normal band")

# Highlight excursions
for exc in result["excursions"]:
    if exc["max_deviation"] > 0.005:
        ax1.axvspan(exc["start"], exc["end"], alpha=0.15, color=NEGATIVE)
        ax1.annotate(
            f"Depeg: {exc['max_deviation']:.1%}",
            xy=(exc["start"], prices[exc["start"]]),
            xytext=(exc["start"] + 20, prices[exc["start"]] - 0.03),
            fontsize=8, fontweight="bold", color=NEGATIVE,
            arrowprops={"arrowstyle": "->", "color": NEGATIVE, "lw": 1},
        )

ax1.legend(loc="lower right", fontsize=8, framealpha=0.9)
ax1.set_ylim(0.85, 1.05)

_style_ax(ax2, "", xlabel="Day", ylabel="Abs deviation")
ax2.fill_between(range(len(result["abs_deviations"])), result["abs_deviations"],
                 color=SECONDARY, alpha=0.6)
ax2.axhline(y=0.005, color=NEGATIVE, linestyle="--", alpha=0.7, linewidth=1)
ax2.text(len(prices) - 5, 0.006, "Threshold", fontsize=8, color=NEGATIVE, ha="right")

plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "stablecoin_deviation_tracker.png"), dpi=150, facecolor=BG)
plt.close()
print("Saved stablecoin_deviation_tracker.png")

print("\nAll stablecoin charts generated successfully.")
