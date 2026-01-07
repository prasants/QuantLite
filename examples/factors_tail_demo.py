"""Tail risk factor analysis demo.

Generates charts for CVaR decomposition, regime factor exposures,
and factor crowding scores.
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from quantlite.factors.tail_risk import (
    factor_crowding_score,
    factor_cvar_decomposition,
    regime_factor_exposure,
)

PRIMARY = "#4E79A7"
SECONDARY = "#F28E2B"
NEGATIVE = "#E15759"
POSITIVE = "#59A14F"
NEUTRAL = "#76B7B2"

IMG_DIR = os.path.join(os.path.dirname(__file__), "..", "docs", "images")
os.makedirs(IMG_DIR, exist_ok=True)

rng = np.random.RandomState(42)
n = 1000

market = rng.normal(0.0004, 0.012, n)
value = rng.normal(0.0001, 0.007, n)
momentum = rng.normal(0.0002, 0.008, n)

# Fund returns with asymmetric factor exposure (higher in tails)
noise = rng.normal(0, 0.004, n)
returns = 0.0001 + 1.0 * market + 0.4 * value + 0.3 * momentum + noise
# Add tail amplification
tail_mask = market < np.percentile(market, 5)
returns[tail_mask] += 0.5 * market[tail_mask]  # Extra market exposure in tails

# --- Chart 1: CVaR decomposition ---
cvar = factor_cvar_decomposition(
    returns, [market, value, momentum], ["Market", "Value", "Momentum"]
)

fig, ax = plt.subplots(figsize=(7, 5))
labels = list(cvar["factor_contributions"].keys()) + ["Residual"]
values_plot = list(cvar["factor_contributions"].values()) + [cvar["residual_contribution"]]
colours = [PRIMARY, SECONDARY, NEUTRAL, NEGATIVE]

bars = ax.barh(labels, [v * 100 for v in values_plot], color=colours,
               edgecolor="none", height=0.5)

for bar, v in zip(bars, values_plot):
    pct = abs(v / cvar["total_cvar"]) * 100 if abs(cvar["total_cvar"]) > 1e-12 else 0
    w = bar.get_width()
    ax.text(w + 0.01 if w >= 0 else w - 0.01, bar.get_y() + bar.get_height() / 2,
            f"{v * 100:.3f}% ({pct:.0f}%)",
            va="center", ha="left" if w >= 0 else "right",
            fontsize=9, fontweight="bold")

ax.axvline(0, color="#999999", linewidth=0.5)
ax.set_xlabel("CVaR Contribution (%)")
ax.set_title(f"Factor CVaR Decomposition (5% tail)\n"
             f"Total CVaR = {cvar['total_cvar'] * 100:.3f}%", fontsize=11)
ax.set_facecolor("white")
fig.patch.set_facecolor("white")
ax.grid(axis="x", alpha=0.3)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "factor_cvar_decomp.png"), dpi=150, facecolor="white")
plt.close()
print("Saved factor_cvar_decomp.png")

# --- Chart 2: Regime factor betas ---
regimes = np.array(["bull"] * 400 + ["bear"] * 350 + ["crisis"] * 250)
# Adjust returns for regime effect
returns_regime = returns.copy()
returns_regime[400:750] -= 0.001  # Bear drag
returns_regime[750:] -= 0.003    # Crisis drag

regime_result = regime_factor_exposure(
    returns_regime, [market, value, momentum],
    ["Market", "Value", "Momentum"], regimes,
)

fig, ax = plt.subplots(figsize=(8, 5))
factor_names = ["Market", "Value", "Momentum"]
regime_names = ["bull", "bear", "crisis"]
regime_labels = ["Bull", "Bear", "Crisis"]
regime_colours = [POSITIVE, SECONDARY, NEGATIVE]

x = np.arange(len(factor_names))
width = 0.25

for i, (regime, label, colour) in enumerate(
    zip(regime_names, regime_labels, regime_colours)
):
    betas = [regime_result[regime]["betas"][f] for f in factor_names]
    bars = ax.bar(x + i * width, betas, width, label=label, color=colour,
                  edgecolor="none")
    for bar, b in zip(bars, betas):
        ax.text(bar.get_x() + bar.get_width() / 2, b + 0.02,
                f"{b:.2f}", ha="center", va="bottom", fontsize=8)

ax.set_xticks(x + width)
ax.set_xticklabels(factor_names)
ax.set_ylabel("Factor Beta")
ax.set_title("Factor Betas Across Market Regimes (Strategy Alpha)", fontsize=11)
ax.legend(frameon=False)
ax.axhline(0, color="#999999", linewidth=0.5)
ax.set_facecolor("white")
fig.patch.set_facecolor("white")
ax.grid(axis="y", alpha=0.3)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "factor_regime_betas.png"), dpi=150, facecolor="white")
plt.close()
print("Saved factor_regime_betas.png")

# --- Chart 3: Factor crowding score ---
# Simulate factors becoming more correlated over time
n_crowd = 500
base = rng.normal(0, 0.01, n_crowd)
factor_a = base + rng.normal(0, 0.01, n_crowd)
# Factor B becomes increasingly correlated with A
correlation_ramp = np.linspace(0.1, 0.9, n_crowd)
factor_b = np.array([
    correlation_ramp[i] * base[i] + (1 - correlation_ramp[i]) * rng.normal(0, 0.01)
    for i in range(n_crowd)
])

crowd = factor_crowding_score([factor_a, factor_b], rolling_window=60)

fig, ax = plt.subplots(figsize=(9, 4.5))
ax.plot(range(60, n_crowd + 1), crowd["crowding_scores"], color=PRIMARY, linewidth=1.5)
ax.axhline(0.7, color=NEGATIVE, linewidth=1, linestyle="--", label="Crowding threshold (0.7)")
ax.fill_between(
    range(60, n_crowd + 1), crowd["crowding_scores"],
    where=[s > 0.7 for s in crowd["crowding_scores"]],
    color=NEGATIVE, alpha=0.2,
)

ax.set_xlabel("Trading Day")
ax.set_ylabel("Crowding Score")
ax.set_title(f"Factor Crowding Score (Rolling 60-Day)\n"
             f"Current = {crowd['current_score']:.3f}, "
             f"Trend = {crowd['trend']:.5f}/day",
             fontsize=11)
ax.legend(frameon=False)
ax.set_ylim(0, 1)
ax.set_facecolor("white")
fig.patch.set_facecolor("white")
ax.grid(axis="y", alpha=0.3)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "factor_crowding.png"), dpi=150, facecolor="white")
plt.close()
print("Saved factor_crowding.png")
