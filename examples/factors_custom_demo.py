"""Custom factor tools demo.

Generates charts for factor decay analysis, factor-sorted portfolio
returns, and factor correlation matrix.
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from quantlite.factors.custom import (
    CustomFactor,
    factor_correlation_matrix,
    factor_decay,
    factor_portfolio,
)

PRIMARY = "#4E79A7"
SECONDARY = "#F28E2B"
NEGATIVE = "#E15759"
POSITIVE = "#59A14F"
NEUTRAL = "#76B7B2"

IMG_DIR = os.path.join(os.path.dirname(__file__), "..", "docs", "images")
os.makedirs(IMG_DIR, exist_ok=True)

rng = np.random.RandomState(42)

# --- Chart 1: Factor decay ---
n = 500
momentum = rng.normal(0, 1, n)
# Returns with decaying predictability
returns = np.zeros(n)
for lag in range(1, n):
    returns[lag] = 0.3 * np.exp(-0.15 * 1) * momentum[lag - 1] + rng.normal(0, 0.8)

decay = factor_decay(returns, momentum, max_lag=15)

fig, ax = plt.subplots(figsize=(8, 4.5))
lags = [d[0] for d in decay["decay_curve"]]
corrs = [d[1] for d in decay["decay_curve"]]

ax.bar(lags, corrs, color=[PRIMARY if c >= 0 else NEGATIVE for c in corrs],
       edgecolor="none", width=0.7)
for lag_val, corr_val in zip(lags, corrs):
    ax.text(lag_val, corr_val + (0.005 if corr_val >= 0 else -0.01),
            f"{corr_val:.3f}", ha="center", va="bottom" if corr_val >= 0 else "top",
            fontsize=8)

if decay["half_life"] is not None:
    ax.axvline(decay["half_life"], color=SECONDARY, linewidth=1.5, linestyle="--",
               label=f"Half-life = {decay['half_life']:.1f} lags")
    ax.legend(frameon=False)

ax.set_xlabel("Lag (periods)")
ax.set_ylabel("Correlation")
ax.set_title("Factor Predictive Power Decay (Momentum Signal)", fontsize=11)
ax.axhline(0, color="#999999", linewidth=0.5)
ax.set_facecolor("white")
fig.patch.set_facecolor("white")
ax.grid(axis="y", alpha=0.3)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "factor_decay.png"), dpi=150, facecolor="white")
plt.close()
print("Saved factor_decay.png")

# --- Chart 2: Factor-sorted quintile returns ---
n_assets = 100
n_periods = 252
returns_df = pd.DataFrame(
    rng.normal(0.0003, 0.015, (n_periods, n_assets)),
    columns=[f"Stock_{i}" for i in range(n_assets)],
)
# Factor that predicts returns
factor_values = rng.normal(0, 1, n_assets)
# Add factor signal to returns
for i in range(n_assets):
    returns_df.iloc[:, i] += factor_values[i] * 0.0005

result = factor_portfolio(returns_df, factor_values, n_quantiles=5)

fig, ax = plt.subplots(figsize=(7, 5))
quintiles = list(result["quantile_returns"].keys())
rets = [result["quantile_returns"][q] * 252 * 100 for q in quintiles]  # Annualised %
colours = [NEGATIVE, SECONDARY, NEUTRAL, PRIMARY, POSITIVE]
bars = ax.bar(
    [f"Q{q}" for q in quintiles], rets,
    color=colours, edgecolor="none", width=0.6,
)
for bar, r in zip(bars, rets):
    ax.text(bar.get_x() + bar.get_width() / 2, r + 0.3,
            f"{r:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

ax.axhline(0, color="#999999", linewidth=0.5)
ax.set_ylabel("Annualised Return (%)")
ax.set_xlabel("Factor Quintile (Q1 = lowest, Q5 = highest)")
ax.set_title(f"Factor-Sorted Portfolio Returns\n"
             f"Long-short spread: {result['spread'] * 252 * 100:.1f}% p.a.",
             fontsize=11)
ax.set_facecolor("white")
fig.patch.set_facecolor("white")
ax.grid(axis="y", alpha=0.3)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "factor_quintiles.png"), dpi=150, facecolor="white")
plt.close()
print("Saved factor_quintiles.png")

# --- Chart 3: Factor correlation matrix ---
momentum_f = CustomFactor("Momentum", rng.normal(0, 0.01, 252))
value_f = CustomFactor("Value", rng.normal(0, 0.008, 252))
size_f = CustomFactor("Size", rng.normal(0, 0.006, 252))
quality_f = CustomFactor("Quality", 0.3 * value_f.values + rng.normal(0, 0.007, 252))

corr_result = factor_correlation_matrix([momentum_f, value_f, size_f, quality_f])

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(corr_result["matrix"], cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
names = corr_result["names"]
ax.set_xticks(range(len(names)))
ax.set_yticks(range(len(names)))
ax.set_xticklabels(names, rotation=45, ha="right")
ax.set_yticklabels(names)

# Annotate cells
for i in range(len(names)):
    for j in range(len(names)):
        val = corr_result["matrix"][i, j]
        colour = "white" if abs(val) > 0.5 else "black"
        ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                fontsize=10, color=colour, fontweight="bold")

ax.set_title("Factor Correlation Matrix", fontsize=11)
fig.colorbar(im, ax=ax, shrink=0.8, label="Correlation")
fig.patch.set_facecolor("white")
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "factor_correlation.png"), dpi=150, facecolor="white")
plt.close()
print("Saved factor_correlation.png")
