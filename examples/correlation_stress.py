#!/usr/bin/env python3
"""Correlation stress analysis: rolling, EWMA, and calm vs crisis matrices.

Generates three charts saved to docs/images/.
"""
from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from quantlite.dependency.correlation import (
    exponential_weighted_correlation,
    rolling_correlation,
    stress_correlation,
)
from quantlite.viz.theme import FEW_PALETTE, apply_few_theme

OUT = os.path.join(os.path.dirname(__file__), "..", "docs", "images")
os.makedirs(OUT, exist_ok=True)
DPI = 150

# --- Two assets with regime-dependent correlation ---
rng = np.random.default_rng(42)

# Calm: low correlation
n_calm = 800
z1_calm = rng.normal(size=n_calm)
z2_calm = 0.2 * z1_calm + np.sqrt(1 - 0.04) * rng.normal(size=n_calm)
r1_calm = 0.0003 + 0.01 * z1_calm
r2_calm = 0.0002 + 0.012 * z2_calm

# Crisis: high correlation, higher vol
n_crisis = 200
z1_crisis = rng.normal(size=n_crisis)
z2_crisis = 0.85 * z1_crisis + np.sqrt(1 - 0.7225) * rng.normal(size=n_crisis)
r1_crisis = -0.001 + 0.035 * z1_crisis
r2_crisis = -0.0015 + 0.04 * z2_crisis

# Recovery: moderate
n_recov = 500
z1_recov = rng.normal(size=n_recov)
z2_recov = 0.4 * z1_recov + np.sqrt(1 - 0.16) * rng.normal(size=n_recov)
r1_recov = 0.0004 + 0.013 * z1_recov
r2_recov = 0.0003 + 0.015 * z2_recov

r1 = np.concatenate([r1_calm, r1_crisis, r1_recov])
r2 = np.concatenate([r2_calm, r2_crisis, r2_recov])

names = ["US Equity", "EU Equity"]
returns_df = pd.DataFrame({names[0]: r1, names[1]: r2})

apply_few_theme()

# ============================================================
# 1. Rolling correlation: calm vs crisis
# ============================================================
fig, ax = plt.subplots(figsize=(10, 4.5))

rolling = rolling_correlation(r1, r2, window=60)
ax.plot(rolling, color=FEW_PALETTE["primary"], linewidth=1.2, label="60-day rolling")

# Shade crisis period
ax.axvspan(n_calm, n_calm + n_crisis, alpha=0.15, color=FEW_PALETTE["negative"], label="Crisis period")

ax.axhline(0, color=FEW_PALETTE["grey_mid"], linewidth=0.8, linestyle="--")
overall = np.corrcoef(r1, r2)[0, 1]
ax.axhline(overall, color=FEW_PALETTE["secondary"], linewidth=1, linestyle=":")
ax.text(len(r1) * 0.02, overall + 0.04, f"Overall: {overall:.3f}",
        fontsize=9, color=FEW_PALETTE["secondary"])

ax.set_xlabel("Trading day")
ax.set_ylabel("Correlation")
ax.set_title("Rolling Correlation: Correlation Spikes During Crisis")
ax.set_ylim(-0.5, 1.05)
ax.legend(fontsize=9, loc="lower right")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "rolling_correlation.png"), dpi=DPI, bbox_inches="tight")
plt.close()
print("  Saved rolling_correlation.png")

# ============================================================
# 2. EWMA correlation vs rolling
# ============================================================
fig, ax = plt.subplots(figsize=(10, 4.5))

ewma = exponential_weighted_correlation(r1, r2, halflife=20)
ax.plot(rolling, color=FEW_PALETTE["grey_mid"], linewidth=1, alpha=0.7, label="Rolling (60d)")
ax.plot(ewma, color=FEW_PALETTE["primary"], linewidth=1.5, label="EWMA (halflife=20d)")

ax.axvspan(n_calm, n_calm + n_crisis, alpha=0.12, color=FEW_PALETTE["negative"])

ax.set_xlabel("Trading day")
ax.set_ylabel("Correlation")
ax.set_title("EWMA vs Rolling Correlation: Faster Reaction to Regime Changes")
ax.set_ylim(-0.5, 1.05)
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "ewma_correlation.png"), dpi=DPI, bbox_inches="tight")
plt.close()
print("  Saved ewma_correlation.png")

# ============================================================
# 3. Stress vs calm correlation matrix (side by side)
# ============================================================
# Build a 4-asset version for a richer matrix
asset_names_4 = ["US Equity", "EU Equity", "Govt Bond", "Gold"]
r3 = -0.15 * r1 + 0.01 * rng.normal(size=len(r1))  # bonds: negative to equity
r4 = 0.02 * r1 + 0.008 * rng.normal(size=len(r1))   # gold: low correlation
returns_4 = pd.DataFrame({"US Equity": r1, "EU Equity": r2, "Govt Bond": r3, "Gold": r4})

calm_corr = returns_4.iloc[:n_calm].corr()
stress_corr_mat = stress_correlation(returns_4, threshold_percentile=10)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

cmap = LinearSegmentedColormap.from_list("few_div",
    [FEW_PALETTE["primary"], "#FFFFFF", FEW_PALETTE["negative"]])

for ax, corr, title in [(axes[0], calm_corr, "Calm Period"),
                          (axes[1], stress_corr_mat, "Stress Period")]:
    n = len(corr)
    im = ax.imshow(corr.values, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
    for i in range(n):
        for j in range(n):
            val = corr.values[i, j]
            colour = "white" if abs(val) > 0.7 else FEW_PALETTE["grey_dark"]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=10, color=colour)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(corr.index, fontsize=9)
    ax.set_title(title)

fig.suptitle("Correlation Breakdown: Calm vs Stress", fontsize=12,
             color=FEW_PALETTE["grey_dark"])
fig.tight_layout()
fig.savefig(os.path.join(OUT, "stress_vs_calm_correlation.png"), dpi=DPI, bbox_inches="tight")
plt.close()
print("  Saved stress_vs_calm_correlation.png")

print("Done: correlation_stress.py")
