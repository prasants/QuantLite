#!/usr/bin/env python3
"""Regime detection: HMM overlay, transition matrix, conditional distributions, changepoints.

Generates four charts saved to docs/images/.
"""
from __future__ import annotations

import os, sys
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from quantlite.viz.theme import apply_few_theme, FEW_PALETTE
from quantlite.regimes.hmm import fit_regime_model
from quantlite.regimes.changepoint import detect_changepoints
from quantlite.data_generation import merton_jump_diffusion, geometric_brownian_motion

OUT = os.path.join(os.path.dirname(__file__), "..", "docs", "images")
os.makedirs(OUT, exist_ok=True)
DPI = 150

# --- Multi-regime synthetic data ---
# Bull market
bull = geometric_brownian_motion(S0=100, mu=0.12, sigma=0.12, steps=400, rng_seed=10)
# Bear / crisis
crisis = merton_jump_diffusion(S0=bull[-1], mu=-0.15, sigma=0.40, lamb=2.0,
                                jump_mean=-0.04, jump_std=0.06, steps=200, rng_seed=20)
# Recovery
recover = geometric_brownian_motion(S0=crisis[-1], mu=0.10, sigma=0.15, steps=300, rng_seed=30)
# Second crisis
crisis2 = merton_jump_diffusion(S0=recover[-1], mu=-0.08, sigma=0.35, lamb=1.5,
                                 jump_mean=-0.03, jump_std=0.05, steps=150, rng_seed=40)
# Final bull
bull2 = geometric_brownian_motion(S0=crisis2[-1], mu=0.15, sigma=0.10, steps=450, rng_seed=50)

prices = np.concatenate([bull, crisis[1:], recover[1:], crisis2[1:], bull2[1:]])
returns = np.diff(prices) / prices[:-1]

apply_few_theme()

REGIME_COLOURS = [FEW_PALETTE["negative"], FEW_PALETTE["neutral"], FEW_PALETTE["positive"]]

# Fit HMM
model = fit_regime_model(returns, n_regimes=3, rng_seed=42)
labels = model.regime_labels

# ============================================================
# 1. Price series with regime overlay
# ============================================================
fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(prices[1:], color=FEW_PALETTE["grey_dark"], linewidth=1, zorder=3)

# Shade regimes
for i in range(len(labels)):
    colour = REGIME_COLOURS[labels[i] % 3]
    ax.axvspan(i - 0.5, i + 0.5, color=colour, alpha=0.12, linewidth=0)

# Legend
for r in sorted(np.unique(labels)):
    vol = np.std(returns[labels == r]) * np.sqrt(252)
    ax.plot([], [], color=REGIME_COLOURS[r % 3], linewidth=8, alpha=0.4,
            label=f"Regime {r} (ann. vol: {vol:.1%})")
ax.legend(fontsize=9, loc="upper left")

ax.set_xlabel("Trading day")
ax.set_ylabel("Price")
ax.set_title("HMM Regime Detection: Bull, Transition, and Crisis Regimes")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "regime_timeline.png"), dpi=DPI)
plt.close()
print("  Saved regime_timeline.png")

# ============================================================
# 2. Transition matrix heatmap
# ============================================================
from matplotlib.colors import LinearSegmentedColormap

fig, ax = plt.subplots(figsize=(5, 4))

trans = model.transition_matrix
n_r = model.n_regimes

cmap = LinearSegmentedColormap.from_list("few_seq", ["#FFFFFF", FEW_PALETTE["primary"]])
ax.imshow(trans, cmap=cmap, vmin=0, vmax=1, aspect="auto")

for i in range(n_r):
    for j in range(n_r):
        val = trans[i, j]
        colour = "white" if val > 0.6 else FEW_PALETTE["grey_dark"]
        ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=12, color=colour)

rlabels = [f"R{i}" for i in range(n_r)]
ax.set_xticks(range(n_r))
ax.set_yticks(range(n_r))
ax.set_xticklabels(rlabels)
ax.set_yticklabels(rlabels)
ax.set_xlabel("To regime")
ax.set_ylabel("From regime")
ax.set_title("Regime Transition Probabilities")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "transition_matrix.png"), dpi=DPI)
plt.close()
print("  Saved transition_matrix.png")

# ============================================================
# 3. Regime-conditional return distributions (overlaid)
# ============================================================
fig, ax = plt.subplots(figsize=(8, 5))

bins = np.linspace(returns.min(), returns.max(), 80)
for r in sorted(np.unique(labels)):
    r_data = returns[labels == r]
    colour = REGIME_COLOURS[r % 3]
    mu = np.mean(r_data)
    vol = np.std(r_data, ddof=1)
    ax.hist(r_data, bins=bins, density=True, alpha=0.45, color=colour,
            edgecolor="white", linewidth=0.3,
            label=f"Regime {r} (mu={mu:.4f}, vol={vol:.4f})")

ax.set_xlabel("Daily return")
ax.set_ylabel("Density")
ax.set_title("Return Distributions by Regime")
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "regime_distributions.png"), dpi=DPI)
plt.close()
print("  Saved regime_distributions.png")

# ============================================================
# 4. Bayesian changepoint detection
# ============================================================
fig, axes = plt.subplots(2, 1, figsize=(12, 6), gridspec_kw={"height_ratios": [3, 1]}, sharex=True)

# Price chart
axes[0].plot(prices[1:], color=FEW_PALETTE["grey_dark"], linewidth=1)

# Detect changepoints
cps = detect_changepoints(returns, method="bayesian", penalty=80)

for cp in cps:
    axes[0].axvline(cp.index, color=FEW_PALETTE["secondary"], linewidth=1.2,
                     linestyle="--", alpha=0.8)
    axes[1].bar(cp.index, cp.confidence, width=3, color=FEW_PALETTE["secondary"], alpha=0.8)

axes[0].set_ylabel("Price")
axes[0].set_title("Bayesian Changepoint Detection")

axes[1].set_ylabel("Confidence")
axes[1].set_xlabel("Trading day")
axes[1].set_ylim(0, 1.1)

# Annotate structural breaks
for cp in cps[:8]:  # show top 8
    axes[0].annotate(f"{cp.direction.replace('_', ' ')}", xy=(cp.index, prices[cp.index + 1]),
                      fontsize=7, color=FEW_PALETTE["secondary"], rotation=45,
                      ha="left", va="bottom")

fig.tight_layout()
fig.savefig(os.path.join(OUT, "changepoint_detection.png"), dpi=DPI)
plt.close()
print("  Saved changepoint_detection.png")

print("Done: regime_detection.py")
