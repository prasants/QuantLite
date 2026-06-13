#!/usr/bin/env python3
"""Extreme Value Theory analysis: GPD tail fit, QQ plot, return levels, Hill plot.

Generates four charts saved to docs/images/.
"""
from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from quantlite.data_generation import merton_jump_diffusion
from quantlite.risk.evt import fit_gpd, hill_estimator, return_level
from quantlite.viz.theme import FEW_PALETTE, apply_few_theme

OUT = os.path.join(os.path.dirname(__file__), "..", "docs", "images")
os.makedirs(OUT, exist_ok=True)
DPI = 150

# --- Generate fat-tailed returns ---
calm = merton_jump_diffusion(S0=100, mu=0.06, sigma=0.15, lamb=0.3,
                              jump_mean=-0.02, jump_std=0.04, steps=252*7, rng_seed=42)
crisis = merton_jump_diffusion(S0=calm[-1], mu=-0.10, sigma=0.45, lamb=2.0,
                                jump_mean=-0.05, jump_std=0.08, steps=252, rng_seed=99)
recovery = merton_jump_diffusion(S0=crisis[-1], mu=0.10, sigma=0.18, lamb=0.4,
                                  jump_mean=-0.01, jump_std=0.03, steps=252*2, rng_seed=7)
prices = np.concatenate([calm, crisis[1:], recovery[1:]])
returns = np.diff(prices) / prices[:-1]
losses = -returns

apply_few_theme()

# ============================================================
# 1. GPD tail fit vs empirical
# ============================================================
gpd = fit_gpd(returns)
threshold = gpd.threshold
exceedances = losses[losses > threshold] - threshold

fig, ax = plt.subplots(figsize=(8, 5))

# Empirical tail
sorted_exc = np.sort(exceedances)
n_exc = len(sorted_exc)
empirical_sf = 1 - np.arange(1, n_exc + 1) / (n_exc + 1)

ax.semilogy(sorted_exc, empirical_sf, "o", color=FEW_PALETTE["grey_mid"],
            markersize=3, alpha=0.6, label="Empirical")

# GPD fit
x_fit = np.linspace(0, sorted_exc.max(), 200)
gpd_sf = stats.genpareto.sf(x_fit, gpd.shape, scale=gpd.scale)
ax.semilogy(x_fit, gpd_sf, color=FEW_PALETTE["primary"], linewidth=2,
            label=f"GPD (xi={gpd.shape:.3f}, sigma={gpd.scale:.4f})")

ax.set_xlabel("Excess loss above threshold")
ax.set_ylabel("Survival probability")
ax.set_title(f"GPD Tail Fit (threshold = {threshold:.4f})")
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "gpd_tail_fit.png"), dpi=DPI, bbox_inches="tight")
plt.close()
print("  Saved gpd_tail_fit.png")

# ============================================================
# 2. QQ plot: empirical vs normal and Student-t
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

# vs Normal
ax = axes[0]
sorted_r = np.sort(returns)
n = len(sorted_r)
theoretical_q = stats.norm.ppf(np.linspace(0.001, 0.999, n))
ax.scatter(theoretical_q, sorted_r, s=2, alpha=0.4, color=FEW_PALETTE["grey_mid"])
lims = [min(theoretical_q.min(), sorted_r.min()), max(theoretical_q.max(), sorted_r.max())]
ax.plot(lims, lims, "--", color=FEW_PALETTE["negative"], linewidth=1)
ax.set_xlabel("Normal theoretical quantiles")
ax.set_ylabel("Sample quantiles")
ax.set_title("QQ Plot vs Normal")

# vs Student-t
ax = axes[1]
df_t, loc_t, scale_t = stats.t.fit(returns)
theoretical_t = stats.t.ppf(np.linspace(0.001, 0.999, n), df_t, loc_t, scale_t)
ax.scatter(theoretical_t, sorted_r, s=2, alpha=0.4, color=FEW_PALETTE["grey_mid"])
lims_t = [min(theoretical_t.min(), sorted_r.min()), max(theoretical_t.max(), sorted_r.max())]
ax.plot(lims_t, lims_t, "--", color=FEW_PALETTE["positive"], linewidth=1)
ax.set_xlabel(f"Student-t (df={df_t:.1f}) theoretical quantiles")
ax.set_ylabel("Sample quantiles")
ax.set_title("QQ Plot vs Student-t")

fig.tight_layout()
fig.savefig(os.path.join(OUT, "qq_plots.png"), dpi=DPI, bbox_inches="tight")
plt.close()
print("  Saved qq_plots.png")

# ============================================================
# 3. Return level plot with confidence bands
# ============================================================
fig, ax = plt.subplots(figsize=(8, 5))

periods = np.logspace(1, np.log10(10000), 60)
levels = [return_level(gpd, rp) for rp in periods]

ax.plot(periods, levels, color=FEW_PALETTE["primary"], linewidth=2)

# Confidence bands (delta method approximation)
se = gpd.scale / np.sqrt(gpd.n_exceedances)
upper = [lv + 1.96 * se for lv in levels]
lower = [lv - 1.96 * se for lv in levels]
ax.fill_between(periods, lower, upper, alpha=0.15, color=FEW_PALETTE["primary"])

# Key return levels
for rp in [100, 1000, 5000]:
    rl = return_level(gpd, rp)
    ax.plot(rp, rl, "o", color=FEW_PALETTE["negative"], markersize=6, zorder=5)
    ax.annotate(f"1-in-{rp}: {rl:.3f}", xy=(rp, rl),
                xytext=(rp * 1.3, rl + 0.002), fontsize=9,
                color=FEW_PALETTE["negative"])

ax.set_xscale("log")
ax.set_xlabel("Return period (trading days)")
ax.set_ylabel("Estimated loss")
ax.set_title("Return Level Plot")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "return_level_plot.png"), dpi=DPI, bbox_inches="tight")
plt.close()
print("  Saved return_level_plot.png")

# ============================================================
# 4. Hill plot: tail index vs number of order statistics
# ============================================================
fig, ax = plt.subplots(figsize=(8, 5))

losses_sorted = np.sort(losses)[::-1]
k_range = range(10, min(500, len(losses_sorted) // 2))
alphas_hill = []
for k in k_range:
    try:
        h = hill_estimator(returns, k=k)
        alphas_hill.append(h.tail_index)
    except (ValueError, ZeroDivisionError):
        alphas_hill.append(np.nan)

ax.plot(list(k_range), alphas_hill, color=FEW_PALETTE["primary"], linewidth=1.2)

# Highlight the stable region
stable_start = 50
stable_end = 200
stable_alpha = np.nanmean(alphas_hill[stable_start-10:stable_end-10])
ax.axhspan(stable_alpha - 0.5, stable_alpha + 0.5, alpha=0.1, color=FEW_PALETTE["positive"])
ax.axhline(stable_alpha, color=FEW_PALETTE["secondary"], linewidth=1, linestyle="--")
ax.text(max(k_range) * 0.7, stable_alpha + 0.3, f"alpha ~ {stable_alpha:.1f}",
        fontsize=10, color=FEW_PALETTE["secondary"])

ax.set_xlabel("k (number of order statistics)")
ax.set_ylabel("Tail index (alpha)")
ax.set_title("Hill Plot: Tail Index Estimation")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "hill_plot.png"), dpi=DPI, bbox_inches="tight")
plt.close()
print("  Saved hill_plot.png")

print("Done: evt_analysis.py")
