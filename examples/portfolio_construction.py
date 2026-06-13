#!/usr/bin/env python3
"""Portfolio construction: efficient frontier, HRP, weight comparison, monthly heatmap.

Generates four charts saved to docs/images/.
"""
from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from quantlite.data_generation import correlated_gbm
from quantlite.dependency.clustering import hierarchical_cluster
from quantlite.portfolio.optimisation import (
    hrp_weights,
    max_sharpe_weights,
    mean_variance_weights,
    min_variance_weights,
    risk_parity_weights,
)
from quantlite.viz.theme import FEW_PALETTE, apply_few_theme

OUT = os.path.join(os.path.dirname(__file__), "..", "docs", "images")
os.makedirs(OUT, exist_ok=True)
DPI = 150

# --- Generate multi-asset returns ---
asset_names = ["US Equity", "EU Equity", "EM Equity", "Govt Bond", "Corp Bond",
               "Gold", "Real Estate", "Commodities"]
n_assets = len(asset_names)
mus = [0.08, 0.06, 0.10, 0.03, 0.04, 0.05, 0.07, 0.04]
vols = [0.16, 0.18, 0.25, 0.05, 0.07, 0.15, 0.14, 0.20]

# Correlation structure
corr = np.eye(n_assets)
# Equity block: high correlation
for i in range(3):
    for j in range(3):
        if i != j:
            corr[i, j] = 0.65
# Bond block
corr[3, 4] = corr[4, 3] = 0.5
# Negative equity-bond
for i in range(3):
    corr[i, 3] = corr[3, i] = -0.2
    corr[i, 4] = corr[4, i] = -0.1
# Gold diversifier
for i in range(3):
    corr[i, 5] = corr[5, i] = 0.05
corr[5, 3] = corr[3, 5] = 0.15

vols_arr = np.array(vols)
cov = np.outer(vols_arr, vols_arr) * corr

prices_df = correlated_gbm(
    S0_list=[100.0] * n_assets,
    mu_list=mus,
    cov_matrix=cov,
    steps=252 * 5,
    rng_seed=42,
    return_as="dataframe",
)
prices_df.columns = asset_names
returns_df = prices_df.pct_change().dropna()

apply_few_theme()

# ============================================================
# 1. Efficient frontier
# ============================================================
fig, ax = plt.subplots(figsize=(8, 5.5))

rng = np.random.default_rng(42)
mu_vec = returns_df.mean().values
cov_mat = returns_df.cov().values
freq = 252

# Random portfolios
rand_rets, rand_vols = [], []
for _ in range(3000):
    w = rng.random(n_assets)
    w /= w.sum()
    ret = float((1 + w @ mu_vec) ** freq - 1)
    vol = float(np.sqrt(w @ cov_mat @ w) * np.sqrt(freq))
    rand_rets.append(ret)
    rand_vols.append(vol)

# Colour by Sharpe
sharpes = [(r / v if v > 0 else 0) for r, v in zip(rand_rets, rand_vols)]
sc = ax.scatter(rand_vols, rand_rets, c=sharpes, cmap="Blues", s=6, alpha=0.5, edgecolors="none")

# Individual assets
for i, name in enumerate(asset_names):
    ann_ret = float((1 + mu_vec[i]) ** freq - 1)
    ann_vol = float(np.sqrt(cov_mat[i, i]) * np.sqrt(freq))
    ax.scatter(ann_vol, ann_ret, s=60, color=FEW_PALETTE["secondary"],
               zorder=5, edgecolors="white", linewidth=0.8)
    ax.annotate(name, (ann_vol + 0.003, ann_ret), fontsize=7, color=FEW_PALETTE["grey_dark"])

# Min variance
mv = min_variance_weights(returns_df, freq=freq)
ax.scatter(mv.expected_risk, mv.expected_return, s=100, color=FEW_PALETTE["positive"],
           zorder=6, marker="o", edgecolors="white", linewidth=1.5)
ax.annotate("Min Variance", (mv.expected_risk + 0.005, mv.expected_return),
            fontsize=9, color=FEW_PALETTE["positive"], fontweight="bold")

# Max Sharpe
ms = max_sharpe_weights(returns_df, freq=freq)
ax.scatter(ms.expected_risk, ms.expected_return, s=100, color=FEW_PALETTE["negative"],
           zorder=6, marker="D", edgecolors="white", linewidth=1.5)
ax.annotate(f"Max Sharpe ({ms.sharpe:.2f})", (ms.expected_risk + 0.005, ms.expected_return),
            fontsize=9, color=FEW_PALETTE["negative"], fontweight="bold")

ax.set_xlabel("Annualised volatility")
ax.set_ylabel("Annualised return")
ax.set_title("Efficient Frontier with Individual Assets")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "efficient_frontier.png"), dpi=DPI, bbox_inches="tight")
plt.close()
print("  Saved efficient_frontier.png")

# ============================================================
# 2. HRP dendrogram + weight allocation
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), gridspec_kw={"width_ratios": [1.2, 1]})

# Dendrogram
corr_df = returns_df.corr()
linkage_mat = hierarchical_cluster(corr_df, method="single")
dendrogram(linkage_mat, labels=asset_names, ax=axes[0],
           leaf_rotation=45, leaf_font_size=9,
           color_threshold=0, above_threshold_color=FEW_PALETTE["primary"])
axes[0].set_title("Hierarchical Clustering (Correlation Distance)")
axes[0].set_ylabel("Distance")

# HRP weights
hrp = hrp_weights(returns_df)
names_sorted = sorted(hrp.weights.keys(), key=lambda k: hrp.weights[k], reverse=True)
weights_sorted = [hrp.weights[n] for n in names_sorted]

axes[1].barh(range(len(names_sorted)), weights_sorted, color=FEW_PALETTE["primary"], height=0.6)
axes[1].set_yticks(range(len(names_sorted)))
axes[1].set_yticklabels(names_sorted, fontsize=9)
axes[1].set_xlabel("Weight")
axes[1].set_title("HRP Weight Allocation")
for i, w in enumerate(weights_sorted):
    axes[1].text(w + 0.005, i, f"{w:.1%}", va="center", fontsize=8, color=FEW_PALETTE["grey_dark"])

fig.tight_layout()
fig.savefig(os.path.join(OUT, "hrp_dendrogram.png"), dpi=DPI, bbox_inches="tight")
plt.close()
print("  Saved hrp_dendrogram.png")

# ============================================================
# 3. Weight comparison: MV vs Risk Parity vs HRP vs Max Sharpe
# ============================================================
fig, ax = plt.subplots(figsize=(10, 5))

mv_w = mean_variance_weights(returns_df, freq=freq)
rp_w = risk_parity_weights(returns_df, freq=freq)
ms_w = max_sharpe_weights(returns_df, freq=freq)

methods_data = [
    ("Mean-Variance", mv_w.weights),
    ("Risk Parity", rp_w.weights),
    ("HRP", hrp.weights),
    ("Max Sharpe", ms_w.weights),
]

x = np.arange(n_assets)
width = 0.2
colours = [FEW_PALETTE["primary"], FEW_PALETTE["secondary"],
           FEW_PALETTE["positive"], FEW_PALETTE["negative"]]

for i, (method_name, wts) in enumerate(methods_data):
    vals = [wts.get(name, 0) for name in asset_names]
    ax.bar(x + i * width, vals, width * 0.9, color=colours[i], label=method_name, alpha=0.85)

ax.set_xticks(x + 1.5 * width)
ax.set_xticklabels(asset_names, rotation=30, ha="right", fontsize=9)
ax.set_ylabel("Weight")
ax.set_title("Portfolio Weight Comparison Across Methods")
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "weight_comparison.png"), dpi=DPI, bbox_inches="tight")
plt.close()
print("  Saved weight_comparison.png")

# ============================================================
# 4. Monthly returns heatmap
# ============================================================
# Use equal-weight portfolio
eq_returns = returns_df.mean(axis=1)
# Create monthly aggregation
dates = pd.date_range("2020-01-02", periods=len(eq_returns), freq="B")
eq_series = pd.Series(eq_returns.values, index=dates)
monthly = eq_series.resample("M").apply(lambda x: (1 + x).prod() - 1)

# Pivot to year x month
monthly_df = pd.DataFrame({"year": monthly.index.year, "month": monthly.index.month, "ret": monthly.values})
pivot = monthly_df.pivot(index="year", columns="month", values="ret")
pivot.columns = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                 "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

fig, ax = plt.subplots(figsize=(10, 3.5))

data = pivot.values
vmax = max(abs(np.nanmin(data)), abs(np.nanmax(data)))
cmap = plt.cm.RdBu_r.copy()
cmap.set_bad(color=FEW_PALETTE["grey_light"])
masked = np.ma.masked_invalid(data)

ax.imshow(masked, cmap=cmap, aspect="auto", vmin=-vmax, vmax=vmax)

for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        if not np.isnan(data[i, j]):
            colour = "white" if abs(data[i, j]) > vmax * 0.5 else FEW_PALETTE["grey_dark"]
            ax.text(j, i, f"{data[i, j]:.1%}", ha="center", va="center", fontsize=8, color=colour)

ax.set_xticks(range(12))
ax.set_xticklabels(pivot.columns, fontsize=9)
ax.set_yticks(range(len(pivot)))
ax.set_yticklabels(pivot.index, fontsize=9)
ax.set_title("Monthly Returns Heatmap (Equal-Weight Portfolio)")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "monthly_returns_heatmap.png"), dpi=DPI, bbox_inches="tight")
plt.close()
print("  Saved monthly_returns_heatmap.png")

print("Done: portfolio_construction.py")
