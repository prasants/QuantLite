"""Contagion demonstration: CoVaR, systemic risk contributions, and causal network.

Generates three charts for the v0.6 documentation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from quantlite.contagion import (
    covar,
    granger_causality,
    systemic_risk_contributions,
)
from quantlite.viz.theme import FEW_PALETTE, apply_few_theme, direct_label

apply_few_theme()

OUT = Path(__file__).resolve().parent.parent / "docs" / "images"
OUT.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(42)
n = 1000

# Correlated multi-asset returns with contagion structure
market = rng.normal(0.0003, 0.012, n)
financials = 0.7 * market + rng.normal(0.0001, 0.015, n)
tech = 0.5 * market + rng.normal(0.0004, 0.018, n)
energy = 0.3 * market + rng.normal(0.0002, 0.020, n)
gold = -0.15 * market + rng.normal(0.0001, 0.010, n)
bonds = -0.25 * market + rng.normal(0.0002, 0.005, n)

returns_df = pd.DataFrame({
    "Financials": financials,
    "Tech": tech,
    "Energy": energy,
    "Gold": gold,
    "Bonds": bonds,
})

# ---------------------------------------------------------------------------
# Chart 1: CoVaR comparison across assets
# ---------------------------------------------------------------------------
assets = list(returns_df.columns)
covar_vals = []
var_vals = []
delta_vals = []

for asset in assets:
    result = covar(market, returns_df[asset].values, alpha=0.05)
    covar_vals.append(result["covar"])
    var_vals.append(result["var_b"])
    delta_vals.append(result["delta_covar"])

fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(assets))
bar_width = 0.35

bars_var = ax.bar(
    x - bar_width / 2,
    [v * 100 for v in var_vals],
    bar_width,
    color=FEW_PALETTE["primary"],
)
bars_covar = ax.bar(
    x + bar_width / 2,
    [v * 100 for v in covar_vals],
    bar_width,
    color=FEW_PALETTE["negative"],
)

# Direct labels
for i, bar in enumerate(bars_var):
    direct_label(
        ax,
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() - 0.1,
        f"{var_vals[i] * 100:.2f}%",
        ha="center",
        va="top",
        fontsize=8,
        colour=FEW_PALETTE["primary"],
    )

for i, bar in enumerate(bars_covar):
    direct_label(
        ax,
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() - 0.1,
        f"{covar_vals[i] * 100:.2f}%",
        ha="center",
        va="top",
        fontsize=8,
        colour=FEW_PALETTE["negative"],
    )

# Series labels on rightmost bars
direct_label(
    ax,
    x[-1] - bar_width / 2 + 0.4,
    var_vals[-1] * 100,
    "Unconditional VaR",
    ha="left",
    va="center",
    fontsize=9,
    colour=FEW_PALETTE["primary"],
)
direct_label(
    ax,
    x[-1] + bar_width / 2 + 0.4,
    covar_vals[-1] * 100,
    "CoVaR (market stress)",
    ha="left",
    va="center",
    fontsize=9,
    colour=FEW_PALETTE["negative"],
)

ax.set_xticks(x)
ax.set_xticklabels(assets)
ax.set_ylabel("Loss (%)")
ax.set_title("CoVaR: How Market Stress Amplifies Individual Asset Risk")
ax.set_xlim(-0.5, len(assets) + 0.8)
fig.savefig(OUT / "covar_comparison.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved {OUT / 'covar_comparison.png'}")

# ---------------------------------------------------------------------------
# Chart 2: Systemic risk contributions (MES)
# ---------------------------------------------------------------------------
contributions = systemic_risk_contributions(returns_df, alpha=0.05)
names = list(contributions.keys())
mes_vals = [contributions[n] * 100 for n in names]

colours = [
    FEW_PALETTE["negative"] if v < -1.0 else FEW_PALETTE["secondary"] if v < 0 else FEW_PALETTE["positive"]
    for v in mes_vals
]

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.barh(range(len(names)), mes_vals, color=colours)

for i, _bar in enumerate(bars):
    val = mes_vals[i]
    ha = "right" if val < 0 else "left"
    offset = -0.05 if val < 0 else 0.05
    direct_label(
        ax,
        val + offset,
        i,
        f"{val:.2f}%",
        ha=ha,
        va="center",
        fontsize=9,
    )

ax.set_yticks(range(len(names)))
ax.set_yticklabels(names)
ax.set_xlabel("Marginal Expected Shortfall (%)")
ax.set_title("Systemic Risk Contributions: Who Hurts Most When Markets Crash?")
ax.axvline(x=0, color=FEW_PALETTE["grey_dark"], linewidth=0.8)
fig.savefig(OUT / "systemic_risk_contributions.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved {OUT / 'systemic_risk_contributions.png'}")

# ---------------------------------------------------------------------------
# Chart 3: Granger causality heatmap
# ---------------------------------------------------------------------------
all_series = {"Market": market, **{c: returns_df[c].values for c in returns_df.columns}}
series_names = list(all_series.keys())
n_series = len(series_names)
p_matrix = np.ones((n_series, n_series))

for i in range(n_series):
    for j in range(n_series):
        if i == j:
            continue
        result = granger_causality(
            list(all_series.values())[i],
            list(all_series.values())[j],
            max_lag=5,
        )
        p_matrix[i, j] = result["a_to_b"]["p_value"]

fig, ax = plt.subplots(figsize=(8, 6))
# Significance mask: -log10(p) for visual impact, capped
significance = -np.log10(np.clip(p_matrix, 1e-10, 1.0))
np.fill_diagonal(significance, 0)

im = ax.imshow(significance, cmap="Blues", aspect="auto", vmin=0, vmax=5)

ax.set_xticks(range(n_series))
ax.set_xticklabels(series_names, rotation=45, ha="right")
ax.set_yticks(range(n_series))
ax.set_yticklabels(series_names)

# Annotate cells
for i in range(n_series):
    for j in range(n_series):
        if i == j:
            continue
        p = p_matrix[i, j]
        marker = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""
        ax.text(
            j,
            i,
            marker,
            ha="center",
            va="center",
            fontsize=10,
            color="white" if significance[i, j] > 2.5 else FEW_PALETTE["grey_dark"],
        )

ax.set_xlabel("Target (effect)")
ax.set_ylabel("Source (cause)")
ax.set_title("Granger Causality: Who Leads Whom?")

# Significance legend as text
ax.text(
    1.02,
    0.5,
    "*** p < 0.01\n** p < 0.05\n* p < 0.10",
    transform=ax.transAxes,
    fontsize=9,
    va="center",
    family="monospace",
)

fig.savefig(OUT / "granger_causality.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved {OUT / 'granger_causality.png'}")
