"""Diversification demonstration: ENB, tail diversification, and marginal tail risk.

Generates three charts for the v0.6 documentation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from quantlite.diversification import (
    diversification_ratio,
    effective_number_of_bets,
    entropy_diversification,
    herfindahl_index,
    marginal_tail_risk_contribution,
    tail_diversification,
)
from quantlite.viz.theme import FEW_PALETTE, apply_few_theme, direct_label

apply_few_theme()

OUT = Path(__file__).resolve().parent.parent / "docs" / "images"
OUT.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(42)
n = 2000

# Build correlated multi-asset returns
market = rng.normal(0.0003, 0.012, n)
assets = {
    "US Equity": 0.9 * market + rng.normal(0, 0.005, n),
    "EU Equity": 0.7 * market + rng.normal(0, 0.007, n),
    "EM Equity": 0.5 * market + rng.normal(0, 0.012, n),
    "Govt Bonds": -0.25 * market + rng.normal(0.0002, 0.004, n),
    "Gold": -0.1 * market + rng.normal(0.0001, 0.008, n),
    "Real Estate": 0.4 * market + rng.normal(0, 0.009, n),
}
returns_df = pd.DataFrame(assets)
cov = returns_df.cov().values
vols = returns_df.std().values

# Portfolio configurations
portfolios = {
    "Concentrated\n(70/30 equity)": np.array([0.70, 0.15, 0.05, 0.05, 0.03, 0.02]),
    "Balanced\n(equal weight)": np.ones(6) / 6,
    "Diversified\n(risk parity)": np.array([0.10, 0.12, 0.08, 0.35, 0.20, 0.15]),
    "Defensive\n(bond heavy)": np.array([0.10, 0.05, 0.05, 0.50, 0.20, 0.10]),
}

# ---------------------------------------------------------------------------
# Chart 1: Diversification metrics comparison across portfolios
# ---------------------------------------------------------------------------
metrics = {}
for name, w in portfolios.items():
    enb = effective_number_of_bets(w, cov)
    ent = entropy_diversification(w)
    dr = diversification_ratio(w, vols, cov)
    hhi = herfindahl_index(w)
    metrics[name] = {"ENB": enb, "Entropy": ent, "DR": dr, "HHI": hhi}

fig, axes = plt.subplots(1, 4, figsize=(14, 5), sharey=True)
metric_names = ["ENB", "Entropy", "DR", "HHI"]
metric_labels = [
    "Effective Number\nof Bets",
    "Entropy\nDiversification",
    "Diversification\nRatio",
    "Herfindahl\nIndex",
]
port_names = list(portfolios.keys())

for idx, (metric, label) in enumerate(zip(metric_names, metric_labels)):  # noqa: B905
    ax = axes[idx]
    vals = [metrics[p][metric] for p in port_names]

    if metric == "HHI":
        colours = [
            FEW_PALETTE["negative"] if v > 0.2
            else FEW_PALETTE["secondary"] if v > 0.15
            else FEW_PALETTE["positive"]
            for v in vals
        ]
    else:
        colours = [
            FEW_PALETTE["positive"] if v > np.median(vals)
            else FEW_PALETTE["secondary"]
            for v in vals
        ]

    bars = ax.barh(range(len(port_names)), vals, color=colours)

    for i, bar in enumerate(bars):
        direct_label(
            ax,
            bar.get_width() + max(vals) * 0.03,
            i,
            f"{vals[i]:.2f}",
            ha="left",
            va="center",
            fontsize=9,
        )

    ax.set_title(label, fontsize=10)
    ax.set_xlim(0, max(vals) * 1.35)
    if idx == 0:
        ax.set_yticks(range(len(port_names)))
        ax.set_yticklabels(port_names, fontsize=9)

fig.suptitle(
    "Diversification Metrics: Four Lenses on the Same Portfolios",
    fontsize=12,
    fontweight="bold",
    y=1.02,
)
fig.tight_layout()
fig.savefig(OUT / "diversification_metrics.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved {OUT / 'diversification_metrics.png'}")

# ---------------------------------------------------------------------------
# Chart 2: Tail diversification breakdown
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 5))
bar_width = 0.35
x = np.arange(len(port_names))

normal_vals = []
tail_vals = []
for _name, w in portfolios.items():
    td = tail_diversification(returns_df, w, alpha=0.05)
    normal_vals.append(td["normal_diversification"])
    tail_vals.append(td["tail_diversification"])

bars_normal = ax.bar(
    x - bar_width / 2,
    normal_vals,
    bar_width,
    color=FEW_PALETTE["positive"],
)
bars_tail = ax.bar(
    x + bar_width / 2,
    tail_vals,
    bar_width,
    color=FEW_PALETTE["negative"],
)

for i in range(len(port_names)):
    direct_label(
        ax,
        x[i] - bar_width / 2,
        normal_vals[i] + 0.01,
        f"{normal_vals[i]:.2f}",
        ha="center",
        va="bottom",
        fontsize=8,
    )
    direct_label(
        ax,
        x[i] + bar_width / 2,
        tail_vals[i] + 0.01,
        f"{tail_vals[i]:.2f}",
        ha="center",
        va="bottom",
        fontsize=8,
    )

# Series labels
direct_label(
    ax,
    x[-1] - bar_width / 2,
    max(normal_vals) + 0.05,
    "Normal times",
    ha="center",
    fontsize=9,
    colour=FEW_PALETTE["positive"],
)
direct_label(
    ax,
    x[-1] + bar_width / 2,
    max(tail_vals) + 0.05,
    "Tail (5th percentile)",
    ha="center",
    fontsize=9,
    colour=FEW_PALETTE["negative"],
)

ax.set_xticks(x)
ax.set_xticklabels(port_names, fontsize=9)
ax.set_ylabel("Diversification ratio")
ax.set_title("The Diversification Illusion: Normal Times vs Crisis")
fig.savefig(OUT / "tail_diversification.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved {OUT / 'tail_diversification.png'}")

# ---------------------------------------------------------------------------
# Chart 3: Marginal tail risk contributions (best portfolio)
# ---------------------------------------------------------------------------
best_weights = portfolios["Diversified\n(risk parity)"]
contributions = marginal_tail_risk_contribution(returns_df, best_weights, alpha=0.05)
asset_names = list(contributions.keys())
contrib_vals = [contributions[a] * 100 for a in asset_names]

# Sort by contribution (most negative first)
sorted_pairs = sorted(
    zip(asset_names, contrib_vals),  # noqa: B905
    key=lambda x: x[1],
)
asset_names = [p[0] for p in sorted_pairs]
contrib_vals = [p[1] for p in sorted_pairs]

colours = [
    FEW_PALETTE["negative"] if v < -0.01
    else FEW_PALETTE["positive"] if v > 0.01
    else FEW_PALETTE["neutral"]
    for v in contrib_vals
]

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.barh(range(len(asset_names)), contrib_vals, color=colours)

for i, _bar in enumerate(bars):
    val = contrib_vals[i]
    ha = "right" if val < 0 else "left"
    offset = -0.002 if val < 0 else 0.002
    direct_label(
        ax,
        val + offset,
        i,
        f"{val:.3f}%",
        ha=ha,
        va="center",
        fontsize=9,
    )

ax.set_yticks(range(len(asset_names)))
ax.set_yticklabels(asset_names)
ax.set_xlabel("Marginal CVaR contribution (%)")
ax.set_title("Where Does Tail Risk Come From? (Risk Parity Portfolio)")
ax.axvline(x=0, color=FEW_PALETTE["grey_dark"], linewidth=0.8)
fig.savefig(OUT / "marginal_tail_risk.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved {OUT / 'marginal_tail_risk.png'}")
