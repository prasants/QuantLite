"""Scenario stress testing demonstration.

Creates a multi-asset portfolio, runs stress tests against all
pre-built scenarios, computes the fragility heatmap, and plots
the results.
"""

from __future__ import annotations

import numpy as np

from quantlite.scenarios import SCENARIO_LIBRARY, fragility_heatmap
from quantlite.viz.theme import FEW_PALETTE, apply_few_theme, few_figure

# ---------------------------------------------------------------------------
# 1. Define a multi-asset portfolio
# ---------------------------------------------------------------------------
weights = {
    "SPX": 0.30,
    "BTC": 0.15,
    "ETH": 0.10,
    "BONDS_10Y": 0.30,
    "GLD": 0.15,
}

print("Portfolio Weights")
print("-" * 40)
for asset, w in weights.items():
    print(f"  {asset:12s}: {w:.0%}")
print()

# ---------------------------------------------------------------------------
# 2. Stress test against all 5 pre-built scenarios
# ---------------------------------------------------------------------------
scenarios = list(SCENARIO_LIBRARY.values())

print("Stress Test Results")
print("-" * 60)
for scenario in scenarios:
    from quantlite.scenarios import stress_test

    result = stress_test(weights, scenario)
    survival = "survives" if result["survival"] else "WIPED OUT"
    print(f"  {result['scenario_name']:20s}: {result['portfolio_impact']:+.2%} ({survival})")
print()

# ---------------------------------------------------------------------------
# 3. Fragility heatmap
# ---------------------------------------------------------------------------
heatmap = fragility_heatmap(weights, scenarios)

print("Fragility Heatmap")
print("-" * 60)
header = f"  {'Scenario':20s}" + "".join(f"{a:>12s}" for a in weights)
print(header)
for scenario_name, impacts in heatmap.items():
    row = f"  {scenario_name:20s}"
    for asset in weights:
        val = impacts.get(asset, 0.0)
        row += f"{val:+12.4f}"
    print(row)
print()

# ---------------------------------------------------------------------------
# 4. Plot the heatmap
# ---------------------------------------------------------------------------
apply_few_theme()

scenario_names = list(heatmap.keys())
asset_names = list(weights.keys())
data = np.array(
    [[heatmap[s].get(a, 0.0) for a in asset_names] for s in scenario_names]
)

fig, ax = few_figure(figsize=(9, 5))

# Diverging colourmap: red for negative, green for positive
vmax = max(abs(data.min()), abs(data.max()))
im = ax.imshow(
    data,
    cmap="RdYlGn",
    aspect="auto",
    vmin=-vmax,
    vmax=vmax,
    interpolation="nearest",
)

# Labels
ax.set_xticks(range(len(asset_names)))
ax.set_xticklabels(asset_names, fontsize=10)
ax.set_yticks(range(len(scenario_names)))
ax.set_yticklabels(scenario_names, fontsize=10)

# Direct labels in cells
for i in range(len(scenario_names)):
    for j in range(len(asset_names)):
        val = data[i, j]
        text_colour = "white" if abs(val) > vmax * 0.6 else FEW_PALETTE["grey_dark"]
        ax.text(
            j,
            i,
            f"{val:+.3f}",
            ha="center",
            va="center",
            fontsize=9,
            color=text_colour,
        )

ax.set_title("Scenario Fragility Heatmap: Impact by Asset and Crisis")

# Minimal colourbar
cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
cbar.set_label("Weighted impact", fontsize=10)

# Remove default grid for the heatmap
ax.grid(False)

fig.savefig("examples/scenario_heatmap.png")
print("Chart saved to examples/scenario_heatmap.png")
