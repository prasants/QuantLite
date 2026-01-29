"""Regime Monte Carlo demonstration: switching paths, stress tests, and reverse stress.

Generates three charts for the v0.9 documentation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from quantlite.simulation.regime_mc import (
    regime_switching_simulation,
    reverse_stress_test,
    simulation_summary,
    stress_test_scenario,
)
from quantlite.viz.theme import FEW_PALETTE, apply_few_theme

apply_few_theme()

OUT = Path(__file__).resolve().parent.parent / "docs" / "images"
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Chart 1: Regime-switching simulation paths
# ---------------------------------------------------------------------------
regime_params = [
    {"mu": 0.0003, "sigma": 0.008},   # calm
    {"mu": -0.001, "sigma": 0.025},    # volatile
]
transition_matrix = np.array([
    [0.95, 0.05],
    [0.10, 0.90],
])

result = regime_switching_simulation(
    regime_params, transition_matrix, n_steps=252, n_scenarios=200, seed=42,
)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), height_ratios=[3, 1])

# Plot cumulative return paths
for i in range(min(50, result["cumulative_returns"].shape[0])):
    colour = FEW_PALETTE["primary"] if result["regimes"][i, -1] == 0 else FEW_PALETTE["negative"]
    ax1.plot(result["cumulative_returns"][i] * 100, alpha=0.2, linewidth=0.5, color=colour)

median_path = np.median(result["cumulative_returns"], axis=0) * 100
p5 = np.percentile(result["cumulative_returns"], 5, axis=0) * 100
p95 = np.percentile(result["cumulative_returns"], 95, axis=0) * 100
ax1.plot(median_path, color=FEW_PALETTE["primary"], linewidth=2, label="Median")
ax1.fill_between(range(252), p5, p95, alpha=0.1, color=FEW_PALETTE["primary"])
ax1.set_ylabel("Cumulative return (%)")
ax1.set_title("Regime-Switching Simulation: 252-Day Paths")
ax1.legend()

# Show regime frequency over time
regime_freq = np.mean(result["regimes"] == 1, axis=0)
ax2.fill_between(range(252), 0, regime_freq, alpha=0.5, color=FEW_PALETTE["negative"],
                 label="Volatile regime")
ax2.set_ylabel("Volatile regime freq.")
ax2.set_xlabel("Trading day")
ax2.set_ylim(0, 1)
ax2.legend()

fig.tight_layout()
fig.savefig(OUT / "regime_simulation.png")
plt.close(fig)
print(f"Saved {OUT / 'regime_simulation.png'}")

# ---------------------------------------------------------------------------
# Chart 2: Stress test scenarios
# ---------------------------------------------------------------------------
rng = np.random.default_rng(42)
returns = rng.standard_t(4, 500) * 0.01

shock_types = ["market_crash", "vol_spike", "liquidity_freeze"]
colours = [FEW_PALETTE["negative"], FEW_PALETTE["secondary"], FEW_PALETTE["neutral"]]

fig, ax = plt.subplots(figsize=(10, 5))
for shock, colour in zip(shock_types, colours):
    st = stress_test_scenario(returns, shock_type=shock, magnitude=0.25, horizon=21)
    cum = np.cumprod(1 + st["stressed_returns"]) - 1
    ax.plot(cum * 100, color=colour, linewidth=2,
            label=f"{shock.replace('_', ' ').title()} ({st['cumulative_impact']:.1%})")

ax.axhline(0, color=FEW_PALETTE["neutral"], linewidth=0.8, linestyle="--")
ax.set_xlabel("Day")
ax.set_ylabel("Cumulative return (%)")
ax.set_title("Stress Test Scenarios: 25% Magnitude Shocks")
ax.legend()
fig.savefig(OUT / "stress_test_scenarios.png")
plt.close(fig)
print(f"Saved {OUT / 'stress_test_scenarios.png'}")

# ---------------------------------------------------------------------------
# Chart 3: Reverse stress test
# ---------------------------------------------------------------------------
rst = reverse_stress_test(returns, target_loss=-0.15, n_scenarios=50000, seed=42)

fig, ax = plt.subplots(figsize=(10, 5))
# Plot individual closest paths
for i in range(min(30, rst["closest_scenarios"].shape[0])):
    path_cum = np.cumprod(1 + rst["closest_scenarios"][i]) - 1
    ax.plot(path_cum * 100, alpha=0.2, color=FEW_PALETTE["negative"], linewidth=0.8)

# Mean path
mean_cum = np.cumprod(1 + rst["mean_path"]) - 1
ax.plot(mean_cum * 100, color=FEW_PALETTE["negative"], linewidth=2.5, label="Mean path")
ax.axhline(-15, color=FEW_PALETTE["secondary"], linewidth=1.5, linestyle="--",
           label="Target: -15%")
ax.set_xlabel("Day")
ax.set_ylabel("Cumulative return (%)")
ax.set_title("Reverse Stress Test: Paths Producing ~15% Loss")
ax.legend()
fig.savefig(OUT / "reverse_stress_test.png")
plt.close(fig)
print(f"Saved {OUT / 'reverse_stress_test.png'}")
