"""EVT simulation demonstration: tail scenarios, bootstrap, and fan charts.

Generates three charts for the v0.9 documentation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from quantlite.simulation.evt_simulation import (
    evt_tail_simulation,
    historical_bootstrap_evt,
    scenario_fan,
)
from quantlite.viz.theme import FEW_PALETTE, apply_few_theme, direct_label

apply_few_theme()

OUT = Path(__file__).resolve().parent.parent / "docs" / "images"
OUT.mkdir(parents=True, exist_ok=True)

# Synthetic fat-tailed returns (Acme Fund daily returns)
rng = np.random.default_rng(42)
body = rng.normal(0.0003, 0.01, 900)
tails = rng.standard_t(3, 100) * 0.03
returns = np.concatenate([body, tails])
rng.shuffle(returns)

# ---------------------------------------------------------------------------
# Chart 1: EVT tail simulation vs historical distribution
# ---------------------------------------------------------------------------
simulated = evt_tail_simulation(returns, n_scenarios=20000, seed=42)

fig, ax = plt.subplots(figsize=(10, 5))
bins = np.linspace(-0.15, 0.15, 120)
ax.hist(returns, bins=bins, density=True, alpha=0.6, color=FEW_PALETTE["primary"],
        label="Historical (Acme Fund)")
ax.hist(simulated, bins=bins, density=True, alpha=0.4, color=FEW_PALETTE["secondary"],
        label="EVT simulated")
ax.set_xlabel("Daily return")
ax.set_ylabel("Density")
ax.set_title("EVT Tail Simulation vs Historical Distribution")
ax.legend()
fig.savefig(OUT / "evt_tail_simulation.png")
plt.close(fig)
print(f"Saved {OUT / 'evt_tail_simulation.png'}")

# ---------------------------------------------------------------------------
# Chart 2: Historical bootstrap EVT comparison
# ---------------------------------------------------------------------------
bootstrap = historical_bootstrap_evt(returns, n_scenarios=20000, seed=42)

fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(returns, bins=bins, density=True, alpha=0.6, color=FEW_PALETTE["primary"],
        label="Historical")
ax.hist(bootstrap, bins=bins, density=True, alpha=0.4, color=FEW_PALETTE["positive"],
        label="Bootstrap EVT")
ax.set_xlabel("Daily return")
ax.set_ylabel("Density")
ax.set_title("Historical Bootstrap with EVT Tails")
ax.legend()
fig.savefig(OUT / "bootstrap_evt.png")
plt.close(fig)
print(f"Saved {OUT / 'bootstrap_evt.png'}")

# ---------------------------------------------------------------------------
# Chart 3: Scenario fan chart
# ---------------------------------------------------------------------------
horizons = [1, 5, 21, 63, 252]
fan_result = scenario_fan(returns, horizons, n_scenarios=5000, seed=42)

fig, ax = plt.subplots(figsize=(10, 5))
x = range(len(horizons))
labels = ["1d", "5d", "21d", "63d", "252d"]

p5 = [fan_result["fans"][h]["5"] for h in horizons]
p25 = [fan_result["fans"][h]["25"] for h in horizons]
p50 = [fan_result["fans"][h]["50"] for h in horizons]
p75 = [fan_result["fans"][h]["75"] for h in horizons]
p95 = [fan_result["fans"][h]["95"] for h in horizons]

ax.fill_between(x, p5, p95, alpha=0.15, color=FEW_PALETTE["primary"], label="5th to 95th")
ax.fill_between(x, p25, p75, alpha=0.3, color=FEW_PALETTE["primary"], label="25th to 75th")
ax.plot(x, p50, color=FEW_PALETTE["primary"], linewidth=2, label="Median")
ax.axhline(0, color=FEW_PALETTE["neutral"], linewidth=0.8, linestyle="--")

ax.set_xticks(list(x))
ax.set_xticklabels(labels)
ax.set_xlabel("Horizon")
ax.set_ylabel("Cumulative return")
ax.set_title("Scenario Fan: Return Distribution by Horizon")
ax.legend()
fig.savefig(OUT / "scenario_fan.png")
plt.close(fig)
print(f"Saved {OUT / 'scenario_fan.png'}")
