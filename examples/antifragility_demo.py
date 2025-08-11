"""Antifragility demonstration.

Creates synthetic fragile, robust, and antifragile return series,
computes antifragility scores, runs Fourth Quadrant detection and
barbell allocation, then plots the results.
"""

from __future__ import annotations

import numpy as np

from quantlite.antifragile import (
    antifragility_score,
    barbell_allocation,
    fourth_quadrant,
)
from quantlite.viz.theme import FEW_PALETTE, apply_few_theme, direct_label, few_figure

# ---------------------------------------------------------------------------
# 1. Generate three synthetic return series
# ---------------------------------------------------------------------------
rng = np.random.default_rng(42)
n = 1000

# Fragile: thin-tailed, negatively skewed (short vol profile)
fragile = -np.abs(rng.normal(0, 0.005, n)) + rng.choice([0.001, 0.001, 0.001, -0.05], n)

# Robust: normal returns, symmetric
robust = rng.normal(0.0003, 0.012, n)

# Antifragile: positively skewed, convex payoff (long vol profile)
antifragile = np.abs(rng.normal(0, 0.005, n)) * -1 + rng.choice(
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.08], n
)

# ---------------------------------------------------------------------------
# 2. Antifragility scores
# ---------------------------------------------------------------------------
scores = {
    "Fragile": antifragility_score(fragile),
    "Robust": antifragility_score(robust),
    "Antifragile": antifragility_score(antifragile),
}

print("Antifragility Scores")
print("-" * 40)
for name, score in scores.items():
    print(f"  {name:15s}: {score:+.4f}")
print()

# ---------------------------------------------------------------------------
# 3. Fourth Quadrant detection
# ---------------------------------------------------------------------------
print("Fourth Quadrant Detection")
print("-" * 40)
for name, rets in [("Fragile", fragile), ("Robust", robust), ("Antifragile", antifragile)]:
    fq = fourth_quadrant(rets)
    status = "YES" if fq["fourth_quadrant"] else "no"
    print(f"  {name:15s}: kurtosis={fq['kurtosis']:6.2f}, fourth_quadrant={status}")
print()

# ---------------------------------------------------------------------------
# 4. Barbell allocation: bonds + crypto
# ---------------------------------------------------------------------------
bonds = rng.normal(0.0001, 0.002, 252)
crypto = rng.standard_t(3, 252) * 0.03

barbell = barbell_allocation(bonds, crypto, conservative_pct=0.90)
print("Barbell Allocation (90% bonds / 10% crypto)")
print("-" * 40)
print(f"  Blended arithmetic return: {barbell['blended_arithmetic']:+.6f}")
print(f"  Blended geometric return:  {barbell['blended_geometric']:+.6f}")
print(f"  Max single-period loss:    {barbell['max_loss']:+.4f}")
print(f"  Upside capture (top 10%):  {barbell['upside_capture']:+.4f}")
print()

# ---------------------------------------------------------------------------
# 5. Plot antifragility scores
# ---------------------------------------------------------------------------
apply_few_theme()
fig, ax = few_figure(figsize=(6, 4))

names = list(scores.keys())
vals = list(scores.values())
colours = [FEW_PALETTE["negative"], FEW_PALETTE["neutral"], FEW_PALETTE["positive"]]

bars = ax.bar(names, vals, color=colours, width=0.5, edgecolor="white", linewidth=0.5)

# Direct labels on bars
for bar_obj, val in zip(bars, vals):  # noqa: B905
    y_pos = val + 0.02 if val >= 0 else val - 0.04
    ha = "center"
    direct_label(
        ax,
        bar_obj.get_x() + bar_obj.get_width() / 2,
        y_pos,
        f"{val:+.3f}",
        fontsize=11,
        ha=ha,
    )

ax.axhline(y=0, color=FEW_PALETTE["grey_mid"], linewidth=0.8)
ax.set_ylabel("Antifragility score")
ax.set_title("Fragile, Robust, and Antifragile: Payoff Asymmetry")

fig.savefig("examples/antifragility_scores.png")
print("Chart saved to examples/antifragility_scores.png")
