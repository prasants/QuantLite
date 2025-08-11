"""Ergodicity economics demonstration.

Generates synthetic fat-tailed returns, computes the ergodicity gap,
finds the Kelly-optimal fraction, and plots how leverage affects
time-average growth rate.
"""

from __future__ import annotations

import numpy as np

from quantlite.ergodicity import (
    ensemble_average,
    ergodicity_gap,
    kelly_fraction,
    leverage_effect,
    time_average,
)
from quantlite.viz.theme import FEW_PALETTE, apply_few_theme, direct_label, few_figure

# ---------------------------------------------------------------------------
# 1. Generate synthetic returns with fat tails (Student-t, nu=4)
# ---------------------------------------------------------------------------
rng = np.random.default_rng(42)
returns = rng.standard_t(df=4, size=2520) * 0.012 + 0.0003

# ---------------------------------------------------------------------------
# 2. Ergodicity gap
# ---------------------------------------------------------------------------
ta = time_average(returns)
ea = ensemble_average(returns)
gap = ergodicity_gap(returns)

print("Ergodicity Gap")
print("-" * 40)
print(f"  Ensemble average (arithmetic): {ea:+.6f}")
print(f"  Time average (geometric):      {ta:+.6f}")
print(f"  Gap:                           {gap:+.6f}")
print()

# ---------------------------------------------------------------------------
# 3. Kelly fraction
# ---------------------------------------------------------------------------
kelly_f = kelly_fraction(returns)
print(f"Optimal Kelly fraction: {kelly_f:.4f}")
print()

# ---------------------------------------------------------------------------
# 4. Leverage effect: growth rate vs leverage multiple
# ---------------------------------------------------------------------------
leverages = [round(x, 2) for x in np.arange(0.25, 5.25, 0.25).tolist()]
lev_results = leverage_effect(returns, leverages=leverages)

print("Leverage vs Time-Average Growth")
print("-" * 40)
for lev, growth in lev_results.items():
    marker = " <-- Kelly" if abs(lev - kelly_f) < 0.3 else ""
    print(f"  {lev:5.2f}x: {growth:+.6f}{marker}")

# ---------------------------------------------------------------------------
# 5. Plot and save
# ---------------------------------------------------------------------------
apply_few_theme()
fig, ax = few_figure(figsize=(8, 4.5))

levs = list(lev_results.keys())
growths = list(lev_results.values())

ax.plot(levs, growths, color=FEW_PALETTE["primary"], linewidth=2)
ax.axhline(y=0, color=FEW_PALETTE["grey_mid"], linewidth=0.8, linestyle="--")

# Mark the Kelly-optimal point
peak_lev = levs[np.argmax(growths)]
peak_growth = max(growths)
ax.plot(peak_lev, peak_growth, "o", color=FEW_PALETTE["secondary"], markersize=8, zorder=5)
direct_label(
    ax,
    peak_lev + 0.15,
    peak_growth,
    f"Kelly optimal ({peak_lev:.1f}x)",
    colour=FEW_PALETTE["secondary"],
    fontsize=10,
)

# Mark where growth turns negative
neg_levs = [lv for lv, gr in zip(levs, growths) if gr < 0]  # noqa: B905
if neg_levs:
    ruin_lev = neg_levs[0]
    ax.axvline(
        x=ruin_lev,
        color=FEW_PALETTE["negative"],
        linewidth=1,
        linestyle=":",
        alpha=0.7,
    )
    direct_label(
        ax,
        ruin_lev + 0.1,
        min(growths) * 0.5,
        f"Ruin zone ({ruin_lev:.1f}x)",
        colour=FEW_PALETTE["negative"],
        fontsize=9,
    )

ax.set_xlabel("Leverage multiple")
ax.set_ylabel("Time-average growth rate (per period)")
ax.set_title("Leverage vs Time-Average Growth: the Peak and the Collapse")

fig.savefig("examples/ergodicity_leverage.png")
print("\nChart saved to examples/ergodicity_leverage.png")
