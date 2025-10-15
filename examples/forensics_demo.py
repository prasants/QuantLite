"""Forensics demonstration: Deflated Sharpe, minimum track record, and signal decay.

Generates three charts for the v0.5 documentation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from quantlite.forensics import deflated_sharpe_ratio, min_track_record_length, signal_decay
from quantlite.viz.theme import FEW_PALETTE, apply_few_theme, direct_label

apply_few_theme()

OUT = Path(__file__).resolve().parent.parent / "docs" / "images"
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Chart 1: Deflated Sharpe Ratio vs number of trials
# ---------------------------------------------------------------------------
trials = [1, 5, 10, 20, 50, 100]
observed = 2.0
n_obs = 252

dsr_values = [deflated_sharpe_ratio(observed, t, n_obs) for t in trials]

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(
    range(len(trials)),
    dsr_values,
    color=FEW_PALETTE["primary"],
    width=0.6,
)
ax.set_xticks(range(len(trials)))
ax.set_xticklabels([str(t) for t in trials])
ax.set_xlabel("Number of trials tested")
ax.set_ylabel("Deflated Sharpe Ratio (probability)")
ax.set_title("The More You Try, the Less You Should Believe")

# Direct labels on bars
for i, val in enumerate(dsr_values):
    direct_label(
        ax, i, val + 0.02, f"{val:.2f}",
        ha="center", va="bottom", fontsize=10,
    )

# Reference line for 95% confidence threshold
ax.axhline(
    y=0.95, color=FEW_PALETTE["secondary"], linewidth=1.5, linestyle="--",
)
ax.annotate(
    "95% confidence threshold",
    xy=(4.5, 0.95),
    fontsize=10,
    color=FEW_PALETTE["secondary"],
    ha="right",
    va="bottom",
    bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "none", "alpha": 0.9},
)

ax.set_ylim(0, 1.15)
fig.savefig(OUT / "deflated_sharpe.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved {OUT / 'deflated_sharpe.png'}")

# ---------------------------------------------------------------------------
# Chart 2: Minimum Track Record Length
# ---------------------------------------------------------------------------
sharpe_range = np.linspace(0.5, 3.0, 100)
min_years = [min_track_record_length(s) / 252 for s in sharpe_range]

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(sharpe_range, min_years, color=FEW_PALETTE["primary"], linewidth=2)
ax.fill_between(
    sharpe_range, 0, min_years,
    color=FEW_PALETTE["negative"], alpha=0.12,
)

# Label the shaded region
ax.text(
    1.0, 2.0, "Insufficient data",
    fontsize=11, color=FEW_PALETTE["negative"], alpha=0.7,
    style="italic",
)

# Direct label on the curve
mid_idx = 30
direct_label(
    ax,
    sharpe_range[mid_idx],
    min_years[mid_idx] + 0.5,
    "Minimum track record",
    colour=FEW_PALETTE["primary"],
    ha="left",
    fontsize=10,
)

ax.set_xlabel("Observed Sharpe Ratio")
ax.set_ylabel("Minimum track record (years)")
ax.set_title("Minimum Track Record Length to Trust a Sharpe Ratio")
ax.set_xlim(0.5, 3.0)
ax.set_ylim(0, max(min_years) * 1.1)
fig.savefig(OUT / "min_track_record.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved {OUT / 'min_track_record.png'}")

# ---------------------------------------------------------------------------
# Chart 3: Signal Decay
# ---------------------------------------------------------------------------
rng = np.random.default_rng(42)
n = 500
returns = rng.normal(0.0005, 0.01, n)
# Signal = lagged returns + increasing noise
signal_vals = np.roll(returns, 1) + rng.normal(0, 0.003, n)
signal_vals[0] = 0.0

lags = [1, 2, 3, 5, 7, 10, 15, 20]
result = signal_decay(returns, signal_vals, lags=lags)

decay_lags = [x[0] for x in result["decay_curve"]]
decay_corrs = [x[1] for x in result["decay_curve"]]

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(
    decay_lags, decay_corrs,
    color=FEW_PALETTE["primary"], linewidth=2, marker="o", markersize=5,
)

# Direct labels
for i in range(len(decay_lags)):
    lg = decay_lags[i]
    corr = decay_corrs[i]
    va = "bottom" if i % 2 == 0 else "top"
    offset = 0.01 if va == "bottom" else -0.01
    direct_label(
        ax, lg, corr + offset, f"{corr:.3f}",
        ha="center", va=va, fontsize=9,
    )

# Mark half-life
if result["half_life"] is not None:
    hl = result["half_life"]
    ax.axvline(
        x=hl, color=FEW_PALETTE["secondary"], linewidth=1.5, linestyle="--",
    )
    ax.annotate(
        f"Half-life: {hl:.0f} periods",
        xy=(hl, max(decay_corrs) * 0.7),
        fontsize=10,
        color=FEW_PALETTE["secondary"],
        ha="left",
        bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "none", "alpha": 0.9},
    )

ax.set_xlabel("Lag (periods)")
ax.set_ylabel("Correlation")
ax.set_title("Signal Decay: How Fast Does Alpha Disappear?")
fig.savefig(OUT / "signal_decay.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved {OUT / 'signal_decay.png'}")
