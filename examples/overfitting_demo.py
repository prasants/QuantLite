"""Overfitting demonstration: strategy mining and walk-forward validation.

Generates two charts for the v0.5 documentation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from quantlite.overfit import TrialTracker, walk_forward_validate
from quantlite.viz.theme import FEW_PALETTE, apply_few_theme, direct_label

apply_few_theme()

OUT = Path(__file__).resolve().parent.parent / "docs" / "images"
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Chart 4: Strategy Mining - 50 Trials, 1 "Winner"
# ---------------------------------------------------------------------------
rng = np.random.default_rng(42)
n_trials = 50
n_obs = 500

with TrialTracker("random_mining") as tracker:
    all_sharpes = []
    for i in range(n_trials):
        ret = rng.normal(0, 0.01, n_obs)
        sharpe = float(np.mean(ret) / np.std(ret, ddof=1) * np.sqrt(252))
        tracker.log(params={"trial": i}, sharpe=sharpe, returns=ret)
        all_sharpes.append(sharpe)

    pbo = tracker.overfitting_probability()
    best = tracker.best_trial

best_sharpe = best["sharpe"]

fig, ax = plt.subplots(figsize=(10, 5))
n_bins, bins, patches = ax.hist(
    all_sharpes, bins=15, color=FEW_PALETTE["grey_mid"], alpha=0.7,
    edgecolor="white", linewidth=0.5,
)

# Highlight the best trial's bin
bin_width = bins[1] - bins[0]
for i in range(len(patches)):
    left_edge = bins[i]
    if left_edge <= best_sharpe < left_edge + bin_width:
        patches[i].set_facecolor(FEW_PALETTE["negative"])
        patches[i].set_alpha(1.0)

ax.axvline(
    x=best_sharpe, color=FEW_PALETTE["negative"], linewidth=1.5, linestyle="--",
)
ax.annotate(
    f'"Best" Sharpe: {best_sharpe:.2f}',
    xy=(best_sharpe, max(n_bins) * 0.85),
    fontsize=10,
    color=FEW_PALETTE["negative"],
    ha="left",
    bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "none", "alpha": 0.9},
)

ax.text(
    0.02, 0.95, f"PBO: {pbo:.0%}",
    transform=ax.transAxes,
    fontsize=12,
    color=FEW_PALETTE["grey_dark"],
    va="top",
    bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": FEW_PALETTE["grey_light"]},
)

ax.set_xlabel("Sharpe Ratio")
ax.set_ylabel("Number of strategies")
ax.set_title("Strategy Mining: 50 Trials, 1 'Winner'")
fig.savefig(OUT / "overfitting_trials.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved {OUT / 'overfitting_trials.png'}")

# ---------------------------------------------------------------------------
# Chart 5: Walk-Forward Validation
# ---------------------------------------------------------------------------
rng2 = np.random.default_rng(123)
returns = rng2.normal(0.0003, 0.01, 1000)


def momentum_strategy(train_returns):
    """Go long if recent mean is positive, else flat."""
    return 1.0 if np.mean(train_returns) > 0 else 0.0


result = walk_forward_validate(returns, momentum_strategy, window=200, step=50)
folds = result["folds"]

fold_sharpes = []
for f in folds:
    test_start = f["test_start"]
    test_end = f["test_end"]
    test_ret = returns[test_start:test_end] * f["weight"]
    std = float(np.std(test_ret, ddof=1)) if len(test_ret) > 1 else 1e-10
    sr = float(np.mean(test_ret) / std * np.sqrt(252)) if std > 1e-10 else 0.0
    fold_sharpes.append(sr)

fig, ax = plt.subplots(figsize=(10, 5))
colours = [
    FEW_PALETTE["positive"] if s >= 0 else FEW_PALETTE["negative"]
    for s in fold_sharpes
]
ax.bar(range(len(fold_sharpes)), fold_sharpes, color=colours, width=0.6)

for i, val in enumerate(fold_sharpes):
    va = "bottom" if val >= 0 else "top"
    offset = 0.15 if val >= 0 else -0.15
    direct_label(
        ax, i, val + offset, f"{val:.1f}",
        ha="center", va=va, fontsize=9,
    )

ax.set_xticks(range(len(fold_sharpes)))
ax.set_xticklabels([f"Fold {i + 1}" for i in range(len(fold_sharpes))], fontsize=9)
ax.set_xlabel("Out-of-sample fold")
ax.set_ylabel("Sharpe Ratio")
ax.set_title("Walk-Forward Validation: Per-Fold Out-of-Sample Sharpe")
ax.axhline(y=0, color=FEW_PALETTE["grey_mid"], linewidth=0.8)
fig.savefig(OUT / "walk_forward.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved {OUT / 'walk_forward.png'}")
