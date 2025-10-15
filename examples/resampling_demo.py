"""Resampling demonstration: bootstrap distributions and confidence intervals.

Generates three charts for the v0.5 documentation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from quantlite.resample import (
    block_bootstrap,
    bootstrap_drawdown_distribution,
    bootstrap_sharpe_distribution,
    stationary_bootstrap,
)
from quantlite.viz.theme import FEW_PALETTE, apply_few_theme

apply_few_theme()

OUT = Path(__file__).resolve().parent.parent / "docs" / "images"
OUT.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(42)
returns = rng.normal(0.0004, 0.01, 500)


def _sharpe(ret):
    """Annualised Sharpe."""
    if len(ret) < 2:
        return 0.0
    std = float(np.std(ret, ddof=1))
    if std < 1e-15:
        return 0.0
    return float(np.mean(ret) / std * np.sqrt(252))


# ---------------------------------------------------------------------------
# Chart 6: Bootstrapped Sharpe Ratio Distribution
# ---------------------------------------------------------------------------
result = bootstrap_sharpe_distribution(returns, n_samples=2000, seed=42)
dist = result["distribution"]
pe = result["point_estimate"]
ci_lo = result["ci_lower"]
ci_hi = result["ci_upper"]

fig, ax = plt.subplots(figsize=(10, 5))
n_bins, bins, patches = ax.hist(
    dist, bins=40, color=FEW_PALETTE["grey_mid"], alpha=0.5,
    edgecolor="white", linewidth=0.5,
)

# Shade 95% CI
bin_width = bins[1] - bins[0]
for i in range(len(patches)):
    left_edge = bins[i]
    right_edge = left_edge + bin_width
    if left_edge >= ci_lo and right_edge <= ci_hi:
        patches[i].set_facecolor(FEW_PALETTE["primary"])
        patches[i].set_alpha(0.4)

ax.axvline(x=pe, color=FEW_PALETTE["secondary"], linewidth=2)
ax.annotate(
    f"Point estimate: {pe:.2f}",
    xy=(pe, max(n_bins) * 0.9),
    fontsize=10,
    color=FEW_PALETTE["secondary"],
    ha="left",
    bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "none", "alpha": 0.9},
)

ax.text(
    0.02, 0.95,
    f"Point estimate: {pe:.2f}\n95% CI: [{ci_lo:.2f}, {ci_hi:.2f}]",
    transform=ax.transAxes,
    fontsize=10,
    color=FEW_PALETTE["grey_dark"],
    va="top",
    bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": FEW_PALETTE["grey_light"]},
)

ax.set_xlabel("Sharpe Ratio")
ax.set_ylabel("Frequency")
ax.set_title("Bootstrapped Sharpe Ratio Distribution")
fig.savefig(OUT / "sharpe_distribution.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved {OUT / 'sharpe_distribution.png'}")

# ---------------------------------------------------------------------------
# Chart 7: Bootstrapped Maximum Drawdown Distribution
# ---------------------------------------------------------------------------
dd_result = bootstrap_drawdown_distribution(returns, n_samples=2000, seed=42)
dd_dist = dd_result["distribution"]
dd_pe = dd_result["point_estimate"]
dd_lo = dd_result["ci_lower"]
dd_hi = dd_result["ci_upper"]

fig, ax = plt.subplots(figsize=(10, 5))
n_bins, bins, patches = ax.hist(
    dd_dist, bins=40, color=FEW_PALETTE["grey_mid"], alpha=0.5,
    edgecolor="white", linewidth=0.5,
)

bin_width = bins[1] - bins[0]
for i in range(len(patches)):
    left_edge = bins[i]
    right_edge = left_edge + bin_width
    if left_edge >= dd_lo and right_edge <= dd_hi:
        patches[i].set_facecolor(FEW_PALETTE["negative"])
        patches[i].set_alpha(0.4)

ax.axvline(x=dd_pe, color=FEW_PALETTE["secondary"], linewidth=2)
ax.annotate(
    f"Point estimate: {dd_pe:.2%}",
    xy=(dd_pe, max(n_bins) * 0.9),
    fontsize=10,
    color=FEW_PALETTE["secondary"],
    ha="left",
    bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "none", "alpha": 0.9},
)

ax.text(
    0.02, 0.95,
    f"Point estimate: {dd_pe:.2%}\n95% CI: [{dd_lo:.2%}, {dd_hi:.2%}]",
    transform=ax.transAxes,
    fontsize=10,
    color=FEW_PALETTE["grey_dark"],
    va="top",
    bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": FEW_PALETTE["grey_light"]},
)

ax.set_xlabel("Maximum Drawdown")
ax.set_ylabel("Frequency")
ax.set_title("Bootstrapped Maximum Drawdown Distribution")
fig.savefig(OUT / "drawdown_ci.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved {OUT / 'drawdown_ci.png'}")

# ---------------------------------------------------------------------------
# Chart 8: Block vs Stationary Bootstrap
# ---------------------------------------------------------------------------
block_size = 20
n_samples = 2000

block_samples = block_bootstrap(returns, block_size, n_samples=n_samples, seed=42)
stat_samples = stationary_bootstrap(returns, block_size, n_samples=n_samples, seed=42)

block_sharpes = [_sharpe(block_samples[i]) for i in range(n_samples)]
stat_sharpes = [_sharpe(stat_samples[i]) for i in range(n_samples)]

fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(
    block_sharpes, bins=40, color=FEW_PALETTE["primary"], alpha=0.5,
    edgecolor="white", linewidth=0.5, label="Block bootstrap",
)
ax.hist(
    stat_sharpes, bins=40, color=FEW_PALETTE["secondary"], alpha=0.5,
    edgecolor="white", linewidth=0.5, label="Stationary bootstrap",
)

ax.legend(
    loc="upper right", frameon=True,
    facecolor="white", edgecolor=FEW_PALETTE["grey_light"],
)
ax.set_xlabel("Sharpe Ratio")
ax.set_ylabel("Frequency")
ax.set_title("Block vs Stationary Bootstrap: Sharpe Ratio Distributions")
fig.savefig(OUT / "block_vs_stationary.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved {OUT / 'block_vs_stationary.png'}")
