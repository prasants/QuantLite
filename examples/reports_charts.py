"""Tearsheet preview chart for the reports documentation.

Generates a 2x2 subplot showing what a QuantLite tearsheet looks like.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from quantlite.viz.theme import FEW_PALETTE, apply_few_theme, direct_label

apply_few_theme()

OUT = Path(__file__).resolve().parent.parent / "docs" / "images"
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Synthetic returns
# ---------------------------------------------------------------------------
rng = np.random.default_rng(42)
n = 500
daily_returns = rng.normal(0.0003, 0.012, n)
# Inject a drawdown period
daily_returns[120:160] -= 0.008
daily_returns[300:330] -= 0.006

dates = pd.bdate_range("2023-01-03", periods=n)
returns = pd.Series(daily_returns, index=dates)
cumulative = (1 + returns).cumprod()

# Drawdown series
rolling_max = cumulative.cummax()
drawdown = (cumulative - rolling_max) / rolling_max

# Rolling Sharpe (63-day window)
rolling_mean = returns.rolling(63).mean()
rolling_std = returns.rolling(63).std()
rolling_sharpe = (rolling_mean / rolling_std) * np.sqrt(252)

# ---------------------------------------------------------------------------
# 2x2 subplot
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle(
    "Tearsheet Preview: What You Get",
    fontsize=14,
    fontweight="bold",
    y=0.97,
)

# Top-left: Equity curve
ax = axes[0, 0]
ax.plot(cumulative.index, cumulative.values, color=FEW_PALETTE["primary"], linewidth=1.5)
ax.set_title("Equity Curve", fontsize=11)
ax.set_ylabel("Portfolio value")
direct_label(
    ax,
    cumulative.index[-1],
    cumulative.iloc[-1],
    f" {cumulative.iloc[-1]:.2f}",
    colour=FEW_PALETTE["primary"],
    ha="left",
    fontsize=9,
)

# Top-right: Drawdown
ax = axes[0, 1]
ax.fill_between(
    drawdown.index,
    drawdown.values,
    0,
    color=FEW_PALETTE["negative"],
    alpha=0.4,
)
ax.plot(drawdown.index, drawdown.values, color=FEW_PALETTE["negative"], linewidth=1)
ax.set_title("Drawdown", fontsize=11)
ax.set_ylabel("Drawdown")
max_dd_idx = drawdown.idxmin()
direct_label(
    ax,
    max_dd_idx,
    drawdown[max_dd_idx] - 0.005,
    f"Max: {drawdown[max_dd_idx]:.1%}",
    colour=FEW_PALETTE["negative"],
    ha="center",
    va="top",
    fontsize=9,
)

# Bottom-left: Rolling Sharpe
ax = axes[1, 0]
valid = rolling_sharpe.dropna()
ax.plot(valid.index, valid.values, color=FEW_PALETTE["positive"], linewidth=1.5)
ax.axhline(y=0, color=FEW_PALETTE["grey_mid"], linewidth=0.8, linestyle="-")
ax.set_title("Rolling Sharpe (63-day)", fontsize=11)
ax.set_ylabel("Sharpe ratio")
direct_label(
    ax,
    valid.index[-1],
    valid.iloc[-1],
    f" {valid.iloc[-1]:.1f}",
    colour=FEW_PALETTE["positive"],
    ha="left",
    fontsize=9,
)

# Bottom-right: Return distribution with VaR
ax = axes[1, 1]
ax.hist(
    returns.values,
    bins=40,
    color=FEW_PALETTE["primary"],
    alpha=0.6,
    edgecolor="white",
    linewidth=0.5,
)
var_95 = np.percentile(returns, 5)
cvar_95 = returns[returns <= var_95].mean()
ax.axvline(x=var_95, color=FEW_PALETTE["negative"], linewidth=1.5, linestyle="--")
ax.axvline(x=cvar_95, color=FEW_PALETTE["secondary"], linewidth=1.5, linestyle="--")
ax.set_title("Return Distribution", fontsize=11)
ax.set_xlabel("Daily return")
ax.set_ylabel("Frequency")
# Labels with white backgrounds
ax.annotate(
    f"VaR 95%: {var_95:.3f}",
    xy=(var_95, ax.get_ylim()[1] * 0.85),
    fontsize=8,
    color=FEW_PALETTE["negative"],
    ha="right",
    bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "none", "alpha": 0.9},
)
ax.annotate(
    f"CVaR 95%: {cvar_95:.3f}",
    xy=(cvar_95, ax.get_ylim()[1] * 0.7),
    fontsize=8,
    color=FEW_PALETTE["secondary"],
    ha="right",
    bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "none", "alpha": 0.9},
)

# Rotate x-axis dates for top row
for a in axes[0]:
    for label in a.get_xticklabels():
        label.set_rotation(30)
        label.set_ha("right")
for a in axes[1, :1]:
    for label in a.get_xticklabels():
        label.set_rotation(30)
        label.set_ha("right")

fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(OUT / "tearsheet_preview.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved {OUT / 'tearsheet_preview.png'}")
