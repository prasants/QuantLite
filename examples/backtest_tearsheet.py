#!/usr/bin/env python3
"""Backtest tearsheet: equity curve, drawdown, monthly heatmap, rolling Sharpe.

Generates four charts saved to docs/images/.
"""
from __future__ import annotations

import os, sys
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from quantlite.viz.theme import apply_few_theme, FEW_PALETTE
from quantlite.risk.metrics import max_drawdown_duration
from quantlite.data_generation import merton_jump_diffusion, geometric_brownian_motion

OUT = os.path.join(os.path.dirname(__file__), "..", "docs", "images")
os.makedirs(OUT, exist_ok=True)
DPI = 150

# --- Strategy vs benchmark ---
np.random.seed(42)
# Benchmark (SPX-like)
bench_prices = np.concatenate([
    geometric_brownian_motion(S0=100, mu=0.08, sigma=0.16, steps=252*3, rng_seed=10),
    merton_jump_diffusion(S0=100, mu=-0.05, sigma=0.35, lamb=1.5,
                           jump_mean=-0.03, jump_std=0.05, steps=252, rng_seed=20),
])
bench_prices[252*3+1:] = bench_prices[252*3] * bench_prices[252*3+1:] / bench_prices[252*3+1]
bench_returns = np.diff(bench_prices) / bench_prices[:-1]

# Strategy: lower drawdown, slightly higher return
strat_returns = bench_returns * 0.7 + 0.0002  # dampened beta + alpha
# Add some alpha periods
strat_returns[100:300] += 0.0005
strat_returns[600:700] += 0.0003

dates = pd.date_range("2020-01-02", periods=len(bench_returns), freq="B")

bench_equity = 100 * np.cumprod(1 + bench_returns)
strat_equity = 100 * np.cumprod(1 + strat_returns)

apply_few_theme()

# ============================================================
# 1. Equity curve with benchmark
# ============================================================
fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(dates, strat_equity, color=FEW_PALETTE["primary"], linewidth=1.5, label="Strategy")
ax.plot(dates, bench_equity, color=FEW_PALETTE["grey_mid"], linewidth=1, label="Benchmark")

# Annotate final values
ax.text(dates[-1], strat_equity[-1], f" {strat_equity[-1]:.0f}", fontsize=9,
        color=FEW_PALETTE["primary"], va="center")
ax.text(dates[-1], bench_equity[-1], f" {bench_equity[-1]:.0f}", fontsize=9,
        color=FEW_PALETTE["grey_mid"], va="center")

ax.set_ylabel("Portfolio value")
ax.set_title("Strategy vs Benchmark")
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "equity_curve.png"), dpi=DPI)
plt.close()
print("  Saved equity_curve.png")

# ============================================================
# 2. Underwater (drawdown) chart
# ============================================================
fig, ax = plt.subplots(figsize=(10, 3.5))

cum_strat = np.cumprod(1 + strat_returns)
roll_max = np.maximum.accumulate(cum_strat)
dd = (cum_strat - roll_max) / roll_max

ax.fill_between(dates, dd, 0, color=FEW_PALETTE["negative"], alpha=0.35)
ax.plot(dates, dd, color=FEW_PALETTE["negative"], linewidth=0.8)

dd_info = max_drawdown_duration(strat_returns)
ax.annotate(f"Max DD: {dd_info.max_drawdown:.1%}",
            xy=(dates[dd_info.end_idx], dd_info.max_drawdown),
            xytext=(dates[min(dd_info.end_idx + 60, len(dates)-1)], dd_info.max_drawdown * 0.5),
            fontsize=9, color=FEW_PALETTE["grey_dark"],
            arrowprops=dict(arrowstyle="->", color=FEW_PALETTE["grey_mid"]))

ax.set_ylabel("Drawdown")
ax.set_title("Underwater Chart")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "backtest_drawdown.png"), dpi=DPI)
plt.close()
print("  Saved backtest_drawdown.png")

# ============================================================
# 3. Monthly returns heatmap
# ============================================================
strat_series = pd.Series(strat_returns, index=dates)
monthly = strat_series.resample("M").apply(lambda x: (1 + x).prod() - 1)
monthly_df = pd.DataFrame({"year": monthly.index.year, "month": monthly.index.month, "ret": monthly.values})
pivot = monthly_df.pivot(index="year", columns="month", values="ret")
pivot.columns = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                 "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

fig, ax = plt.subplots(figsize=(10, 3))
data = pivot.values
vmax = max(abs(np.nanmin(data)), abs(np.nanmax(data)))
cmap = plt.cm.RdBu_r.copy()
cmap.set_bad(color=FEW_PALETTE["grey_light"])
masked = np.ma.masked_invalid(data)

ax.imshow(masked, cmap=cmap, aspect="auto", vmin=-vmax, vmax=vmax)
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        if not np.isnan(data[i, j]):
            colour = "white" if abs(data[i, j]) > vmax * 0.5 else FEW_PALETTE["grey_dark"]
            ax.text(j, i, f"{data[i, j]:.1%}", ha="center", va="center", fontsize=8, color=colour)

ax.set_xticks(range(12))
ax.set_xticklabels(pivot.columns, fontsize=9)
ax.set_yticks(range(len(pivot)))
ax.set_yticklabels(pivot.index, fontsize=9)
ax.set_title("Monthly Returns")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "backtest_monthly_returns.png"), dpi=DPI)
plt.close()
print("  Saved backtest_monthly_returns.png")

# ============================================================
# 4. Rolling Sharpe ratio
# ============================================================
fig, ax = plt.subplots(figsize=(10, 4))

window = 126  # 6 months
rolling_mean = pd.Series(strat_returns).rolling(window).mean()
rolling_std = pd.Series(strat_returns).rolling(window).std()
rolling_sharpe = (rolling_mean / rolling_std) * np.sqrt(252)

ax.plot(dates, rolling_sharpe, color=FEW_PALETTE["primary"], linewidth=1.2)
ax.axhline(0, color=FEW_PALETTE["grey_mid"], linewidth=0.8, linestyle="--")
ax.axhline(1.0, color=FEW_PALETTE["positive"], linewidth=0.8, linestyle=":", alpha=0.7)
ax.text(dates[5], 1.05, "Sharpe = 1.0", fontsize=8, color=FEW_PALETTE["positive"])

# Overall Sharpe
overall_sharpe = (np.mean(strat_returns) / np.std(strat_returns, ddof=1)) * np.sqrt(252)
ax.axhline(overall_sharpe, color=FEW_PALETTE["secondary"], linewidth=1, linestyle=":")
ax.text(dates[5], overall_sharpe + 0.1, f"Overall: {overall_sharpe:.2f}",
        fontsize=9, color=FEW_PALETTE["secondary"])

ax.set_ylabel("Sharpe ratio (ann.)")
ax.set_title("Rolling 6-Month Sharpe Ratio")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "rolling_sharpe.png"), dpi=DPI)
plt.close()
print("  Saved rolling_sharpe.png")

print("Done: backtest_tearsheet.py")
