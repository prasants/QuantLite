#!/usr/bin/env python3
"""Risk dashboard: VaR/CVaR comparison, fat-tail distribution, drawdown, and bullet graphs.

Generates four publication-quality charts saved to docs/images/.
"""
from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from quantlite.data_generation import merton_jump_diffusion
from quantlite.risk.metrics import (
    calmar_ratio,
    cvar,
    max_drawdown_duration,
    omega_ratio,
    sortino_ratio,
    value_at_risk,
)
from quantlite.viz.theme import FEW_PALETTE, apply_few_theme, bullet_graph

OUT = os.path.join(os.path.dirname(__file__), "..", "docs", "images")
os.makedirs(OUT, exist_ok=True)
DPI = 150

# --- Realistic synthetic data: 10 years of daily returns with fat tails ---
np.random.seed(42)
# Calm period (7 years)
calm = merton_jump_diffusion(
    S0=100, mu=0.08, sigma=0.15, lamb=0.3, jump_mean=-0.02, jump_std=0.04,
    steps=252 * 7, rng_seed=42,
)
# Crisis period (1 year): higher vol, more jumps
crisis = merton_jump_diffusion(
    S0=calm[-1], mu=-0.10, sigma=0.45, lamb=2.0, jump_mean=-0.05, jump_std=0.08,
    steps=252, rng_seed=99,
)
# Recovery (2 years)
recovery = merton_jump_diffusion(
    S0=crisis[-1], mu=0.12, sigma=0.20, lamb=0.5, jump_mean=-0.01, jump_std=0.03,
    steps=252 * 2, rng_seed=7,
)
prices = np.concatenate([calm, crisis[1:], recovery[1:]])
returns = np.diff(prices) / prices[:-1]

# ============================================================
# 1. VaR/CVaR fan chart: historical vs parametric vs Cornish-Fisher
# ============================================================
apply_few_theme()
fig, ax = plt.subplots(figsize=(10, 5.5))

alphas = [0.10, 0.05, 0.025, 0.01]
methods = ["historical", "parametric", "cornish-fisher"]
colours = [FEW_PALETTE["primary"], FEW_PALETTE["secondary"], FEW_PALETTE["negative"]]
labels = ["Historical", "Parametric (Gaussian)", "Cornish-Fisher"]

x = np.arange(len(alphas))
width = 0.22

for i, (method, colour, label) in enumerate(zip(methods, colours, labels)):
    vars_ = [value_at_risk(returns, alpha=a, method=method) for a in alphas]
    bars = ax.bar(x + i * width, [-v for v in vars_], width * 0.9, color=colour, alpha=0.8, label=f"{label} VaR")
    # CVaR markers
    cvar_vals = [-cvar(returns, alpha=a) for a in alphas]
    ax.scatter(x + i * width, cvar_vals,
               color=colour, marker="_", s=200, linewidths=2.5, zorder=5,
               label=f"{label} CVaR" if i == 0 else None)

ax.set_xticks(x + width)
ax.set_xticklabels([f"{int(a*100)}%" for a in alphas])
ax.set_xlabel("Significance level")
ax.set_ylabel("Loss magnitude")
ax.set_title("VaR Comparison: Historical vs Parametric vs Cornish-Fisher\n",
             fontsize=13)
ax.text(0.5, 1.01, "Bars = VaR    Horizontal marks = CVaR (Expected Shortfall)",
        transform=ax.transAxes, fontsize=9, ha="center", va="bottom",
        color=FEW_PALETTE["grey_mid"], fontstyle="italic")

# Legend outside the chart area, below the subtitle
ax.legend(fontsize=8.5, loc="upper left", frameon=False, ncol=1)

# Add breathing room on the right
ax.set_xlim(-0.3, len(alphas) - 0.3 + width * 3)
ax.margins(y=0.1)

fig.tight_layout()
fig.savefig(os.path.join(OUT, "var_cvar_comparison.png"), dpi=DPI, bbox_inches="tight")
plt.close()
print("  Saved var_cvar_comparison.png")

# ============================================================
# 2. Return distribution with fat-tail overlay
# ============================================================
fig, ax = plt.subplots(figsize=(8, 5))

ax.hist(returns, bins=120, density=True, alpha=0.5,
        color=FEW_PALETTE["grey_light"], edgecolor=FEW_PALETTE["grey_mid"], label="_nolegend_")

x_range = np.linspace(returns.min(), returns.max(), 500)
mu_r, sigma_r = returns.mean(), returns.std()

# Normal overlay
normal_pdf = stats.norm.pdf(x_range, mu_r, sigma_r)
ax.plot(x_range, normal_pdf, color=FEW_PALETTE["grey_mid"], linewidth=1.5, linestyle="--", label="Gaussian")

# Fit Student-t
df_t, loc_t, scale_t = stats.t.fit(returns)
t_pdf = stats.t.pdf(x_range, df_t, loc_t, scale_t)
ax.plot(x_range, t_pdf, color=FEW_PALETTE["primary"], linewidth=2, label=f"Student-t (df={df_t:.1f})")

# Shade the tails where Gaussian underestimates
left_mask = x_range < np.percentile(returns, 2)
right_mask = x_range > np.percentile(returns, 98)
for mask in [left_mask, right_mask]:
    ax.fill_between(x_range[mask], normal_pdf[mask], t_pdf[mask],
                    where=t_pdf[mask] > normal_pdf[mask],
                    alpha=0.3, color=FEW_PALETTE["negative"])

ax.annotate("Gaussian\nunderestimates\ntail risk here",
            xy=(np.percentile(returns, 1), 0.5), fontsize=9,
            color=FEW_PALETTE["negative"],
            arrowprops=dict(arrowstyle="->", color=FEW_PALETTE["negative"]),
            xytext=(np.percentile(returns, 1) - 0.02, 8))

ax.set_xlabel("Daily return")
ax.set_ylabel("Density")
ax.set_title("Return Distribution: Fat Tails vs Gaussian")
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "return_distribution_fat_tails.png"), dpi=DPI, bbox_inches="tight")
plt.close()
print("  Saved return_distribution_fat_tails.png")

# ============================================================
# 3. Drawdown chart
# ============================================================
fig, ax = plt.subplots(figsize=(10, 4))

cum = np.cumprod(1 + returns)
roll_max = np.maximum.accumulate(cum)
drawdowns = (cum - roll_max) / roll_max

ax.fill_between(range(len(drawdowns)), drawdowns, 0,
                color=FEW_PALETTE["negative"], alpha=0.35)
ax.plot(drawdowns, color=FEW_PALETTE["negative"], linewidth=0.8)

dd_info = max_drawdown_duration(returns)
ax.annotate(
    f"Max drawdown: {dd_info.max_drawdown:.1%}\nDuration: {dd_info.duration} days",
    xy=(dd_info.end_idx, dd_info.max_drawdown),
    xytext=(dd_info.end_idx + 100, dd_info.max_drawdown * 0.5),
    fontsize=10, color=FEW_PALETTE["grey_dark"],
    arrowprops=dict(arrowstyle="->", color=FEW_PALETTE["grey_mid"]),
)

ax.set_xlabel("Trading day")
ax.set_ylabel("Drawdown")
ax.set_title("Underwater Chart")
ax.set_xlim(0, len(drawdowns) - 1)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "drawdown_chart.png"), dpi=DPI, bbox_inches="tight")
plt.close()
print("  Saved drawdown_chart.png")

# ============================================================
# 4. Bullet graphs for Sortino, Calmar, Omega
# ============================================================
fig, axes = plt.subplots(3, 1, figsize=(8, 3.5))

sortino = sortino_ratio(returns)
calmar = calmar_ratio(returns)
omega = omega_ratio(returns)

metrics = [
    ("Sortino", sortino, 1.5, [1.0, 2.0, 3.0]),
    ("Calmar", calmar, 1.0, [0.5, 1.5, 2.5]),
    ("Omega", omega, 1.5, [1.0, 2.0, 3.0]),
]

for ax, (name, val, target, ranges) in zip(axes, metrics):
    bullet_graph(ax, min(val, max(ranges)), target, ranges, label=name,
                 colour=FEW_PALETTE["primary"])
    ax.text(min(val, max(ranges) * 0.95), 0.35, f"{val:.2f}", fontsize=9,
            color=FEW_PALETTE["grey_dark"], ha="center")

fig.suptitle("Risk-Adjusted Return Metrics", fontsize=12, y=1.02,
             color=FEW_PALETTE["grey_dark"])
fig.tight_layout()
fig.savefig(os.path.join(OUT, "risk_bullet_graphs.png"), dpi=DPI, bbox_inches="tight")
plt.close()
print("  Saved risk_bullet_graphs.png")

print("Done: risk_dashboard.py")
