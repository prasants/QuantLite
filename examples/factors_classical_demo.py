"""Factor Models: Classical factor analysis demo.

Generates charts for Fama-French three-factor regression results,
factor attribution breakdown, and factor summary statistics.
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from quantlite.factors.classical import (
    factor_attribution,
    factor_summary,
    fama_french_three,
)

# Palette
PRIMARY = "#4E79A7"
SECONDARY = "#F28E2B"
NEGATIVE = "#E15759"
POSITIVE = "#59A14F"
NEUTRAL = "#76B7B2"

IMG_DIR = os.path.join(os.path.dirname(__file__), "..", "docs", "images")
os.makedirs(IMG_DIR, exist_ok=True)

# Generate synthetic data
rng = np.random.RandomState(42)
n = 504  # Two years of daily data

market = rng.normal(0.0004, 0.012, n)
smb = rng.normal(0.0001, 0.006, n)
hml = rng.normal(0.00005, 0.005, n)

# Fund with known factor exposures
alpha_true = 0.0002
returns = alpha_true + 1.15 * market + 0.35 * smb - 0.25 * hml + rng.normal(0, 0.004, n)

# --- Chart 1: Factor betas with confidence intervals ---
result = fama_french_three(returns, market, smb, hml)
summary = factor_summary(returns, [market, smb, hml], ["Market", "SMB", "HML"])

factors = ["Market", "SMB", "HML"]
betas = [summary["betas"][f] for f in factors]
# Approximate 95% CI from t-stats
ci = [1.96 * abs(b / t) if abs(t) > 0.01 else 0.5
      for b, t in zip(betas, [summary["t_stats"][f] for f in factors])]

fig, ax = plt.subplots(figsize=(8, 5))
colours = [PRIMARY, SECONDARY, NEGATIVE]
bars = ax.barh(factors, betas, xerr=ci, color=colours, edgecolor="none",
               capsize=4, height=0.5)

for i, (b, name) in enumerate(zip(betas, factors)):
    t = summary["t_stats"][name]
    label = f"{b:+.3f} (t={t:.1f})"
    offset = ci[i] + 0.02
    ax.text(b + offset if b >= 0 else b - offset, i, label,
            va="center", ha="left" if b >= 0 else "right",
            fontsize=10, fontweight="bold")

ax.axvline(0, color="#999999", linewidth=0.8, linestyle="-")
ax.set_xlabel("Factor Beta")
ax.set_title(f"Fama-French Three-Factor Betas (Acme Fund)\n"
             f"R-squared = {summary['r_squared']:.3f}, "
             f"Alpha = {summary['alpha']:.5f} (t={summary['alpha_t']:.2f})",
             fontsize=11)
ax.set_facecolor("white")
fig.patch.set_facecolor("white")
ax.grid(axis="x", alpha=0.3)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "factor_betas.png"), dpi=150, facecolor="white")
plt.close()
print("Saved factor_betas.png")

# --- Chart 2: Factor attribution pie ---
attr = factor_attribution(returns, [market, smb, hml], ["Market", "SMB", "HML"])

labels = list(attr["factor_contributions"].keys()) + ["Unexplained"]
values = list(attr["factor_contributions"].values()) + [attr["unexplained"]]
# Use absolute values for sizing, track sign
abs_values = [abs(v) for v in values]
colours_pie = [PRIMARY, SECONDARY, NEGATIVE, NEUTRAL]

fig, ax = plt.subplots(figsize=(7, 5))
wedges, texts, autotexts = ax.pie(
    abs_values, labels=labels, colors=colours_pie,
    autopct=lambda pct: f"{pct:.1f}%",
    startangle=90, textprops={"fontsize": 10},
)
for t in autotexts:
    t.set_fontweight("bold")
ax.set_title("Return Attribution by Factor (Acme Fund)", fontsize=12)
fig.patch.set_facecolor("white")
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "factor_attribution.png"), dpi=150, facecolor="white")
plt.close()
print("Saved factor_attribution.png")

# --- Chart 3: Rolling R-squared ---
window = 63  # Quarterly
rolling_r2 = []
for i in range(window, n):
    s = factor_summary(
        returns[i - window:i],
        [market[i - window:i], smb[i - window:i], hml[i - window:i]],
        ["Market", "SMB", "HML"],
    )
    rolling_r2.append(s["r_squared"])

fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(range(window, n), rolling_r2, color=PRIMARY, linewidth=1.5)
ax.axhline(np.mean(rolling_r2), color=SECONDARY, linewidth=1, linestyle="--",
           label=f"Mean = {np.mean(rolling_r2):.3f}")
ax.set_xlabel("Trading Day")
ax.set_ylabel("R-squared")
ax.set_title("Rolling 63-Day Factor Model R-squared (Acme Fund)", fontsize=11)
ax.set_ylim(0, 1)
ax.legend(frameon=False)
ax.set_facecolor("white")
fig.patch.set_facecolor("white")
ax.grid(axis="y", alpha=0.3)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "factor_rolling_r2.png"), dpi=150, facecolor="white")
plt.close()
print("Saved factor_rolling_r2.png")
