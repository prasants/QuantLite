#!/usr/bin/env python3
"""Copula showcase: contour plots, tail dependence, and dependency structures.

Generates three charts saved to docs/images/.
"""
from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde, rankdata

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from quantlite.data_generation import merton_jump_diffusion
from quantlite.dependency.copulas import (
    ClaytonCopula,
    FrankCopula,
    GaussianCopula,
    GumbelCopula,
    StudentTCopula,
)
from quantlite.viz.theme import FEW_PALETTE, apply_few_theme

OUT = os.path.join(os.path.dirname(__file__), "..", "docs", "images")
os.makedirs(OUT, exist_ok=True)
DPI = 150

# --- Generate correlated fat-tailed data ---
rng = np.random.default_rng(42)
# Two assets with jump diffusion
p1 = merton_jump_diffusion(S0=100, mu=0.06, sigma=0.18, lamb=0.5,
                            jump_mean=-0.03, jump_std=0.05, steps=2000, rng_seed=42)
p2 = merton_jump_diffusion(S0=100, mu=0.04, sigma=0.22, lamb=0.7,
                            jump_mean=-0.04, jump_std=0.06, steps=2000, rng_seed=43)
r1 = np.diff(p1) / p1[:-1]
r2 = np.diff(p2) / p2[:-1]
# Induce tail dependence: during large losses of asset 1, asset 2 also drops
crisis_mask = r1 < np.percentile(r1, 5)
r2[crisis_mask] = r2[crisis_mask] * 1.8 - 0.01
data = np.column_stack([r1, r2])

apply_few_theme()

# ============================================================
# 1. Side-by-side copula contour plots
# ============================================================
copulas = [
    ("Gaussian", GaussianCopula()),
    ("Student-t", StudentTCopula()),
    ("Clayton", ClaytonCopula()),
    ("Gumbel", GumbelCopula()),
    ("Frank", FrankCopula()),
]

fig, axes = plt.subplots(1, 5, figsize=(16, 3.2))

n = data.shape[0]
u_data = np.column_stack([rankdata(data[:, j], method="ordinal") / (n + 1) for j in range(2)])

grid = np.linspace(0.01, 0.99, 40)
X, Y = np.meshgrid(grid, grid)

for ax, (name, cop) in zip(axes, copulas):
    cop.fit(data)
    samples = cop.simulate(5000, rng_seed=42)

    ax.scatter(u_data[:, 0], u_data[:, 1], s=1, alpha=0.15, color=FEW_PALETTE["grey_mid"])

    try:
        kde = gaussian_kde(samples.T)
        Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
        ax.contour(X, Y, Z, levels=6, colors=FEW_PALETTE["primary"], linewidths=0.8)
    except Exception:
        pass

    td = cop.tail_dependence()
    ax.set_title(name, fontsize=10)
    ax.text(0.05, 0.05, f"L={td['lower']:.2f}", transform=ax.transAxes,
            fontsize=7, color=FEW_PALETTE["negative"])
    ax.text(0.65, 0.95, f"U={td['upper']:.2f}", transform=ax.transAxes,
            fontsize=7, color=FEW_PALETTE["positive"], va="top")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    if ax != axes[0]:
        ax.set_yticklabels([])

axes[0].set_ylabel("U2")
for ax in axes:
    ax.set_xlabel("U1")

fig.suptitle("Copula Contours: Five Families Fitted to the Same Data", fontsize=12,
             color=FEW_PALETTE["grey_dark"])
fig.tight_layout()
fig.savefig(os.path.join(OUT, "copula_contours.png"), dpi=DPI, bbox_inches="tight")
plt.close()
print("  Saved copula_contours.png")

# ============================================================
# 2. Tail dependence comparison bar chart
# ============================================================
fig, ax = plt.subplots(figsize=(8, 4))

names = []
lower_tds = []
upper_tds = []
for name, cop in copulas:
    cop.fit(data)
    td = cop.tail_dependence()
    names.append(name)
    lower_tds.append(td["lower"])
    upper_tds.append(td["upper"])

x = np.arange(len(names))
w = 0.35
ax.bar(x - w/2, lower_tds, w, color=FEW_PALETTE["negative"], alpha=0.8, label="Lower tail")
ax.bar(x + w/2, upper_tds, w, color=FEW_PALETTE["positive"], alpha=0.8, label="Upper tail")

ax.set_xticks(x)
ax.set_xticklabels(names)
ax.set_ylabel("Tail dependence coefficient")
ax.set_title("Tail Dependence by Copula Family")
ax.legend(fontsize=9)

for i, (ltd, utd) in enumerate(zip(lower_tds, upper_tds)):
    if ltd > 0.01:
        ax.text(i - w/2, ltd + 0.01, f"{ltd:.2f}", ha="center", fontsize=8, color=FEW_PALETTE["grey_dark"])
    if utd > 0.01:
        ax.text(i + w/2, utd + 0.01, f"{utd:.2f}", ha="center", fontsize=8, color=FEW_PALETTE["grey_dark"])

fig.tight_layout()
fig.savefig(os.path.join(OUT, "tail_dependence_comparison.png"), dpi=DPI, bbox_inches="tight")
plt.close()
print("  Saved tail_dependence_comparison.png")

# ============================================================
# 3. Simulated scatter: different copulas, different structures
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

# Clayton (lower tail), Gumbel (upper tail), Student-t (symmetric)
showcase = [
    ("Clayton", ClaytonCopula()),
    ("Student-t", StudentTCopula()),
    ("Gumbel", GumbelCopula()),
]

for ax, (name, cop) in zip(axes, showcase):
    cop.fit(data)
    samples = cop.simulate(2000, rng_seed=42)
    ax.scatter(samples[:, 0], samples[:, 1], s=4, alpha=0.3, color=FEW_PALETTE["primary"])

    # Highlight tail clusters
    lower_mask = (samples[:, 0] < 0.1) & (samples[:, 1] < 0.1)
    upper_mask = (samples[:, 0] > 0.9) & (samples[:, 1] > 0.9)
    ax.scatter(samples[lower_mask, 0], samples[lower_mask, 1], s=8,
               color=FEW_PALETTE["negative"], alpha=0.6, label="Joint crash")
    ax.scatter(samples[upper_mask, 0], samples[upper_mask, 1], s=8,
               color=FEW_PALETTE["positive"], alpha=0.6, label="Joint rally")

    td = cop.tail_dependence()
    ax.set_title(f"{name}\nLower={td['lower']:.2f}, Upper={td['upper']:.2f}", fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("U1")

axes[0].set_ylabel("U2")
axes[1].legend(fontsize=8, loc="upper left")
fig.suptitle("Dependency Structures: How Different Copulas Capture Tail Risk",
             fontsize=12, color=FEW_PALETTE["grey_dark"])
fig.tight_layout()
fig.savefig(os.path.join(OUT, "copula_scatter.png"), dpi=DPI, bbox_inches="tight")
plt.close()
print("  Saved copula_scatter.png")

print("Done: copula_showcase.py")
