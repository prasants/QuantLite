"""Copula Monte Carlo demonstration: Gaussian vs t-copula and stressed correlations.

Generates three charts for the v0.9 documentation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from quantlite.simulation.copula_mc import (
    gaussian_copula_mc,
    joint_tail_probability,
    stress_correlation_mc,
    t_copula_mc,
)
from quantlite.viz.theme import FEW_PALETTE, apply_few_theme

apply_few_theme()

OUT = Path(__file__).resolve().parent.parent / "docs" / "images"
OUT.mkdir(parents=True, exist_ok=True)

# Synthetic marginals: Fund A and Fund B
rng = np.random.default_rng(42)
fund_a = rng.standard_t(4, 1000) * 0.012
fund_b = 0.5 * fund_a + rng.standard_t(4, 1000) * 0.010
marginals = [fund_a, fund_b]
corr = np.array([[1.0, 0.5], [0.5, 1.0]])

# ---------------------------------------------------------------------------
# Chart 1: Gaussian vs t-copula scatter
# ---------------------------------------------------------------------------
gauss_sim = gaussian_copula_mc(marginals, corr, n_scenarios=5000, seed=42)
t_sim = t_copula_mc(marginals, corr, df=3, n_scenarios=5000, seed=42)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.scatter(gauss_sim[:, 0], gauss_sim[:, 1], alpha=0.15, s=4,
            color=FEW_PALETTE["primary"])
ax1.set_xlabel("Fund A return")
ax1.set_ylabel("Fund B return")
ax1.set_title("Gaussian Copula")
ax1.set_xlim(-0.08, 0.08)
ax1.set_ylim(-0.08, 0.08)

ax2.scatter(t_sim[:, 0], t_sim[:, 1], alpha=0.15, s=4,
            color=FEW_PALETTE["negative"])
ax2.set_xlabel("Fund A return")
ax2.set_ylabel("Fund B return")
ax2.set_title("Student-t Copula (df=3)")
ax2.set_xlim(-0.08, 0.08)
ax2.set_ylim(-0.08, 0.08)

fig.suptitle("Gaussian vs t-Copula: Joint Return Scenarios", fontsize=13)
fig.tight_layout()
fig.savefig(OUT / "copula_comparison.png")
plt.close(fig)
print(f"Saved {OUT / 'copula_comparison.png'}")

# ---------------------------------------------------------------------------
# Chart 2: Stressed correlations
# ---------------------------------------------------------------------------
normal_sim = gaussian_copula_mc(marginals, corr, n_scenarios=5000, seed=42)
stressed_sim = stress_correlation_mc(marginals, corr, stress_factor=1.8, n_scenarios=5000, seed=42)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.scatter(normal_sim[:, 0], normal_sim[:, 1], alpha=0.15, s=4,
            color=FEW_PALETTE["primary"])
ax1.set_xlabel("Fund A return")
ax1.set_ylabel("Fund B return")
ax1.set_title("Normal Correlations")
rho_n = np.corrcoef(normal_sim[:, 0], normal_sim[:, 1])[0, 1]
ax1.text(0.05, 0.95, f"rho = {rho_n:.2f}", transform=ax1.transAxes,
         fontsize=11, verticalalignment="top")

ax2.scatter(stressed_sim[:, 0], stressed_sim[:, 1], alpha=0.15, s=4,
            color=FEW_PALETTE["negative"])
ax2.set_xlabel("Fund A return")
ax2.set_ylabel("Fund B return")
ax2.set_title("Stressed Correlations (1.8x)")
rho_s = np.corrcoef(stressed_sim[:, 0], stressed_sim[:, 1])[0, 1]
ax2.text(0.05, 0.95, f"rho = {rho_s:.2f}", transform=ax2.transAxes,
         fontsize=11, verticalalignment="top")

fig.suptitle("Correlation Stress: Diversification Failure", fontsize=13)
fig.tight_layout()
fig.savefig(OUT / "stressed_correlations.png")
plt.close(fig)
print(f"Saved {OUT / 'stressed_correlations.png'}")

# ---------------------------------------------------------------------------
# Chart 3: Joint tail probability across thresholds
# ---------------------------------------------------------------------------
thresholds = np.linspace(-0.01, -0.06, 20)
joint_probs = []
for thresh in thresholds:
    jtp = joint_tail_probability(t_sim, [thresh, thresh])
    joint_probs.append(jtp["joint_probability"])

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(thresholds * 100, joint_probs, color=FEW_PALETTE["negative"], linewidth=2)
ax.set_xlabel("Threshold (%, both assets)")
ax.set_ylabel("Joint breach probability")
ax.set_title("Joint Tail Probability: Both Funds Breaching Simultaneously")
ax.fill_between(thresholds * 100, 0, joint_probs, alpha=0.15,
                color=FEW_PALETTE["negative"])
fig.savefig(OUT / "joint_tail_probability.png")
plt.close(fig)
print(f"Saved {OUT / 'joint_tail_probability.png'}")
