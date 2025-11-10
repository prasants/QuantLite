"""Network analysis demonstration: correlation network, cascade simulation, and communities.

Generates three charts for the v0.6 documentation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from quantlite.network import (
    cascade_simulation,
    network_summary,
)
from quantlite.viz.theme import FEW_PALETTE, apply_few_theme, direct_label

apply_few_theme()

OUT = Path(__file__).resolve().parent.parent / "docs" / "images"
OUT.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(42)
n = 1000

# Build correlated asset universe
market = rng.normal(0.0003, 0.012, n)
assets = {
    "SPX": 0.9 * market + rng.normal(0, 0.005, n),
    "NDX": 0.85 * market + rng.normal(0, 0.008, n),
    "FTSE": 0.7 * market + rng.normal(0, 0.007, n),
    "DAX": 0.65 * market + rng.normal(0, 0.008, n),
    "NKY": 0.5 * market + rng.normal(0, 0.010, n),
    "Gold": -0.15 * market + rng.normal(0.0001, 0.008, n),
    "Bonds": -0.3 * market + rng.normal(0.0002, 0.004, n),
    "Oil": 0.3 * market + rng.normal(0, 0.018, n),
}
returns_df = pd.DataFrame(assets)

# ---------------------------------------------------------------------------
# Chart 1: Correlation network with centrality-scaled nodes
# ---------------------------------------------------------------------------
summary = network_summary(returns_df, threshold=0.3)
net = summary["network"]
adj = net["adjacency_matrix"]
centrality = summary["centrality"]
communities = summary["communities"]
nodes = net["nodes"]
n_nodes = len(nodes)

# Spring layout (simple force-directed)
np.random.seed(42)
pos = np.random.randn(n_nodes, 2)
for _ in range(200):
    for i in range(n_nodes):
        force = np.zeros(2)
        for j in range(n_nodes):
            if i == j:
                continue
            diff = pos[i] - pos[j]
            dist = max(np.linalg.norm(diff), 0.01)
            # Repulsion
            force += 0.5 * diff / (dist ** 2)
            # Attraction if connected
            if abs(adj[i, j]) > 0:
                force -= 0.1 * diff * abs(adj[i, j])
        pos[i] += 0.05 * force

comm_colours = [
    FEW_PALETTE["primary"],
    FEW_PALETTE["secondary"],
    FEW_PALETTE["positive"],
    FEW_PALETTE["neutral"],
    FEW_PALETTE["negative"],
]

fig, ax = plt.subplots(figsize=(10, 8))

# Draw edges
for i in range(n_nodes):
    for j in range(i + 1, n_nodes):
        if abs(adj[i, j]) > 0:
            weight = abs(adj[i, j])
            ax.plot(
                [pos[i, 0], pos[j, 0]],
                [pos[i, 1], pos[j, 1]],
                color=FEW_PALETTE["grey_mid"],
                linewidth=weight * 2,
                alpha=0.4,
            )

# Draw nodes
for i in range(n_nodes):
    size = 300 + centrality[i] * 5000
    colour = comm_colours[int(communities[i]) % len(comm_colours)]
    ax.scatter(pos[i, 0], pos[i, 1], s=size, c=colour, zorder=5, edgecolors="white", linewidth=1.5)
    ax.annotate(
        nodes[i],
        (pos[i, 0], pos[i, 1]),
        fontsize=9,
        ha="center",
        va="center",
        fontweight="bold",
        zorder=6,
    )

ax.set_title("Correlation Network: Node Size Shows Systemic Importance")
ax.set_aspect("equal")
ax.axis("off")
fig.savefig(OUT / "correlation_network.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved {OUT / 'correlation_network.png'}")

# ---------------------------------------------------------------------------
# Chart 2: Cascade simulation from most central node
# ---------------------------------------------------------------------------
most_central = int(np.argmax(centrality))
cascade = cascade_simulation(
    adj,
    shock_node=most_central,
    shock_magnitude=-0.50,
    propagation_factor=0.7,
    max_rounds=8,
)

rounds = cascade["per_round"]
n_rounds = len(rounds)

fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(n_rounds)

# Plot cumulative impact per node
cumulative = np.zeros((n_rounds, n_nodes))
cumulative[0] = rounds[0]
for r in range(1, n_rounds):
    cumulative[r] = cumulative[r - 1] + rounds[r]

for i in range(n_nodes):
    colour = comm_colours[int(communities[i]) % len(comm_colours)]
    vals = cumulative[:, i] * 100
    ax.plot(x, vals, marker="o", markersize=4, color=colour, linewidth=1.5)
    # Direct label at final point
    direct_label(
        ax,
        x[-1] + 0.15,
        vals[-1],
        nodes[i],
        ha="left",
        va="center",
        fontsize=8,
        colour=colour,
    )

ax.set_xlabel("Propagation round")
ax.set_ylabel("Cumulative impact (%)")
ax.set_title(f"Shock Cascade: {nodes[most_central]} Fails, Who Suffers?")
ax.set_xticks(x)
ax.set_xlim(-0.2, n_rounds + 1.5)
fig.savefig(OUT / "cascade_simulation.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved {OUT / 'cascade_simulation.png'}")

# ---------------------------------------------------------------------------
# Chart 3: Eigenvector centrality ranking
# ---------------------------------------------------------------------------
sorted_idx = np.argsort(centrality)[::-1]
sorted_names = [nodes[i] for i in sorted_idx]
sorted_centrality = [centrality[i] for i in sorted_idx]
sorted_communities = [int(communities[i]) for i in sorted_idx]

fig, ax = plt.subplots(figsize=(10, 5))
colours = [comm_colours[c % len(comm_colours)] for c in sorted_communities]
bars = ax.barh(range(len(sorted_names)), sorted_centrality, color=colours)

for i, bar in enumerate(bars):
    direct_label(
        ax,
        bar.get_width() + 0.002,
        i,
        f"{sorted_centrality[i]:.3f}",
        ha="left",
        va="center",
        fontsize=9,
    )

ax.set_yticks(range(len(sorted_names)))
ax.set_yticklabels(sorted_names)
ax.set_xlabel("Eigenvector centrality")
ax.set_title("Systemic Importance: Which Assets Are Most Connected?")
ax.invert_yaxis()
fig.savefig(OUT / "eigenvector_centrality.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved {OUT / 'eigenvector_centrality.png'}")
