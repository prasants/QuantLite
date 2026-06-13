# Financial Network Analysis

## Overview

Markets are networks, not collections of independent assets.

When you hold a portfolio of 10 assets, you are not making 10 independent bets. You are holding a position in a complex web of interconnections where shocks propagate, amplify, and cascade. The 2008 crisis showed that Lehman Brothers' failure did not stay contained; it cascaded through counterparty networks, funding channels, and correlated positions until the entire system seized.

The `quantlite.network` module provides tools to model financial markets as networks and analyse their structure:

1. **Correlation Network** builds an undirected graph from pairwise correlations
2. **Eigenvector Centrality** identifies the most systemically important nodes
3. **Cascade Simulation** models how a shock to one asset propagates through the network
4. **Community Detection** finds clusters of assets that move together
5. **Network Summary** combines all four into a single diagnostic call

Understanding network structure reveals hidden risks that correlation matrices alone cannot show: which assets are hubs, which are bridges between clusters, and where shocks will propagate fastest.

## API Reference

### `correlation_network`

```python
correlation_network(
    returns_df: pandas.DataFrame,
    threshold: float = 0.5,
) -> dict
```

Build an undirected network from a correlation matrix. Edges exist where the absolute correlation exceeds the threshold. Lower thresholds produce denser networks; higher thresholds reveal only the strongest relationships.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `returns_df` | DataFrame | Asset returns, one column per asset |
| `threshold` | float | Minimum absolute correlation for an edge (default 0.5) |

**Returns:** Dictionary with:

| Key | Description |
|-----|-------------|
| `adjacency_matrix` | NumPy array of edge weights (correlation values) |
| `edges` | List of `(asset_a, asset_b, correlation)` tuples |
| `nodes` | List of column names |

**Interpretation:**

| Threshold | Network Type |
|-----------|-------------|
| 0.3 | Dense network, captures weak relationships |
| 0.5 | Moderate, good default for most analyses |
| 0.7 | Sparse, only the strongest connections |
| 0.9 | Near-identical assets only |

**Example:**

```python
import pandas as pd
from quantlite.network import correlation_network

returns_df = pd.DataFrame({
    "SPX": spx_returns,
    "NDX": ndx_returns,
    "FTSE": ftse_returns,
    "Gold": gold_returns,
    "Bonds": bond_returns,
})

net = correlation_network(returns_df, threshold=0.4)
print(f"Nodes: {len(net['nodes'])}")
print(f"Edges: {len(net['edges'])}")
for a, b, corr in net["edges"]:
    print(f"  {a} -- {b}: {corr:.3f}")
```

![Correlation Network](images/correlation_network.png)

### `eigenvector_centrality`

```python
eigenvector_centrality(
    adjacency_matrix: array-like,
    max_iter: int = 100,
) -> numpy.ndarray
```

Compute eigenvector centrality for each node. Unlike simple degree centrality (which just counts connections), eigenvector centrality weights connections by the importance of the connected nodes. A node connected to other important nodes scores higher.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `adjacency_matrix` | array-like | Square adjacency matrix (symmetric for undirected networks) |
| `max_iter` | int | Maximum power iteration steps (default 100) |

**Returns:** Array of centrality scores, normalised to sum to 1.

**Interpretation:**

| Centrality Score | Meaning |
|-----------------|---------|
| Much higher than 1/n | Systemically important hub |
| Close to 1/n | Average connectivity |
| Much lower than 1/n | Peripheral asset, loosely connected |
| Near zero | Effectively disconnected from the network |

**Example:**

```python
from quantlite.network import correlation_network, eigenvector_centrality

net = correlation_network(returns_df, threshold=0.4)
centrality = eigenvector_centrality(net["adjacency_matrix"])

for name, score in zip(net["nodes"], centrality):
    print(f"{name:10s} centrality: {score:.4f}")
# The most central asset is the biggest single point of failure.
```

![Eigenvector Centrality](images/eigenvector_centrality.png)

### `cascade_simulation`

```python
cascade_simulation(
    adjacency_matrix: array-like,
    shock_node: int,
    shock_magnitude: float = -0.5,
    propagation_factor: float = 0.7,
    max_rounds: int = 10,
) -> dict
```

Simulate shock propagation through a network. At each round, shocked nodes transmit `shock * propagation_factor * edge_weight` to their neighbours. The simulation converges when transmitted shocks become negligible.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `adjacency_matrix` | array-like | Square adjacency matrix with edge weights |
| `shock_node` | int | Index of the initially shocked node |
| `shock_magnitude` | float | Initial shock magnitude (default -0.5, i.e., 50% loss) |
| `propagation_factor` | float | Fraction of shock transmitted per edge (default 0.7) |
| `max_rounds` | int | Maximum simulation rounds (default 10) |

**Returns:** Dictionary with:

| Key | Description |
|-----|-------------|
| `per_round` | List of arrays, one per round, showing new shocks at each step |
| `final_state` | Array of cumulative impact per node |

**Interpretation:**

| Final State Value | Meaning |
|-------------------|---------|
| Close to shock_magnitude | Directly hit or first-order contagion |
| Moderate negative | Second or third-order contagion |
| Near zero | Well insulated from the shock source |
| Zero | Disconnected from the shocked node |

**Example:**

```python
from quantlite.network import correlation_network, cascade_simulation

net = correlation_network(returns_df, threshold=0.4)
adj = net["adjacency_matrix"]

# Shock the most central node (index 0 = SPX)
cascade = cascade_simulation(
    adj,
    shock_node=0,
    shock_magnitude=-0.50,
    propagation_factor=0.7,
)

for name, impact in zip(net["nodes"], cascade["final_state"]):
    print(f"{name:10s} cumulative impact: {impact:+.4f}")
print(f"Propagation rounds: {len(cascade['per_round'])}")
```

![Cascade Simulation](images/cascade_simulation.png)

### `community_detection`

```python
community_detection(
    adjacency_matrix: array-like,
    n_communities: int | None = None,
) -> numpy.ndarray
```

Detect communities using spectral clustering on the graph Laplacian. Automatically determines the number of communities via the eigengap heuristic if `n_communities` is not specified.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `adjacency_matrix` | array-like | Square adjacency matrix (symmetric) |
| `n_communities` | int or None | Number of communities; if None, uses eigengap heuristic |

**Returns:** Array of integer community labels, one per node.

**Example:**

```python
from quantlite.network import correlation_network, community_detection

net = correlation_network(returns_df, threshold=0.4)
communities = community_detection(net["adjacency_matrix"])

for name, comm in zip(net["nodes"], communities):
    print(f"{name:10s} community: {comm}")
# Assets in the same community are more tightly connected to each
# other than to assets in other communities.
```

### `network_summary`

```python
network_summary(
    returns_df: pandas.DataFrame,
    threshold: float = 0.5,
) -> dict
```

One-call summary that builds the correlation network, computes eigenvector centrality, and detects communities. Use this as a starting point for network analysis.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `returns_df` | DataFrame | Asset returns, one column per asset |
| `threshold` | float | Correlation threshold for edge inclusion (default 0.5) |

**Returns:** Dictionary with:

| Key | Description |
|-----|-------------|
| `network` | Full `correlation_network` result |
| `centrality` | Eigenvector centrality scores |
| `communities` | Community labels |
| `nodes` | List of column names |

**Example:**

```python
from quantlite.network import network_summary

summary = network_summary(returns_df, threshold=0.4)

print("Asset          Centrality  Community")
print("-" * 40)
for i, name in enumerate(summary["nodes"]):
    cent = summary["centrality"][i]
    comm = summary["communities"][i]
    print(f"{name:15s} {cent:.4f}     {comm}")

# Use the adjacency matrix for cascade simulation
adj = summary["network"]["adjacency_matrix"]
```

## Practical Guidance

### Choosing the Right Threshold

The correlation threshold is the single most important parameter. Too low and everything connects to everything (useless). Too high and you miss important relationships.

**Recommended approach:** Start at 0.5 and examine the network density. If every node has edges to every other node, increase the threshold. If the network is mostly disconnected, decrease it. A good network typically has 2 to 4 edges per node on average.

### Cascade Simulation Parameters

| Parameter | Conservative | Moderate | Aggressive |
|-----------|-------------|----------|-----------|
| `propagation_factor` | 0.3 | 0.5 | 0.7 |
| `shock_magnitude` | -0.10 | -0.25 | -0.50 |

Use conservative parameters for routine stress testing and aggressive parameters for worst-case analysis.

### Data Requirements

- **Minimum observations:** 250 for stable correlation estimates
- **Frequency:** Daily returns recommended; weekly returns smooth out important short-term dynamics
- **Universe size:** Works well up to ~50 assets; beyond that, consider sector-level aggregation
- **Stationarity:** Use returns, not prices. Consider rolling windows for time-varying network analysis.

### Limitations

- Correlation networks capture linear relationships only; non-linear dependence (e.g., tail dependence) requires copula-based approaches
- Community detection via spectral clustering may produce unstable results when communities are not well separated
- Cascade simulation uses a simplified linear propagation model; real contagion involves non-linear feedback loops, margin calls, and liquidity spirals
- Eigenvector centrality can behave unexpectedly on disconnected graphs
