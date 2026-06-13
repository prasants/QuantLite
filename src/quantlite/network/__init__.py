"""Network risk analysis for financial systems.

Provides tools for building correlation networks, computing centrality
measures, simulating shock cascades, and detecting communities of
interconnected risk. Uses numpy and scipy for all graph operations;
no networkx dependency required.
"""

import numpy as np
from scipy import linalg


def correlation_network(returns_df, threshold=0.5):
    """Build an undirected network from a correlation matrix.

    Edges exist where the absolute correlation exceeds the threshold.

    Parameters
    ----------
    returns_df : pandas.DataFrame
        DataFrame of asset returns, one column per asset.
    threshold : float, optional
        Minimum absolute correlation for an edge (default 0.5).

    Returns
    -------
    dict
        Dictionary with 'adjacency_matrix' (numpy array),
        'edges' (list of (a, b, correlation) tuples), and
        'nodes' (list of column names).
    """
    cols = list(returns_df.columns)
    corr = np.corrcoef(returns_df.values.T)
    n = len(cols)
    adj = np.zeros((n, n))
    edges = []

    for i in range(n):
        for j in range(i + 1, n):
            c = corr[i, j]
            if abs(c) > threshold:
                adj[i, j] = c
                adj[j, i] = c
                edges.append((cols[i], cols[j], float(c)))

    return {
        "adjacency_matrix": adj,
        "edges": edges,
        "nodes": cols,
    }


def eigenvector_centrality(adjacency_matrix, max_iter=100):
    """Compute eigenvector centrality for each node.

    Nodes connected to other important nodes score higher. Uses
    power iteration on the absolute values of the adjacency matrix.

    Parameters
    ----------
    adjacency_matrix : array-like
        Square adjacency matrix (symmetric for undirected networks).
    max_iter : int, optional
        Maximum number of iterations (default 100).

    Returns
    -------
    numpy.ndarray
        Array of centrality scores, normalised to sum to 1.
    """
    A = np.abs(np.asarray(adjacency_matrix, dtype=float))
    n = A.shape[0]
    if n == 0:
        return np.array([])

    # Power iteration
    x = np.ones(n) / n
    for _ in range(max_iter):
        x_new = A @ x
        norm = np.linalg.norm(x_new)
        if norm < 1e-15:
            return np.ones(n) / n
        x_new = x_new / norm
        if np.allclose(x, x_new, atol=1e-10):
            break
        x = x_new

    # Normalise to sum to 1
    total = x.sum()
    if total > 0:
        x = x / total
    return x


def cascade_simulation(
    adjacency_matrix,
    shock_node,
    shock_magnitude=-0.5,
    propagation_factor=0.7,
    max_rounds=10,
):
    """Simulate shock propagation through a network.

    At each round, shocked nodes transmit
    (shock * propagation_factor * edge_weight) to their neighbours.

    Parameters
    ----------
    adjacency_matrix : array-like
        Square adjacency matrix with edge weights.
    shock_node : int
        Index of the initially shocked node.
    shock_magnitude : float, optional
        Initial shock magnitude (default -0.5).
    propagation_factor : float, optional
        Fraction of shock transmitted per edge (default 0.7).
    max_rounds : int, optional
        Maximum simulation rounds (default 10).

    Returns
    -------
    dict
        Dictionary with 'per_round' (list of arrays, one per round)
        and 'final_state' (array of cumulative impact per node).
    """
    A = np.abs(np.asarray(adjacency_matrix, dtype=float))
    n = A.shape[0]
    state = np.zeros(n)
    state[shock_node] = shock_magnitude
    cumulative = state.copy()
    per_round = [state.copy()]

    for _ in range(max_rounds - 1):
        new_shocks = np.zeros(n)
        for i in range(n):
            if abs(state[i]) > 1e-15:
                for j in range(n):
                    if i != j and A[i, j] > 0:
                        transmitted = state[i] * propagation_factor * A[i, j]
                        new_shocks[j] += transmitted

        if np.max(np.abs(new_shocks)) < 1e-12:
            break

        state = new_shocks
        cumulative += state
        per_round.append(state.copy())

    return {
        "per_round": per_round,
        "final_state": cumulative,
    }


def community_detection(adjacency_matrix, n_communities=None):
    """Detect communities using spectral clustering on the graph Laplacian.

    Parameters
    ----------
    adjacency_matrix : array-like
        Square adjacency matrix (symmetric).
    n_communities : int or None, optional
        Number of communities. If None, uses the eigengap heuristic
        (default None).

    Returns
    -------
    numpy.ndarray
        Array of integer community labels, one per node.
    """
    A = np.abs(np.asarray(adjacency_matrix, dtype=float))
    n = A.shape[0]

    if n <= 1:
        return np.zeros(n, dtype=int)

    # Graph Laplacian
    D = np.diag(A.sum(axis=1))
    L = D - A

    # Eigendecomposition
    eigenvalues, eigenvectors = linalg.eigh(L)

    # Eigengap heuristic if n_communities not specified
    if n_communities is None:
        # Count near-zero eigenvalues (disconnected components)
        near_zero = int(np.sum(eigenvalues < 1e-8))
        if near_zero >= 2:
            n_communities = near_zero
        else:
            gaps = np.diff(eigenvalues[:min(n, 10)])
            n_communities = int(np.argmax(gaps[1:]) + 2) if len(gaps) > 1 else 1

    n_communities = min(n_communities, n)

    if n_communities <= 1:
        return np.zeros(n, dtype=int)

    # Use first k eigenvectors (after the trivial one)
    V = eigenvectors[:, :n_communities]

    # Simple k-means-style assignment
    # Normalise rows
    norms = np.linalg.norm(V, axis=1, keepdims=True)
    norms = np.where(norms < 1e-15, 1.0, norms)
    V_norm = V / norms

    # K-means with numpy (no sklearn dependency)
    labels = _kmeans(V_norm, n_communities)
    return labels


def _kmeans(X, k, max_iter=50):
    """Simple k-means clustering.

    Parameters
    ----------
    X : numpy.ndarray
        Data matrix (n_samples, n_features).
    k : int
        Number of clusters.
    max_iter : int, optional
        Maximum iterations (default 50).

    Returns
    -------
    numpy.ndarray
        Cluster labels.
    """
    n = X.shape[0]
    # K-means++ initialisation
    rng = np.random.RandomState(42)
    idx = [rng.randint(n)]
    for _ in range(1, min(k, n)):
        dists = np.min([np.sum((X - X[c]) ** 2, axis=1) for c in idx], axis=0)
        total = dists.sum()
        if total < 1e-15:
            remaining = [i for i in range(n) if i not in idx]
            idx.append(rng.choice(remaining))
        else:
            probs = dists / total
            idx.append(rng.choice(n, p=probs))
    centres = X[idx].copy()

    labels = np.zeros(n, dtype=int)
    for _ in range(max_iter):
        # Assign
        dists = np.array([np.sum((X - c) ** 2, axis=1) for c in centres])
        new_labels = np.argmin(dists, axis=0)

        if np.array_equal(new_labels, labels):
            break
        labels = new_labels

        # Update
        for j in range(k):
            mask = labels == j
            if mask.sum() > 0:
                centres[j] = X[mask].mean(axis=0)

    return labels


def network_summary(returns_df, threshold=0.5):
    """One-call summary: builds network, computes centrality, detects communities.

    Parameters
    ----------
    returns_df : pandas.DataFrame
        DataFrame of asset returns, one column per asset.
    threshold : float, optional
        Correlation threshold for edge inclusion (default 0.5).

    Returns
    -------
    dict
        Dictionary with 'network' (correlation_network result),
        'centrality' (eigenvector centrality scores),
        'communities' (community labels), and 'nodes' (column names).
    """
    net = correlation_network(returns_df, threshold=threshold)
    adj = net["adjacency_matrix"]
    cent = eigenvector_centrality(adj)
    comm = community_detection(adj)

    return {
        "network": net,
        "centrality": cent,
        "communities": comm,
        "nodes": net["nodes"],
    }
