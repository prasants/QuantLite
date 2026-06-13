"""Hierarchical clustering and risk parity allocation.

Implements Lopez de Prado's Hierarchical Risk Parity (HRP) method,
which uses hierarchical clustering on the correlation matrix to
produce diversified, stable portfolio weights without requiring
covariance matrix inversion.
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage

__all__ = [
    "correlation_distance",
    "hierarchical_cluster",
    "quasi_diagonalise",
    "hrp_weights",
]


def correlation_distance(corr_matrix: pd.DataFrame | np.ndarray) -> np.ndarray:
    """Convert a correlation matrix to a distance matrix.

    Uses the standard transformation: d(i,j) = sqrt(0.5 * (1 - rho(i,j))).

    Args:
        corr_matrix: Correlation matrix (square, symmetric).

    Returns:
        Distance matrix as a NumPy array.
    """
    corr = np.asarray(corr_matrix, dtype=float)
    return np.sqrt(0.5 * (1 - corr))


def hierarchical_cluster(
    corr_matrix: pd.DataFrame | np.ndarray,
    method: str = "single",
) -> np.ndarray:
    """Perform hierarchical clustering on a correlation matrix.

    Args:
        corr_matrix: Correlation matrix (square, symmetric).
        method: Linkage method (``"single"``, ``"complete"``,
            ``"average"``, or ``"ward"``).

    Returns:
        Linkage matrix from ``scipy.cluster.hierarchy.linkage``.
    """
    dist = correlation_distance(corr_matrix)
    # Convert distance matrix to condensed form
    n = dist.shape[0]
    condensed = dist[np.triu_indices(n, k=1)]
    return linkage(condensed, method=method)


def quasi_diagonalise(
    link: np.ndarray,
    corr_matrix: pd.DataFrame | np.ndarray,
) -> pd.DataFrame | np.ndarray:
    """Reorder correlation matrix by hierarchical cluster leaves.

    Places highly correlated assets adjacent to each other, producing
    a quasi-diagonal structure that reveals cluster boundaries.

    Args:
        link: Linkage matrix from ``hierarchical_cluster``.
        corr_matrix: Original correlation matrix.

    Returns:
        Reordered correlation matrix (same type as input).
    """
    order = list(leaves_list(link).astype(int))

    if isinstance(corr_matrix, pd.DataFrame):
        cols = [corr_matrix.columns[i] for i in order]
        return corr_matrix.loc[cols, cols]

    arr = np.asarray(corr_matrix)
    return arr[np.ix_(order, order)]


def _get_cluster_var(
    cov: pd.DataFrame,
    cluster_items: list[int],
) -> float:
    """Compute the inverse-variance portfolio variance for a cluster.

    Args:
        cov: Covariance matrix.
        cluster_items: Indices of assets in the cluster.

    Returns:
        Portfolio variance of the inverse-variance sub-portfolio.
    """
    sub_cov = cov.iloc[cluster_items, cluster_items]
    ivp = 1.0 / np.diag(sub_cov.values)
    ivp = ivp / ivp.sum()
    return float(ivp @ sub_cov.values @ ivp)


def _recursive_bisection(
    cov: pd.DataFrame,
    sorted_idx: list[int],
    weights: dict[int, float],
) -> None:
    """Recursively bisect the sorted index and allocate weights.

    This is the core of the HRP algorithm: at each step, split the
    cluster into two halves, compute each half's variance, and allocate
    weight inversely proportional to variance.

    Args:
        cov: Full covariance matrix.
        sorted_idx: Current list of asset indices (cluster-sorted).
        weights: Mutable dict accumulating final weights.
    """
    if len(sorted_idx) == 1:
        weights[sorted_idx[0]] = 1.0
        return

    mid = len(sorted_idx) // 2
    left = sorted_idx[:mid]
    right = sorted_idx[mid:]

    var_left = _get_cluster_var(cov, left)
    var_right = _get_cluster_var(cov, right)

    alpha = 1 - var_left / (var_left + var_right)

    _recursive_bisection(cov, left, weights)
    _recursive_bisection(cov, right, weights)

    # Scale weights
    for idx in left:
        weights[idx] *= alpha
    for idx in right:
        weights[idx] *= (1 - alpha)


def hrp_weights(returns_df: pd.DataFrame) -> OrderedDict[str, float]:
    """Compute Hierarchical Risk Parity portfolio weights.

    Implements the full HRP algorithm from Lopez de Prado (2016):
    1. Compute correlation and covariance matrices.
    2. Hierarchically cluster assets by correlation distance.
    3. Quasi-diagonalise the covariance matrix.
    4. Recursively bisect, allocating weight inversely proportional
       to cluster variance.

    Args:
        returns_df: DataFrame of asset returns (assets as columns).

    Returns:
        OrderedDict mapping asset names to weights (sum to 1, all positive).
    """
    corr = returns_df.corr()
    cov = returns_df.cov()

    link = hierarchical_cluster(corr, method="single")
    sorted_order = list(leaves_list(link).astype(int))

    weights: dict[int, float] = {}
    _recursive_bisection(cov, sorted_order, weights)

    # Map indices back to column names and normalise
    total = sum(weights.values())
    result = OrderedDict()
    for idx in sorted(weights.keys()):
        name = returns_df.columns[idx]
        result[name] = weights[idx] / total

    return result
