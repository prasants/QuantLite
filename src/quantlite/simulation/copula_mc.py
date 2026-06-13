"""Copula-based multivariate Monte Carlo simulation.

Generates joint return scenarios that preserve correlation structure
while allowing fat-tailed marginals. Supports Gaussian and Student-t
copulas, stressed correlations, and joint tail probability estimation.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

__all__ = [
    "gaussian_copula_mc",
    "t_copula_mc",
    "stress_correlation_mc",
    "joint_tail_probability",
]


def _to_array(x: np.ndarray | list[float]) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    return arr[~np.isnan(arr)]


def _ensure_positive_definite(matrix: np.ndarray) -> np.ndarray:
    """Ensure a correlation matrix is positive semi-definite via eigenvalue clipping."""
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    eigenvalues = np.maximum(eigenvalues, 1e-8)
    fixed = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    # Re-normalise to correlation matrix
    d = np.sqrt(np.diag(fixed))
    fixed = fixed / np.outer(d, d)
    np.fill_diagonal(fixed, 1.0)
    return fixed


def _empirical_cdf(data: np.ndarray) -> np.ndarray:
    """Compute empirical CDF values (rank-based, in (0,1))."""
    from scipy.stats import rankdata
    n = len(data)
    return rankdata(data, method="ordinal") / (n + 1)


def _inverse_empirical(sorted_data: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Map uniform samples back to data space via empirical quantile function."""
    n = len(sorted_data)
    indices = np.clip((u * n).astype(int), 0, n - 1)
    return sorted_data[indices]


def gaussian_copula_mc(
    marginals: list[np.ndarray] | np.ndarray,
    correlation_matrix: np.ndarray,
    n_scenarios: int = 10000,
    seed: int = 42,
) -> np.ndarray:
    """Multivariate simulation using a Gaussian copula with empirical marginals.

    Generates correlated uniform samples via a Gaussian copula, then
    maps them back to the empirical marginal distributions. This
    preserves the correlation structure while retaining the fat tails
    of each individual asset's return distribution.

    Args:
        marginals: List of 1-D arrays, one per asset, containing
            historical returns. Each array defines the marginal
            distribution for that asset.
        correlation_matrix: Square correlation matrix of shape
            ``(n_assets, n_assets)``.
        n_scenarios: Number of joint scenarios to generate.
        seed: Random seed.

    Returns:
        Array of shape ``(n_scenarios, n_assets)`` with simulated returns.
    """
    rng = np.random.default_rng(seed)
    n_assets = len(marginals)
    corr = _ensure_positive_definite(np.asarray(correlation_matrix, dtype=float))

    # Generate correlated normal samples
    L = np.linalg.cholesky(corr)
    z = rng.standard_normal((n_scenarios, n_assets))
    correlated_z = z @ L.T

    # Transform to uniform via normal CDF
    u = stats.norm.cdf(correlated_z)

    # Map to empirical marginals
    result = np.empty((n_scenarios, n_assets))
    for j in range(n_assets):
        sorted_m = np.sort(_to_array(marginals[j]))
        result[:, j] = _inverse_empirical(sorted_m, u[:, j])

    return result


def t_copula_mc(
    marginals: list[np.ndarray] | np.ndarray,
    correlation_matrix: np.ndarray,
    df: int = 4,
    n_scenarios: int = 10000,
    seed: int = 42,
) -> np.ndarray:
    """Multivariate simulation using a Student-t copula.

    The t-copula generates tail dependence: extreme events across
    assets are more likely to co-occur than under a Gaussian copula.
    Lower degrees of freedom produce stronger tail dependence.

    Args:
        marginals: List of 1-D arrays, one per asset.
        correlation_matrix: Correlation matrix.
        df: Degrees of freedom for the t-copula (default 4).
        n_scenarios: Number of scenarios.
        seed: Random seed.

    Returns:
        Array of shape ``(n_scenarios, n_assets)``.
    """
    rng = np.random.default_rng(seed)
    n_assets = len(marginals)
    corr = _ensure_positive_definite(np.asarray(correlation_matrix, dtype=float))

    L = np.linalg.cholesky(corr)
    z = rng.standard_normal((n_scenarios, n_assets))
    correlated_z = z @ L.T

    # Scale by chi-squared for t-distribution
    chi2 = rng.chisquare(df, size=n_scenarios)
    scaling = np.sqrt(df / chi2)
    t_samples = correlated_z * scaling[:, np.newaxis]

    # Transform to uniform via t CDF
    u = stats.t.cdf(t_samples, df=df)

    result = np.empty((n_scenarios, n_assets))
    for j in range(n_assets):
        sorted_m = np.sort(_to_array(marginals[j]))
        result[:, j] = _inverse_empirical(sorted_m, u[:, j])

    return result


def stress_correlation_mc(
    marginals: list[np.ndarray] | np.ndarray,
    correlation_matrix: np.ndarray,
    stress_factor: float = 1.5,
    n_scenarios: int = 10000,
    seed: int = 42,
) -> np.ndarray:
    """Simulate under stressed correlations.

    In crises, correlations tend to move towards 1.0. This function
    scales off-diagonal correlations by ``stress_factor``, capping
    at 1.0, then simulates using a Gaussian copula.

    Args:
        marginals: List of 1-D arrays, one per asset.
        correlation_matrix: Base correlation matrix.
        stress_factor: Multiplier for off-diagonal correlations
            (default 1.5). Values > 1 increase correlation.
        n_scenarios: Number of scenarios.
        seed: Random seed.

    Returns:
        Array of shape ``(n_scenarios, n_assets)``.
    """
    corr = np.asarray(correlation_matrix, dtype=float).copy()
    n = corr.shape[0]

    # Stress the off-diagonal elements
    for i in range(n):
        for j in range(n):
            if i != j:
                corr[i, j] = np.clip(corr[i, j] * stress_factor, -1.0, 1.0)

    corr = _ensure_positive_definite(corr)
    return gaussian_copula_mc(marginals, corr, n_scenarios=n_scenarios, seed=seed)


def joint_tail_probability(
    simulated_returns: np.ndarray,
    thresholds: list[float] | np.ndarray,
) -> dict[str, float]:
    """Compute joint tail probabilities from simulation output.

    Estimates the probability of multiple assets simultaneously
    breaching their respective thresholds (all falling below
    their threshold in the same scenario).

    Args:
        simulated_returns: Array of shape ``(n_scenarios, n_assets)``
            from a copula simulation.
        thresholds: List of return thresholds, one per asset
            (negative for losses, e.g. ``[-0.05, -0.03]``).

    Returns:
        Dictionary with:

        - ``"joint_probability"``: probability all assets breach
          simultaneously
        - ``"marginal_probabilities"``: list of individual breach
          probabilities
        - ``"conditional_probabilities"``: probability of joint breach
          given each asset breaches
        - ``"n_joint_breaches"``: count of joint breach scenarios
        - ``"n_scenarios"``: total scenarios
    """
    thresholds_arr = np.asarray(thresholds, dtype=float)
    n_scenarios = simulated_returns.shape[0]
    n_assets = simulated_returns.shape[1]

    # Individual breaches
    breaches = simulated_returns < thresholds_arr[np.newaxis, :]
    marginal_counts = breaches.sum(axis=0)
    marginal_probs = [float(c / n_scenarios) for c in marginal_counts]

    # Joint breach: all assets breach simultaneously
    joint_breach = np.all(breaches, axis=1)
    joint_count = int(joint_breach.sum())
    joint_prob = joint_count / n_scenarios

    # Conditional probabilities
    conditional_probs = []
    for j in range(n_assets):
        if marginal_counts[j] > 0:
            conditional_probs.append(float(joint_count / marginal_counts[j]))
        else:
            conditional_probs.append(0.0)

    return {
        "joint_probability": joint_prob,
        "marginal_probabilities": marginal_probs,
        "conditional_probabilities": conditional_probs,
        "n_joint_breaches": joint_count,
        "n_scenarios": n_scenarios,
    }
