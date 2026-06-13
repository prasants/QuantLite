"""Stochastic process generators for asset price simulation.

All functions use NumPy's modern Generator API (``default_rng``) for
thread-safe, reproducible random number generation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.random import Generator

__all__ = [
    "geometric_brownian_motion",
    "correlated_gbm",
    "ornstein_uhlenbeck",
    "merton_jump_diffusion",
]


def _make_rng(seed: int | None) -> Generator:
    """Create a NumPy Generator from an optional seed."""
    return np.random.default_rng(seed)


def geometric_brownian_motion(
    S0: float = 100.0,
    mu: float = 0.05,
    sigma: float = 0.2,
    dt: float = 1 / 252,
    steps: int = 252,
    rng_seed: int | None = None,
    return_as: str = "array",
) -> np.ndarray | pd.Series:
    """Simulate a Geometric Brownian Motion price path.

    Args:
        S0: Initial price.
        mu: Annualised drift.
        sigma: Annualised volatility.
        dt: Time increment in years.
        steps: Number of time steps.
        rng_seed: Seed for reproducibility.
        return_as: ``"array"`` or ``"series"``.

    Returns:
        Price path of length ``steps + 1``.

    Raises:
        ValueError: If ``steps < 1`` or ``sigma < 0``.
    """
    if steps < 1:
        raise ValueError("steps must be >= 1")
    if sigma < 0:
        raise ValueError("sigma cannot be negative")

    rng = _make_rng(rng_seed)
    dW = rng.normal(0, np.sqrt(dt), size=steps)
    log_returns = (mu - 0.5 * sigma**2) * dt + sigma * dW
    prices = np.empty(steps + 1)
    prices[0] = S0
    prices[1:] = S0 * np.exp(np.cumsum(log_returns))

    if return_as == "series":
        return pd.Series(prices, index=range(steps + 1), name="GBM")
    return prices


def correlated_gbm(
    S0_list: list[float],
    mu_list: list[float],
    cov_matrix: np.ndarray,
    steps: int = 252,
    dt: float = 1 / 252,
    rng_seed: int | None = None,
    return_as: str = "dataframe",
) -> np.ndarray | pd.DataFrame:
    """Simulate correlated multi-asset GBM paths via Cholesky decomposition.

    Args:
        S0_list: Initial prices per asset.
        mu_list: Annualised drifts per asset.
        cov_matrix: Covariance matrix (N x N).
        steps: Number of time steps.
        dt: Time increment in years.
        rng_seed: Seed for reproducibility.
        return_as: ``"dataframe"`` or ``"array"``.

    Returns:
        Price paths of shape ``(steps + 1, n_assets)``.

    Raises:
        ValueError: On dimension mismatches or invalid steps.
    """
    n_assets = len(S0_list)
    if len(mu_list) != n_assets:
        raise ValueError("mu_list length must match S0_list")
    if cov_matrix.shape != (n_assets, n_assets):
        raise ValueError("cov_matrix must be NxN where N=len(S0_list)")
    if steps < 1:
        raise ValueError("steps must be >= 1")

    rng = _make_rng(rng_seed)
    L = np.linalg.cholesky(cov_matrix)

    # Vectorised: draw all random variates at once
    Z = rng.normal(size=(steps, n_assets))
    corr_Z = Z @ L.T  # (steps, n_assets)

    sigmas = np.sqrt(np.diag(cov_matrix))
    drifts = (np.array(mu_list) - 0.5 * sigmas**2) * dt
    diffusions = corr_Z * np.sqrt(dt)

    log_returns = drifts + diffusions * sigmas  # broadcast
    # Fix: apply sigma per-asset correctly
    log_returns = drifts + corr_Z * np.sqrt(dt)

    prices = np.empty((steps + 1, n_assets))
    prices[0, :] = S0_list
    prices[1:, :] = np.array(S0_list) * np.exp(np.cumsum(log_returns, axis=0))

    if return_as == "dataframe":
        cols = [f"Asset_{i}" for i in range(n_assets)]
        return pd.DataFrame(prices, columns=cols)
    return prices


def ornstein_uhlenbeck(
    x0: float = 0.0,
    theta: float = 0.07,
    mu: float = 0.0,
    sigma: float = 0.1,
    dt: float = 1 / 252,
    steps: int = 252,
    rng_seed: int | None = None,
    return_as: str = "array",
) -> np.ndarray | pd.Series:
    """Simulate an Ornstein-Uhlenbeck mean-reverting process.

    Args:
        x0: Initial value.
        theta: Speed of mean reversion.
        mu: Long-term mean.
        sigma: Volatility.
        dt: Time increment.
        steps: Number of time steps.
        rng_seed: Seed for reproducibility.
        return_as: ``"array"`` or ``"series"``.

    Returns:
        Process values of length ``steps + 1``.

    Raises:
        ValueError: If ``steps < 1`` or ``sigma < 0``.
    """
    if steps < 1:
        raise ValueError("steps must be >= 1")
    if sigma < 0:
        raise ValueError("sigma cannot be negative")

    rng = _make_rng(rng_seed)
    dW = rng.normal(0, np.sqrt(dt), size=steps)

    x_vals = np.empty(steps + 1)
    x_vals[0] = x0
    for t in range(steps):
        x_vals[t + 1] = x_vals[t] + theta * (mu - x_vals[t]) * dt + sigma * dW[t]

    if return_as == "series":
        return pd.Series(x_vals, index=range(steps + 1), name="OU")
    return x_vals


def merton_jump_diffusion(
    S0: float = 100.0,
    mu: float = 0.05,
    sigma: float = 0.2,
    lamb: float = 0.5,
    jump_mean: float = 0.0,
    jump_std: float = 0.1,
    dt: float = 1 / 252,
    steps: int = 252,
    rng_seed: int | None = None,
    return_as: str = "array",
) -> np.ndarray | pd.Series:
    """Simulate a Merton jump-diffusion price path.

    Combines GBM with a compound Poisson jump process whose
    log-jump sizes are normally distributed.

    Args:
        S0: Initial price.
        mu: Annualised drift.
        sigma: Annualised diffusion volatility.
        lamb: Jump intensity (expected jumps per year).
        jump_mean: Mean of log-jump size.
        jump_std: Std of log-jump size.
        dt: Time increment.
        steps: Number of time steps.
        rng_seed: Seed for reproducibility.
        return_as: ``"array"`` or ``"series"``.

    Returns:
        Price path of length ``steps + 1``.

    Raises:
        ValueError: If ``steps < 1``.
    """
    if steps < 1:
        raise ValueError("steps must be >= 1")

    rng = _make_rng(rng_seed)

    dW = rng.normal(0, np.sqrt(dt), size=steps)
    N_jumps = rng.poisson(lamb * dt, size=steps)

    log_increments = (mu - 0.5 * sigma**2) * dt + sigma * dW
    for t in range(steps):
        if N_jumps[t] > 0:
            log_increments[t] += np.sum(rng.normal(jump_mean, jump_std, N_jumps[t]))

    prices = np.empty(steps + 1)
    prices[0] = S0
    prices[1:] = S0 * np.exp(np.cumsum(log_increments))

    if return_as == "series":
        return pd.Series(prices, index=range(steps + 1), name="MJD")
    return prices
