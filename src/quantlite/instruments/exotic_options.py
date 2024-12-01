"""Monte Carlo pricing for exotic options (barrier, Asian).

All simulations are fully vectorised using NumPy broadcasting.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = [
    "barrier_option_knock_out",
    "asian_option_arithmetic",
]


def barrier_option_knock_out(
    S0: float,
    K: float,
    H: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
    barrier_type: str = "down-and-out",
    steps: int = 1000,
    sims: int = 10000,
    rng_seed: int | None = None,
) -> float:
    """Price a knock-out barrier option via vectorised Monte Carlo.

    Args:
        S0: Initial spot price.
        K: Strike price.
        H: Barrier level.
        T: Time to maturity in years.
        r: Risk-free rate.
        sigma: Volatility.
        option_type: ``"call"`` or ``"put"``.
        barrier_type: ``"down-and-out"`` (more types in future).
        steps: Time steps per path.
        sims: Number of simulated paths.
        rng_seed: Seed for reproducibility.

    Returns:
        Estimated option price.
    """
    rng = np.random.default_rng(rng_seed)
    dt = T / steps
    disc_factor = math.exp(-r * T)

    # Vectorised path generation: (sims, steps)
    Z = rng.normal(size=(sims, steps))
    log_increments = (r - 0.5 * sigma**2) * dt + sigma * math.sqrt(dt) * Z
    log_paths = np.cumsum(log_increments, axis=1)
    paths = S0 * np.exp(log_paths)  # (sims, steps)

    # Barrier check
    if barrier_type == "down-and-out":
        knocked_out = np.any(paths <= H, axis=1)
    else:
        knocked_out = np.zeros(sims, dtype=bool)

    final_prices = paths[:, -1]

    if option_type == "call":
        payoffs = np.maximum(final_prices - K, 0.0)
    else:
        payoffs = np.maximum(K - final_prices, 0.0)

    payoffs[knocked_out] = 0.0
    return float(disc_factor * np.mean(payoffs))


def asian_option_arithmetic(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
    steps: int = 1000,
    sims: int = 10000,
    rng_seed: int | None = None,
) -> float:
    """Price an arithmetic average Asian option via vectorised Monte Carlo.

    Args:
        S0: Initial spot price.
        K: Strike price.
        T: Time to maturity in years.
        r: Risk-free rate.
        sigma: Volatility.
        option_type: ``"call"`` or ``"put"``.
        steps: Time steps per path.
        sims: Number of simulated paths.
        rng_seed: Seed for reproducibility.

    Returns:
        Estimated option price.
    """
    rng = np.random.default_rng(rng_seed)
    dt = T / steps
    disc_factor = math.exp(-r * T)

    Z = rng.normal(size=(sims, steps))
    log_increments = (r - 0.5 * sigma**2) * dt + sigma * math.sqrt(dt) * Z
    log_paths = np.cumsum(log_increments, axis=1)
    paths = S0 * np.exp(log_paths)

    # Include S0 in average
    full_paths = np.column_stack([np.full(sims, S0), paths])
    avg_prices = np.mean(full_paths, axis=1)

    if option_type == "call":
        payoffs = np.maximum(avg_prices - K, 0.0)
    else:
        payoffs = np.maximum(K - avg_prices, 0.0)

    return float(disc_factor * np.mean(payoffs))
