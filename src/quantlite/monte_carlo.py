"""Monte Carlo simulation utilities for backtesting and scenario analysis."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from .backtesting.legacy import legacy_run_backtest as run_backtest

__all__ = ["run_monte_carlo_sims"]


def run_monte_carlo_sims(
    price_data: pd.Series,
    signal_function: Callable[[int, pd.Series], int],
    n_sims: int = 100,
    noise_scale: float = 0.01,
    mode: str = "perturb",
    rng_seed: int | None = None,
) -> list[dict[str, Any]]:
    """Run multiple backtests on perturbed or synthetic price paths.

    Args:
        price_data: Base price series.
        signal_function: Signal callable ``(idx, price_data) -> {-1, 0, 1}``.
        n_sims: Number of simulations.
        noise_scale: Scale of perturbation noise.
        mode: One of ``"perturb"``, ``"gbm"``, or ``"replace"``.
        rng_seed: Seed for reproducibility.

    Returns:
        List of backtest result dicts.

    Raises:
        ValueError: On invalid ``mode`` or non-Series input.
    """
    rng = np.random.default_rng(rng_seed)

    if not isinstance(price_data, pd.Series):
        raise ValueError("price_data must be a pandas Series")

    results: list[dict[str, Any]] = []
    base_index = price_data.index
    base_values = price_data.values.astype(float)

    for _ in range(n_sims):
        if mode == "perturb":
            noise = rng.normal(loc=0.0, scale=noise_scale, size=len(base_values))
            sim_vals = base_values * (1 + noise)
        elif mode == "gbm":
            sim_vals = _simulate_gbm_series(base_values[0], len(base_values), rng=rng)
        elif mode == "replace":
            daily_returns = rng.normal(loc=0.0, scale=noise_scale, size=len(base_values) - 1)
            cum_returns = np.concatenate([[0.0], np.cumsum(np.log1p(daily_returns))])
            sim_vals = base_values[0] * np.exp(cum_returns)
        else:
            raise ValueError("Unknown mode. Choose from ['perturb', 'gbm', 'replace'].")

        sim_prices = pd.Series(sim_vals, index=base_index)
        result = run_backtest(sim_prices, signal_function)
        results.append(result)

    return results


def _simulate_gbm_series(
    S0: float,
    n_points: int,
    mu: float = 0.05,
    sigma: float = 0.2,
    dt: float = 1 / 252,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate a single GBM path (helper for MC sims)."""
    if rng is None:
        rng = np.random.default_rng()
    steps = n_points - 1
    if steps <= 0:
        return np.array([S0])
    dW = rng.normal(0, np.sqrt(dt), size=steps)
    log_returns = (mu - 0.5 * sigma**2) * dt + sigma * dW
    prices = np.empty(n_points)
    prices[0] = S0
    prices[1:] = S0 * np.exp(np.cumsum(log_returns))
    return prices
