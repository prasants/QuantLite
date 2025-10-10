"""Resampled backtesting: bootstrap methods for robust performance estimation.

Provides block and stationary bootstrap implementations, along with
convenience wrappers for computing confidence intervals on common
performance metrics such as Sharpe ratio and maximum drawdown.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

__all__ = [
    "block_bootstrap",
    "stationary_bootstrap",
    "bootstrap_confidence_interval",
    "bootstrap_sharpe_distribution",
    "bootstrap_drawdown_distribution",
]


def block_bootstrap(
    returns: np.ndarray | Sequence[float],
    block_size: int,
    n_samples: int = 1000,
    seed: int | None = None,
) -> np.ndarray:
    """Non-overlapping block bootstrap preserving autocorrelation.

    Resamples blocks of consecutive returns to generate synthetic
    return series that preserve the serial dependence structure.

    Parameters
    ----------
    returns : array-like
        Original return series.
    block_size : int
        Size of each block.
    n_samples : int
        Number of bootstrap samples to generate (default 1000).
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    numpy.ndarray
        Array of shape (n_samples, len(returns)).
    """
    ret = np.asarray(returns, dtype=float)
    n = len(ret)
    rng = np.random.RandomState(seed)

    n_blocks = int(math.ceil(n / block_size))
    max_start = n - block_size

    if max_start < 0:
        raise ValueError("block_size must not exceed the length of returns.")

    result = np.empty((n_samples, n), dtype=float)
    for i in range(n_samples):
        starts = rng.randint(0, max_start + 1, size=n_blocks)
        pieces = [ret[s: s + block_size] for s in starts]
        sample = np.concatenate(pieces)[:n]
        result[i] = sample

    return result


def stationary_bootstrap(
    returns: np.ndarray | Sequence[float],
    avg_block_size: int,
    n_samples: int = 1000,
    seed: int | None = None,
) -> np.ndarray:
    """Stationary bootstrap with geometrically distributed block lengths.

    Better suited than the fixed-block bootstrap when the autocorrelation
    structure is unknown, as block lengths vary randomly.

    Parameters
    ----------
    returns : array-like
        Original return series.
    avg_block_size : int
        Average block size (the probability of ending a block at each
        step is 1 / avg_block_size).
    n_samples : int
        Number of bootstrap samples to generate (default 1000).
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    numpy.ndarray
        Array of shape (n_samples, len(returns)).
    """
    ret = np.asarray(returns, dtype=float)
    n = len(ret)
    rng = np.random.RandomState(seed)

    if avg_block_size < 1:
        raise ValueError("avg_block_size must be at least 1.")

    p = 1.0 / avg_block_size
    result = np.empty((n_samples, n), dtype=float)

    for i in range(n_samples):
        sample = np.empty(n, dtype=float)
        pos = 0
        idx = rng.randint(0, n)
        while pos < n:
            sample[pos] = ret[idx % n]
            pos += 1
            if rng.random() < p:
                idx = rng.randint(0, n)
            else:
                idx += 1
        result[i] = sample

    return result


def bootstrap_confidence_interval(
    returns: np.ndarray | Sequence[float],
    metric_fn: Callable[[np.ndarray], float],
    n_samples: int = 1000,
    confidence: float = 0.95,
    method: str = "block",
    block_size: int | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Confidence interval for any metric via bootstrap.

    Parameters
    ----------
    returns : array-like
        Original return series.
    metric_fn : callable
        Function that takes a returns array and returns a float metric.
    n_samples : int
        Number of bootstrap samples (default 1000).
    confidence : float
        Confidence level (default 0.95).
    method : str
        Bootstrap method: "block" or "stationary" (default "block").
    block_size : int, optional
        Block size. Defaults to int(sqrt(len(returns))).
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    dict
        Dictionary with keys:

        - ``point_estimate``: metric computed on original returns.
        - ``ci_lower``: lower bound of confidence interval.
        - ``ci_upper``: upper bound of confidence interval.
        - ``distribution``: array of bootstrap metric values.
    """
    ret = np.asarray(returns, dtype=float)
    if block_size is None:
        block_size = max(1, int(math.sqrt(len(ret))))

    if method == "stationary":
        samples = stationary_bootstrap(ret, block_size, n_samples, seed)
    else:
        samples = block_bootstrap(ret, block_size, n_samples, seed)

    distribution = np.array([metric_fn(samples[i]) for i in range(n_samples)])
    point_estimate = float(metric_fn(ret))

    alpha = 1.0 - confidence
    ci_lower = float(np.percentile(distribution, 100 * alpha / 2))
    ci_upper = float(np.percentile(distribution, 100 * (1 - alpha / 2)))

    return {
        "point_estimate": point_estimate,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "distribution": distribution,
    }


def _sharpe(returns: np.ndarray) -> float:
    """Compute annualised Sharpe ratio from a returns array.

    Parameters
    ----------
    returns : numpy.ndarray
        Array of returns.

    Returns
    -------
    float
        Annualised Sharpe ratio.
    """
    if len(returns) < 2:
        return 0.0
    std = float(np.std(returns, ddof=1))
    if std < 1e-15:
        return 0.0
    return float(np.mean(returns) / std * math.sqrt(252))


def _max_drawdown(returns: np.ndarray) -> float:
    """Compute maximum drawdown from a returns array.

    Parameters
    ----------
    returns : numpy.ndarray
        Array of returns.

    Returns
    -------
    float
        Maximum drawdown as a positive fraction.
    """
    cumret = np.cumprod(1.0 + returns)
    peak = np.maximum.accumulate(cumret)
    dd = (peak - cumret) / peak
    return float(np.max(dd)) if len(dd) > 0 else 0.0


def bootstrap_sharpe_distribution(
    returns: np.ndarray | Sequence[float],
    n_samples: int = 1000,
    block_size: int | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Bootstrap confidence interval for the Sharpe ratio.

    Convenience wrapper around ``bootstrap_confidence_interval`` using
    the annualised Sharpe ratio as the metric.

    Parameters
    ----------
    returns : array-like
        Original return series.
    n_samples : int
        Number of bootstrap samples (default 1000).
    block_size : int, optional
        Block size for block bootstrap.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    dict
        Same structure as ``bootstrap_confidence_interval``.
    """
    return bootstrap_confidence_interval(
        returns, _sharpe, n_samples=n_samples,
        block_size=block_size, seed=seed,
    )


def bootstrap_drawdown_distribution(
    returns: np.ndarray | Sequence[float],
    n_samples: int = 1000,
    block_size: int | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Bootstrap confidence interval for maximum drawdown.

    Convenience wrapper around ``bootstrap_confidence_interval`` using
    maximum drawdown as the metric.

    Parameters
    ----------
    returns : array-like
        Original return series.
    n_samples : int
        Number of bootstrap samples (default 1000).
    block_size : int, optional
        Block size for block bootstrap.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    dict
        Same structure as ``bootstrap_confidence_interval``.
    """
    return bootstrap_confidence_interval(
        returns, _max_drawdown, n_samples=n_samples,
        block_size=block_size, seed=seed,
    )
