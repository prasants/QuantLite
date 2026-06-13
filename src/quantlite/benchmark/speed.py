"""Speed benchmarks for QuantLite operations.

Times key operations across different data sizes and compares
against naive baseline implementations.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import stats

from ..risk.evt import fit_gpd
from ..risk.metrics import cvar, value_at_risk

# ---------------------------------------------------------------------------
# Naive baselines
# ---------------------------------------------------------------------------

def _naive_var(returns: np.ndarray, alpha: float = 0.05) -> float:
    """Naive historical VaR via simple percentile.

    Args:
        returns: Return series.
        alpha: Significance level.

    Returns:
        VaR estimate.
    """
    return float(np.percentile(returns, alpha * 100))


def _naive_cvar(returns: np.ndarray, alpha: float = 0.05) -> float:
    """Naive CVaR via simple mean of tail.

    Args:
        returns: Return series.
        alpha: Significance level.

    Returns:
        CVaR estimate.
    """
    threshold = np.percentile(returns, alpha * 100)
    tail = returns[returns <= threshold]
    return float(np.mean(tail)) if len(tail) > 0 else threshold


def _naive_evt(returns: np.ndarray) -> float:
    """Naive EVT: just fit a normal and return the 99th percentile loss.

    Args:
        returns: Return series.

    Returns:
        Extreme loss estimate.
    """
    mu, sigma = np.mean(returns), np.std(returns)
    return float(mu + sigma * stats.norm.ppf(0.01))


# ---------------------------------------------------------------------------
# Operations to benchmark
# ---------------------------------------------------------------------------

def _generate_data(n: int, seed: int = 42) -> np.ndarray:
    """Generate fat-tailed test data.

    Args:
        n: Number of observations.
        seed: Random seed.

    Returns:
        Array of returns.
    """
    rng = np.random.RandomState(seed)
    return rng.standard_t(df=5, size=n) * 0.01


def _time_operation(fn: Any, *args: Any, repeats: int = 3) -> float:
    """Time an operation, returning the minimum time across repeats.

    Args:
        fn: Callable to time.
        *args: Arguments to pass.
        repeats: Number of repetitions.

    Returns:
        Minimum execution time in seconds.
    """
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(*args)
        times.append(time.perf_counter() - t0)
    return min(times)


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

@dataclass
class SpeedResult:
    """Result of a speed benchmark.

    Attributes:
        operation: Name of the operation.
        data_size: Number of observations.
        quantlite_time: QuantLite execution time (seconds).
        baseline_time: Baseline execution time (seconds).
        speedup: Ratio of baseline_time / quantlite_time.
    """
    operation: str
    data_size: int
    quantlite_time: float
    baseline_time: float
    speedup: float


def run_speed_benchmarks(
    sizes: Sequence[int] | None = None,
    operations: Sequence[str] | None = None,
    seed: int = 42,
    repeats: int = 3,
) -> list[SpeedResult]:
    """Run speed benchmarks across data sizes and operations.

    Args:
        sizes: Data sizes to test. Defaults to
            ``[100, 1000, 10000, 100000]``.
        operations: Operations to benchmark. Options:
            ``"var"``, ``"cvar"``, ``"evt_fitting"``,
            ``"full_pipeline"``. Defaults to all.
        seed: Random seed.
        repeats: Number of timing repetitions per measurement.

    Returns:
        List of ``SpeedResult`` objects.
    """
    if sizes is None:
        sizes = [100, 1_000, 10_000, 100_000]
    if operations is None:
        operations = ["var", "cvar", "evt_fitting", "full_pipeline"]

    results = []

    for n in sizes:
        data = _generate_data(n, seed=seed)

        if "var" in operations:
            ql_time = _time_operation(
                value_at_risk, data, 0.05, "cornish-fisher", repeats=repeats
            )
            bl_time = _time_operation(_naive_var, data, 0.05, repeats=repeats)
            results.append(SpeedResult(
                operation="VaR (Cornish-Fisher vs percentile)",
                data_size=n,
                quantlite_time=ql_time,
                baseline_time=bl_time,
                speedup=bl_time / max(ql_time, 1e-10),
            ))

        if "cvar" in operations:
            ql_time = _time_operation(cvar, data, 0.05, repeats=repeats)
            bl_time = _time_operation(_naive_cvar, data, 0.05, repeats=repeats)
            results.append(SpeedResult(
                operation="CVaR (QuantLite vs naive)",
                data_size=n,
                quantlite_time=ql_time,
                baseline_time=bl_time,
                speedup=bl_time / max(ql_time, 1e-10),
            ))

        if "evt_fitting" in operations and n >= 500:
                ql_time = _time_operation(fit_gpd, data, repeats=repeats)
                bl_time = _time_operation(_naive_evt, data, repeats=repeats)
                results.append(SpeedResult(
                    operation="EVT fitting (GPD vs Gaussian)",
                    data_size=n,
                    quantlite_time=ql_time,
                    baseline_time=bl_time,
                    speedup=bl_time / max(ql_time, 1e-10),
                ))

        if "full_pipeline" in operations:
            def _full_pipeline(d: np.ndarray) -> None:
                value_at_risk(d, 0.05, "cornish-fisher")
                cvar(d, 0.05)
                if len(d) >= 500:
                    fit_gpd(d)

            def _naive_pipeline(d: np.ndarray) -> None:
                _naive_var(d, 0.05)
                _naive_cvar(d, 0.05)
                _naive_evt(d)

            ql_time = _time_operation(_full_pipeline, data, repeats=repeats)
            bl_time = _time_operation(_naive_pipeline, data, repeats=repeats)
            results.append(SpeedResult(
                operation="Full pipeline",
                data_size=n,
                quantlite_time=ql_time,
                baseline_time=bl_time,
                speedup=bl_time / max(ql_time, 1e-10),
            ))

    return results
