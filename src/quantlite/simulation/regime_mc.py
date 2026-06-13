"""Regime-switching Monte Carlo simulation, stress testing, and reverse stress tests.

Simulates return paths where the underlying distribution shifts
between regimes (e.g. calm, volatile, crisis) according to a
Markov transition matrix. Also provides stress testing frameworks
and simulation summary statistics.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats

__all__ = [
    "regime_switching_simulation",
    "stress_test_scenario",
    "reverse_stress_test",
    "simulation_summary",
]


def _to_array(returns: np.ndarray | list[float]) -> np.ndarray:
    arr = np.asarray(returns, dtype=float)
    return arr[~np.isnan(arr)]


def regime_switching_simulation(
    regime_params: list[dict[str, float]],
    transition_matrix: np.ndarray | list[list[float]],
    n_steps: int = 252,
    n_scenarios: int = 5000,
    seed: int = 42,
) -> dict[str, Any]:
    """Simulate return paths with regime-switching dynamics.

    Each regime has its own mean and volatility. At each step, the
    regime may change according to the Markov transition matrix.

    Args:
        regime_params: List of dicts, each with keys ``"mu"`` (daily
            mean return) and ``"sigma"`` (daily volatility). One dict
            per regime.
        transition_matrix: Square matrix of shape ``(n_regimes, n_regimes)``
            where entry ``[i][j]`` is the probability of transitioning
            from regime ``i`` to regime ``j``.
        n_steps: Number of time steps per path (default 252 = 1 year).
        n_scenarios: Number of simulation paths.
        seed: Random seed.

    Returns:
        Dictionary with:

        - ``"returns"``: array of shape ``(n_scenarios, n_steps)``
        - ``"regimes"``: array of shape ``(n_scenarios, n_steps)``
          with regime index at each step
        - ``"cumulative_returns"``: array of shape ``(n_scenarios, n_steps)``
        - ``"regime_params"``: the input regime parameters
    """
    rng = np.random.default_rng(seed)
    trans = np.asarray(transition_matrix, dtype=float)
    n_regimes = len(regime_params)

    # Validate transition matrix rows sum to 1
    row_sums = trans.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-6):
        raise ValueError(
            f"Transition matrix rows must sum to 1.0, got {row_sums}"
        )

    returns_out = np.empty((n_scenarios, n_steps))
    regimes_out = np.empty((n_scenarios, n_steps), dtype=int)

    # Start in the stationary distribution (or regime 0)
    for s in range(n_scenarios):
        regime = 0
        for t in range(n_steps):
            regimes_out[s, t] = regime
            mu = regime_params[regime]["mu"]
            sigma = regime_params[regime]["sigma"]
            returns_out[s, t] = rng.normal(mu, sigma)
            # Transition
            regime = int(rng.choice(n_regimes, p=trans[regime]))

    cum_returns = np.cumprod(1 + returns_out, axis=1) - 1

    return {
        "returns": returns_out,
        "regimes": regimes_out,
        "cumulative_returns": cum_returns,
        "regime_params": regime_params,
    }


def stress_test_scenario(
    returns: np.ndarray | list[float],
    shock_type: str,
    magnitude: float,
    horizon: int = 21,
) -> dict[str, Any]:
    """Apply predefined stress scenarios to a return series.

    Supported shock types:

    - ``"market_crash"``: sudden drop of ``magnitude`` spread over horizon
    - ``"vol_spike"``: volatility multiplied by ``magnitude``
    - ``"correlation_breakdown"``: not applicable to single series;
      returns impact assuming diversification fails
    - ``"liquidity_freeze"``: returns compressed toward zero with
      occasional large drops

    Args:
        returns: Historical return series (used to calibrate baseline).
        shock_type: One of ``"market_crash"``, ``"vol_spike"``,
            ``"correlation_breakdown"``, ``"liquidity_freeze"``.
        magnitude: Severity of the shock. For crash: total loss fraction.
            For vol_spike: volatility multiplier. For others: scaling.
        horizon: Number of periods over which the shock unfolds.

    Returns:
        Dictionary with:

        - ``"stressed_returns"``: array of stressed return path
        - ``"cumulative_impact"``: total cumulative return
        - ``"max_drawdown"``: worst peak-to-trough within the scenario
        - ``"shock_type"``: echo of input
        - ``"magnitude"``: echo of input
        - ``"horizon"``: echo of input
    """
    arr = _to_array(returns)
    hist_mean = float(np.mean(arr))
    hist_std = float(np.std(arr))

    valid_types = ("market_crash", "vol_spike", "correlation_breakdown", "liquidity_freeze")
    if shock_type not in valid_types:
        raise ValueError(
            f"shock_type must be one of {valid_types}, got {shock_type!r}"
        )

    rng = np.random.default_rng(0)  # deterministic stress

    if shock_type == "market_crash":
        # Distribute the crash over the horizon with front-loading
        weights = np.exp(-np.linspace(0, 2, horizon))
        weights /= weights.sum()
        daily_shocks = -abs(magnitude) * weights
        noise = rng.normal(0, hist_std * 0.5, horizon)
        stressed = daily_shocks + noise

    elif shock_type == "vol_spike":
        stressed = rng.normal(hist_mean * 0.5, hist_std * magnitude, horizon)

    elif shock_type == "correlation_breakdown":
        # Assume diversification benefit disappears; returns become
        # more negative with higher variance
        stressed = rng.normal(
            -abs(hist_mean) * magnitude,
            hist_std * (1 + magnitude * 0.5),
            horizon,
        )

    elif shock_type == "liquidity_freeze":
        # Mostly zero returns with occasional large drops
        stressed = np.zeros(horizon)
        n_drops = max(1, int(horizon * 0.15))
        drop_indices = rng.choice(horizon, size=n_drops, replace=False)
        stressed[drop_indices] = -abs(magnitude) / n_drops * rng.uniform(0.5, 1.5, n_drops)

    cum = float(np.prod(1 + stressed) - 1)

    # Max drawdown
    prices = np.cumprod(1 + stressed)
    peak = np.maximum.accumulate(prices)
    drawdowns = (prices - peak) / peak
    max_dd = float(np.min(drawdowns))

    return {
        "stressed_returns": stressed,
        "cumulative_impact": cum,
        "max_drawdown": max_dd,
        "shock_type": shock_type,
        "magnitude": magnitude,
        "horizon": horizon,
    }


def reverse_stress_test(
    returns: np.ndarray | list[float],
    target_loss: float,
    n_scenarios: int = 50000,
    seed: int = 42,
) -> dict[str, Any]:
    """Find scenarios that produce a specific loss level.

    Simulates many paths and identifies the ones closest to the
    target loss, revealing what combination of moves would cause
    (for example) a 20% drawdown.

    Args:
        returns: Historical return series.
        target_loss: Target cumulative loss (negative, e.g. -0.20
            for a 20% drawdown).
        n_scenarios: Number of scenarios to simulate.
        seed: Random seed.

    Returns:
        Dictionary with:

        - ``"target_loss"``: echo of input
        - ``"closest_scenarios"``: array of shape ``(n_closest, horizon)``
          with the scenarios nearest to the target
        - ``"closest_cumulative"``: cumulative returns of closest scenarios
        - ``"mean_path"``: average of the closest scenario paths
        - ``"n_scenarios_searched"``: total simulated
        - ``"worst_day_mean"``: average worst single-day return in
          matching scenarios
    """
    arr = _to_array(returns)
    rng = np.random.default_rng(seed)

    # Use a 21-day horizon for the reverse stress test
    horizon = 21
    n_closest = min(50, n_scenarios // 100)

    # Bootstrap daily returns
    paths = np.empty((n_scenarios, horizon))
    for t in range(horizon):
        paths[:, t] = rng.choice(arr, size=n_scenarios)

    cum_returns = np.prod(1 + paths, axis=1) - 1

    # Find scenarios closest to target
    distances = np.abs(cum_returns - target_loss)
    closest_idx = np.argsort(distances)[:n_closest]

    closest_paths = paths[closest_idx]
    closest_cum = cum_returns[closest_idx]
    mean_path = np.mean(closest_paths, axis=0)
    worst_days = np.min(closest_paths, axis=1)

    return {
        "target_loss": target_loss,
        "closest_scenarios": closest_paths,
        "closest_cumulative": closest_cum,
        "mean_path": mean_path,
        "n_scenarios_searched": n_scenarios,
        "worst_day_mean": float(np.mean(worst_days)),
    }


def simulation_summary(
    simulated_returns: np.ndarray,
) -> dict[str, Any]:
    """Compute comprehensive risk statistics from simulation output.

    Calculates VaR, CVaR, max drawdown distribution, and probability
    of ruin at various confidence levels.

    Args:
        simulated_returns: Array of simulated returns. Can be 1-D
            (single-period scenarios) or 2-D ``(n_scenarios, n_steps)``
            for path-based simulations.

    Returns:
        Dictionary with:

        - ``"var"``: dict of VaR at 90%, 95%, 99% confidence
        - ``"cvar"``: dict of CVaR at 90%, 95%, 99%
        - ``"max_drawdown"``: dict with mean, median, worst, best
        - ``"probability_of_ruin"``: dict of probability of losing
          more than 10%, 20%, 50%
        - ``"mean_return"``: mean of terminal returns
        - ``"median_return"``: median of terminal returns
        - ``"std_return"``: standard deviation
        - ``"skewness"``: skewness of returns
        - ``"kurtosis"``: excess kurtosis
    """
    arr = np.asarray(simulated_returns, dtype=float)

    if arr.ndim == 1:
        terminal = arr
        # For max drawdown, treat each value as a single-step path
        max_dds = -np.abs(np.minimum(arr, 0))
    else:
        # 2-D: (n_scenarios, n_steps)
        terminal = np.prod(1 + arr, axis=1) - 1
        # Compute max drawdown for each path
        prices = np.cumprod(1 + arr, axis=1)
        peak = np.maximum.accumulate(prices, axis=1)
        dd = (prices - peak) / np.where(peak > 0, peak, 1.0)
        max_dds = np.min(dd, axis=1)

    # VaR (loss quantiles, reported as positive losses)
    var_levels = {"90%": 0.10, "95%": 0.05, "99%": 0.01}
    var_dict = {}
    cvar_dict = {}
    for label, alpha in var_levels.items():
        q = float(np.percentile(terminal, alpha * 100))
        var_dict[label] = -q  # positive number = loss
        tail = terminal[terminal <= q]
        cvar_dict[label] = -float(np.mean(tail)) if len(tail) > 0 else -q

    # Max drawdown stats
    dd_stats = {
        "mean": float(np.mean(max_dds)),
        "median": float(np.median(max_dds)),
        "worst": float(np.min(max_dds)),
        "best": float(np.max(max_dds)),
    }

    # Probability of ruin
    ruin_levels = {"10%": -0.10, "20%": -0.20, "50%": -0.50}
    ruin_probs = {}
    for label, threshold in ruin_levels.items():
        ruin_probs[label] = float(np.mean(terminal < threshold))

    return {
        "var": var_dict,
        "cvar": cvar_dict,
        "max_drawdown": dd_stats,
        "probability_of_ruin": ruin_probs,
        "mean_return": float(np.mean(terminal)),
        "median_return": float(np.median(terminal)),
        "std_return": float(np.std(terminal)),
        "skewness": float(stats.skew(terminal)),
        "kurtosis": float(stats.kurtosis(terminal)),
    }
