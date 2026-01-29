"""EVT-based Monte Carlo simulation.

Generates return scenarios using Generalised Pareto Distribution (GPD)
for the tails and empirical or normal distributions for the body.
This produces realistic fat-tailed scenario sets that respect the
observed tail behaviour of financial returns.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

__all__ = [
    "evt_tail_simulation",
    "parametric_tail_simulation",
    "historical_bootstrap_evt",
    "scenario_fan",
]


def _to_array(returns: np.ndarray | list[float]) -> np.ndarray:
    """Coerce to float64 array, dropping NaNs."""
    arr = np.asarray(returns, dtype=float)
    return arr[~np.isnan(arr)]


def _fit_gpd_tail(losses: np.ndarray, threshold: float) -> tuple:
    """Fit GPD to exceedances above threshold. Returns (shape, scale)."""
    exceedances = losses[losses > threshold] - threshold
    if len(exceedances) < 5:
        scale = float(np.mean(exceedances)) if len(exceedances) > 0 else 1.0
        return 0.0, scale
    shape, _loc, scale = stats.genpareto.fit(exceedances, floc=0)
    return float(shape), float(scale)


def _gpd_rvs(shape: float, scale: float, size: int, rng: np.random.Generator) -> np.ndarray:
    """Vectorised GPD random variates using inverse CDF."""
    u = rng.random(size)
    if abs(shape) < 1e-10:
        return -scale * np.log(1 - u)
    return (scale / shape) * ((1 - u) ** (-shape) - 1)


def evt_tail_simulation(
    returns: np.ndarray | list[float],
    n_scenarios: int = 10000,
    alpha: float = 0.05,
    seed: int = 42,
) -> np.ndarray:
    """Generate scenarios using fitted GPD for tails, empirical for body.

    The left and right tails (below the ``alpha`` and above the
    ``1 - alpha`` quantiles) are modelled with separate GPD fits.
    The body is sampled from the empirical distribution.

    Args:
        returns: Historical return series.
        n_scenarios: Number of scenarios to generate.
        alpha: Tail fraction on each side (default 5%).
        seed: Random seed for reproducibility.

    Returns:
        Array of shape ``(n_scenarios,)`` with simulated returns.
    """
    arr = _to_array(returns)
    rng = np.random.default_rng(seed)

    lower_q = np.percentile(arr, alpha * 100)
    upper_q = np.percentile(arr, (1 - alpha) * 100)

    body = arr[(arr >= lower_q) & (arr <= upper_q)]

    # Fit GPD to left tail (losses = -returns)
    left_losses = -arr[arr < lower_q]
    left_shape, left_scale = _fit_gpd_tail(left_losses, 0.0)

    # Fit GPD to right tail
    right_exceedances = arr[arr > upper_q]
    right_shape, right_scale = _fit_gpd_tail(right_exceedances, 0.0)

    # Assign regions
    region_idx = rng.choice(3, size=n_scenarios, p=[alpha, 1 - 2 * alpha, alpha])

    left_mask = region_idx == 0
    body_mask = region_idx == 1
    right_mask = region_idx == 2

    n_left = int(left_mask.sum())
    n_body = int(body_mask.sum())
    n_right = int(right_mask.sum())

    result = np.empty(n_scenarios)

    if n_body > 0:
        result[body_mask] = rng.choice(body, size=n_body)
    if n_left > 0:
        exc = _gpd_rvs(left_shape, left_scale, n_left, rng)
        result[left_mask] = lower_q - exc
    if n_right > 0:
        exc = _gpd_rvs(right_shape, right_scale, n_right, rng)
        result[right_mask] = upper_q + exc

    return result


def parametric_tail_simulation(
    shape: float,
    scale: float,
    threshold: float,
    n_body: int,
    body_mean: float,
    body_std: float,
    n_scenarios: int = 10000,
    seed: int = 42,
) -> np.ndarray:
    """Simulate from explicit GPD parameters plus a normal body.

    The body is drawn from ``N(body_mean, body_std)``, truncated at
    ``[-threshold, threshold]``. Tails beyond the threshold are
    drawn from a GPD.

    Args:
        shape: GPD shape parameter (xi).
        scale: GPD scale parameter (sigma).
        threshold: Tail threshold (positive, applied symmetrically).
        n_body: Approximate number of historical observations in body
            (used to set tail probability).
        body_mean: Mean of the normal body.
        body_std: Standard deviation of the normal body.
        n_scenarios: Number of scenarios.
        seed: Random seed.

    Returns:
        Array of ``n_scenarios`` simulated returns.
    """
    rng = np.random.default_rng(seed)

    # Tail probability from threshold and body normal
    tail_prob = float(stats.norm.sf(threshold, loc=body_mean, scale=body_std))
    tail_prob = max(tail_prob, 0.005)  # floor at 0.5%

    u = rng.random(n_scenarios)
    left_mask = u < tail_prob
    right_mask = u > (1 - tail_prob)
    body_mask = ~left_mask & ~right_mask

    n_left = int(left_mask.sum())
    n_right = int(right_mask.sum())
    n_mid = int(body_mask.sum())

    result = np.empty(n_scenarios)

    if n_left > 0:
        exc = _gpd_rvs(shape, scale, n_left, rng)
        result[left_mask] = -threshold - exc
    if n_right > 0:
        exc = _gpd_rvs(shape, scale, n_right, rng)
        result[right_mask] = threshold + exc
    if n_mid > 0:
        vals = rng.normal(body_mean, body_std, n_mid)
        result[body_mask] = np.clip(vals, -threshold, threshold)

    return result


def historical_bootstrap_evt(
    returns: np.ndarray | list[float],
    n_scenarios: int = 10000,
    tail_fraction: float = 0.05,
    seed: int = 42,
) -> np.ndarray:
    """Hybrid bootstrap: empirical body, GPD-sampled tails.

    Bootstraps from the body of the distribution and uses GPD
    sampling for tail scenarios. This preserves the empirical
    centre while allowing tail extrapolation beyond observed data.

    Args:
        returns: Historical return series.
        n_scenarios: Number of scenarios.
        tail_fraction: Fraction of each tail (default 5%).
        seed: Random seed.

    Returns:
        Array of ``n_scenarios`` simulated returns.
    """
    arr = _to_array(returns)
    rng = np.random.default_rng(seed)

    lower_q = np.percentile(arr, tail_fraction * 100)
    upper_q = np.percentile(arr, (1 - tail_fraction) * 100)

    body = arr[(arr >= lower_q) & (arr <= upper_q)]

    # Fit GPD to left tail
    left_tail = arr[arr < lower_q]
    left_losses = -left_tail
    if len(left_losses) >= 5:
        left_shape, _loc, left_scale = stats.genpareto.fit(
            left_losses - np.min(left_losses), floc=0,
        )
    else:
        left_shape, left_scale = 0.0, float(np.std(left_losses)) if len(left_losses) > 0 else 0.01

    # Fit GPD to right tail
    right_tail = arr[arr > upper_q]
    if len(right_tail) >= 5:
        right_shape, _loc, right_scale = stats.genpareto.fit(
            right_tail - np.min(right_tail), floc=0,
        )
    else:
        right_shape, right_scale = 0.0, float(np.std(right_tail)) if len(right_tail) > 0 else 0.01

    region_idx = rng.choice(
        3, size=n_scenarios,
        p=[tail_fraction, 1 - 2 * tail_fraction, tail_fraction],
    )

    left_mask = region_idx == 0
    body_mask = region_idx == 1
    right_mask = region_idx == 2

    n_left = int(left_mask.sum())
    n_body = int(body_mask.sum())
    n_right = int(right_mask.sum())

    result = np.empty(n_scenarios)

    if n_body > 0:
        result[body_mask] = rng.choice(body, size=n_body)
    if n_left > 0:
        exc = _gpd_rvs(left_shape, left_scale, n_left, rng)
        result[left_mask] = lower_q - np.abs(exc)
    if n_right > 0:
        exc = _gpd_rvs(right_shape, right_scale, n_right, rng)
        result[right_mask] = upper_q + np.abs(exc)

    return result


def scenario_fan(
    returns: np.ndarray | list[float],
    horizons: list[int] | np.ndarray,
    n_scenarios: int = 5000,
    seed: int = 42,
) -> dict[str, np.ndarray | dict[int, dict[str, float]]]:
    """Project return distributions at multiple horizons.

    For each horizon, simulates cumulative returns by drawing daily
    returns from an EVT-based model and compounding over the horizon.

    Args:
        returns: Daily return series.
        horizons: List of horizons in days (e.g. [1, 5, 21, 63, 252]).
        n_scenarios: Number of scenarios per horizon.
        seed: Random seed.

    Returns:
        Dictionary with keys:

        - ``"horizons"``: the input horizons
        - ``"percentiles"``: list of percentile levels [5, 25, 50, 75, 95]
        - ``"fans"``: dict mapping each horizon to a dict of
          ``{percentile: cumulative_return_value}``
        - ``"scenarios"``: dict mapping each horizon to the full
          array of cumulative returns
    """
    arr = _to_array(returns)
    horizons_list = list(horizons)

    max_horizon = max(horizons_list)
    total_draws = n_scenarios * max_horizon
    daily_scenarios = evt_tail_simulation(arr, n_scenarios=total_draws, seed=seed)

    percentile_levels = [5, 25, 50, 75, 95]
    fans = {}
    scenarios = {}

    for h in horizons_list:
        n_usable = (total_draws // h) * h
        if n_usable < n_scenarios * h:
            extra = evt_tail_simulation(
                arr, n_scenarios=n_scenarios * h, seed=seed + h,
            )
            block = extra[: n_scenarios * h].reshape(n_scenarios, h)
        else:
            block = daily_scenarios[: n_scenarios * h].reshape(n_scenarios, h)

        cum_returns = np.prod(1 + block, axis=1) - 1
        scenarios[h] = cum_returns

        fan = {}
        for p in percentile_levels:
            fan[str(p)] = float(np.percentile(cum_returns, p))
        fans[h] = fan

    return {
        "horizons": horizons_list,
        "percentiles": percentile_levels,
        "fans": fans,
        "scenarios": scenarios,
    }
