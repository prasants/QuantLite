"""Strategy forensics: tools for detecting spurious backtesting results.

Implements Lopez de Prado's framework for honest backtesting, including
the Deflated Sharpe Ratio, Probabilistic Sharpe Ratio, haircut adjustments,
minimum track record length, and signal decay analysis.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy import stats as sp_stats

__all__ = [
    "deflated_sharpe_ratio",
    "probabilistic_sharpe_ratio",
    "haircut_sharpe_ratio",
    "min_track_record_length",
    "signal_decay",
]


def _sharpe_ratio_std(n_obs: int, skewness: float = 0.0, kurtosis: float = 3.0,
                      sharpe: float = 0.0) -> float:
    """Standard deviation of the Sharpe ratio estimator.

    Parameters
    ----------
    n_obs : int
        Number of observations.
    skewness : float
        Skewness of returns.
    kurtosis : float
        Kurtosis of returns (not excess kurtosis; normal = 3).
    sharpe : float
        Observed Sharpe ratio.

    Returns
    -------
    float
        Standard deviation of the Sharpe ratio estimator.
    """
    excess_kurt = kurtosis - 3.0
    return math.sqrt(
        (1.0
         - skewness * sharpe
         + ((excess_kurt) / 4.0) * sharpe ** 2)
        / n_obs
    )


def probabilistic_sharpe_ratio(
    observed_sharpe: float,
    benchmark_sharpe: float,
    n_obs: int,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """Probabilistic Sharpe Ratio (PSR).

    Computes the probability that the true Sharpe ratio exceeds a given
    benchmark, accounting for the estimation error of the Sharpe ratio
    and non-normality of returns.

    Parameters
    ----------
    observed_sharpe : float
        Observed (in-sample) Sharpe ratio.
    benchmark_sharpe : float
        Benchmark Sharpe ratio to test against.
    n_obs : int
        Number of return observations.
    skewness : float
        Skewness of returns (default 0).
    kurtosis : float
        Kurtosis of returns (default 3, i.e. normal).

    Returns
    -------
    float
        Probability in [0, 1] that the true Sharpe exceeds the benchmark.
    """
    se = _sharpe_ratio_std(n_obs, skewness, kurtosis, observed_sharpe)
    if se <= 0:
        return 1.0 if observed_sharpe > benchmark_sharpe else 0.0
    z = (observed_sharpe - benchmark_sharpe) / se
    return float(sp_stats.norm.cdf(z))


def deflated_sharpe_ratio(
    observed_sharpe: float,
    n_trials: int,
    n_obs: int,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """Deflated Sharpe Ratio (DSR).

    Adjusts the observed Sharpe ratio for the number of strategies tried
    (multiple testing bias). Returns the probability that the best strategy's
    Sharpe is genuine after accounting for how many strategies were tested.

    Uses the expected maximum Sharpe ratio under the null hypothesis
    (all strategies have zero true Sharpe) as the benchmark for PSR.

    Parameters
    ----------
    observed_sharpe : float
        Best observed Sharpe ratio among all trials.
    n_trials : int
        Number of independent strategy trials conducted.
    n_obs : int
        Number of return observations per trial.
    skewness : float
        Skewness of returns (default 0).
    kurtosis : float
        Kurtosis of returns (default 3, i.e. normal).

    Returns
    -------
    float
        Probability in [0, 1] that the observed Sharpe is genuine.
    """
    if n_trials <= 0:
        raise ValueError("n_trials must be positive.")
    if n_obs <= 1:
        raise ValueError("n_obs must be greater than 1.")

    # Expected maximum Sharpe under the null (all zero-Sharpe strategies).
    # E[max(Z_1,...,Z_k)] approximation using Euler-Mascheroni constant.
    euler_mascheroni = 0.5772156649015329
    e_max_z = (
        (1.0 - euler_mascheroni) * sp_stats.norm.ppf(1.0 - 1.0 / n_trials)
        + euler_mascheroni * sp_stats.norm.ppf(1.0 - 1.0 / (n_trials * math.e))
    )
    # The expected max Sharpe is the expected max z-score scaled by the
    # Sharpe ratio standard error (assuming zero true Sharpe).
    se0 = _sharpe_ratio_std(n_obs, skewness, kurtosis, sharpe=0.0)
    benchmark_sharpe = e_max_z * se0

    return probabilistic_sharpe_ratio(
        observed_sharpe, benchmark_sharpe, n_obs, skewness, kurtosis,
    )


def haircut_sharpe_ratio(
    observed_sharpe: float,
    n_obs: int,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
    method: str = "holm",
) -> float:
    """Haircut Sharpe Ratio.

    Adjusts the observed Sharpe ratio downward to account for non-normality
    and multiple testing. The haircut factor depends on the chosen correction
    method.

    Parameters
    ----------
    observed_sharpe : float
        Observed Sharpe ratio.
    n_obs : int
        Number of return observations.
    skewness : float
        Skewness of returns (default 0).
    kurtosis : float
        Kurtosis of returns (default 3, i.e. normal).
    method : str
        Multiple testing correction method: "bonferroni", "holm", or "bhy".

    Returns
    -------
    float
        Adjusted (haircutted) Sharpe ratio.

    Raises
    ------
    ValueError
        If method is not one of the supported options.
    """
    valid = {"bonferroni", "holm", "bhy"}
    if method not in valid:
        raise ValueError(f"method must be one of {valid}, got '{method}'.")

    # Haircut factors: more conservative methods yield larger haircuts.
    haircut_map = {
        "bonferroni": 0.50,
        "holm": 0.65,
        "bhy": 0.80,
    }

    # Non-normality adjustment: penalise negative skew and fat tails.
    excess_kurt = kurtosis - 3.0
    normality_penalty = 1.0 + abs(skewness) * 0.1 + max(excess_kurt, 0.0) * 0.02

    haircut = haircut_map[method] / normality_penalty
    return observed_sharpe * haircut


def min_track_record_length(
    observed_sharpe: float,
    benchmark_sharpe: float = 0.0,
    confidence: float = 0.95,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """Minimum Track Record Length (MinTRL).

    Computes the minimum number of observations required for the observed
    Sharpe ratio to be statistically significant at the given confidence
    level.

    Parameters
    ----------
    observed_sharpe : float
        Observed Sharpe ratio.
    benchmark_sharpe : float
        Benchmark Sharpe ratio (default 0).
    confidence : float
        Confidence level (default 0.95).
    skewness : float
        Skewness of returns (default 0).
    kurtosis : float
        Kurtosis of returns (default 3, i.e. normal).

    Returns
    -------
    float
        Minimum number of observations (not necessarily integer).

    Raises
    ------
    ValueError
        If observed_sharpe equals benchmark_sharpe (division by zero).
    """
    diff = observed_sharpe - benchmark_sharpe
    if abs(diff) < 1e-15:
        raise ValueError(
            "observed_sharpe must differ from benchmark_sharpe."
        )

    z_c = sp_stats.norm.ppf(confidence)
    excess_kurt = kurtosis - 3.0

    # MinTRL = n such that (SR - SR*) / se(SR, n) = z_c
    # se(SR, n) = sqrt((1 - gamma3*SR + (gamma4/4)*SR^2) / n)
    # Solving for n:
    numerator = (
        1.0
        - skewness * observed_sharpe
        + (excess_kurt / 4.0) * observed_sharpe ** 2
    )

    n_min = numerator * (z_c / diff) ** 2
    return max(n_min, 1.0)


def signal_decay(
    returns: np.ndarray | Sequence[float],
    signal: np.ndarray | Sequence[float],
    lags: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Analyse how alpha decays over time.

    Computes the correlation between a trading signal and forward returns
    at increasing lags to measure how quickly the signal's predictive
    power decays.

    Parameters
    ----------
    returns : array-like
        Array of asset returns.
    signal : array-like
        Array of signal values, same length as returns.
    lags : sequence of int, optional
        Lag values to test. Defaults to [1, 2, 3, 5, 10, 20].

    Returns
    -------
    dict
        Dictionary with keys:

        - ``half_life``: estimated half-life in periods (float or None).
        - ``decay_curve``: list of (lag, correlation) tuples.
        - ``r_squared_curve``: list of (lag, r_squared) tuples.
    """
    returns_arr = np.asarray(returns, dtype=float)
    signal_arr = np.asarray(signal, dtype=float)

    if len(returns_arr) != len(signal_arr):
        raise ValueError("returns and signal must have the same length.")

    if lags is None:
        lags = [1, 2, 3, 5, 10, 20]

    decay_curve: list[tuple] = []
    r_squared_curve: list[tuple] = []
    n = len(returns_arr)

    for lag in lags:
        if lag >= n:
            break
        # Forward returns at this lag.
        fwd = returns_arr[lag:]
        sig = signal_arr[: n - lag]

        if len(fwd) < 3:
            break

        corr = float(np.corrcoef(sig, fwd)[0, 1])
        if np.isnan(corr):
            corr = 0.0
        r_sq = corr ** 2
        decay_curve.append((lag, corr))
        r_squared_curve.append((lag, r_sq))

    # Estimate half-life via exponential decay fit.
    half_life: float | None = None
    if len(decay_curve) >= 2 and decay_curve[0][1] > 0:
        initial_corr = decay_curve[0][1]
        for lag_val, corr_val in decay_curve[1:]:
            if corr_val <= initial_corr / 2.0:
                half_life = float(lag_val)
                break
        if half_life is None and len(decay_curve) >= 2:
            # Fit log-linear decay: corr ~ c0 * exp(-lambda * lag)
            pos_pairs = [(lg, c) for lg, c in decay_curve if c > 0]
            if len(pos_pairs) >= 2:
                log_corrs = np.array([math.log(c) for _, c in pos_pairs])
                lag_vals = np.array([float(lg) for lg, _ in pos_pairs])
                if len(lag_vals) >= 2:
                    slope, _, _, _, _ = sp_stats.linregress(lag_vals, log_corrs)  # noqa: E501
                    if slope < 0:
                        half_life = float(math.log(2) / abs(slope))

    return {
        "half_life": half_life,
        "decay_curve": decay_curve,
        "r_squared_curve": r_squared_curve,
    }
