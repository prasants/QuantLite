"""Overfitting detection: tools to identify and quantify backtest overfitting.

Provides the Combinatorially Symmetric Cross-Validation (CSCV) method for
estimating the Probability of Backtest Overfitting (PBO), multiple testing
corrections, minimum backtest length estimation, walk-forward validation,
and a TrialTracker context manager for logging backtest trials.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from scipy import stats as sp_stats

__all__ = [
    "TrialTracker",
    "probability_of_backtest_overfitting",
    "multiple_testing_correction",
    "min_backtest_length",
    "walk_forward_validate",
]


def multiple_testing_correction(
    p_values: np.ndarray | Sequence[float],
    method: str = "bhy",
) -> np.ndarray:
    """Adjust p-values for multiple testing.

    Parameters
    ----------
    p_values : array-like
        Raw p-values to adjust.
    method : str
        Correction method: "bonferroni", "holm", or "bhy"
        (Benjamini-Hochberg-Yekutieli).

    Returns
    -------
    numpy.ndarray
        Adjusted p-values, clipped to [0, 1].

    Raises
    ------
    ValueError
        If method is not one of the supported options.
    """
    valid = {"bonferroni", "holm", "bhy"}
    if method not in valid:
        raise ValueError(f"method must be one of {valid}, got '{method}'.")

    pv = np.asarray(p_values, dtype=float)
    m = len(pv)
    if m == 0:
        return pv.copy()

    if method == "bonferroni":
        adjusted = pv * m

    elif method == "holm":
        order = np.argsort(pv)
        adjusted = np.empty(m, dtype=float)
        cummax = 0.0
        for rank, idx in enumerate(order):
            val = pv[idx] * (m - rank)
            cummax = max(cummax, val)
            adjusted[idx] = cummax

    elif method == "bhy":
        order = np.argsort(pv)
        # Harmonic number for BHY correction.
        c_m = sum(1.0 / i for i in range(1, m + 1))
        adjusted = np.empty(m, dtype=float)
        cummin = 1.0
        for i in range(m - 1, -1, -1):
            idx = order[i]
            rank = i + 1
            val = pv[idx] * m * c_m / rank
            cummin = min(cummin, val)
            adjusted[idx] = cummin

    return np.clip(adjusted, 0.0, 1.0)


def probability_of_backtest_overfitting(
    trial_returns: np.ndarray | list[np.ndarray],
    n_splits: int = 10,
) -> dict[str, Any]:
    """Probability of Backtest Overfitting via CSCV.

    Implements Combinatorially Symmetric Cross-Validation. Splits the data
    into S subsets, forms all C(S, S/2) train/test combinations, and computes
    the rank correlation of in-sample vs out-of-sample performance. PBO is
    the fraction of combinations where rank correlation is negative.

    Parameters
    ----------
    trial_returns : array-like or list of array-like
        If a 2D array, shape is (n_trials, n_obs). Each row is one
        strategy's return series. If a list, each element is a 1D array
        of returns for one strategy.
    n_splits : int
        Number of temporal subsets to split data into (must be even,
        default 10).

    Returns
    -------
    dict
        Dictionary with keys:

        - ``pbo``: Probability of Backtest Overfitting in [0, 1].
        - ``rank_correlations``: list of rank correlations per combination.
        - ``n_combinations``: number of train/test combinations evaluated.
    """
    if n_splits % 2 != 0:
        raise ValueError("n_splits must be even.")

    # Convert to 2D array: (n_trials, n_obs).
    if isinstance(trial_returns, list):
        trial_returns = np.array(trial_returns, dtype=float)
    else:
        trial_returns = np.asarray(trial_returns, dtype=float)

    if trial_returns.ndim == 1:
        raise ValueError("trial_returns must be 2D (n_trials x n_obs).")

    n_trials, n_obs = trial_returns.shape
    if n_trials < 2:
        raise ValueError("Need at least 2 trials.")

    # Split time axis into n_splits subsets.
    split_size = n_obs // n_splits
    if split_size < 1:
        raise ValueError("Not enough observations for the requested n_splits.")

    # Compute performance per trial per split.
    perf = np.zeros((n_trials, n_splits))
    for s in range(n_splits):
        start = s * split_size
        end = start + split_size if s < n_splits - 1 else n_obs
        perf[:, s] = trial_returns[:, start:end].sum(axis=1)

    # Generate all C(S, S/2) combinations for train sets.
    half = n_splits // 2
    all_indices = list(range(n_splits))
    combinations = list(itertools.combinations(all_indices, half))

    rank_correlations: list[float] = []
    for train_idx in combinations:
        test_idx = tuple(i for i in all_indices if i not in train_idx)
        is_perf = perf[:, list(train_idx)].sum(axis=1)
        oos_perf = perf[:, list(test_idx)].sum(axis=1)

        if n_trials == 2:
            # Spearman is undefined for n=2; use sign agreement.
            rc = 1.0 if (is_perf[0] - is_perf[1]) * (oos_perf[0] - oos_perf[1]) > 0 else -1.0
        else:
            rc, _ = sp_stats.spearmanr(is_perf, oos_perf)
            if np.isnan(rc):
                rc = 0.0
        rank_correlations.append(float(rc))

    n_neg = sum(1 for rc in rank_correlations if rc < 0)
    pbo = n_neg / len(rank_correlations) if rank_correlations else 0.0

    return {
        "pbo": pbo,
        "rank_correlations": rank_correlations,
        "n_combinations": len(rank_correlations),
    }


def min_backtest_length(
    sharpe: float,
    confidence: float = 0.95,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """Minimum backtest length to trust a given Sharpe ratio.

    Parameters
    ----------
    sharpe : float
        Observed annualised Sharpe ratio.
    confidence : float
        Desired confidence level (default 0.95).
    skewness : float
        Skewness of returns (default 0).
    kurtosis : float
        Kurtosis of returns (default 3, i.e. normal).

    Returns
    -------
    float
        Minimum number of observations needed.

    Raises
    ------
    ValueError
        If sharpe is zero (cannot determine minimum length).
    """
    if abs(sharpe) < 1e-15:
        raise ValueError("sharpe must be non-zero.")

    z_c = sp_stats.norm.ppf(confidence)
    excess_kurt = kurtosis - 3.0
    numerator = (
        1.0
        - skewness * sharpe
        + (excess_kurt / 4.0) * sharpe ** 2
    )
    n_min = numerator * (z_c / sharpe) ** 2
    return max(float(n_min), 1.0)


def walk_forward_validate(
    returns: np.ndarray | Sequence[float],
    strategy_fn: Callable[[np.ndarray], np.ndarray],
    window: int,
    step: int,
    expanding: bool = False,
) -> dict[str, Any]:
    """Walk-forward validation of a trading strategy.

    Iterates through the return series, calling ``strategy_fn`` on the
    training window to obtain weights, then evaluating performance on
    the subsequent test window.

    Parameters
    ----------
    returns : array-like
        1D array of asset returns.
    strategy_fn : callable
        Function that takes a training returns array and returns a float
        or array of weights/signals for the test period. If a scalar is
        returned, it is applied uniformly to the test window.
    window : int
        Training window size (number of observations).
    step : int
        Test window size (step forward per fold).
    expanding : bool
        If True, use an expanding window instead of rolling (default False).

    Returns
    -------
    dict
        Dictionary with keys:

        - ``folds``: list of per-fold dicts with train_start, train_end,
          test_start, test_end, test_return, and weight.
        - ``aggregate_return``: total return across all test folds.
        - ``n_folds``: number of folds evaluated.
        - ``mean_fold_return``: mean return per fold.
    """
    ret = np.asarray(returns, dtype=float)
    n = len(ret)
    folds: list[dict[str, Any]] = []

    pos = window
    while pos + step <= n:
        train_start = 0 if expanding else pos - window
        train_end = pos
        test_start = pos
        test_end = pos + step

        train_ret = ret[train_start:train_end]
        test_ret = ret[test_start:test_end]

        weight = strategy_fn(train_ret)
        weight_scalar = float(np.mean(weight))

        fold_return = float(np.sum(test_ret * weight_scalar))

        folds.append({
            "train_start": train_start,
            "train_end": train_end,
            "test_start": test_start,
            "test_end": test_end,
            "test_return": fold_return,
            "weight": weight_scalar,
        })

        pos += step

    agg = sum(f["test_return"] for f in folds) if folds else 0.0
    mean_ret = agg / len(folds) if folds else 0.0

    return {
        "folds": folds,
        "aggregate_return": agg,
        "n_folds": len(folds),
        "mean_fold_return": mean_ret,
    }


class TrialTracker:
    """Context manager for logging and analysing backtest trials.

    Tracks multiple strategy trials and provides overfitting diagnostics.

    Parameters
    ----------
    name : str
        Name of the strategy family being tested.

    Examples
    --------
    >>> with TrialTracker("momentum") as tracker:
    ...     tracker.log(params={"lookback": 20}, sharpe=1.2, returns=[0.01, 0.02])
    ...     tracker.log(params={"lookback": 40}, sharpe=0.8, returns=[0.005, 0.01])
    ...     prob = tracker.overfitting_probability()
    """

    def __init__(self, name: str = "unnamed") -> None:
        self.name = name
        self.trials: list[dict[str, Any]] = []

    def __enter__(self) -> TrialTracker:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        pass

    def log(
        self,
        params: dict[str, Any] | None = None,
        sharpe: float = 0.0,
        returns: np.ndarray | Sequence[float] | None = None,
    ) -> None:
        """Log a single backtest trial.

        Parameters
        ----------
        params : dict, optional
            Strategy parameters used in this trial.
        sharpe : float
            Sharpe ratio achieved.
        returns : array-like, optional
            Return series for this trial.
        """
        ret_arr = np.asarray(returns, dtype=float) if returns is not None else None
        self.trials.append({
            "params": params or {},
            "sharpe": sharpe,
            "returns": ret_arr,
        })

    def overfitting_probability(self, n_splits: int = 10) -> float:
        """Estimate the probability of backtest overfitting.

        Uses CSCV if return series are available for at least 2 trials;
        otherwise returns a heuristic based on the number of trials.

        Parameters
        ----------
        n_splits : int
            Number of splits for CSCV (default 10).

        Returns
        -------
        float
            Estimated probability of overfitting in [0, 1].
        """
        trials_with_returns = [
            t for t in self.trials if t["returns"] is not None
        ]

        if len(trials_with_returns) >= 2:
            # Align lengths.
            min_len = min(len(t["returns"]) for t in trials_with_returns)
            trial_mat = np.array([
                t["returns"][:min_len] for t in trials_with_returns
            ])
            # Ensure enough observations for splits.
            effective_splits = min(n_splits, min_len)
            if effective_splits % 2 != 0:
                effective_splits = max(effective_splits - 1, 2)
            result = probability_of_backtest_overfitting(
                trial_mat, n_splits=effective_splits,
            )
            return result["pbo"]

        # Heuristic fallback: more trials means higher overfitting risk.
        n = len(self.trials)
        if n <= 1:
            return 0.0
        return min(1.0 - 1.0 / n, 0.99)

    @property
    def best_trial(self) -> dict[str, Any] | None:
        """Return the trial with the highest Sharpe ratio.

        Returns
        -------
        dict or None
            Best trial dict, or None if no trials logged.
        """
        if not self.trials:
            return None
        return max(self.trials, key=lambda t: t["sharpe"])
