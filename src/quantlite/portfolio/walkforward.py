"""Walk-forward optimisation framework.

Implements expanding and sliding window walk-forward analysis with
in-sample optimisation and out-of-sample evaluation. Supports multiple
scoring functions and produces a results object with per-fold metrics.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = [
    "WalkForwardFold",
    "WalkForwardResult",
    "walk_forward",
    "sharpe_score",
    "sortino_score",
    "calmar_score",
    "max_drawdown_score",
]


@dataclass(frozen=True)
class WalkForwardFold:
    """Metrics for a single walk-forward fold.

    Attributes:
        fold_index: Zero-based fold number.
        is_start: Start index of the in-sample window.
        is_end: End index of the in-sample window.
        oos_start: Start index of the out-of-sample window.
        oos_end: End index of the out-of-sample window.
        is_score: In-sample score.
        oos_score: Out-of-sample score.
        weights: Optimised weights from in-sample.
        oos_returns: Out-of-sample portfolio returns.
    """

    fold_index: int
    is_start: int
    is_end: int
    oos_start: int
    oos_end: int
    is_score: float
    oos_score: float
    weights: dict[str, float]
    oos_returns: np.ndarray


@dataclass(frozen=True)
class WalkForwardResult:
    """Aggregate results from walk-forward optimisation.

    Attributes:
        folds: List of per-fold results.
        aggregate_score: Mean out-of-sample score across folds.
        aggregate_std: Standard deviation of out-of-sample scores.
        equity_curve: Concatenated out-of-sample equity curve.
        scoring_function: Name of the scoring function used.
        window_type: ``"expanding"`` or ``"sliding"``.
    """

    folds: list[WalkForwardFold]
    aggregate_score: float
    aggregate_std: float
    equity_curve: np.ndarray
    scoring_function: str
    window_type: str


def sharpe_score(returns: np.ndarray, risk_free: float = 0.0) -> float:
    """Compute annualised Sharpe ratio.

    Args:
        returns: Array of simple returns.
        risk_free: Annualised risk-free rate.

    Returns:
        Annualised Sharpe ratio.
    """
    if len(returns) < 2:
        return 0.0
    mu = float(np.mean(returns))
    sigma = float(np.std(returns, ddof=1))
    if sigma < 1e-12:
        return 0.0
    rf_periodic = (1 + risk_free) ** (1 / 252) - 1
    return float((mu - rf_periodic) / sigma * np.sqrt(252))


def sortino_score(returns: np.ndarray, risk_free: float = 0.0) -> float:
    """Compute annualised Sortino ratio.

    Args:
        returns: Array of simple returns.
        risk_free: Annualised risk-free rate.

    Returns:
        Annualised Sortino ratio.
    """
    if len(returns) < 2:
        return 0.0
    mu = float(np.mean(returns))
    rf_periodic = (1 + risk_free) ** (1 / 252) - 1
    downside = returns[returns < rf_periodic] - rf_periodic
    if len(downside) < 1:
        return 0.0
    downside_std = float(np.sqrt(np.mean(downside ** 2)))
    if downside_std < 1e-12:
        return 0.0
    return float((mu - rf_periodic) / downside_std * np.sqrt(252))


def calmar_score(returns: np.ndarray) -> float:
    """Compute Calmar ratio (annualised return / max drawdown).

    Args:
        returns: Array of simple returns.

    Returns:
        Calmar ratio (positive is better).
    """
    if len(returns) < 2:
        return 0.0
    equity = np.cumprod(1.0 + returns)
    running_max = np.maximum.accumulate(equity)
    drawdowns = (equity - running_max) / running_max
    max_dd = float(np.min(drawdowns))
    if abs(max_dd) < 1e-12:
        return 0.0
    ann_ret = float(equity[-1] ** (252.0 / len(returns)) - 1.0)
    return ann_ret / abs(max_dd)


def max_drawdown_score(returns: np.ndarray) -> float:
    """Compute maximum drawdown (returned as negative value; less negative is better).

    Args:
        returns: Array of simple returns.

    Returns:
        Maximum drawdown (negative number).
    """
    if len(returns) < 2:
        return 0.0
    equity = np.cumprod(1.0 + returns)
    running_max = np.maximum.accumulate(equity)
    drawdowns = (equity - running_max) / running_max
    return float(np.min(drawdowns))


def walk_forward(
    returns_df: pd.DataFrame,
    optimiser: Callable[[pd.DataFrame], dict[str, float]],
    is_window: int = 252,
    oos_window: int = 63,
    window_type: str = "expanding",
    scoring: str | Callable[[np.ndarray], float] = "sharpe",
    min_is_observations: int = 60,
) -> WalkForwardResult:
    """Run walk-forward optimisation.

    Splits the data into sequential in-sample/out-of-sample windows,
    optimises weights on each in-sample period, and evaluates on the
    subsequent out-of-sample period.

    Args:
        returns_df: DataFrame of asset returns (assets as columns).
        optimiser: Callable that takes an in-sample returns DataFrame
            and returns a weight dictionary.
        is_window: In-sample window size in periods. For expanding
            windows, this is the minimum initial window.
        oos_window: Out-of-sample window size in periods.
        window_type: ``"expanding"`` or ``"sliding"``.
        scoring: Scoring function. One of ``"sharpe"``, ``"sortino"``,
            ``"calmar"``, ``"max_dd"``, or a callable.
        min_is_observations: Minimum in-sample observations required.

    Returns:
        ``WalkForwardResult`` with per-fold and aggregate metrics.

    Raises:
        ValueError: If window_type is not recognised or data is too short.
    """
    if window_type not in ("expanding", "sliding"):
        raise ValueError(f"window_type must be 'expanding' or 'sliding', got {window_type!r}")

    # Resolve scoring function
    score_fn: Callable[[np.ndarray], float]
    score_name: str
    if callable(scoring):
        score_fn = scoring
        score_name = getattr(scoring, "__name__", "custom")
    elif scoring == "sharpe":
        score_fn = sharpe_score
        score_name = "sharpe"
    elif scoring == "sortino":
        score_fn = sortino_score
        score_name = "sortino"
    elif scoring == "calmar":
        score_fn = calmar_score
        score_name = "calmar"
    elif scoring == "max_dd":
        score_fn = max_drawdown_score
        score_name = "max_drawdown"
    else:
        raise ValueError(f"Unknown scoring function: {scoring!r}")

    n = len(returns_df)
    names = list(returns_df.columns)

    if n < is_window + oos_window:
        raise ValueError(
            f"Data length ({n}) is shorter than is_window + oos_window "
            f"({is_window + oos_window})"
        )

    folds: list[WalkForwardFold] = []
    all_oos_returns: list[np.ndarray] = []
    fold_idx = 0

    oos_start = is_window

    while oos_start + oos_window <= n:
        oos_end = oos_start + oos_window

        is_start = 0 if window_type == "expanding" else max(0, oos_start - is_window)

        is_end = oos_start

        if (is_end - is_start) < min_is_observations:
            oos_start += oos_window
            continue

        # In-sample optimisation
        is_data = returns_df.iloc[is_start:is_end]
        weights = optimiser(is_data)

        # Out-of-sample evaluation
        oos_data = returns_df.iloc[oos_start:oos_end]
        w_arr = np.array([weights.get(name, 0.0) for name in names])
        oos_rets = oos_data.values @ w_arr

        is_rets = is_data.values @ w_arr
        is_score = float(score_fn(is_rets))
        oos_score = float(score_fn(oos_rets))

        folds.append(WalkForwardFold(
            fold_index=fold_idx,
            is_start=is_start,
            is_end=is_end,
            oos_start=oos_start,
            oos_end=oos_end,
            is_score=is_score,
            oos_score=oos_score,
            weights=weights,
            oos_returns=oos_rets,
        ))

        all_oos_returns.append(oos_rets)
        fold_idx += 1
        oos_start += oos_window

    if not folds:
        raise ValueError("No folds could be generated with the given parameters")

    # Aggregate
    oos_scores = [f.oos_score for f in folds]
    agg_score = float(np.mean(oos_scores))
    agg_std = float(np.std(oos_scores, ddof=1)) if len(oos_scores) > 1 else 0.0

    # Build equity curve
    concat_rets = np.concatenate(all_oos_returns)
    equity = np.cumprod(1.0 + concat_rets)
    equity = np.insert(equity, 0, 1.0)

    return WalkForwardResult(
        folds=folds,
        aggregate_score=agg_score,
        aggregate_std=agg_std,
        equity_curve=equity,
        scoring_function=score_name,
        window_type=window_type,
    )
