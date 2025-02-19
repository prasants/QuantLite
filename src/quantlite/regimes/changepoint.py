"""Change-point detection in financial return series.

Provides CUSUM-based and Bayesian online change-point detection
(Adams and MacKay, 2007) for identifying structural breaks in the
mean or variance of returns.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

__all__ = [
    "ChangePoint",
    "detect_changepoints",
]


@dataclass(frozen=True)
class ChangePoint:
    """A detected structural break in the return series.

    Attributes:
        index: Integer position in the series.
        date: Datetime if the input had a DatetimeIndex, else ``None``.
        confidence: Detection confidence (0 to 1).
        direction: Description of the change (e.g. ``"increase_mean"``).
    """

    index: int
    date: object | None
    confidence: float
    direction: str

    def __repr__(self) -> str:
        loc = self.date if self.date is not None else self.index
        return (
            f"ChangePoint(loc={loc}, confidence={self.confidence:.3f}, "
            f"direction={self.direction!r})"
        )


def _cusum_detection(
    returns: np.ndarray,
    penalty: float,
) -> list[tuple[int, float, str]]:
    """Detect change points via CUSUM on standardised cumulative sums.

    Args:
        returns: Return array.
        penalty: Threshold for detection (higher = fewer points).

    Returns:
        List of (index, confidence, direction) tuples.
    """
    n = len(returns)
    mu = returns.mean()
    sigma = returns.std(ddof=1)
    if sigma < 1e-15:
        return []

    standardised = (returns - mu) / sigma
    cusum_pos = np.zeros(n)
    cusum_neg = np.zeros(n)

    changepoints: list[tuple[int, float, str]] = []

    for i in range(1, n):
        cusum_pos[i] = max(0, cusum_pos[i - 1] + standardised[i])
        cusum_neg[i] = max(0, cusum_neg[i - 1] - standardised[i])

        if cusum_pos[i] > penalty:
            confidence = min(1.0, cusum_pos[i] / (2 * penalty))
            changepoints.append((i, confidence, "increase_mean"))
            cusum_pos[i] = 0
        elif cusum_neg[i] > penalty:
            confidence = min(1.0, cusum_neg[i] / (2 * penalty))
            changepoints.append((i, confidence, "decrease_mean"))
            cusum_neg[i] = 0

    return changepoints


def _bayesian_online_detection(
    returns: np.ndarray,
    penalty: float,
) -> list[tuple[int, float, str]]:
    """Bayesian online change-point detection (Adams and MacKay, 2007).

    Implements the recursive message-passing algorithm with a
    normal-inverse-gamma conjugate prior and Student-t predictive
    distribution.

    Args:
        returns: Return array.
        penalty: Hazard rate parameter (1/penalty = prior probability
            of change at each step).

    Returns:
        List of (index, confidence, direction) tuples.
    """
    from scipy.stats import t as student_t

    n = len(returns)
    hazard = 1.0 / penalty

    # Weakly informative prior: use data scale for beta0
    data_var = max(np.var(returns), 1e-15)
    mu0 = float(np.mean(returns))
    kappa0 = 0.01  # weak prior on mean
    alpha0 = 0.01  # weak prior on variance
    beta0 = data_var * alpha0  # centres prior variance near data variance

    max_rl = n + 1
    run_probs = np.zeros(max_rl)
    run_probs[0] = 1.0

    muN = np.zeros(max_rl) + mu0
    kappaN = np.zeros(max_rl) + kappa0
    alphaN = np.zeros(max_rl) + alpha0
    betaN = np.zeros(max_rl) + beta0

    map_run_lengths = np.zeros(n, dtype=int)
    changepoints: list[tuple[int, float, str]] = []

    # Track segment means for direction detection
    segment_start = 0

    for t in range(n):
        x = returns[t]

        # Student-t predictive distribution
        df = 2 * alphaN[:t + 1]
        pred_scale = np.sqrt(betaN[:t + 1] * (kappaN[:t + 1] + 1) / (
            alphaN[:t + 1] * kappaN[:t + 1]
        ))
        pred_scale = np.maximum(pred_scale, 1e-15)
        df = np.maximum(df, 0.01)

        pred_probs = student_t.pdf(x, df=df, loc=muN[:t + 1], scale=pred_scale)
        pred_probs = np.maximum(pred_probs, 1e-300)

        growth = run_probs[:t + 1] * pred_probs * (1 - hazard)
        cp = np.sum(run_probs[:t + 1] * pred_probs * hazard)

        new_probs = np.zeros(max_rl)
        new_probs[0] = cp
        new_probs[1:t + 2] = growth
        evidence = new_probs[:t + 2].sum()
        if evidence > 0:
            new_probs[:t + 2] /= evidence
        run_probs = new_probs

        # Update sufficient statistics
        old_mu = muN[:t + 1].copy()
        old_kappa = kappaN[:t + 1].copy()
        old_alpha = alphaN[:t + 1].copy()
        old_beta = betaN[:t + 1].copy()

        new_kappaN_vals = old_kappa + 1
        new_muN_vals = (old_kappa * old_mu + x) / new_kappaN_vals
        new_alphaN_vals = old_alpha + 0.5
        new_betaN_vals = (
            old_beta
            + 0.5 * old_kappa * (x - old_mu) ** 2 / new_kappaN_vals
        )

        # Shift: index r+1 gets the updated stats from run length r
        new_muN = np.zeros(max_rl) + mu0
        new_kappaN = np.zeros(max_rl) + kappa0
        new_alphaN = np.zeros(max_rl) + alpha0
        new_betaN = np.zeros(max_rl) + beta0

        new_muN[1:t + 2] = new_muN_vals
        new_kappaN[1:t + 2] = new_kappaN_vals
        new_alphaN[1:t + 2] = new_alphaN_vals
        new_betaN[1:t + 2] = new_betaN_vals

        muN = new_muN
        kappaN = new_kappaN
        alphaN = new_alphaN
        betaN = new_betaN

        map_rl = int(np.argmax(run_probs[:t + 2]))
        map_run_lengths[t] = map_rl

        # Detect: MAP run length drops significantly
        if t > 10 and map_run_lengths[t] < 5 and map_run_lengths[t - 1] > 10:
            prev_mean = float(np.mean(returns[segment_start:t]))
            # Estimate new segment mean from recent observations
            confidence = min(1.0, float(
                np.sum(run_probs[:5]) / max(np.sum(run_probs[:t + 2]), 1e-15)
            ))
            direction = (
                "increase_mean" if x > prev_mean
                else "decrease_mean"
            )
            changepoints.append((t, max(confidence, 0.3), direction))
            segment_start = t

    return changepoints


def detect_changepoints(
    returns: np.ndarray | pd.Series,
    method: str = "cusum",
    penalty: float = 5.0,
) -> list[ChangePoint]:
    """Detect structural breaks in a return series.

    Args:
        returns: Simple periodic returns.
        method: Detection algorithm: ``"cusum"`` or ``"bayesian"``.
        penalty: Sensitivity parameter. For CUSUM, the detection
            threshold (higher = fewer detections). For Bayesian,
            the inverse hazard rate.

    Returns:
        List of ``ChangePoint`` dataclasses sorted by index.

    Raises:
        ValueError: On unknown method.
    """
    has_dates = isinstance(returns, pd.Series) and isinstance(
        returns.index, pd.DatetimeIndex
    )
    dates = returns.index if has_dates else None
    arr = np.asarray(returns, dtype=float)

    if method == "cusum":
        raw = _cusum_detection(arr, penalty)
    elif method == "bayesian":
        raw = _bayesian_online_detection(arr, penalty)
    else:
        raise ValueError(
            f"Unknown method: {method}. Use 'cusum' or 'bayesian'."
        )

    result: list[ChangePoint] = []
    for idx, conf, direction in raw:
        date = dates[idx] if dates is not None else None
        result.append(ChangePoint(
            index=idx, date=date, confidence=conf, direction=direction,
        ))

    return sorted(result, key=lambda cp: cp.index)
