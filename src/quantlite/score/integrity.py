"""Track-record integrity checks for the QuantLite Score.

A score is only as trustworthy as its inputs. These checks detect the
common ways a track record misleads: return smoothing, dependence on a
handful of lucky days, the short-volatility signature (high win rate,
brutal left tail), cherry-picked start dates, and records too short to
say anything at all.

Every check is deterministic and threshold-driven. Thresholds are module
constants, documented in the QuantLite Score specification, and frozen
within a methodology version.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

__all__ = [
    "IntegrityFlag",
    "validate_track_record",
]

# Severity levels.
WARNING = "warning"
CRITICAL = "critical"

# Thresholds (QLS-1.0). Frozen for the lifetime of the methodology version.
MIN_OBSERVATIONS = 30
SHORT_RECORD_FRACTION = 0.5  # of one year, i.e. freq * 0.5 observations
SMOOTHING_AUTOCORR_WARNING = 0.30
SMOOTHING_AUTOCORR_CRITICAL = 0.50
OUTLIER_TOP_N = 5
OUTLIER_SHARE_THRESHOLD = 0.50
WIN_RATE_THRESHOLD = 0.85
WIN_RATE_SKEW_THRESHOLD = -1.0
START_SENSITIVITY_TRIM = 0.10
START_SENSITIVITY_DECAY = 0.50
ZERO_VARIANCE_EPS = 1e-12


@dataclass(frozen=True)
class IntegrityFlag:
    """A single integrity finding on a track record.

    Attributes
    ----------
    code : str
        Stable machine-readable identifier (e.g. ``"excessive_smoothing"``).
    severity : str
        Either ``"warning"`` or ``"critical"``.
    detail : str
        Human-readable explanation with the measured value.
    """

    code: str
    severity: str
    detail: str


def _lag1_autocorrelation(arr: np.ndarray) -> float:
    """Lag-1 autocorrelation of a return series.

    Parameters
    ----------
    arr : numpy.ndarray
        Return series.

    Returns
    -------
    float
        Lag-1 autocorrelation, or 0.0 if undefined.
    """
    if len(arr) < 3:
        return 0.0
    a, b = arr[:-1], arr[1:]
    sa, sb = np.std(a), np.std(b)
    if sa < ZERO_VARIANCE_EPS or sb < ZERO_VARIANCE_EPS:
        return 0.0
    corr = float(np.corrcoef(a, b)[0, 1])
    return 0.0 if np.isnan(corr) else corr


def _per_period_sharpe(arr: np.ndarray) -> float:
    """Non-annualised Sharpe ratio (mean over standard deviation).

    Parameters
    ----------
    arr : numpy.ndarray
        Return series.

    Returns
    -------
    float
        Per-period Sharpe, or 0.0 if volatility is zero.
    """
    if len(arr) < 2:
        return 0.0
    std = float(np.std(arr, ddof=1))
    if std < ZERO_VARIANCE_EPS:
        return 0.0
    return float(np.mean(arr)) / std


def validate_track_record(
    returns: np.ndarray | Sequence[float],
    freq: int = 252,
) -> tuple[IntegrityFlag, ...]:
    """Run all QLS-1.0 integrity checks on a return series.

    Checks performed:

    - ``non_finite_values``: NaN or infinite returns (critical).
    - ``zero_variance``: constant returns (critical).
    - ``short_record``: fewer than half a year of observations (critical).
    - ``excessive_smoothing``: high lag-1 autocorrelation, the signature
      of smoothed or stale marks (warning above 0.30, critical above 0.50).
    - ``outlier_dependence``: the top 5 days contribute more than half of
      all positive log returns (warning).
    - ``short_volatility_signature``: win rate above 85% combined with
      skewness below -1, the classic pattern of strategies that sell tail
      risk (warning).
    - ``start_date_sensitivity``: dropping the first 10% of observations
      cuts the Sharpe ratio by more than half, suggesting a cherry-picked
      start date (warning).

    Parameters
    ----------
    returns : array-like
        Simple periodic returns.
    freq : int
        Periods per year (default 252).

    Returns
    -------
    tuple of IntegrityFlag
        All triggered flags, possibly empty. Order is deterministic.
    """
    arr = np.asarray(returns, dtype=float)
    flags: list[IntegrityFlag] = []

    n = len(arr)
    if n == 0 or not np.all(np.isfinite(arr)):
        flags.append(IntegrityFlag(
            code="non_finite_values",
            severity=CRITICAL,
            detail="Track record is empty or contains NaN/infinite returns.",
        ))
        return tuple(flags)

    if float(np.std(arr)) < ZERO_VARIANCE_EPS:
        flags.append(IntegrityFlag(
            code="zero_variance",
            severity=CRITICAL,
            detail="Returns are constant; no risk was taken or marks are synthetic.",
        ))
        return tuple(flags)

    short_threshold = max(MIN_OBSERVATIONS, int(freq * SHORT_RECORD_FRACTION))
    if n < short_threshold:
        flags.append(IntegrityFlag(
            code="short_record",
            severity=CRITICAL,
            detail=(
                f"Only {n} observations; at least {short_threshold} required "
                f"for a meaningful score at freq={freq}."
            ),
        ))

    ac1 = _lag1_autocorrelation(arr)
    if ac1 > SMOOTHING_AUTOCORR_CRITICAL:
        flags.append(IntegrityFlag(
            code="excessive_smoothing",
            severity=CRITICAL,
            detail=(
                f"Lag-1 autocorrelation {ac1:.2f} exceeds "
                f"{SMOOTHING_AUTOCORR_CRITICAL}; returns appear smoothed or stale."
            ),
        ))
    elif ac1 > SMOOTHING_AUTOCORR_WARNING:
        flags.append(IntegrityFlag(
            code="excessive_smoothing",
            severity=WARNING,
            detail=(
                f"Lag-1 autocorrelation {ac1:.2f} exceeds "
                f"{SMOOTHING_AUTOCORR_WARNING}; possible return smoothing."
            ),
        ))

    # Outlier dependence: share of positive log growth from the best days.
    log_returns = np.log1p(np.clip(arr, -0.9999, None))
    positive = log_returns[log_returns > 0]
    if len(positive) > OUTLIER_TOP_N:
        top = np.sort(positive)[-OUTLIER_TOP_N:]
        share = float(np.sum(top) / np.sum(positive))
        if share > OUTLIER_SHARE_THRESHOLD:
            flags.append(IntegrityFlag(
                code="outlier_dependence",
                severity=WARNING,
                detail=(
                    f"Top {OUTLIER_TOP_N} days contribute {share:.0%} of all "
                    f"positive log returns; performance hinges on a few outliers."
                ),
            ))

    # Short-volatility signature: steady small wins, rare catastrophic losses.
    win_rate = float(np.mean(arr > 0))
    if n >= MIN_OBSERVATIONS and win_rate > WIN_RATE_THRESHOLD:
        centred = arr - np.mean(arr)
        std = float(np.std(arr, ddof=1))
        skewness = float(np.mean(centred**3) / std**3) if std > ZERO_VARIANCE_EPS else 0.0
        if skewness < WIN_RATE_SKEW_THRESHOLD:
            flags.append(IntegrityFlag(
                code="short_volatility_signature",
                severity=WARNING,
                detail=(
                    f"Win rate {win_rate:.0%} with skewness {skewness:.2f}; "
                    f"pattern consistent with selling tail risk."
                ),
            ))

    # Start-date sensitivity: does the edge survive without the opening run?
    trim = int(n * START_SENSITIVITY_TRIM)
    if trim >= 1 and n - trim >= MIN_OBSERVATIONS:
        full_sharpe = _per_period_sharpe(arr)
        trimmed_sharpe = _per_period_sharpe(arr[trim:])
        if full_sharpe > 0 and trimmed_sharpe < full_sharpe * START_SENSITIVITY_DECAY:
            flags.append(IntegrityFlag(
                code="start_date_sensitivity",
                severity=WARNING,
                detail=(
                    f"Sharpe falls from {full_sharpe:.3f} to {trimmed_sharpe:.3f} "
                    f"when the first {START_SENSITIVITY_TRIM:.0%} of observations "
                    f"are dropped; the start date may be cherry-picked."
                ),
            ))

    return tuple(flags)
