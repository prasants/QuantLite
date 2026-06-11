"""The QuantLite Score engine.

Combines the library's strategy forensics, bootstrap resampling, and tail
risk metrics into a single deterministic 0-100 score with a letter grade.
The methodology is open, versioned (``QLS-1.0``), and documented in
``docs/score.md``. Same inputs, same parameters, same library version:
same score, bit for bit.

Components and weights (QLS-1.0):

- **skill (35%):** Deflated Sharpe Ratio. The probability the observed
  Sharpe is genuine after accounting for how many strategies were tried.
- **robustness (20%):** 5th percentile of the bootstrapped annualised
  Sharpe distribution, mapped through a logistic squash. Rewards edges
  that survive resampling, not point estimates.
- **tail (20%):** Maximum drawdown and CVaR(95%), blended. Punishes
  strategies that win small and lose catastrophically.
- **consistency (15%):** Fraction of rolling windows with positive mean
  return. Rewards edges that show up throughout the record.
- **sufficiency (10%):** Track record length relative to the Minimum
  Track Record Length for the observed Sharpe.

Integrity flags cap the composite: any critical flag caps the score at
40, any warning at 70. The flags travel with the artifact, so a capped
score is always explainable.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np

from .. import __version__ as _library_version
from ..forensics import deflated_sharpe_ratio, min_track_record_length
from ..metrics import annualised_return, annualised_volatility, sharpe_ratio
from ..resample import stationary_bootstrap
from ..risk.metrics import cvar
from .artifact import ScoreArtifact, input_digest
from .integrity import (
    CRITICAL,
    MIN_OBSERVATIONS,
    WARNING,
    ZERO_VARIANCE_EPS,
    IntegrityFlag,
    validate_track_record,
)

__all__ = [
    "METHODOLOGY_VERSION",
    "DEFAULT_SEED",
    "COMPONENT_WEIGHTS",
    "ScoreResult",
    "compute_score",
    "verify_artifact",
]

METHODOLOGY_VERSION = "QLS-1.0"
DEFAULT_SEED = 1729
DEFAULT_BOOTSTRAP_SAMPLES = 1000

COMPONENT_WEIGHTS: dict[str, float] = {
    "skill": 0.35,
    "robustness": 0.20,
    "tail": 0.20,
    "consistency": 0.15,
    "sufficiency": 0.10,
}

# Score caps applied when integrity flags are present.
CRITICAL_FLAG_CAP = 40.0
WARNING_FLAG_CAP = 70.0

# Component calibration constants (QLS-1.0, frozen).
ROBUSTNESS_LOGISTIC_SLOPE = 1.2
TAIL_DRAWDOWN_FLOOR = 0.50      # drawdowns at or beyond -50% score zero
TAIL_CVAR_FLOOR = 0.80          # annualised-unit CVaR at or beyond 80% scores zero
TAIL_DRAWDOWN_WEIGHT = 0.6
TAIL_CVAR_WEIGHT = 0.4
CONSISTENCY_MIN_WINDOW = 20
SUFFICIENCY_FLOOR_YEARS = 1.0

GRADE_BANDS: tuple[tuple[float, str], ...] = (
    (90.0, "A+"),
    (80.0, "A"),
    (65.0, "B"),
    (50.0, "C"),
    (35.0, "D"),
    (0.0, "F"),
)


@dataclass(frozen=True)
class ScoreResult:
    """Result of a QuantLite Score computation.

    Attributes
    ----------
    score : float
        Composite score in [0, 100], after integrity caps.
    grade : str
        Letter grade: A+, A, B, C, D, or F.
    components : dict
        Component scores in [0, 100] before weighting.
    metrics : dict
        Supporting raw metrics (annualised Sharpe, max drawdown, etc.).
    flags : tuple of IntegrityFlag
        Integrity findings on the track record.
    artifact : ScoreArtifact
        Portable, hashable record of this computation.
    """

    score: float
    grade: str
    components: dict[str, float]
    metrics: dict[str, float]
    flags: tuple[IntegrityFlag, ...]
    artifact: ScoreArtifact


def _grade(score: float) -> str:
    """Map a composite score to a letter grade.

    Parameters
    ----------
    score : float
        Composite score in [0, 100].

    Returns
    -------
    str
        Letter grade.
    """
    for threshold, grade in GRADE_BANDS:
        if score >= threshold:
            return grade
    return "F"


def _max_drawdown(arr: np.ndarray) -> float:
    """Maximum drawdown as a negative fraction.

    Parameters
    ----------
    arr : numpy.ndarray
        Return series.

    Returns
    -------
    float
        Maximum drawdown (e.g. -0.25 for a 25% drawdown).
    """
    cum = np.cumprod(1.0 + arr)
    roll_max = np.maximum.accumulate(cum)
    dd = (cum - roll_max) / roll_max
    return float(dd.min()) if len(dd) else 0.0


def _annualised_sharpe(arr: np.ndarray, freq: int) -> float:
    """Annualised Sharpe ratio of a returns array.

    Parameters
    ----------
    arr : numpy.ndarray
        Return series.
    freq : int
        Periods per year.

    Returns
    -------
    float
        Annualised Sharpe, or 0.0 if volatility is zero.
    """
    if len(arr) < 2:
        return 0.0
    std = float(np.std(arr, ddof=1))
    if std < ZERO_VARIANCE_EPS:
        return 0.0
    return float(np.mean(arr) / std * math.sqrt(freq))


def _skill_component(arr: np.ndarray, n_trials: int) -> tuple[float, float]:
    """Skill component: Deflated Sharpe Ratio scaled to [0, 100].

    Parameters
    ----------
    arr : numpy.ndarray
        Return series.
    n_trials : int
        Number of strategy trials conducted.

    Returns
    -------
    tuple of float
        ``(component_score, dsr)``.
    """
    n = len(arr)
    std = float(np.std(arr, ddof=1))
    per_period_sharpe = float(np.mean(arr)) / std if std > ZERO_VARIANCE_EPS else 0.0
    centred = arr - np.mean(arr)
    skewness = float(np.mean(centred**3) / std**3) if std > ZERO_VARIANCE_EPS else 0.0
    kurt = float(np.mean(centred**4) / std**4) if std > ZERO_VARIANCE_EPS else 3.0
    dsr = deflated_sharpe_ratio(
        per_period_sharpe, n_trials=n_trials, n_obs=n,
        skewness=skewness, kurtosis=kurt,
    )
    return dsr * 100.0, dsr


def _robustness_component(
    arr: np.ndarray, freq: int, n_bootstrap: int, seed: int,
) -> tuple[float, float]:
    """Robustness component: bootstrapped 5th percentile Sharpe, squashed.

    Parameters
    ----------
    arr : numpy.ndarray
        Return series.
    freq : int
        Periods per year.
    n_bootstrap : int
        Number of stationary bootstrap samples.
    seed : int
        Random seed.

    Returns
    -------
    tuple of float
        ``(component_score, sharpe_p5)``.
    """
    block = max(1, int(math.sqrt(len(arr))))
    samples = stationary_bootstrap(arr, block, n_samples=n_bootstrap, seed=seed)
    sharpes = np.array([_annualised_sharpe(samples[i], freq) for i in range(n_bootstrap)])
    sharpe_p5 = float(np.percentile(sharpes, 5))
    component = 100.0 / (1.0 + math.exp(-ROBUSTNESS_LOGISTIC_SLOPE * sharpe_p5))
    return component, sharpe_p5


def _tail_component(arr: np.ndarray, freq: int) -> tuple[float, float, float]:
    """Tail component: blended drawdown and CVaR(95%) penalty.

    Parameters
    ----------
    arr : numpy.ndarray
        Return series.
    freq : int
        Periods per year.

    Returns
    -------
    tuple of float
        ``(component_score, max_drawdown, cvar_95)``.
    """
    max_dd = _max_drawdown(arr)
    cvar_95 = float(cvar(arr, alpha=0.05))
    dd_score = max(0.0, 1.0 - abs(max_dd) / TAIL_DRAWDOWN_FLOOR)
    cvar_annual_unit = abs(cvar_95) * math.sqrt(freq)
    cvar_score = max(0.0, 1.0 - cvar_annual_unit / TAIL_CVAR_FLOOR)
    component = (TAIL_DRAWDOWN_WEIGHT * dd_score + TAIL_CVAR_WEIGHT * cvar_score) * 100.0
    return component, max_dd, cvar_95


def _consistency_component(arr: np.ndarray, freq: int) -> tuple[float, float]:
    """Consistency component: share of rolling windows with positive mean.

    Window length is one quarter (freq // 4, floored at 20), stepped by a
    third of the window, so windows overlap but the count stays small and
    deterministic.

    Parameters
    ----------
    arr : numpy.ndarray
        Return series.
    freq : int
        Periods per year.

    Returns
    -------
    tuple of float
        ``(component_score, positive_window_fraction)``.
    """
    window = max(CONSISTENCY_MIN_WINDOW, freq // 4)
    n = len(arr)
    if n < window:
        fraction = float(np.mean(arr) > 0)
        return fraction * 100.0, fraction
    step = max(1, window // 3)
    means = [float(np.mean(arr[s: s + window])) for s in range(0, n - window + 1, step)]
    fraction = float(np.mean([m > 0 for m in means]))
    return fraction * 100.0, fraction


def _sufficiency_component(arr: np.ndarray, freq: int) -> tuple[float, float]:
    """Sufficiency component: record length versus the Minimum Track Record Length.

    A non-positive Sharpe scores zero: no amount of history makes a
    non-positive edge sufficient. Otherwise the score is the ratio of
    observations to MinTRL (floored at one year), capped at 100.

    Parameters
    ----------
    arr : numpy.ndarray
        Return series.
    freq : int
        Periods per year.

    Returns
    -------
    tuple of float
        ``(component_score, min_trl)``. ``min_trl`` is ``inf`` when the
        Sharpe is non-positive.
    """
    n = len(arr)
    std = float(np.std(arr, ddof=1))
    per_period_sharpe = float(np.mean(arr)) / std if std > ZERO_VARIANCE_EPS else 0.0
    if per_period_sharpe <= 0:
        return 0.0, float("inf")
    centred = arr - np.mean(arr)
    skewness = float(np.mean(centred**3) / std**3)
    kurt = float(np.mean(centred**4) / std**4)
    min_trl = min_track_record_length(
        per_period_sharpe, benchmark_sharpe=0.0,
        skewness=skewness, kurtosis=kurt,
    )
    floor = freq * SUFFICIENCY_FLOOR_YEARS
    ratio = n / max(min_trl, floor)
    return min(1.0, ratio) * 100.0, float(min_trl)


def compute_score(
    returns: np.ndarray | Sequence[float],
    n_trials: int = 1,
    freq: int = 252,
    n_bootstrap: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = DEFAULT_SEED,
) -> ScoreResult:
    """Compute the QuantLite Score (QLS-1.0) for a track record.

    The computation is deterministic: the same returns, parameters, and
    library version always produce the same score and the same artifact
    content hash.

    Parameters
    ----------
    returns : array-like
        Simple periodic returns of the strategy or trader.
    n_trials : int
        Number of strategy variants tried before this track record was
        selected (default 1). Understating this inflates the skill
        component; verification services should require it to be logged.
    freq : int
        Periods per year (default 252 for daily returns).
    n_bootstrap : int
        Number of stationary bootstrap samples for the robustness
        component (default 1000).
    seed : int
        Random seed for all stochastic steps (default 1729).

    Returns
    -------
    ScoreResult
        Composite score, grade, components, metrics, integrity flags,
        and a verifiable artifact.

    Raises
    ------
    ValueError
        If the series is shorter than 30 observations or contains
        non-finite values.
    """
    arr = np.asarray(returns, dtype=float)
    if len(arr) < MIN_OBSERVATIONS:
        raise ValueError(
            f"Need at least {MIN_OBSERVATIONS} observations to score, got {len(arr)}."
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError("Returns contain NaN or infinite values.")
    if n_trials < 1:
        raise ValueError("n_trials must be at least 1.")

    flags = validate_track_record(arr, freq=freq)

    skill, dsr = _skill_component(arr, n_trials)
    robustness, sharpe_p5 = _robustness_component(arr, freq, n_bootstrap, seed)
    tail, max_dd, cvar_95 = _tail_component(arr, freq)
    consistency, positive_fraction = _consistency_component(arr, freq)
    sufficiency, min_trl = _sufficiency_component(arr, freq)

    components = {
        "skill": round(skill, 6),
        "robustness": round(robustness, 6),
        "tail": round(tail, 6),
        "consistency": round(consistency, 6),
        "sufficiency": round(sufficiency, 6),
    }

    composite = sum(COMPONENT_WEIGHTS[name] * value for name, value in components.items())

    severities = {f.severity for f in flags}
    if CRITICAL in severities:
        composite = min(composite, CRITICAL_FLAG_CAP)
    elif WARNING in severities:
        composite = min(composite, WARNING_FLAG_CAP)

    composite = round(float(np.clip(composite, 0.0, 100.0)), 6)
    grade = _grade(composite)

    metrics = {
        "annualised_return": round(annualised_return(arr, freq=freq), 6),
        "annualised_volatility": round(annualised_volatility(arr, freq=freq), 6),
        "sharpe_ratio": round(sharpe_ratio(arr, freq=freq), 6),
        "deflated_sharpe_ratio": round(dsr, 6),
        "bootstrap_sharpe_p5": round(sharpe_p5, 6),
        "max_drawdown": round(max_dd, 6),
        "cvar_95": round(cvar_95, 6),
        "positive_window_fraction": round(positive_fraction, 6),
        "min_track_record_length": round(min_trl, 6) if math.isfinite(min_trl) else -1.0,
    }

    artifact = ScoreArtifact(
        methodology_version=METHODOLOGY_VERSION,
        library_version=_library_version,
        input_digest=input_digest(arr),
        n_obs=len(arr),
        freq=freq,
        n_trials=n_trials,
        n_bootstrap=n_bootstrap,
        seed=seed,
        score=composite,
        grade=grade,
        components=components,
        metrics=metrics,
        flags=flags,
        created_at=datetime.now(timezone.utc).isoformat(),
    )

    return ScoreResult(
        score=composite,
        grade=grade,
        components=components,
        metrics=metrics,
        flags=flags,
        artifact=artifact,
    )


def verify_artifact(
    artifact: ScoreArtifact | str,
    returns: np.ndarray | Sequence[float],
) -> bool:
    """Independently verify a score artifact against the original returns.

    Recomputes the score using the parameters recorded in the artifact
    and confirms (a) the artifact's content hash matches its fields,
    (b) the input digest matches the supplied returns, and (c) the
    recomputed score and components match exactly.

    Requires the same QuantLite version that produced the artifact;
    methodology versions are frozen, but verification is bit-exact only
    within a library version.

    Parameters
    ----------
    artifact : ScoreArtifact or str
        The artifact, or its JSON representation.
    returns : array-like
        The original return series the artifact claims to score.

    Returns
    -------
    bool
        True if the artifact is authentic and reproducible.
    """
    if isinstance(artifact, str):
        artifact = ScoreArtifact.from_json(artifact)

    if not artifact.is_internally_consistent():
        return False
    if artifact.input_digest != input_digest(returns):
        return False

    recomputed = compute_score(
        returns,
        n_trials=artifact.n_trials,
        freq=artifact.freq,
        n_bootstrap=artifact.n_bootstrap,
        seed=artifact.seed,
    )
    return (
        recomputed.score == artifact.score
        and recomputed.components == artifact.components
        and recomputed.artifact.content_hash == artifact.content_hash
    )
