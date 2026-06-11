"""The QuantLite Score: an open, verifiable rating for trading track records.

Raw Sharpe ratios are trivially gamed: test fifty variants, publish the
winner, start the chart at the bottom of a drawdown. The QuantLite Score
is the antidote, built from the library's own forensics stack: the
Deflated Sharpe Ratio, bootstrap robustness, tail risk penalties,
consistency, and track record sufficiency, with integrity checks that
flag smoothed returns, outlier dependence, and cherry-picked start dates.

The methodology is open and versioned (``QLS-1.0``, see ``docs/score.md``).
Every score ships as a portable artifact whose content hash anyone can
reproduce from the original returns::

    from quantlite.score import compute_score, verify_artifact

    result = compute_score(returns, n_trials=20)
    print(result.score, result.grade)        # e.g. 72.4 B
    payload = result.artifact.to_json()      # publish or store this
    assert verify_artifact(payload, returns)  # anyone can check it
"""

from .artifact import ScoreArtifact, input_digest
from .engine import (
    COMPONENT_WEIGHTS,
    DEFAULT_SEED,
    METHODOLOGY_VERSION,
    ScoreResult,
    compute_score,
    verify_artifact,
)
from .integrity import IntegrityFlag, validate_track_record

__all__ = [
    "METHODOLOGY_VERSION",
    "DEFAULT_SEED",
    "COMPONENT_WEIGHTS",
    "compute_score",
    "verify_artifact",
    "validate_track_record",
    "input_digest",
    "ScoreResult",
    "ScoreArtifact",
    "IntegrityFlag",
]
