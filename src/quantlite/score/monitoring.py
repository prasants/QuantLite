"""Continuous monitoring and early warning for track-record scores.

A point-in-time score answers "how good is this record today". Monitoring
answers the question an allocator actually loses sleep over: "is this record
*deteriorating*, and will I see it before it costs me". The firewall
(:mod:`~quantlite.score.provenance`) guarantees each score is reproducible
and independently sourced; this layer runs that machinery forward through
time and raises an alarm when the picture turns.

The core idea is **dual-window scoring**. The *expanding* window scores the
entire history -- stable, hard to game, the authoritative published number.
But an expanding window is slow to notice trouble: a long, strong history
mathematically buries a weak recent stretch. So each observation is also
scored over a *trailing* window of recent periods, compared against the
preceding window of equal length. The two windows are the same size, so the
comparison is length-fair: any gap reflects a change in performance, not the
length-dependent sufficiency and Deflated-Sharpe effects that contaminate a
naive comparison against the full history. When the recent window scores
materially below the comparable prior window, the record is deteriorating
*now*, well before the headline grade moves -- which is how deterioration is
pre-empted rather than discovered after the fact.

Every observation appends a :class:`ScoreSnapshot` to an append-only,
hash-chained :class:`ScoreHistory`: the monitoring analogue of the artifact
content hash, proving the history was not back-edited. Alerts are emitted as
verifiable data structures, not delivered; webhook, dashboard, and polling
adapters wrap them in the hosted layer.

    from quantlite.score import Monitor, Cadence, DataSource

    monitor = Monitor(
        cadence=Cadence.DAILY,
        source=DataSource.EXCHANGE_CUSTODY,
        attester="exchange:quantmarket",
        account_ref="acct-7a3f",
    )
    alerts = monitor.observe(returns, as_of="2025-01-31")
    history = monitor.history
    assert history.is_chain_valid()
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from enum import Enum

import numpy as np

from .engine import compute_score
from .provenance import AttestedScore, DataSource, SourceAttestation

__all__ = [
    "Cadence",
    "AlertKind",
    "Severity",
    "Alert",
    "ScoreSnapshot",
    "ScoreHistory",
    "Monitor",
]

# Periods per year for each cadence, used for annualisation.
_CADENCE_FREQ = {
    "daily": 252,
    "weekly": 52,
    "monthly": 12,
    "yearly": 1,
}

# Letter grades from best to worst, for ordering comparisons.
_GRADE_ORDER = ("A+", "A", "B", "C", "D", "F")
_GRADE_RANK = {g: i for i, g in enumerate(_GRADE_ORDER)}


class Cadence(str, Enum):
    """How often a strategy is re-scored."""

    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    YEARLY = "yearly"

    @property
    def freq(self) -> int:
        """Periods per year for this cadence.

        Returns
        -------
        int
            Annualisation factor (e.g. 252 for daily).
        """
        return _CADENCE_FREQ[self.value]


class Severity(str, Enum):
    """Alert severity."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertKind(str, Enum):
    """The kind of deterioration an alert reports.

    Attributes
    ----------
    NEW_CRITICAL_FLAG
        A critical integrity flag is present now that was absent in the
        prior snapshot (e.g. a smoothing or short-volatility signature
        emerging over time).
    GRADE_DROP
        The authoritative (expanding-window) letter grade fell.
    SCORE_DROP
        The expanding-window score fell by more than the configured
        threshold versus the prior snapshot.
    TRAILING_DIVERGENCE
        The recent trailing window's annualised Sharpe fell materially below
        the preceding equal-length window's: the length-fair early-warning
        signal that recent performance has decayed before the headline number
        reflects it.
    """

    NEW_CRITICAL_FLAG = "new_critical_flag"
    GRADE_DROP = "grade_drop"
    SCORE_DROP = "score_drop"
    TRAILING_DIVERGENCE = "trailing_divergence"


@dataclass(frozen=True)
class Alert:
    """A single deterioration finding from one monitoring step.

    Attributes
    ----------
    kind : str
        An :class:`AlertKind` value.
    severity : str
        A :class:`Severity` value.
    as_of : str
        ISO-8601 date of the snapshot that triggered the alert.
    detail : str
        Human-readable explanation with the measured values.
    snapshot_hash : str
        Hash of the snapshot that triggered the alert, for traceability.
    """

    kind: str
    severity: str
    as_of: str
    detail: str
    snapshot_hash: str

    def to_dict(self) -> dict:
        """Convert to a JSON-compatible dictionary.

        Returns
        -------
        dict
            All fields of the alert.
        """
        return asdict(self)


def _canonical_json(payload: dict) -> str:
    """Serialise a payload deterministically (QLS-1.0 rules).

    Parameters
    ----------
    payload : dict
        JSON-compatible dictionary.

    Returns
    -------
    str
        Canonical JSON: sorted keys, compact separators.
    """
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _critical_flag_codes(attested: AttestedScore) -> tuple[str, ...]:
    """Sorted codes of the critical integrity flags on a score.

    Parameters
    ----------
    attested : AttestedScore
        The attested expanding-window score.

    Returns
    -------
    tuple of str
        Critical flag codes, sorted for stable comparison.
    """
    return tuple(sorted(f.code for f in attested.artifact.flags if f.severity == "critical"))


@dataclass(frozen=True)
class ScoreSnapshot:
    """A timestamped, chain-linked score for one monitoring step.

    Each snapshot binds the authoritative expanding-window attested score to
    a trailing-window summary and the hash of the previous snapshot, forming
    an append-only chain that cannot be silently reordered or edited.

    Attributes
    ----------
    as_of : str
        ISO-8601 date this snapshot was taken.
    cadence : str
        A :class:`Cadence` value.
    expanding : AttestedScore
        Authoritative score over the full history. Independently
        verifiable via :meth:`AttestedScore.verify`.
    trailing_score : float or None
        Score over the trailing window, or None when the window held too
        few observations to score.
    trailing_grade : str or None
        Letter grade over the trailing window, or None.
    critical_flags : tuple of str
        Critical integrity flag codes present on the expanding score.
    prev_hash : str
        ``snapshot_hash`` of the prior snapshot, or "" for the first.
    snapshot_hash : str
        SHA-256 over this snapshot's content and ``prev_hash``.
    """

    as_of: str
    cadence: str
    expanding: AttestedScore
    trailing_score: float | None
    trailing_grade: str | None
    critical_flags: tuple[str, ...]
    prev_hash: str
    snapshot_hash: str = field(default="")

    def __post_init__(self) -> None:
        if not self.snapshot_hash:
            object.__setattr__(self, "snapshot_hash", self._compute_hash())

    def _hashable_payload(self) -> dict:
        """Build the payload covered by the snapshot hash.

        The expanding score is represented by its already-verifiable content
        and attestation hashes rather than re-serialised in full, so the
        chain is tied to the firewall artifacts without duplicating them.

        Returns
        -------
        dict
            Deterministic summary of this snapshot.
        """
        return {
            "as_of": self.as_of,
            "cadence": self.cadence,
            "content_hash": self.expanding.artifact.content_hash,
            "attestation_hash": self.expanding.attestation.attestation_hash,
            "trailing_score": self.trailing_score,
            "trailing_grade": self.trailing_grade,
            "critical_flags": list(self.critical_flags),
            "prev_hash": self.prev_hash,
        }

    def _compute_hash(self) -> str:
        """SHA-256 over the canonical JSON of the hashable payload.

        Returns
        -------
        str
            Hex-encoded SHA-256 digest.
        """
        return hashlib.sha256(_canonical_json(self._hashable_payload()).encode("utf-8")).hexdigest()

    def is_internally_consistent(self) -> bool:
        """Check that the stored hash matches the snapshot's contents.

        Returns
        -------
        bool
            True if the snapshot has not been tampered with.
        """
        return self.snapshot_hash == self._compute_hash()

    @property
    def score(self) -> float:
        """Authoritative expanding-window score.

        Returns
        -------
        float
            Composite score in [0, 100].
        """
        return self.expanding.artifact.score

    @property
    def grade(self) -> str:
        """Authoritative expanding-window letter grade.

        Returns
        -------
        str
            Letter grade.
        """
        return self.expanding.artifact.grade


@dataclass(frozen=True)
class ScoreHistory:
    """An append-only, hash-chained sequence of score snapshots.

    Attributes
    ----------
    snapshots : tuple of ScoreSnapshot
        Snapshots in chronological order.
    """

    snapshots: tuple[ScoreSnapshot, ...] = ()

    def append(self, snapshot: ScoreSnapshot) -> ScoreHistory:
        """Return a new history with one snapshot appended.

        Parameters
        ----------
        snapshot : ScoreSnapshot
            The snapshot to add; its ``prev_hash`` should link the current
            tip.

        Returns
        -------
        ScoreHistory
            A new history (the dataclass is immutable).
        """
        return ScoreHistory(self.snapshots + (snapshot,))

    @property
    def latest(self) -> ScoreSnapshot | None:
        """The most recent snapshot, or None if the history is empty.

        Returns
        -------
        ScoreSnapshot or None
            The chain tip.
        """
        return self.snapshots[-1] if self.snapshots else None

    def is_chain_valid(self) -> bool:
        """Verify the hash chain end to end.

        Confirms every snapshot is internally consistent and that each
        ``prev_hash`` matches the preceding snapshot's ``snapshot_hash``,
        proving the sequence was not reordered, inserted into, or edited.

        Returns
        -------
        bool
            True if the chain is intact.
        """
        prev = ""
        for snap in self.snapshots:
            if not snap.is_internally_consistent():
                return False
            if snap.prev_hash != prev:
                return False
            prev = snap.snapshot_hash
        return True

    def to_json(self) -> str:
        """Serialise the whole history to canonical JSON.

        Returns
        -------
        str
            Canonical JSON: an object with a ``snapshots`` array.
        """
        payload = {
            "snapshots": [
                {
                    "as_of": s.as_of,
                    "cadence": s.cadence,
                    "expanding": s.expanding.artifact.to_dict(),
                    "attestation": s.expanding.attestation.to_dict(),
                    "trailing_score": s.trailing_score,
                    "trailing_grade": s.trailing_grade,
                    "critical_flags": list(s.critical_flags),
                    "prev_hash": s.prev_hash,
                    "snapshot_hash": s.snapshot_hash,
                }
                for s in self.snapshots
            ]
        }
        return _canonical_json(payload)


@dataclass(frozen=True)
class Monitor:
    """Runs dual-window scoring forward through time and emits alerts.

    The monitor is configured once with a cadence, an independent data
    source, and the deterioration thresholds. Each call to :meth:`observe`
    scores the full history (expanding) and the recent window (trailing),
    appends a chain-linked snapshot, and returns any alerts triggered by the
    transition from the prior snapshot.

    Attributes
    ----------
    cadence : Cadence
        Scoring frequency; also sets the annualisation factor.
    source : DataSource
        Independent origin of the returns (must satisfy the firewall).
    attester : str
        Party vouching for the data; must not be the rated manager.
    account_ref : str
        Reference to the account or strategy at the source.
    trailing_window : int or None
        Number of recent periods for the trailing score. Defaults to half a
        year at the cadence's frequency; None or 0 disables divergence.
        Divergence needs two such windows of history before it can fire.
    n_trials : int
        Strategy-trial count passed to the Deflated Sharpe Ratio.
    score_drop_threshold : float
        Expanding-score fall (points) that triggers a SCORE_DROP alert.
    divergence_threshold : float
        Drop in annualised Sharpe by which the preceding window must exceed
        the recent window to trigger a TRAILING_DIVERGENCE alert.
    history : ScoreHistory
        The accumulated snapshot chain.
    """

    cadence: Cadence
    source: DataSource
    attester: str
    account_ref: str
    trailing_window: int | None = None
    n_trials: int = 1
    score_drop_threshold: float = 5.0
    divergence_threshold: float = 0.75
    history: ScoreHistory = field(default_factory=ScoreHistory)

    @property
    def _effective_trailing_window(self) -> int:
        """Trailing window in periods, defaulting to half a year.

        Returns
        -------
        int
            Number of trailing periods (0 disables divergence).
        """
        if self.trailing_window is not None:
            return self.trailing_window
        return self.cadence.freq // 2

    def observe(
        self,
        returns: np.ndarray | Sequence[float],
        as_of: str,
        period_start: str | None = None,
        ingested_at: str | None = None,
    ) -> tuple[Alert, ...]:
        """Score a new observation, append a snapshot, and return alerts.

        Parameters
        ----------
        returns : array-like
            The full independently-sourced return history as of ``as_of``.
        as_of : str
            ISO-8601 date of this observation; also the attestation period
            end and, by default, the ingestion timestamp date.
        period_start : str, optional
            ISO-8601 start of the attested period. Defaults to ``as_of``.
        ingested_at : str, optional
            UTC ISO-8601 ingestion timestamp. Defaults to ``as_of``.

        Returns
        -------
        tuple of Alert
            Alerts triggered by the transition from the prior snapshot;
            empty for the first observation or a stable transition.
        """
        arr = np.asarray(returns, dtype=float)
        freq = self.cadence.freq
        ingested = ingested_at or f"{as_of}T00:00:00Z"
        start = period_start or as_of

        expanding_result = compute_score(arr, n_trials=self.n_trials, freq=freq)
        attestation = SourceAttestation.create(
            source=self.source,
            account_ref=self.account_ref,
            period_start=start,
            period_end=as_of,
            ingested_at=ingested,
            returns=arr,
            attester=self.attester,
        )
        attested = AttestedScore(expanding_result.artifact, attestation)

        recent_score, recent_grade, recent_sharpe, prior_sharpe = self._windows(arr, freq)

        prev = self.history.latest
        snapshot = ScoreSnapshot(
            as_of=as_of,
            cadence=self.cadence.value,
            expanding=attested,
            trailing_score=recent_score,
            trailing_grade=recent_grade,
            critical_flags=_critical_flag_codes(attested),
            prev_hash=prev.snapshot_hash if prev else "",
        )
        alerts = self._evaluate(prev, snapshot, recent_sharpe, prior_sharpe)
        object.__setattr__(self, "history", self.history.append(snapshot))
        return alerts

    def _windows(
        self, arr: np.ndarray, freq: int
    ) -> tuple[float | None, str | None, float | None, float | None]:
        """Score the recent window and the preceding equal-length window.

        Returns both the composite score of the recent window (for display in
        the snapshot) and the annualised Sharpe of the recent and preceding
        windows (for the divergence signal). The windows are identical in
        length, so their Sharpes are directly comparable; Sharpe is used
        rather than the composite score because the composite is capped by
        integrity rules and saturates on sub-year windows, whereas Sharpe
        stays continuous and sensitive.

        Parameters
        ----------
        arr : numpy.ndarray
            The full return history.
        freq : int
            Annualisation factor.

        Returns
        -------
        tuple
            ``(recent_score, recent_grade, recent_sharpe, prior_sharpe)``.
            The recent fields are None when the window is disabled or too
            short; ``prior_sharpe`` is None until the history spans two
            full windows.
        """
        window = self._effective_trailing_window
        if window <= 0 or len(arr) < window:
            return None, None, None, None
        recent_score, recent_grade, recent_sharpe = self._score_window(arr[-window:], freq)
        prior_sharpe = None
        if len(arr) >= 2 * window:
            _, _, prior_sharpe = self._score_window(arr[-2 * window : -window], freq)
        return recent_score, recent_grade, recent_sharpe, prior_sharpe

    def _score_window(
        self, window: np.ndarray, freq: int
    ) -> tuple[float | None, str | None, float | None]:
        """Score a single window, tolerating windows too short for the engine.

        Parameters
        ----------
        window : numpy.ndarray
            The slice of returns to score.
        freq : int
            Annualisation factor.

        Returns
        -------
        tuple
            ``(score, grade, annualised_sharpe)``, or ``(None, None, None)``
            if the window is too short for the scoring engine.
        """
        try:
            result = compute_score(window, n_trials=self.n_trials, freq=freq)
        except ValueError:
            return None, None, None
        return result.score, result.grade, result.metrics["sharpe_ratio"]

    def _evaluate(
        self,
        prev: ScoreSnapshot | None,
        current: ScoreSnapshot,
        recent_sharpe: float | None,
        prior_sharpe: float | None,
    ) -> tuple[Alert, ...]:
        """Compare windows and snapshots and emit deterioration alerts.

        Parameters
        ----------
        prev : ScoreSnapshot or None
            The prior snapshot, or None for the first observation.
        current : ScoreSnapshot
            The snapshot just computed.
        recent_sharpe : float or None
            Annualised Sharpe of the recent window.
        prior_sharpe : float or None
            Annualised Sharpe of the window immediately preceding the recent
            window, used for the length-fair divergence signal.

        Returns
        -------
        tuple of Alert
            Alerts for every deterioration rule that fired.
        """
        alerts: list[Alert] = []

        # Length-fair divergence: the recent window's Sharpe fell materially
        # below the preceding equal-length window's. Evaluable as soon as the
        # history spans two windows; it is an intra-snapshot momentum signal.
        if recent_sharpe is not None and prior_sharpe is not None:
            gap = prior_sharpe - recent_sharpe
            if gap >= self.divergence_threshold:
                alerts.append(
                    Alert(
                        kind=AlertKind.TRAILING_DIVERGENCE.value,
                        severity=Severity.WARNING.value,
                        as_of=current.as_of,
                        detail=(
                            f"Recent-window Sharpe {recent_sharpe:.2f} is "
                            f"{gap:.2f} below the preceding comparable window "
                            f"at {prior_sharpe:.2f}; performance is "
                            "deteriorating ahead of the headline grade."
                        ),
                        snapshot_hash=current.snapshot_hash,
                    )
                )

        if prev is None:
            return tuple(alerts)

        new_critical = set(current.critical_flags) - set(prev.critical_flags)
        if new_critical:
            codes = ", ".join(sorted(new_critical))
            alerts.append(
                Alert(
                    kind=AlertKind.NEW_CRITICAL_FLAG.value,
                    severity=Severity.CRITICAL.value,
                    as_of=current.as_of,
                    detail=(f"New critical integrity flag(s) since the prior snapshot: {codes}."),
                    snapshot_hash=current.snapshot_hash,
                )
            )

        if _GRADE_RANK[current.grade] > _GRADE_RANK[prev.grade]:
            alerts.append(
                Alert(
                    kind=AlertKind.GRADE_DROP.value,
                    severity=Severity.WARNING.value,
                    as_of=current.as_of,
                    detail=(f"Grade fell from {prev.grade} to {current.grade}."),
                    snapshot_hash=current.snapshot_hash,
                )
            )

        drop = prev.score - current.score
        if drop >= self.score_drop_threshold:
            alerts.append(
                Alert(
                    kind=AlertKind.SCORE_DROP.value,
                    severity=Severity.WARNING.value,
                    as_of=current.as_of,
                    detail=(
                        f"Score fell {drop:.1f} points, from {prev.score:.1f} "
                        f"to {current.score:.1f}."
                    ),
                    snapshot_hash=current.snapshot_hash,
                )
            )

        return tuple(alerts)
