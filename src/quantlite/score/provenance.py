"""Data provenance and the QuantLite firewall.

A score is only as trustworthy as the data behind it, and the most
dangerous data is data supplied by the party being rated. The original
sin of credit ratings was *issuer-pays*: the rated entity hands over the
inputs and signs the cheque, so the rating drifts from measurement toward
marketing. The QuantLite firewall is the structural answer.

The :mod:`~quantlite.score.artifact` layer proves *what* the returns were
(an ``input_digest``). This layer proves *where they came from*. A
:class:`SourceAttestation` binds an input digest to an independently
sourced origin -- an exchange's own fills, a fund administrator's books, a
prime broker's statements -- recorded by an attester who is not the rated
manager. A score is *firewall-clean* only when its inputs are attested to
such an independent source.

For a marketplace operator this is the whole game. When the venue is the
custodian of record (a crypto exchange scoring the strategies on its own
quant marketplace, say), the returns are derived from venue-side fills the
manager never touches, so the record cannot be smoothed, back-painted, or
cherry-picked at source. The firewall is not a policy page; it is a
property the verifier can check::

    from quantlite.score import (
        compute_score, DataSource, SourceAttestation, AttestedScore,
    )

    result = compute_score(returns, n_trials=20)
    attestation = SourceAttestation.create(
        source=DataSource.EXCHANGE_CUSTODY,
        account_ref="acct-7a3f",
        period_start="2024-01-01",
        period_end="2024-12-31",
        ingested_at="2025-01-04T09:00:00Z",
        returns=returns,
        attester="exchange:quantmarket",
    )
    attested = AttestedScore(artifact=result.artifact, attestation=attestation)
    report = attested.verify(returns)
    assert report.ok and report.firewall_clean
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from enum import Enum

import numpy as np

from .artifact import ScoreArtifact, input_digest
from .engine import verify_artifact

__all__ = [
    "DataSource",
    "SourceAttestation",
    "AttestedScore",
    "FirewallReport",
    "FirewallError",
    "is_independent",
    "assert_firewall",
]


class DataSource(str, Enum):
    """Origin of a return series, ordered by independence from the rated party.

    The first three sources are *independent*: the data is produced and
    vouched for by a party other than the manager being scored. The last
    two are not, and a score built on them can never be firewall-clean.
    """

    EXCHANGE_CUSTODY = "exchange_custody"
    FUND_ADMINISTRATOR = "fund_administrator"
    PRIME_BROKER = "prime_broker"
    ALLOCATOR_SUPPLIED = "allocator_supplied"
    MANAGER_SUBMITTED = "manager_submitted"


# Sources for which the rated manager does not control the inputs.
_INDEPENDENT_SOURCES = frozenset(
    {
        DataSource.EXCHANGE_CUSTODY,
        DataSource.FUND_ADMINISTRATOR,
        DataSource.PRIME_BROKER,
        DataSource.ALLOCATOR_SUPPLIED,
    }
)


def is_independent(source: DataSource | str) -> bool:
    """Whether a data source is independent of the rated manager.

    Parameters
    ----------
    source : DataSource or str
        The origin of the return series.

    Returns
    -------
    bool
        True if the inputs come from a party other than the manager being
        scored, and therefore could not have been altered at source.
    """
    return DataSource(source) in _INDEPENDENT_SOURCES


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


class FirewallError(Exception):
    """Raised when a score's inputs are not attested to an independent source."""


@dataclass(frozen=True)
class SourceAttestation:
    """An assertion about where a return series came from.

    The attestation binds an input digest to a named, independently
    sourced origin and the party vouching for it. Its content hash covers
    every field except the hash itself, so it can be re-serialised without
    changing identity but cannot be silently edited.

    Attributes
    ----------
    source : str
        A :class:`DataSource` value.
    account_ref : str
        Opaque reference to the account or strategy at the source. Not a
        secret; an identifier the source and verifier both recognise.
    period_start : str
        ISO-8601 date of the first observation covered.
    period_end : str
        ISO-8601 date of the last observation covered.
    ingested_at : str
        UTC ISO-8601 timestamp of when the data was pulled from the source.
    input_digest : str
        SHA-256 digest of the canonicalised returns (see
        :func:`~quantlite.score.artifact.input_digest`). Must match the
        digest carried by the score artifact.
    attester : str
        Identifier of the party making the attestation (e.g. the venue),
        which must not be the rated manager.
    attestation_hash : str
        SHA-256 over the canonical JSON of all other fields.
    """

    source: str
    account_ref: str
    period_start: str
    period_end: str
    ingested_at: str
    input_digest: str
    attester: str
    attestation_hash: str = field(default="")

    def __post_init__(self) -> None:
        # Normalise the source to its enum value for stable hashing.
        object.__setattr__(self, "source", DataSource(self.source).value)
        computed = self._compute_hash()
        if not self.attestation_hash:
            object.__setattr__(self, "attestation_hash", computed)

    @classmethod
    def create(
        cls,
        source: DataSource | str,
        account_ref: str,
        period_start: str,
        period_end: str,
        ingested_at: str,
        returns: np.ndarray | Sequence[float],
        attester: str,
    ) -> SourceAttestation:
        """Build an attestation directly from a return series.

        Computes the canonical input digest so the attestation and the
        score artifact commit to byte-identical inputs.

        Parameters
        ----------
        source : DataSource or str
            Origin of the returns.
        account_ref : str
            Reference to the account or strategy at the source.
        period_start, period_end : str
            ISO-8601 dates bounding the record.
        ingested_at : str
            UTC ISO-8601 timestamp of ingestion.
        returns : array-like
            The return series, used only to derive the input digest.
        attester : str
            Party vouching for the data; must not be the rated manager.

        Returns
        -------
        SourceAttestation
            A sealed attestation with its content hash populated.
        """
        return cls(
            source=DataSource(source).value,
            account_ref=account_ref,
            period_start=period_start,
            period_end=period_end,
            ingested_at=ingested_at,
            input_digest=input_digest(returns),
            attester=attester,
        )

    def _hashable_payload(self) -> dict:
        """Build the payload covered by the content hash.

        Returns
        -------
        dict
            All fields except ``attestation_hash``.
        """
        payload = asdict(self)
        payload.pop("attestation_hash", None)
        return payload

    def _compute_hash(self) -> str:
        """SHA-256 over the canonical JSON of the hashable payload.

        Returns
        -------
        str
            Hex-encoded SHA-256 digest.
        """
        return hashlib.sha256(_canonical_json(self._hashable_payload()).encode("utf-8")).hexdigest()

    def is_internally_consistent(self) -> bool:
        """Check that the stored hash matches the fields.

        Returns
        -------
        bool
            True if the attestation has not been tampered with.
        """
        return self.attestation_hash == self._compute_hash()

    def is_firewall_clean(self) -> bool:
        """Whether this attestation satisfies the firewall.

        Returns
        -------
        bool
            True if the attestation is internally consistent and names an
            independent source.
        """
        return self.is_internally_consistent() and is_independent(self.source)

    def to_dict(self) -> dict:
        """Convert to a JSON-compatible dictionary.

        Returns
        -------
        dict
            All fields of the attestation.
        """
        return asdict(self)

    def to_json(self) -> str:
        """Serialise to canonical JSON.

        Returns
        -------
        str
            Canonical JSON string (sorted keys, compact separators).
        """
        return _canonical_json(self.to_dict())

    @classmethod
    def from_json(cls, payload: str) -> SourceAttestation:
        """Reconstruct an attestation from its JSON representation.

        Parameters
        ----------
        payload : str
            JSON produced by :meth:`to_json`.

        Returns
        -------
        SourceAttestation
            The reconstructed attestation. Use
            :meth:`is_internally_consistent` to confirm integrity.
        """
        return cls(**json.loads(payload))


def assert_firewall(attestation: SourceAttestation) -> None:
    """Raise unless an attestation is firewall-clean.

    Parameters
    ----------
    attestation : SourceAttestation
        The attestation to enforce.

    Raises
    ------
    FirewallError
        If the attestation has been tampered with or names a source the
        rated manager controls.
    """
    if not attestation.is_internally_consistent():
        raise FirewallError("attestation hash does not match its contents")
    if not is_independent(attestation.source):
        raise FirewallError(
            f"source {attestation.source!r} is controlled by the rated party; "
            "a firewall-clean score requires an independent source"
        )


@dataclass(frozen=True)
class FirewallReport:
    """Structured result of verifying an attested score.

    Attributes
    ----------
    artifact_consistent : bool
        The score artifact's content hash matches its fields.
    attestation_consistent : bool
        The attestation's content hash matches its fields.
    digests_match : bool
        The attestation, the artifact, and the supplied returns all share
        one input digest.
    score_reproduces : bool
        Recomputing the score from the returns reproduces the artifact.
    firewall_clean : bool
        The attestation names an independent source.
    """

    artifact_consistent: bool
    attestation_consistent: bool
    digests_match: bool
    score_reproduces: bool
    firewall_clean: bool

    @property
    def ok(self) -> bool:
        """Whether every verification check passed.

        Returns
        -------
        bool
            True only if the score is reproducible, untampered, and bound
            to independently sourced inputs.
        """
        return (
            self.artifact_consistent
            and self.attestation_consistent
            and self.digests_match
            and self.score_reproduces
            and self.firewall_clean
        )


@dataclass(frozen=True)
class AttestedScore:
    """A score artifact bound to the provenance of its inputs.

    Attributes
    ----------
    artifact : ScoreArtifact
        The verifiable score.
    attestation : SourceAttestation
        Where the scored returns came from.
    """

    artifact: ScoreArtifact
    attestation: SourceAttestation

    def verify(self, returns: np.ndarray | Sequence[float]) -> FirewallReport:
        """Verify the score and its provenance against the original returns.

        Confirms, in one pass, that the score reproduces from the returns,
        that artifact and attestation are untampered, that all three agree
        on the input digest, and that the source is independent.

        Parameters
        ----------
        returns : array-like
            The original return series.

        Returns
        -------
        FirewallReport
            The per-check breakdown; inspect :attr:`FirewallReport.ok` for
            the overall verdict.
        """
        digest = input_digest(returns)
        digests_match = self.attestation.input_digest == self.artifact.input_digest == digest
        return FirewallReport(
            artifact_consistent=self.artifact.is_internally_consistent(),
            attestation_consistent=self.attestation.is_internally_consistent(),
            digests_match=digests_match,
            score_reproduces=verify_artifact(self.artifact.to_json(), returns),
            firewall_clean=self.attestation.is_firewall_clean(),
        )
