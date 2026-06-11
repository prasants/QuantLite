"""Verifiable score artifacts.

A score that cannot be independently reproduced is just a claim. The
artifact is the reproducibility contract: it records the methodology
version, every scoring parameter, a digest of the input returns, and a
content hash over the canonical JSON representation. Anyone holding the
original return series can recompute the score with the same library
version and confirm the artifact bit for bit.

Canonicalisation rules (QLS-1.0):

- Input returns are cast to little-endian float64 before hashing, so the
  digest is identical across platforms.
- Artifact JSON is serialised with sorted keys, no whitespace, and floats
  via ``repr`` round-tripping (Python's shortest exact representation).
- The content hash is SHA-256 over the canonical JSON of every field
  except ``created_at`` and ``content_hash`` itself, so re-issuing the
  same score at a different time yields the same hash.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field

import numpy as np

from .integrity import IntegrityFlag

__all__ = [
    "ScoreArtifact",
    "input_digest",
]

# Fields excluded from the content hash.
_UNHASHED_FIELDS = {"created_at", "content_hash"}


def input_digest(returns: np.ndarray | Sequence[float]) -> str:
    """SHA-256 digest of a return series in canonical form.

    The series is cast to little-endian float64 and hashed as raw bytes,
    making the digest platform-independent and bit-exact.

    Parameters
    ----------
    returns : array-like
        Simple periodic returns.

    Returns
    -------
    str
        Hex-encoded SHA-256 digest.
    """
    arr = np.ascontiguousarray(np.asarray(returns, dtype=float).astype("<f8"))
    return hashlib.sha256(arr.tobytes()).hexdigest()


def _canonical_json(payload: dict) -> str:
    """Serialise a payload deterministically.

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


@dataclass(frozen=True)
class ScoreArtifact:
    """Portable, verifiable record of a computed QuantLite Score.

    Attributes
    ----------
    methodology_version : str
        Scoring methodology identifier (e.g. ``"QLS-1.0"``).
    library_version : str
        QuantLite version that produced the artifact.
    input_digest : str
        SHA-256 digest of the canonicalised input returns.
    n_obs : int
        Number of return observations.
    freq : int
        Periods per year used for annualisation.
    n_trials : int
        Number of strategy trials declared for the Deflated Sharpe Ratio.
    n_bootstrap : int
        Number of bootstrap samples used for the robustness component.
    seed : int
        Random seed used for all stochastic steps.
    score : float
        Composite score in [0, 100].
    grade : str
        Letter grade derived from the score.
    components : dict
        Component scores in [0, 100], keyed by component name.
    metrics : dict
        Supporting raw metrics (Sharpe, drawdown, CVaR, etc.).
    flags : tuple of IntegrityFlag
        Integrity findings on the track record.
    created_at : str
        UTC ISO-8601 timestamp. Excluded from the content hash.
    content_hash : str
        SHA-256 over the canonical JSON of all other fields except
        ``created_at``.
    """

    methodology_version: str
    library_version: str
    input_digest: str
    n_obs: int
    freq: int
    n_trials: int
    n_bootstrap: int
    seed: int
    score: float
    grade: str
    components: dict[str, float]
    metrics: dict[str, float]
    flags: tuple[IntegrityFlag, ...]
    created_at: str
    content_hash: str = field(default="")

    def __post_init__(self) -> None:
        computed = self._compute_content_hash()
        if not self.content_hash:
            object.__setattr__(self, "content_hash", computed)

    def _hashable_payload(self) -> dict:
        """Build the payload covered by the content hash.

        Returns
        -------
        dict
            All fields except ``created_at`` and ``content_hash``.
        """
        payload = asdict(self)
        for key in _UNHASHED_FIELDS:
            payload.pop(key, None)
        payload["flags"] = [asdict(f) for f in self.flags]
        return payload

    def _compute_content_hash(self) -> str:
        """SHA-256 over the canonical JSON of the hashable payload.

        Returns
        -------
        str
            Hex-encoded SHA-256 digest.
        """
        return hashlib.sha256(
            _canonical_json(self._hashable_payload()).encode("utf-8")
        ).hexdigest()

    def is_internally_consistent(self) -> bool:
        """Check that the stored content hash matches the fields.

        Returns
        -------
        bool
            True if the artifact has not been tampered with.
        """
        return self.content_hash == self._compute_content_hash()

    def to_dict(self) -> dict:
        """Convert the artifact to a JSON-compatible dictionary.

        Returns
        -------
        dict
            All fields, with flags as a list of dictionaries.
        """
        payload = asdict(self)
        payload["flags"] = [asdict(f) for f in self.flags]
        return payload

    def to_json(self) -> str:
        """Serialise the artifact to canonical JSON.

        Returns
        -------
        str
            Canonical JSON string (sorted keys, compact separators).
        """
        return _canonical_json(self.to_dict())

    @classmethod
    def from_json(cls, payload: str) -> ScoreArtifact:
        """Reconstruct an artifact from its JSON representation.

        Parameters
        ----------
        payload : str
            JSON produced by :meth:`to_json`.

        Returns
        -------
        ScoreArtifact
            The reconstructed artifact. Use
            :meth:`is_internally_consistent` to confirm integrity.
        """
        data = json.loads(payload)
        data["flags"] = tuple(IntegrityFlag(**f) for f in data.get("flags", []))
        return cls(**data)
