"""Tests for data provenance and the firewall (quantlite.score.provenance)."""

from __future__ import annotations

import numpy as np
import pytest

from quantlite.score import (
    AttestedScore,
    DataSource,
    FirewallError,
    SourceAttestation,
    assert_firewall,
    compute_score,
    input_digest,
    is_independent,
)


def _returns(n: int = 504, seed: int = 11) -> np.ndarray:
    """A plausible daily track record."""
    rng = np.random.RandomState(seed)
    return rng.normal(0.0005, 0.01, n)


def _attestation(returns, source=DataSource.EXCHANGE_CUSTODY, **kw) -> SourceAttestation:
    params = dict(
        source=source,
        account_ref="acct-7a3f",
        period_start="2024-01-01",
        period_end="2024-12-31",
        ingested_at="2025-01-04T09:00:00Z",
        attester="exchange:quantmarket",
    )
    params.update(kw)
    return SourceAttestation.create(returns=returns, **params)


class TestIndependence:
    def test_exchange_and_admin_are_independent(self):
        assert is_independent(DataSource.EXCHANGE_CUSTODY)
        assert is_independent(DataSource.FUND_ADMINISTRATOR)
        assert is_independent(DataSource.PRIME_BROKER)
        assert is_independent(DataSource.ALLOCATOR_SUPPLIED)

    def test_manager_submitted_is_not_independent(self):
        assert not is_independent(DataSource.MANAGER_SUBMITTED)

    def test_accepts_raw_string_value(self):
        assert is_independent("exchange_custody")
        assert not is_independent("manager_submitted")

    def test_unknown_source_rejected(self):
        with pytest.raises(ValueError):
            is_independent("anonymous_tip")


class TestSourceAttestation:
    def test_create_binds_input_digest(self):
        returns = _returns()
        att = _attestation(returns)
        assert att.input_digest == input_digest(returns)

    def test_hash_populated_and_consistent(self):
        att = _attestation(_returns())
        assert att.attestation_hash
        assert att.is_internally_consistent()

    def test_source_normalised_to_value(self):
        att = _attestation(_returns())
        assert att.source == "exchange_custody"

    def test_json_round_trip_preserves_hash(self):
        att = _attestation(_returns())
        restored = SourceAttestation.from_json(att.to_json())
        assert restored == att
        assert restored.is_internally_consistent()

    def test_reissuing_same_facts_is_stable(self):
        returns = _returns()
        first = _attestation(returns)
        second = _attestation(returns)
        assert first.attestation_hash == second.attestation_hash

    def test_tampering_breaks_consistency(self):
        att = _attestation(_returns())
        forged = SourceAttestation(
            source=att.source,
            account_ref=att.account_ref,
            period_start=att.period_start,
            period_end=att.period_end,
            ingested_at=att.ingested_at,
            input_digest="0" * 64,
            attester=att.attester,
            attestation_hash=att.attestation_hash,
        )
        assert not forged.is_internally_consistent()

    def test_independent_source_is_firewall_clean(self):
        assert _attestation(_returns()).is_firewall_clean()

    def test_manager_submitted_is_not_firewall_clean(self):
        att = _attestation(_returns(), source=DataSource.MANAGER_SUBMITTED)
        assert not att.is_firewall_clean()


class TestAssertFirewall:
    def test_passes_for_independent_source(self):
        assert_firewall(_attestation(_returns()))

    def test_raises_for_manager_submitted(self):
        att = _attestation(_returns(), source=DataSource.MANAGER_SUBMITTED)
        with pytest.raises(FirewallError):
            assert_firewall(att)

    def test_raises_for_tampered_attestation(self):
        att = _attestation(_returns())
        forged = SourceAttestation(
            source=att.source,
            account_ref="acct-OTHER",
            period_start=att.period_start,
            period_end=att.period_end,
            ingested_at=att.ingested_at,
            input_digest=att.input_digest,
            attester=att.attester,
            attestation_hash=att.attestation_hash,
        )
        with pytest.raises(FirewallError):
            assert_firewall(forged)


class TestAttestedScore:
    def test_clean_score_verifies(self):
        returns = _returns()
        result = compute_score(returns, n_trials=20)
        attested = AttestedScore(result.artifact, _attestation(returns))
        report = attested.verify(returns)
        assert report.ok
        assert report.firewall_clean
        assert report.digests_match
        assert report.score_reproduces

    def test_manager_submitted_fails_firewall_only(self):
        returns = _returns()
        result = compute_score(returns, n_trials=20)
        att = _attestation(returns, source=DataSource.MANAGER_SUBMITTED)
        report = AttestedScore(result.artifact, att).verify(returns)
        # The score is real and reproducible; it simply is not firewall-clean.
        assert report.score_reproduces
        assert report.digests_match
        assert not report.firewall_clean
        assert not report.ok

    def test_swapped_returns_fail_digest_and_reproduction(self):
        returns = _returns()
        other = _returns(seed=99)
        result = compute_score(returns, n_trials=20)
        attested = AttestedScore(result.artifact, _attestation(returns))
        report = attested.verify(other)
        assert not report.digests_match
        assert not report.score_reproduces
        assert not report.ok

    def test_attestation_for_wrong_series_fails_digest_match(self):
        returns = _returns()
        wrong = _returns(seed=42)
        result = compute_score(returns, n_trials=20)
        # Attestation digests a different series than the artifact scored.
        attested = AttestedScore(result.artifact, _attestation(wrong))
        report = attested.verify(returns)
        assert not report.digests_match
        assert not report.ok
