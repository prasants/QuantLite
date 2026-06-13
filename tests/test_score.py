"""Tests for the QuantLite Score (quantlite.score)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from quantlite.score import (
    COMPONENT_WEIGHTS,
    METHODOLOGY_VERSION,
    ScoreArtifact,
    compute_score,
    input_digest,
    validate_track_record,
    verify_artifact,
)


def _solid_returns(n: int = 756, seed: int = 7) -> np.ndarray:
    """A respectable daily track record: positive drift, realistic vol."""
    rng = np.random.RandomState(seed)
    return rng.normal(0.0006, 0.01, n)


def _poor_returns(n: int = 756, seed: int = 7) -> np.ndarray:
    """A losing track record."""
    rng = np.random.RandomState(seed)
    return rng.normal(-0.0008, 0.02, n)


class TestComputeScore:
    def test_score_in_range(self):
        result = compute_score(_solid_returns())
        assert 0.0 <= result.score <= 100.0
        assert result.grade in {"A+", "A", "B", "C", "D", "F"}

    def test_components_complete_and_bounded(self):
        result = compute_score(_solid_returns())
        assert set(result.components) == set(COMPONENT_WEIGHTS)
        for value in result.components.values():
            assert 0.0 <= value <= 100.0

    def test_good_strategy_outscores_bad(self):
        good = compute_score(_solid_returns())
        bad = compute_score(_poor_returns())
        assert good.score > bad.score

    def test_more_trials_lowers_skill(self):
        returns = _solid_returns()
        few = compute_score(returns, n_trials=1)
        many = compute_score(returns, n_trials=100)
        assert many.components["skill"] <= few.components["skill"]

    def test_rejects_short_series(self):
        with pytest.raises(ValueError, match="at least 30"):
            compute_score([0.01] * 10)

    def test_rejects_nan(self):
        returns = _solid_returns()
        returns[5] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            compute_score(returns)

    def test_rejects_bad_n_trials(self):
        with pytest.raises(ValueError, match="n_trials"):
            compute_score(_solid_returns(), n_trials=0)

    def test_metrics_present(self):
        result = compute_score(_solid_returns())
        for key in ("sharpe_ratio", "deflated_sharpe_ratio", "max_drawdown",
                    "cvar_95", "bootstrap_sharpe_p5"):
            assert key in result.metrics


class TestDeterminism:
    def test_identical_inputs_identical_artifacts(self):
        returns = _solid_returns()
        a = compute_score(returns, n_trials=20)
        b = compute_score(returns, n_trials=20)
        assert a.score == b.score
        assert a.components == b.components
        assert a.artifact.content_hash == b.artifact.content_hash

    def test_different_seed_different_robustness(self):
        returns = _solid_returns()
        a = compute_score(returns, seed=1)
        b = compute_score(returns, seed=2)
        # Bootstrap percentile shifts with the seed; hash must too.
        assert a.artifact.content_hash != b.artifact.content_hash

    def test_input_digest_platform_canonical(self):
        returns = [0.01, -0.02, 0.005]
        assert input_digest(returns) == input_digest(np.asarray(returns))


class TestIntegrityFlags:
    def test_clean_record_has_no_flags(self):
        flags = validate_track_record(_solid_returns())
        assert flags == ()

    def test_short_record_critical(self):
        flags = validate_track_record(_solid_returns(60))
        codes = {(f.code, f.severity) for f in flags}
        assert ("short_record", "critical") in codes

    def test_smoothing_detected(self):
        rng = np.random.RandomState(3)
        noise = rng.normal(0.0005, 0.01, 756)
        smoothed = np.empty_like(noise)
        smoothed[0] = noise[0]
        for i in range(1, len(noise)):
            smoothed[i] = 0.6 * smoothed[i - 1] + 0.4 * noise[i]
        codes = {f.code for f in validate_track_record(smoothed)}
        assert "excessive_smoothing" in codes

    def test_zero_variance_critical(self):
        flags = validate_track_record(np.full(300, 0.001))
        assert flags[0].code == "zero_variance"
        assert flags[0].severity == "critical"

    def test_short_volatility_signature(self):
        # Steady small wins, rare large losses: high win rate, negative skew.
        rng = np.random.RandomState(11)
        returns = np.full(756, 0.002)
        crash_days = rng.choice(756, 20, replace=False)
        returns[crash_days] = -0.06
        codes = {f.code for f in validate_track_record(returns)}
        assert "short_volatility_signature" in codes

    def test_critical_flag_caps_score(self):
        # Excellent but short record: score must be capped at 40.
        result = compute_score(_solid_returns(60))
        assert result.score <= 40.0
        assert any(f.severity == "critical" for f in result.flags)


class TestArtifact:
    def test_roundtrip_json(self):
        result = compute_score(_solid_returns(), n_trials=20)
        restored = ScoreArtifact.from_json(result.artifact.to_json())
        assert restored == result.artifact
        assert restored.is_internally_consistent()

    def test_verify_authentic_artifact(self):
        returns = _solid_returns()
        result = compute_score(returns, n_trials=20)
        assert verify_artifact(result.artifact, returns)
        assert verify_artifact(result.artifact.to_json(), returns)

    def test_verify_rejects_wrong_returns(self):
        returns = _solid_returns()
        result = compute_score(returns, n_trials=20)
        assert not verify_artifact(result.artifact, _poor_returns())

    def test_verify_rejects_tampered_score(self):
        returns = _solid_returns()
        result = compute_score(returns, n_trials=20)
        payload = json.loads(result.artifact.to_json())
        payload["score"] = 99.9
        payload["grade"] = "A+"
        tampered = json.dumps(payload)
        assert not verify_artifact(tampered, returns)

    def test_content_hash_excludes_timestamp(self):
        returns = _solid_returns()
        a = compute_score(returns).artifact
        b = compute_score(returns).artifact
        # created_at may differ; the content hash must not.
        assert a.content_hash == b.content_hash

    def test_methodology_version_recorded(self):
        result = compute_score(_solid_returns())
        assert result.artifact.methodology_version == METHODOLOGY_VERSION
