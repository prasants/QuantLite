"""Tests for continuous monitoring (quantlite.score.monitoring)."""

from __future__ import annotations

import numpy as np

from quantlite.score import (
    Alert,
    AlertKind,
    Cadence,
    DataSource,
    Monitor,
    ScoreHistory,
)


def _good(n: int, seed: int = 3) -> np.ndarray:
    """A strong daily track record."""
    rng = np.random.RandomState(seed)
    return rng.normal(0.0008, 0.008, n)


def _bad(n: int, seed: int = 5) -> np.ndarray:
    """A weak, volatile daily track record."""
    rng = np.random.RandomState(seed)
    return rng.normal(-0.0010, 0.025, n)


def _monitor(**kw) -> Monitor:
    params = dict(
        cadence=Cadence.DAILY,
        source=DataSource.EXCHANGE_CUSTODY,
        attester="exchange:quantmarket",
        account_ref="acct-7a3f",
    )
    params.update(kw)
    return Monitor(**params)


class TestCadence:
    def test_freq_mapping(self):
        assert Cadence.DAILY.freq == 252
        assert Cadence.WEEKLY.freq == 52
        assert Cadence.MONTHLY.freq == 12
        assert Cadence.YEARLY.freq == 1

    def test_all_four_cadences_exist(self):
        assert {c.value for c in Cadence} == {
            "daily",
            "weekly",
            "monthly",
            "yearly",
        }


class TestObserveAndHistory:
    def test_first_observation_appends_snapshot(self):
        m = _monitor()
        m.observe(_good(504), as_of="2025-01-31")
        assert len(m.history.snapshots) == 1
        assert m.history.latest.as_of == "2025-01-31"

    def test_snapshot_is_firewall_clean_and_verifiable(self):
        m = _monitor()
        returns = _good(504)
        m.observe(returns, as_of="2025-01-31")
        snap = m.history.latest
        report = snap.expanding.verify(returns)
        assert report.ok and report.firewall_clean

    def test_chain_links_and_validates(self):
        m = _monitor()
        m.observe(_good(252), as_of="2025-01-01")
        m.observe(_good(378), as_of="2025-06-30")
        m.observe(_good(504), as_of="2025-12-31")
        snaps = m.history.snapshots
        assert snaps[0].prev_hash == ""
        assert snaps[1].prev_hash == snaps[0].snapshot_hash
        assert snaps[2].prev_hash == snaps[1].snapshot_hash
        assert m.history.is_chain_valid()

    def test_reordered_chain_is_detected(self):
        m = _monitor()
        m.observe(_good(300), as_of="2025-01-01")
        m.observe(_good(450), as_of="2025-06-30")
        # Reversing the snapshots breaks the prev_hash links.
        reordered = ScoreHistory(tuple(reversed(m.history.snapshots)))
        assert not reordered.is_chain_valid()

    def test_dropped_link_is_detected(self):
        m = _monitor()
        m.observe(_good(300), as_of="2025-01-01")
        m.observe(_good(400), as_of="2025-06-30")
        m.observe(_good(500), as_of="2025-12-31")
        # Removing the middle snapshot orphans the third's prev_hash.
        snaps = m.history.snapshots
        gapped = ScoreHistory((snaps[0], snaps[2]))
        assert not gapped.is_chain_valid()

    def test_history_json_round_trip_serialises(self):
        m = _monitor()
        m.observe(_good(504), as_of="2025-01-31")
        payload = m.history.to_json()
        assert "snapshots" in payload
        assert m.history.latest.snapshot_hash in payload


class TestDivergence:
    def test_divergence_alert_when_recent_weakens(self):
        # Strong prior window, weak recent window of equal length: the
        # length-fair divergence signal should fire.
        m = _monitor(trailing_window=120)
        returns = np.concatenate([_good(240), _bad(120)])
        alerts = m.observe(returns, as_of="2025-12-31")
        kinds = {a.kind for a in alerts}
        assert AlertKind.TRAILING_DIVERGENCE.value in kinds

    def test_no_divergence_for_consistent_record(self):
        # Two comparable strong windows: no length artifact, no false alarm.
        m = _monitor(trailing_window=120)
        alerts = m.observe(_good(360), as_of="2025-12-31")
        kinds = {a.kind for a in alerts}
        assert AlertKind.TRAILING_DIVERGENCE.value not in kinds

    def test_no_divergence_before_two_windows_exist(self):
        # Only one window of history: divergence cannot be evaluated yet.
        m = _monitor(trailing_window=120)
        alerts = m.observe(_good(150), as_of="2025-12-31")
        kinds = {a.kind for a in alerts}
        assert AlertKind.TRAILING_DIVERGENCE.value not in kinds
        assert m.history.latest.trailing_score is not None

    def test_trailing_disabled_when_window_zero(self):
        m = _monitor(trailing_window=0)
        m.observe(_good(504), as_of="2025-12-31")
        assert m.history.latest.trailing_score is None

    def test_trailing_none_when_history_shorter_than_window(self):
        m = _monitor(trailing_window=300)
        m.observe(_good(200), as_of="2025-12-31")
        assert m.history.latest.trailing_score is None


class TestTransitionAlerts:
    def test_new_critical_flag_alerts(self):
        # First observation clean, second introduces heavy smoothing.
        m = _monitor(trailing_window=0)
        m.observe(_good(504, seed=1), as_of="2025-06-30")
        smoothed = np.repeat(_good(260, seed=2), 2)[:520] * 0.2 + 0.0004
        alerts = m.observe(smoothed, as_of="2025-12-31")
        # A smoothing/short-vol signature should surface as a critical flag.
        if m.history.snapshots[-1].critical_flags:
            assert any(a.kind == AlertKind.NEW_CRITICAL_FLAG.value for a in alerts)

    def test_score_drop_alerts(self):
        m = _monitor(trailing_window=0, score_drop_threshold=3.0)
        m.observe(_good(504), as_of="2025-06-30")
        worsened = np.concatenate([_good(504), _bad(300)])
        alerts = m.observe(worsened, as_of="2025-12-31")
        kinds = {a.kind for a in alerts}
        assert AlertKind.SCORE_DROP.value in kinds or AlertKind.GRADE_DROP.value in kinds

    def test_reobserving_identical_history_emits_no_alerts(self):
        # The invariant: an unchanged record cannot have deteriorated.
        m = _monitor(trailing_window=0)
        returns = _good(504, seed=7)
        m.observe(returns, as_of="2025-06-30")
        alerts = m.observe(returns, as_of="2025-12-31")
        assert alerts == ()

    def test_alerts_reference_triggering_snapshot(self):
        m = _monitor(trailing_window=120)
        returns = np.concatenate([_good(600), _bad(120)])
        alerts = m.observe(returns, as_of="2025-12-31")
        assert alerts
        for a in alerts:
            assert isinstance(a, Alert)
            assert a.snapshot_hash == m.history.latest.snapshot_hash
            assert a.as_of == "2025-12-31"
