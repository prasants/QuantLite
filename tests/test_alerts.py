"""Tests for the alert system."""

from __future__ import annotations

import pytest

from quantlite.alerts import (
    Alert,
    AlertManager,
    AlertRule,
    AlertStatus,
    ThresholdDirection,
)


class TestAlertManager:
    def test_add_rule(self):
        mgr = AlertManager()
        rule = mgr.add_rule("BTC-USD", condition="regime_change")
        assert isinstance(rule, AlertRule)
        assert rule.metric == "BTC-USD"
        assert rule.condition == "regime_change"
        assert rule.name in mgr.rules

    def test_add_threshold(self):
        mgr = AlertManager()
        rule = mgr.add_threshold("portfolio_var", threshold=0.05, direction="above")
        assert rule.threshold == 0.05
        assert rule.direction == ThresholdDirection.ABOVE

    def test_remove_rule(self):
        mgr = AlertManager()
        mgr.add_rule("BTC-USD", name="test_rule")
        mgr.remove_rule("test_rule")
        assert "test_rule" not in mgr.rules

    def test_remove_nonexistent_raises(self):
        mgr = AlertManager()
        with pytest.raises(KeyError):
            mgr.remove_rule("nope")

    def test_enable_disable(self):
        mgr = AlertManager()
        mgr.add_rule("BTC-USD", name="r1")
        mgr.disable_rule("r1")
        assert not mgr.rules["r1"].enabled
        mgr.enable_rule("r1")
        assert mgr.rules["r1"].enabled

    def test_regime_change_fires(self):
        mgr = AlertManager()
        received = []
        mgr.add_rule(
            "BTC-USD",
            condition="regime_change",
            callback=lambda a: received.append(a),
            cooldown_s=0,
        )

        alerts = mgr.check("BTC-USD", regime=1, previous_regime=0)
        assert len(alerts) == 1
        assert alerts[0].message.startswith("Regime change")
        assert len(received) == 1

    def test_regime_change_same_regime_no_fire(self):
        mgr = AlertManager()
        mgr.add_rule("BTC-USD", condition="regime_change", cooldown_s=0)
        alerts = mgr.check("BTC-USD", regime=1, previous_regime=1)
        assert len(alerts) == 0

    def test_threshold_above(self):
        mgr = AlertManager()
        mgr.add_threshold("var", threshold=0.05, direction="above", cooldown_s=0)
        alerts = mgr.check("var", value=0.06)
        assert len(alerts) == 1

    def test_threshold_above_no_fire(self):
        mgr = AlertManager()
        mgr.add_threshold("var", threshold=0.05, direction="above", cooldown_s=0)
        alerts = mgr.check("var", value=0.04)
        assert len(alerts) == 0

    def test_threshold_below(self):
        mgr = AlertManager()
        mgr.add_threshold("price", threshold=100, direction="below", cooldown_s=0)
        alerts = mgr.check("price", value=95)
        assert len(alerts) == 1

    def test_threshold_cross(self):
        mgr = AlertManager()
        rule = mgr.add_threshold(
            "price", threshold=100, direction="cross", cooldown_s=0
        )
        # Set up initial value below threshold
        rule._last_value = 99

        alerts = mgr.check("price", value=101)
        assert len(alerts) == 1

    def test_cooldown(self):
        mgr = AlertManager()
        mgr.add_threshold("var", threshold=0.05, direction="above", cooldown_s=60)
        mgr.check("var", value=0.06)
        # Second check within cooldown should not fire
        alerts = mgr.check("var", value=0.07)
        assert len(alerts) == 0

    def test_disabled_rule_no_fire(self):
        mgr = AlertManager()
        mgr.add_rule("BTC-USD", name="r1", condition="regime_change", cooldown_s=0)
        mgr.disable_rule("r1")
        alerts = mgr.check("BTC-USD", regime=1, previous_regime=0)
        assert len(alerts) == 0

    def test_history(self):
        mgr = AlertManager()
        mgr.add_threshold("var", threshold=0.05, direction="above", cooldown_s=0)
        mgr.check("var", value=0.06)
        mgr.check("var", value=0.07)
        assert len(mgr.history) == 2

    def test_clear_history(self):
        mgr = AlertManager()
        mgr.add_threshold("var", threshold=0.05, direction="above", cooldown_s=0)
        mgr.check("var", value=0.06)
        mgr.clear_history()
        assert len(mgr.history) == 0

    def test_global_callback(self):
        mgr = AlertManager()
        received = []
        mgr.on_alert(lambda a: received.append(a))
        mgr.add_threshold("var", threshold=0.05, direction="above", cooldown_s=0)
        mgr.check("var", value=0.06)
        assert len(received) == 1

    def test_check_many(self):
        mgr = AlertManager()
        mgr.add_threshold("var", threshold=0.05, direction="above", cooldown_s=0)
        mgr.add_rule("BTC-USD", condition="regime_change", cooldown_s=0)
        alerts = mgr.check_many([
            {"metric": "var", "value": 0.06},
            {"metric": "BTC-USD", "regime": 1, "previous_regime": 0},
        ])
        assert len(alerts) == 2

    def test_max_history(self):
        mgr = AlertManager(max_history=5)
        mgr.add_threshold("var", threshold=0.05, direction="above", cooldown_s=0)
        for i in range(10):
            mgr.check("var", value=0.06 + i * 0.001)
        assert len(mgr.history) <= 5


class TestAlert:
    def test_defaults(self):
        a = Alert(rule_name="r1", metric="BTC-USD", message="test")
        assert a.status == AlertStatus.TRIGGERED
        assert a.timestamp > 0
        assert a.metadata == {}
